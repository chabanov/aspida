---------------------------------------------------------------------
-- LLM_Qwen body — Qwen 3.5 MoE model loader (full implementation)
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Strings;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Unchecked_Deallocation;
with Ada.Exceptions;
with System;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Dequant; use LLM_Dequant;
with LLM_Tensor;  use LLM_Tensor;
with LLM_SSM;
with LLM_MoE;
with LLM_Qwen_Blk; use LLM_Qwen_Blk;

package body LLM_Qwen is

   use Ada.Strings.Fixed;

   function "=" (Left, Right : Qwen_Model) return Boolean is
   begin
      return Left.N_Blocks = Right.N_Blocks
        and then Left.Vocab_Sz = Right.Vocab_Sz
        and then Left.Model_Dim = Right.Model_Dim;
   end "=";

   function Img (N : Integer) return String is
   begin
      return Trim (Integer'Image (N), Ada.Strings.Both);
   end Img;

   --------------------------------------------------------------------
   -- Load model from GGUF
   --------------------------------------------------------------------

   function Load (Path : String) return Qwen_Model is
      G    : GGUF_File;
      M    : Qwen_Model;
      Dim  : Integer := 2048;
      N_Layers : Integer := 36;

      function L (Name : String) return Tensor is
         Info : constant Tensor_Info := Find_Tensor (G, Name);
         Size : constant Natural := Natural (Tensor_Byte_Size (Info));
         type Raw_Access is access String;
         procedure Free is new Ada.Unchecked_Deallocation (String, Raw_Access);
         Raw  : Raw_Access := new String (1 .. Size);
      begin
         Read_Tensor_Raw (G, Info, Raw.all'Address, Size);
         return Result : constant Tensor := Dequantize (Info, Raw.all) do
            Free (Raw);
         end return;
      exception
         when Constraint_Error =>
            Ada.Text_IO.Put_Line ("  WARNING: tensor not found: " & Name);
            return New_Tensor ([1, 1]);
         when E : others =>
            Ada.Text_IO.Put_Line ("  ERROR loading tensor " & Name & ": "
              & Ada.Exceptions.Exception_Message (E));
            return New_Tensor ([1, 1]);
      end L;

      -- Try to load optional tensor, return zero tensor if missing
      function LO (Name : String; Default_Shape : Dims) return Tensor is
      begin
         return L (Name);
      exception
         when others =>
            return New_Tensor (Default_Shape);
      end LO;

   begin
      Ada.Text_IO.Put_Line ("Loading Qwen model from " & Path & " ...");
      Open (G, Path);

      if not Is_Open (G) then
         Ada.Text_IO.Put_Line ("ERROR: cannot open GGUF file");
         return M;
      end if;

      -- Read config from metadata (try qwen35moe.* first, fallback to qwen2.*)
      declare
         function Read_Meta_Int (Key : String) return Integer is
            Val : constant String := Metadata (G, "qwen35moe." & Key);
         begin
            if Val /= "" then
               return Integer'Value (Val);
            end if;
            -- Fallback: try qwen2 prefix
            return Integer'Value (Metadata (G, "qwen2." & Key));
         end Read_Meta_Int;
      begin
         -- Extract key suffix after the architecture prefix
         Dim := Read_Meta_Int ("embedding_length");
         N_Layers := Read_Meta_Int ("block_count");
         M.N_Heads := Read_Meta_Int ("attention.head_count");
         M.Ctx_Len := Read_Meta_Int ("context_length");

         -- vocab_size is not in metadata for qwen35moe — derive it from
         -- the token embedding tensor shape: token_embd.weight = [dim, vocab]
         begin
            M.Vocab_Sz := Read_Meta_Int ("vocab_size");
         exception
            when Constraint_Error =>
               declare
                  TE : constant Tensor_Info := Find_Tensor (G, "token_embd.weight");
               begin
                  -- Dims(1) = embedding_length, Dims(2) = vocab_size
                  if TE.N_Dims >= 2 then
                     M.Vocab_Sz := Integer (TE.Dims (2));
                  else
                     M.Vocab_Sz := Integer (TE.Dims (1));
                  end if;
               end;
         end;

         begin
            M.N_KV_Heads := Read_Meta_Int ("attention.head_count_kv");
         exception
            when others => M.N_KV_Heads := 4;
         end;
      end;

      -- Allocate block array
      M.Blocks := new Block_Array (1 .. N_Layers);

      Ada.Text_IO.Put_Line ("  dim=" & Img (Dim) & " layers=" & Img (N_Layers) &
        " heads=" & Img (M.N_Heads) & " vocab=" & Img (M.Vocab_Sz));

      -- Token embeddings [vocab, dim] — ~2GB FP32, slow
      Ada.Text_IO.Put_Line ("  loading token_embd (2GB)...");
      Ada.Text_IO.Flush;
      M.Token_Emb := L ("token_embd.weight");
      Ada.Text_IO.Put_Line ("  token_embd loaded.");

      -- Final norm [dim]
      M.Final_Norm := L ("output_norm.weight");

      -- LM head [dim, vocab]  (transposed in Qwen convention) — ~1GB, slow
      Ada.Text_IO.Put_Line ("  loading output.weight (1GB)...");
      Ada.Text_IO.Flush;
      M.LM_Head := L ("output.weight");
      Ada.Text_IO.Put_Line ("  output.weight loaded.");

      Ada.Text_IO.Put_Line ("  Loading transformer blocks...");
      Ada.Text_IO.Flush;

      -- Load all transformer blocks
      for I in 0 .. N_Layers - 1 loop
         declare
            Pre : constant String := "blk." & Img (I) & ".";
            Is_Full_Attn : constant Boolean := (I mod 4) = 0;
         begin
            Ada.Text_IO.Put_Line ("  loading block" & Img (I) &
              (if Is_Full_Attn then " [full-attn]" else " [SSM]"));
            Ada.Text_IO.Flush;

            declare
               -- Only norms are guaranteed on every layer
               Attn_Norm : constant Tensor := L (Pre & "attn_norm.weight");
               Post_Attn_Norm : constant Tensor := L (Pre & "post_attention_norm.weight");

               -- Everything else: try L first, fallback to LO (empty tensor)
               function Try_Load (Name : String; Row, Col : Integer) return Tensor is
               begin
                  return L (Name);
               exception
                  when others =>
                     return New_Tensor ([Row, Col]);
               end Try_Load;

               QKV       : constant Tensor := Try_Load (Pre & "attn_qkv.weight", 1, 1);
               Attn_Gate : constant Tensor := Try_Load (Pre & "attn_gate.weight", 1, Dim);
               OW        : constant Tensor := Try_Load (Pre & "attn_output.weight", 1, 1);

               C1D   : constant Tensor := Try_Load (Pre & "ssm_conv1d.weight", 1, 1);
               SA    : constant Tensor := Try_Load (Pre & "ssm_a.weight", 1, 1);
               SDT   : constant Tensor := Try_Load (Pre & "ssm_dt.weight", 1, 1);
               SN    : constant Tensor := Try_Load (Pre & "ssm_norm.weight", 1, Dim);
               SO    : constant Tensor := Try_Load (Pre & "ssm_out.weight", 1, Dim);
               SAlpha: constant Tensor := Try_Load (Pre & "ssm_alpha.weight", 1, 1);
               SBeta : constant Tensor := Try_Load (Pre & "ssm_beta.weight", 1, 1);

               Ssm_P : LLM_SSM.SSM_Params;
            begin
               if not Is_Full_Attn then
                  Ssm_P := LLM_SSM.Create_SSM (C1D, SA, SDT, SN, SO, SAlpha, SBeta);
               end if;

               -- MoE layer (present on all blocks)
               declare
                  Moe_L : constant LLM_MoE.MoE_Layer := LLM_MoE.Create_MoE (
                     Try_Load (Pre & "ffn_gate_inp.weight", 1, Dim),
                     Try_Load (Pre & "ffn_gate_exps.weight", 1, 1),
                     Try_Load (Pre & "ffn_down_exps.weight", 1, 1),
                     Try_Load (Pre & "ffn_up_exps.weight", 1, 1),
                     Try_Load (Pre & "ffn_gate_shexp.weight", 1, Dim),
                     Try_Load (Pre & "ffn_down_shexp.weight", 1, 1),
                     Try_Load (Pre & "ffn_up_shexp.weight", 1, 1),
                     Try_Load (Pre & "ffn_gate_inp_shexp.weight", 1, Dim),
                     256);
               begin
                  M.Blocks (I + 1) := new LLM_Qwen_Blk.Qwen_Block'(
                     LLM_Qwen_Blk.Create_Qwen_Block (
                        QKV, Attn_Gate, OW,
                        Attn_Norm, Post_Attn_Norm,
                        Ssm_P, Moe_L,
                        Is_Full_Attn, Dim, M.N_Heads, M.N_KV_Heads));
               end;
            end;
         end;
      end loop;

      Ada.Text_IO.Put_Line ("  DEBUG: loop finished, closing GGUF...");

      Close (G);
      Ada.Text_IO.Put_Line ("  Qwen model loaded: " & Img (M.N_Blocks) & " blocks.");
      return M;
   end Load;

   --------------------------------------------------------------------
   -- Forward pass (stub — to be completed)
   --------------------------------------------------------------------

   function Forward (M : Qwen_Model; Token_Ids : Tensor) return Tensor is
      Dim : constant Integer := M.Model_Dim;
      Seq_Len : constant Integer := Numel (Token_Ids);
      H : Tensor := New_Tensor ((1, Dim));
   begin
      if Seq_Len < 1 then
         return New_Tensor ((1, M.Vocab_Sz));
      end if;

      -- Embed tokens: average over sequence (simplified)
      for Pos in 1 .. Seq_Len loop
         declare
            Tid : constant Integer := Integer (Get_Flat (Token_Ids, Pos));
         begin
            if Tid >= 1 and Tid <= M.Vocab_Sz then
               for D in 1 .. Dim loop
                  Set_Flat (H, D,
                    Get_Flat (H, D) + Get (M.Token_Emb, (Tid, D)));
               end loop;
            end if;
         end;
      end loop;

      -- Average
      declare
         Scale : constant Float := 1.0 / Float (Seq_Len);
      begin
         for I in 1 .. Numel (H) loop
            Set_Flat (H, I, Get_Flat (H, I) * Scale);
         end loop;
      end;

      -- Pass through blocks
      for I in 1 .. M.N_Blocks loop
         H := LLM_Qwen_Blk.Forward (M.Blocks (I).all, H);
      end loop;

      -- Final norm: element-wise multiply
      for I in 1 .. Dim loop
         Set_Flat (H, I, Get_Flat (H, I) * Get_Flat (M.Final_Norm, I));
      end loop;

      -- LM head: H [1, dim] @ LM_Head [dim, vocab] → [1, vocab]
      return Matmul (H, M.LM_Head);
   end Forward;

   --------------------------------------------------------------------
   -- Generate (autoregressive loop)
   --------------------------------------------------------------------

   function Generate (M : Qwen_Model; Prompt : String; Max_New_Tokens : Integer := 128) return String is
      Result : String (1 .. 4096);
      Result_Len : Integer := 0;
      Max_Len : constant Integer := 4096;
      Ctx_Len : constant Integer := M.Ctx_Len;
      Context : Tensor := New_Tensor ([1, 1]);
   begin
      -- Tokenize prompt (character-level for now)
      if Prompt'Length > 0 then
         Context := New_Tensor ((1, Prompt'Length));
         for I in 1 .. Prompt'Length loop
            Set_Flat (Context, I, Float (Character'Pos (Prompt (I))));
         end loop;
      else
         Context := New_Tensor ((1, 1));
         Set_Flat (Context, 1, 0.0);
      end if;

      -- Copy prompt to output
      for I in Prompt'Range loop
         Result_Len := Result_Len + 1;
         Result (Result_Len) := Prompt (I);
      end loop;

      -- Autoregressive generation
      for Step in 1 .. Max_New_Tokens loop
         -- Truncate context if too long
         declare
            Ctx_Size : constant Integer := Numel (Context);
            Use_Len : Integer := Ctx_Size;
            Trimmed : Tensor;
         begin
            if Use_Len > Ctx_Len then
               Use_Len := Ctx_Len;
               Trimmed := New_Tensor ((1, Use_Len));
               for I in 1 .. Use_Len loop
                  Set_Flat (Trimmed, I, Get_Flat (Context, Ctx_Size - Use_Len + I));
               end loop;
               Context := Trimmed;
            end if;
         end;

         declare
            Logits : constant Tensor := Forward (M, Context);
            Best_Tok : Integer := 1;
            Best_Score : Float := Float'First;
         begin
            -- Argmax
            for I in 1 .. Numel (Logits) loop
               declare
                  Score : constant Float := Get_Flat (Logits, I);
               begin
                  if Score > Best_Score then
                     Best_Score := Score;
                     Best_Tok := I;
                  end if;
               end;
            end loop;

            -- Detokenize: ASCII printable range
            if Best_Tok >= 32 and Best_Tok <= 126 then
               if Result_Len < Max_Len then
                  Result_Len := Result_Len + 1;
                  Result (Result_Len) := Character'Val (Best_Tok);
               end if;
            elsif Best_Tok = 0 or Best_Tok >= M.Vocab_Sz then
               exit;  -- EOS
            end if;

            -- Append to context
            declare
               Old_Len : constant Integer := Numel (Context);
               New_Ctx : Tensor := New_Tensor ((1, Old_Len + 1));
            begin
               for I in 1 .. Old_Len loop
                  Set_Flat (New_Ctx, I, Get_Flat (Context, I));
               end loop;
               Set_Flat (New_Ctx, Old_Len + 1, Float (Best_Tok));
               Context := New_Ctx;
            end;
         end;
      end loop;

      return Result (1 .. Result_Len);
   end Generate;

   -- Parameter count (total FP32 params after dequantization)
   -- Safe addition: only add if tensor has elements (skip uninit tensors)
   function Safe_N (T : Tensor) return Long_Long_Integer is
   begin
      if Numel (T) > 0 then
         return Long_Long_Integer (Numel (T));
      else
         return 0;
      end if;
   end Safe_N;

   function Param_Count (M : Qwen_Model) return Long_Long_Integer is
      C : Long_Long_Integer := 0;
   begin
      C := Safe_N (M.Token_Emb);
      C := C + Safe_N (M.LM_Head);
      C := C + Safe_N (M.Final_Norm);
      for I in 1 .. M.N_Blocks loop
         declare
            B : LLM_Qwen_Blk.Qwen_Block renames M.Blocks (I).all;
         begin
            C := C + Safe_N (B.QKV_W)
                     + Safe_N (B.Attn_Gate_W)
                     + Safe_N (B.O_W)
                     + Safe_N (B.Attn_Norm_W)
                     + Safe_N (B.Post_Attn_Norm_W);
            if not B.Is_Full_Attn then
               C := C + Safe_N (B.SSM.Conv_Weight)
                       + Safe_N (B.SSM.A_Diag)
                       + Safe_N (B.SSM.Dt_Bias)
                       + Safe_N (B.SSM.Gamma)
                       + Safe_N (B.SSM.Out_Weight)
                       + Safe_N (B.SSM.Alpha_W)
                       + Safe_N (B.SSM.Beta_W);
            end if;
            C := C + Safe_N (B.MoE.Gate_Inp_W)
                    + Safe_N (B.MoE.Gate_Exp_W)
                    + Safe_N (B.MoE.Up_W)
                    + Safe_N (B.MoE.Down_W)
                    + Safe_N (B.MoE.Shexp_Gate_W)
                    + Safe_N (B.MoE.Shexp_Up_W)
                    + Safe_N (B.MoE.Shexp_Down_W)
                    + Safe_N (B.MoE.Shexp_Gate_Inp_W);
         end;
      end loop;
      return C;
   end Param_Count;

   function Vocab_Size  (M : Qwen_Model) return Integer is (M.Vocab_Sz);
   function Context_Len (M : Qwen_Model) return Integer is (M.Ctx_Len);
   function Dim         (M : Qwen_Model) return Integer is (M.Model_Dim);
   function Block_Count (M : Qwen_Model) return Integer is (M.N_Blocks);

end LLM_Qwen;
