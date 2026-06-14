---------------------------------------------------------------------
-- LLM_Qwen body — Qwen 3.5 MoE model loader (full implementation)
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Strings;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Unchecked_Deallocation;
with Ada.Exceptions;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Dequant; use LLM_Dequant;
with LLM_Tensor;  use LLM_Tensor;
with LLM_SSM;
with LLM_MoE;
with LLM_RMSNorm;

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

      -- Persist dimensions on the model record. Without this, M.Model_Dim and
      -- M.N_Blocks stay uninitialized and Forward/Param_Count read garbage.
      M.Model_Dim := Dim;
      M.N_Blocks  := N_Layers;

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

      -- Build the tokenizer from the GGUF vocab/merges (byte-level fallback
      -- if the file has no tokenizer arrays).
      M.Tok := LLM_Tokenizer.Create;
      LLM_Tokenizer.Load_From_GGUF (M.Tok, G);
      Ada.Text_IO.Put_Line ("  tokenizer: " &
        Img (LLM_Tokenizer.Vocab_Size (M.Tok)) & " tokens.");

      Close (G);
      Ada.Text_IO.Put_Line ("  Qwen model loaded: " & Img (M.N_Blocks) & " blocks.");
      return M;
   end Load;

   --------------------------------------------------------------------
   -- Forward pass: token_ids [seq_len] → next-token logits [1, vocab]
   --------------------------------------------------------------------

   function Forward (M : Qwen_Model; Token_Ids : Tensor) return Tensor is
      Dim     : constant Integer := M.Model_Dim;
      Seq_Len : constant Integer := Numel (Token_Ids);
      H       : Tensor;
   begin
      if Seq_Len < 1 then
         return New_Tensor ([1, M.Vocab_Sz]);
      end if;

      -- Build the embedding sequence [Seq_Len, Dim]: one row per token.
      H := New_Tensor ([Seq_Len, Dim]);
      for Pos in 1 .. Seq_Len loop
         declare
            Tid : Integer := Integer (Get_Flat (Token_Ids, Pos));
         begin
            if Tid < 1 then
               Tid := 1;
            elsif Tid > M.Vocab_Sz then
               Tid := M.Vocab_Sz;
            end if;
            for D in 1 .. Dim loop
               Set (H, [Pos, D], Get (M.Token_Emb, [Tid, D]));
            end loop;
         end;
      end loop;

      -- Run the transformer blocks over the whole sequence.
      for I in 1 .. M.N_Blocks loop
         H := LLM_Qwen_Blk.Forward (M.Blocks (I).all, H);
      end loop;

      -- Final RMSNorm on the last position, then project to vocab logits.
      declare
         Last : Tensor := New_Tensor ([1, Dim]);
      begin
         for D in 1 .. Dim loop
            Set_Flat (Last, D, Get (H, [Seq_Len, D]));
         end loop;
         declare
            Normed : constant Tensor := LLM_RMSNorm.Forward (Last, M.Final_Norm);
            Logits : Tensor := New_Tensor ([1, M.Vocab_Sz]);
         begin
            -- output.weight is [vocab, dim] (row-major); logits[v] = Normed . row v
            for V in 1 .. M.Vocab_Sz loop
               declare
                  Acc : Float := 0.0;
               begin
                  for D in 1 .. Dim loop
                     Acc := Acc + Get_Flat (Normed, D) * Get (M.LM_Head, [V, D]);
                  end loop;
                  Set_Flat (Logits, V, Acc);
               end;
            end loop;
            return Logits;
         end;
      end;
   end Forward;

   --------------------------------------------------------------------
   -- Generate (autoregressive loop)
   --------------------------------------------------------------------

   function Generate (M : Qwen_Model; Prompt : String; Max_New_Tokens : Integer := 128) return String is
      Ctx_Len : constant Integer := M.Ctx_Len;
      Ids     : constant LLM_Tokenizer.Token_Array := LLM_Tokenizer.Encode (M.Tok, Prompt);
      Cap     : constant Integer := Integer'Max (1, Ids'Length + Max_New_Tokens);
      Ctx     : array (1 .. Cap) of Integer := [others => 1];  -- 1-based embed rows
      L       : Natural := 0;
      Out_Buf : Unbounded_String := To_Unbounded_String (Prompt);
   begin
      -- Seed the context with the prompt tokens as 1-based embedding rows
      -- (row = id + 1, since token ids are 0-based but tensor rows are 1-based).
      for I in Ids'Range loop
         L := L + 1;
         Ctx (L) := Ids (I) + 1;
      end loop;
      if L = 0 then
         L := 1;
         Ctx (1) := 1;
      end if;

      -- Autoregressive generation: greedy argmax decoding.
      for Step in 1 .. Max_New_Tokens loop
         declare
            Start   : constant Integer := Integer'Max (1, L - Ctx_Len + 1);
            Use_Len : constant Integer := L - Start + 1;
            Toks    : Tensor := New_Tensor ([1, Use_Len]);
         begin
            for I in 1 .. Use_Len loop
               Set_Flat (Toks, I, Float (Ctx (Start + I - 1)));
            end loop;

            declare
               Logits   : constant Tensor := Forward (M, Toks);
               Best_Row : Integer := 1;
               Best_S   : Float := Float'First;
            begin
               for I in 1 .. Numel (Logits) loop
                  if Get_Flat (Logits, I) > Best_S then
                     Best_S := Get_Flat (Logits, I);
                     Best_Row := I;
                  end if;
               end loop;

               exit when Best_Row < 1 or else Best_Row > M.Vocab_Sz;

               -- Best_Row is the 1-based embedding row; token id is Best_Row - 1.
               Append (Out_Buf, LLM_Tokenizer.Decode_One (M.Tok, Best_Row - 1));

               exit when L >= Cap;
               L := L + 1;
               Ctx (L) := Best_Row;
            end;
         end;
      end loop;

      return To_String (Out_Buf);
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
