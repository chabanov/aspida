---------------------------------------------------------------------
-- LLM_Gemma body — gemma4 (Gemma 3n E4B) loader + forward.
--
-- The forward graph (incl. the per-layer-embedding / PLE mechanism and the
-- exact scale constants) was reconstructed from llama.cpp's eval-callback
-- trace and validated against it; see the inline references.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Environment_Variables;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;
with Ada.Strings.Fixed;
with Ada.Strings;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with LLM_GGUF;    use LLM_GGUF;
with LLM_Dequant;
with LLM_Tensor;  use LLM_Tensor;
with LLM_Tokenizer;
with LLM_RMSNorm;
with LLM_RoPE;
with LLM_Weight;

package body LLM_Gemma is

   use Ada.Strings.Fixed;
   use type LLM_Qwen.Role_Kind;

   function Img (N : Integer) return String is
     (Trim (Integer'Image (N), Ada.Strings.Both));

   Dbg : constant Boolean := Ada.Environment_Variables.Exists ("LLM_GDBG");
   procedure D3 (Name : String; T : Tensor) is
   begin
      if Dbg then
         Ada.Text_IO.Put_Line ("  [" & Name & "]"
           & Float'Image (Get_Flat (T, 1)) & Float'Image (Get_Flat (T, 2))
           & Float'Image (Get_Flat (T, 3)));
      end if;
   end D3;

   type G_Block is record
      Attn_Norm, Post_Attn_Norm     : Tensor;   -- raw weights (no +1)
      Ffn_Norm, Post_Ffw_Norm       : Tensor;
      Post_Norm                     : Tensor;
      Q_Norm, K_Norm                : Tensor;
      W_Q, W_K, W_V, W_O            : LLM_Weight.Weight;
      W_Gate, W_Up, W_Down          : LLM_Weight.Weight;
      Inp_Gate, Proj                : LLM_Weight.Weight;
      Layer_Out_Scale               : Float := 1.0;
      Is_SWA                        : Boolean := True;
   end record;

   type Block_Arr is array (Positive range <>) of G_Block;
   type Block_Arr_Ptr is access Block_Arr;

   type Gemma_Model_Rec is record
      Tok_Emb       : LLM_Weight.Weight;   -- token_embd (lookup + tied output)
      --  per_layer_token_embd is >2 GiB (won't fit one Ada String): keep the
      --  GGUF open and stream the row for each token on demand.
      Gf            : GGUF_File;
      PLE_Tok_Info  : Tensor_Info;
      PLE_Row_Bytes : Natural := 0;
      PLE_Proj      : LLM_Weight.Weight;   -- per_layer_model_proj
      PLE_Proj_Norm : Tensor;              -- per_layer_proj_norm
      Out_Norm      : Tensor;
      Blocks        : Block_Arr_Ptr;
      Dim, N_Blocks, N_Heads, N_KV, Head_Dim, FFN, Vocab, Ctx : Integer := 0;
      PL_Dim         : Integer := 256;     -- per-layer input dim
      Sliding_Window : Integer := 512;
      RoPE_Glob, RoPE_SWA : LLM_RoPE.RoPE_Params;
      Tok       : LLM_Tokenizer.Tokenizer;
      Bos, Eos, SOT, EOT : Integer := -1;
   end record;

   --  rmsnorm(x) * weight (Gemma's +1 is already folded into the GGUF weights;
   --  the eval-callback graph multiplies by the weight directly).
   function GN (X, W : Tensor) return Tensor is (LLM_RMSNorm.Forward (X, W));

   --  Stream + dequantize one row of the (>2 GiB) per-layer embedding table.
   function PLE_Row (M : Gemma_Model; Tok : Integer) return Tensor is
      RI : Tensor_Info := M.PLE_Tok_Info;
      B  : aliased String (1 .. M.PLE_Row_Bytes);
   begin
      RI.N_Dims := 2;
      RI.Dims   := [M.PLE_Tok_Info.Dims (1), 1, 0, 0];
      Read_Tensor_Range (M.Gf, M.PLE_Tok_Info,
        U64 (Tok) * U64 (M.PLE_Row_Bytes), B'Address, M.PLE_Row_Bytes);
      return LLM_Dequant.Dequantize (RI, B);
   end PLE_Row;

   --  Slice [Lo .. Lo+Len-1] (1-based, flat) of T into a fresh [1, Len] tensor.
   function Slice (T : Tensor; Lo, Len : Integer) return Tensor is
      R : Tensor := New_Tensor ([1, Len]);
   begin
      for I in 1 .. Len loop Set_Flat (R, I, Get_Flat (T, Lo + I - 1)); end loop;
      return R;
   end Slice;

   --------------------------------------------------------------------
   -- Load
   --------------------------------------------------------------------

   function Load (Path : String) return Gemma_Model is
      M : constant Gemma_Model := new Gemma_Model_Rec;
      G : GGUF_File renames M.Gf;   -- opened in place; kept open after Load

      function MI (Key : String; D : Integer) return Integer is
         V : constant String := Metadata (G, "gemma4." & Key);
      begin
         return (if V = "" then D else Integer'Value (V));
      exception when others => return D; end MI;

      function MF (Key : String; D : Float) return Float is
         V : constant String := Metadata (G, "gemma4." & Key);
      begin
         return (if V = "" then D else Float'Value (V));
      exception when others => return D; end MF;

      function LQ (Name : String) return LLM_Weight.Weight is
         Info : constant Tensor_Info := Find_Tensor (G, Name);
         Size : constant Natural := Natural (Tensor_Byte_Size (Info));
         B    : constant LLM_Weight.Byte_Data := new String (1 .. Size);
      begin
         Read_Tensor_Raw (G, Info, B.all'Address, Size);
         return LLM_Weight.From_Quant (Info, B);
      exception
         when E : others =>
            raise Model_Load_Error with "weight " & Name & ": "
              & Ada.Exceptions.Exception_Message (E);
      end LQ;

      --  A 1-D norm/scale tensor as F32 (the whole tensor is "row 0").
      function LT (Name : String) return Tensor is (LLM_Weight.Get_Row (LQ (Name), 0));

      Glob_Base, SWA_Base : Float;
   begin
      Ada.Text_IO.Put_Line ("Loading Gemma (gemma4) model from " & Path & " ...");
      Open (G, Path);
      if not Is_Open (G) then
         raise Model_Load_Error with "cannot open GGUF file: " & Path;
      end if;

      M.Dim      := MI ("embedding_length", 2560);
      M.N_Blocks := MI ("block_count", 42);
      M.N_Heads  := MI ("attention.head_count", 8);
      M.N_KV     := MI ("attention.head_count_kv", 2);
      M.FFN      := MI ("feed_forward_length", 10240);
      M.Ctx      := MI ("context_length", 131072);
      M.PL_Dim   := MI ("embedding_length_per_layer_input", 256);
      M.Sliding_Window := MI ("attention.sliding_window", 512);
      Glob_Base := MF ("rope.freq_base", 1_000_000.0);
      SWA_Base  := MF ("rope.freq_base_swa", 10_000.0);

      M.Tok_Emb       := LQ ("token_embd.weight");
      declare
         I  : constant Tensor_Info := Find_Tensor (G, "per_layer_token_embd.weight");
         RI : Tensor_Info := I;
      begin
         M.PLE_Tok_Info := I;
         RI.N_Dims := 2; RI.Dims := [I.Dims (1), 1, 0, 0];
         M.PLE_Row_Bytes := Natural (Tensor_Byte_Size (RI));
      end;
      M.PLE_Proj      := LQ ("per_layer_model_proj.weight");
      M.PLE_Proj_Norm := LT ("per_layer_proj_norm.weight");
      M.Out_Norm      := LT ("output_norm.weight");
      M.Vocab         := LLM_Weight.Rows (M.Tok_Emb);
      M.Blocks        := new Block_Arr (1 .. M.N_Blocks);

      for I in 1 .. M.N_Blocks loop
         declare
            P  : constant String := "blk." & Img (I - 1) & ".";
            Bk : G_Block;
         begin
            Bk.Attn_Norm      := LT (P & "attn_norm.weight");
            Bk.Post_Attn_Norm := LT (P & "post_attention_norm.weight");
            Bk.Ffn_Norm       := LT (P & "ffn_norm.weight");
            Bk.Post_Ffw_Norm  := LT (P & "post_ffw_norm.weight");
            Bk.Post_Norm      := LT (P & "post_norm.weight");
            Bk.Q_Norm         := LT (P & "attn_q_norm.weight");
            Bk.K_Norm         := LT (P & "attn_k_norm.weight");
            Bk.W_Q := LQ (P & "attn_q.weight");
            Bk.W_K := LQ (P & "attn_k.weight");
            Bk.W_V := LQ (P & "attn_v.weight");
            Bk.W_O := LQ (P & "attn_output.weight");
            Bk.W_Gate := LQ (P & "ffn_gate.weight");
            Bk.W_Up   := LQ (P & "ffn_up.weight");
            Bk.W_Down := LQ (P & "ffn_down.weight");
            Bk.Inp_Gate := LQ (P & "inp_gate.weight");
            Bk.Proj     := LQ (P & "proj.weight");
            Bk.Layer_Out_Scale :=
              Get_Flat (LLM_Weight.Get_Row (LQ (P & "layer_output_scale.weight"), 0), 1);
            Bk.Is_SWA := (I - 1) mod 6 /= 5;     -- 5 local : 1 global
            M.Blocks (I) := Bk;
         end;
      end loop;

      M.Head_Dim := LLM_Weight.Rows (M.Blocks (1).W_Q) / M.N_Heads;
      M.RoPE_Glob := LLM_RoPE.Create_Qwen_RoPE (M.Head_Dim, Glob_Base, M.Ctx);
      M.RoPE_SWA  := LLM_RoPE.Create_Qwen_RoPE (M.Head_Dim, SWA_Base, M.Ctx);

      M.Tok := LLM_Tokenizer.Create;
      LLM_Tokenizer.Load_From_GGUF (M.Tok, G);
      begin M.Bos := Integer'Value (Metadata (G, "tokenizer.ggml.bos_token_id"));
      exception when others => M.Bos := 2; end;
      begin M.Eos := Integer'Value (Metadata (G, "tokenizer.ggml.eos_token_id"));
      exception when others => M.Eos := 1; end;
      --  This finetune uses custom turn markers <|turn> (105) / <turn|> (106).
      M.SOT := LLM_Tokenizer.Token_To_Id (M.Tok, "<|turn>");
      M.EOT := LLM_Tokenizer.Token_To_Id (M.Tok, "<turn|>");

      Ada.Text_IO.Put_Line ("  gemma4: dim=" & Img (M.Dim)
        & " layers=" & Img (M.N_Blocks) & " heads=" & Img (M.N_Heads)
        & "/" & Img (M.N_KV) & " head_dim=" & Img (M.Head_Dim)
        & " pl=" & Img (M.PL_Dim) & " vocab=" & Img (M.Vocab));
      Ada.Text_IO.Put_Line ("  NOTE: forward validated bit-exact through the V "
        & "projection; attention-output scaling is still being matched to the "
        & "reference, so generated text is not yet correct.");
      return M;   -- G (= M.Gf) stays open for on-demand PLE row reads
   end Load;

   --------------------------------------------------------------------
   -- Forward over a token sequence -> logits for the LAST position.
   --------------------------------------------------------------------

   function Forward_Logits
     (M : Gemma_Model; Ids : LLM_Tokenizer.Token_Array) return Tensor
   is
      D    : constant Integer := M.Dim;
      HD   : constant Integer := M.Head_Dim;
      NH   : constant Integer := M.N_Heads;
      NKV  : constant Integer := M.N_KV;
      PL   : constant Integer := M.PL_Dim;
      Seq  : constant Integer := Ids'Length;
      AScale  : constant Float := 1.0 / Sqrt (Float (HD));
      E_Scale : constant Float := Sqrt (Float (D));
      P_Scale : constant Float := Sqrt (Float (PL));
      Inv_D   : constant Float := 1.0 / Sqrt (Float (D));
      Inv_2   : constant Float := 1.0 / Sqrt (2.0);

      type TArr is array (Positive range <>) of Tensor;
      H   : TArr (1 .. Seq);    -- residual stream, each [1, D]
      IPL : TArr (1 .. Seq);    -- per-layer inputs, each [1, N_Blocks*PL]

      procedure Add_To (A : in out Tensor; B : Tensor) is
      begin
         for I in 1 .. D loop Set_Flat (A, I, Get_Flat (A, I) + Get_Flat (B, I)); end loop;
      end Add_To;
   begin
      --  Token + per-layer embeddings and the PLE setup (per position).
      for T in 1 .. Seq loop
         declare
            Tok    : constant Integer := Ids (Ids'First + T - 1);
            Scaled : Tensor := LLM_Weight.Get_Row (M.Tok_Emb, Tok);
            PTok   : constant Tensor := PLE_Row (M, Tok);
         begin
            for I in 1 .. D loop Set_Flat (Scaled, I, Get_Flat (Scaled, I) * E_Scale); end loop;
            H (T) := Scaled;
            IPL (T) := New_Tensor ([1, M.N_Blocks * PL]);
            declare
               --  per_layer_model_proj is applied to the SCALED embedding.
               PProj : Tensor := LLM_Weight.MatVec (M.PLE_Proj, Scaled);
            begin
               for I in 1 .. M.N_Blocks * PL loop
                  Set_Flat (PProj, I, Get_Flat (PProj, I) * Inv_D);
               end loop;
               for Lr in 0 .. M.N_Blocks - 1 loop
                  declare
                     Sel : constant Tensor := Slice (PTok, Lr * PL + 1, PL);
                     Prj : constant Tensor :=
                       GN (Slice (PProj, Lr * PL + 1, PL), M.PLE_Proj_Norm);
                  begin
                     for I in 1 .. PL loop
                        Set_Flat (IPL (T), Lr * PL + I,
                          (Get_Flat (Prj, I) + Get_Flat (Sel, I) * P_Scale) * Inv_2);
                     end loop;
                  end;
               end loop;
            end;
         end;
      end loop;
      D3 ("inp_scaled", H (1));

      for Lr in 1 .. M.N_Blocks loop
         declare
            B    : G_Block renames M.Blocks (Lr);
            RoPE : LLM_RoPE.RoPE_Params :=
              (if B.Is_SWA then M.RoPE_SWA else M.RoPE_Glob);
            Win  : constant Integer := (if B.Is_SWA then M.Sliding_Window else Seq);
            Qs, Ks, Vs : TArr (1 .. Seq);
            Attn : TArr (1 .. Seq);
         begin
            --  Q/K/V + QK-norm + RoPE.
            for T in 1 .. Seq loop
               declare
                  X : constant Tensor := GN (H (T), B.Attn_Norm);
                  Q : Tensor := LLM_Weight.MatVec (B.W_Q, X);
                  K : Tensor := LLM_Weight.MatVec (B.W_K, X);
               begin
                  if Lr = 1 and then T = Seq then
                     D3 ("attn_norm-0", X);
                     D3 ("Vcur-0", LLM_Weight.MatVec (B.W_V, X));
                  end if;
                  for Hh in 0 .. NH - 1 loop
                     declare
                        S : constant Tensor := LLM_RoPE.Apply
                          (RoPE, GN (Slice (Q, Hh * HD + 1, HD), B.Q_Norm), T - 1);
                     begin
                        for J in 1 .. HD loop Set_Flat (Q, Hh * HD + J, Get_Flat (S, J)); end loop;
                     end;
                  end loop;
                  for Hh in 0 .. NKV - 1 loop
                     declare
                        S : constant Tensor := LLM_RoPE.Apply
                          (RoPE, GN (Slice (K, Hh * HD + 1, HD), B.K_Norm), T - 1);
                     begin
                        for J in 1 .. HD loop Set_Flat (K, Hh * HD + J, Get_Flat (S, J)); end loop;
                     end;
                  end loop;
                  Qs (T) := Q; Ks (T) := K; Vs (T) := LLM_Weight.MatVec (B.W_V, X);
               end;
            end loop;

            --  Causal (+ sliding-window) attention.
            for T in 1 .. Seq loop
               declare
                  Ctx_O : Tensor := New_Tensor ([1, NH * HD]);
                  Lo : constant Integer := Integer'Max (1, T - Win + 1);
               begin
                  for Hh in 0 .. NH - 1 loop
                     declare
                        KV  : constant Integer := Hh / (NH / NKV);
                        Scr : Tensor := New_Tensor ([1, Seq]);
                        Mx  : Float := Float'First;
                        Den : Float := 0.0;
                     begin
                        for S in Lo .. T loop
                           declare Dp : Float := 0.0; begin
                              for J in 1 .. HD loop
                                 Dp := Dp + Get_Flat (Qs (T), Hh * HD + J)
                                          * Get_Flat (Ks (S), KV * HD + J);
                              end loop;
                              Set_Flat (Scr, S, Dp * AScale);
                              Mx := Float'Max (Mx, Dp * AScale);
                           end;
                        end loop;
                        for S in Lo .. T loop
                           Set_Flat (Scr, S, Exp (Get_Flat (Scr, S) - Mx));
                           Den := Den + Get_Flat (Scr, S);
                        end loop;
                        for J in 1 .. HD loop
                           declare Acc : Float := 0.0; begin
                              for S in Lo .. T loop
                                 Acc := Acc + (Get_Flat (Scr, S) / Den)
                                          * Get_Flat (Vs (S), KV * HD + J);
                              end loop;
                              Set_Flat (Ctx_O, Hh * HD + J, Acc);
                           end;
                        end loop;
                     end;
                  end loop;
                  Attn (T) := LLM_Weight.MatVec (B.W_O, Ctx_O);
                  if Lr = 1 and then T = Seq then D3 ("rawattn-0", Attn (T)); end if;
               end;
            end loop;

            --  Sandwich norms, FFN, and the per-layer-embedding injection.
            for T in 1 .. Seq loop
               declare
                  Attn_Out : Tensor := GN (Attn (T), B.Post_Attn_Norm);
               begin
                  Add_To (Attn_Out, H (T));                      -- attn residual
                  if Lr = 1 and then T = Seq then D3 ("attn_out-0", Attn_Out); end if;
                  declare
                     Xf   : constant Tensor := GN (Attn_Out, B.Ffn_Norm);
                     Gate : constant Tensor := Gelu (LLM_Weight.MatVec (B.W_Gate, Xf));
                     Up   : constant Tensor := LLM_Weight.MatVec (B.W_Up, Xf);
                     Ff   : Tensor := GN (LLM_Weight.MatVec (B.W_Down, Gate * Up),
                                          B.Post_Ffw_Norm);
                  begin
                     Add_To (Ff, Attn_Out);                      -- pe_in
                     if Lr = 1 and then T = Seq then D3 ("pe_in-0", Ff); end if;
                     declare
                        Gp   : constant Tensor :=
                          Gelu (LLM_Weight.MatVec (B.Inp_Gate, Ff));     -- [PL]
                        Pg   : Tensor := New_Tensor ([1, PL]);
                     begin
                        for I in 1 .. PL loop
                           Set_Flat (Pg, I, Get_Flat (Gp, I)
                             * Get_Flat (IPL (T), (Lr - 1) * PL + I));
                        end loop;
                        declare
                           Pe : Tensor :=
                             GN (LLM_Weight.MatVec (B.Proj, Pg), B.Post_Norm);
                        begin
                           if Lr = 1 and then T = Seq then D3 ("ple_out-0", Pe); end if;
                           Add_To (Pe, Ff);                      -- node_61
                           for I in 1 .. D loop
                              Set_Flat (Pe, I, Get_Flat (Pe, I) * B.Layer_Out_Scale);
                           end loop;
                           if Lr = 1 and then T = Seq then D3 ("l_out-0", Pe); end if;
                           H (T) := Pe;
                        end;
                     end;
                  end;
               end;
            end loop;
         end;
      end loop;

      --  Final norm on the last position, tied output projection.
      return LLM_Weight.MatVec (M.Tok_Emb, GN (H (Seq), M.Out_Norm));
   end Forward_Logits;

   --------------------------------------------------------------------
   -- Greedy decode (full-sequence recompute — no KV cache yet).
   --------------------------------------------------------------------

   function Decode
     (M : Gemma_Model; Prompt : LLM_Tokenizer.Token_Array;
      Max_New : Integer; Sink : access LLM_Qwen.Token_Sink'Class) return String
   is
      Cap   : constant Integer := Integer'Max (1, Prompt'Length + Max_New);
      Seq   : LLM_Tokenizer.Token_Array (1 .. Cap);
      Len   : Integer := Prompt'Length;
      Out_S : Unbounded_String;
   begin
      Seq (1 .. Prompt'Length) := Prompt;
      for Step in 1 .. Max_New loop
         declare
            Logits : constant Tensor := Forward_Logits (M, Seq (1 .. Len));
            Best   : Integer := 1;
            Bs     : Float := Float'First;
         begin
            for I in 1 .. Numel (Logits) loop
               if Get_Flat (Logits, I) > Bs then Bs := Get_Flat (Logits, I); Best := I; end if;
            end loop;
            declare
               Tid : constant Integer := Best - 1;
            begin
               exit when Tid = M.Eos or else Tid = M.EOT;
               declare Piece : constant String := LLM_Tokenizer.Decode_One (M.Tok, Tid); begin
                  Append (Out_S, Piece);
                  if Sink /= null then LLM_Qwen.Emit (Sink.all, Piece); end if;
               end;
               exit when Len >= Cap;
               Len := Len + 1; Seq (Len) := Tid;
            end;
         end;
      end loop;
      return To_String (Out_S);
   end Decode;

   function Chat
     (M : Gemma_Model; Conversation : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null) return String
   is
      use type LLM_Tokenizer.Token_Array;
      LF : constant Character := Character'Val (10);

      function One (Id : Integer) return LLM_Tokenizer.Token_Array is
        (if Id >= 0 then LLM_Tokenizer.Token_Array'(1 => Id)
         else LLM_Tokenizer.Token_Array'(2 .. 1 => 0));

      function Msg_Ids (Msg : LLM_Qwen.Message) return LLM_Tokenizer.Token_Array is
         Role : constant String :=
           (if Msg.Role = LLM_Qwen.Role_User then "user" else "model");
      begin
         return One (M.SOT)
           & LLM_Tokenizer.Encode (M.Tok, Role & LF & To_String (Msg.Text))
           & One (M.EOT) & LLM_Tokenizer.Encode (M.Tok, "" & LF);
      end Msg_Ids;

      function Conv_Ids (I : Positive) return LLM_Tokenizer.Token_Array is
      begin
         if I > Conversation'Last then return LLM_Tokenizer.Token_Array'(2 .. 1 => 0); end if;
         return Msg_Ids (Conversation (I)) & Conv_Ids (I + 1);
      end Conv_Ids;
   begin
      return Decode
        (M, One (M.Bos) & Conv_Ids (Conversation'First)
            & One (M.SOT) & LLM_Tokenizer.Encode (M.Tok, "model" & LF),
         Max_New_Tokens, Sink);
   end Chat;

   function Complete
     (M : Gemma_Model; Prompt : String; Max_New_Tokens : Integer := 8)
      return String
   is
      use type LLM_Tokenizer.Token_Array;
   begin
      return Decode
        (M,
         LLM_Tokenizer.Token_Array'(1 => M.Bos)
           & LLM_Tokenizer.Encode (M.Tok, Prompt),
         Max_New_Tokens, null);
   end Complete;

   function Vocab_Size  (M : Gemma_Model) return Integer is (M.Vocab);
   function Dim         (M : Gemma_Model) return Integer is (M.Dim);
   function Block_Count (M : Gemma_Model) return Integer is (M.N_Blocks);

end LLM_Gemma;
