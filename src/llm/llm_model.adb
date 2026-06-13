---------------------------------------------------------------------
-- LLM_Model body
---------------------------------------------------------------------

with Ada.Strings.Fixed;
with Ada.Text_IO;
with LLM_Autograd;
with LLM_Weights;

package body LLM_Model is

   use LLM_Tensor;
   use LLM_Autograd;

   function Trim_Img (N : Integer) return String is
      S : constant String := Ada.Strings.Fixed.Trim (Integer'Image (N), Ada.Strings.Both);
   begin
      return S;
   end Trim_Img;

   --------------------------------------------------------------------
   -- Construction
   --------------------------------------------------------------------

   function New_GPT2_Small return GPT_Model is
      C : constant Model_Config := (others => <>);
   begin
      return New_Tiny (C.Dim, C.N_Layers);
   end New_GPT2_Small;

   function New_Tiny (Dim, N_Layers : Integer) return GPT_Model is
      M : GPT_Model;
   begin
      M.Config := (
         Vocab_Size => 256,
         Dim        => Dim,
         N_Layers   => N_Layers,
         N_Heads    => Dim / 64,
         Max_Seq_Len => 256
      );

      M.Token_Emb  := LLM_Layer.New_Embedding (M.Config.Vocab_Size, Dim);
      M.Pos_Emb    := LLM_Layer.New_Embedding (M.Config.Max_Seq_Len, Dim);
      M.Final_Norm := LLM_Layer.New_LayerNorm (Dim);
      M.LM_Head    := LLM_Layer.New_Linear (Dim, M.Config.Vocab_Size);

      M.Blocks := new Block_Array (1 .. N_Layers);
      for I in 1 .. N_Layers loop
         M.Blocks (I) := LLM_Block.New_Block (Dim, M.Config.N_Heads);
      end loop;

      return M;
   end New_Tiny;

   --------------------------------------------------------------------
   -- Load GPT-2 pretrained weights
   --------------------------------------------------------------------

   function Load_GPT2 (Dir : String) return GPT_Model is
      use Ada.Text_IO;
      Cfg_File : File_Type;
      Cfg_Line : String (1 .. 256);
      Cfg_Last : Natural;
      Vocab : Integer := 50257;
      Dim   : Integer := 768;
      Heads : Integer := 12;
      N_L   : Integer := 12;
      Max_S : Integer := 1024;
      Equals : Integer;
      
      M : GPT_Model;
      
      function F (Name : String) return String is
      begin
         return Dir & "/" & Name & ".bin";
      end F;
      
   begin
      -- Read config.txt
      Open (Cfg_File, In_File, Dir & "/config.txt");
      while not End_Of_File (Cfg_File) loop
         Get_Line (Cfg_File, Cfg_Line, Cfg_Last);
         Equals := 1;
         while Equals <= Cfg_Last and then Cfg_Line (Equals) /= '=' loop
            Equals := Equals + 1;
         end loop;
         if Equals <= Cfg_Last then
            declare
               Key : constant String := Cfg_Line (1 .. Equals - 1);
               Val : constant String := Cfg_Line (Equals + 1 .. Cfg_Last);
            begin
               if Key = "vocab_size" then
                  Vocab := Integer'Value (Val);
               elsif Key = "dim" then
                  Dim := Integer'Value (Val);
               elsif Key = "n_heads" then
                  Heads := Integer'Value (Val);
               elsif Key = "n_layers" then
                  N_L := Integer'Value (Val);
               elsif Key = "max_seq_len" then
                  Max_S := Integer'Value (Val);
               end if;
            end;
         end if;
      end loop;
      Close (Cfg_File);
      
      -- Build model with correct config
      M.Config := (Vocab, Dim, N_L, Heads, Max_S);
      M.Token_Emb  := LLM_Layer.New_Embedding (Vocab, Dim);
      M.Pos_Emb    := LLM_Layer.New_Embedding (Max_S, Dim);
      M.Final_Norm := LLM_Layer.New_LayerNorm (Dim);
      M.LM_Head    := LLM_Layer.New_Linear (Dim, Vocab);
      M.Blocks := new Block_Array (1 .. N_L);
      for I in 1 .. N_L loop
         M.Blocks (I) := LLM_Block.New_Block (Dim, Heads);
      end loop;
      
      -- Load weights (overwrite random init)
      M.Token_Emb.W := LLM_Weights.Load_Matrix (F ("wte"), Vocab, Dim);
      M.Pos_Emb.W   := LLM_Weights.Load_Matrix (F ("wpe"), Max_S, Dim);
      
      for I in 0 .. N_L - 1 loop
         declare
            Prefix : constant String := "h_" & Trim_Img (I);
            use LLM_Weights;
         begin
            M.Blocks (I + 1).Attn.Q_Proj.W := Load_Matrix (F (Prefix & "_qw"), Dim, Dim);
            M.Blocks (I + 1).Attn.Q_Proj.B := Load_Vector (F (Prefix & "_qb"), Dim);
            M.Blocks (I + 1).Attn.K_Proj.W := Load_Matrix (F (Prefix & "_kw"), Dim, Dim);
            M.Blocks (I + 1).Attn.K_Proj.B := Load_Vector (F (Prefix & "_kb"), Dim);
            M.Blocks (I + 1).Attn.V_Proj.W := Load_Matrix (F (Prefix & "_vw"), Dim, Dim);
            M.Blocks (I + 1).Attn.V_Proj.B := Load_Vector (F (Prefix & "_vb"), Dim);
            M.Blocks (I + 1).Attn.Out_Proj.W := Load_Matrix (F (Prefix & "_ow"), Dim, Dim);
            M.Blocks (I + 1).Attn.Out_Proj.B := Load_Vector (F (Prefix & "_ob"), Dim);
            M.Blocks (I + 1).Attn_Norm.Gamma := Load_Vector (F (Prefix & "_ln1w"), Dim);
            M.Blocks (I + 1).Attn_Norm.Beta  := Load_Vector (F (Prefix & "_ln1b"), Dim);
            M.Blocks (I + 1).MLP_Norm.Gamma  := Load_Vector (F (Prefix & "_ln2w"), Dim);
            M.Blocks (I + 1).MLP_Norm.Beta   := Load_Vector (F (Prefix & "_ln2b"), Dim);
            M.Blocks (I + 1).MLP.FC1.W := Load_Matrix (F (Prefix & "_fcw"), Dim, 4 * Dim);
            M.Blocks (I + 1).MLP.FC1.B := Load_Vector (F (Prefix & "_fcb"), 4 * Dim);
            M.Blocks (I + 1).MLP.FC2.W := Load_Matrix (F (Prefix & "_pw"), 4 * Dim, Dim);
            M.Blocks (I + 1).MLP.FC2.B := Load_Vector (F (Prefix & "_pb"), Dim);
         end;
      end loop;
      
      M.Final_Norm.Gamma := LLM_Weights.Load_Vector (F ("lnfw"), Dim);
      M.Final_Norm.Beta  := LLM_Weights.Load_Vector (F ("lnfb"), Dim);
      
      -- Weight tying: LM head uses transposed token embedding
      M.LM_Head.W := New_Var (Transpose (Data (M.Token_Emb.W)));
      M.LM_Head.B := New_Var (New_Tensor ([1, Vocab]));
      
      return M;
   end Load_GPT2;

   --------------------------------------------------------------------
   -- Parameter count
   --------------------------------------------------------------------

   function Param_Count (M : GPT_Model) return Integer is
      C : Integer := 0;

      function Len (V : LLM_Autograd.Var) return Integer is
      begin
         return Numel (LLM_Autograd.Data (V));
      end Len;
   begin
      C := C + Len (M.Token_Emb.W);
      C := C + Len (M.Pos_Emb.W);
      C := C + Len (M.Final_Norm.Gamma);
      C := C + Len (M.Final_Norm.Beta);
      C := C + Len (M.LM_Head.W);
      C := C + Len (M.LM_Head.B);
      for I in 1 .. M.Blocks'Length loop
         declare
            B : constant LLM_Block.Transformer_Block := M.Blocks (I);
         begin
            C := C + Len (B.Attn.Q_Proj.W);
            C := C + Len (B.Attn.Q_Proj.B);
            C := C + Len (B.Attn.K_Proj.W);
            C := C + Len (B.Attn.K_Proj.B);
            C := C + Len (B.Attn.V_Proj.W);
            C := C + Len (B.Attn.V_Proj.B);
            C := C + Len (B.Attn.Out_Proj.W);
            C := C + Len (B.Attn.Out_Proj.B);
            C := C + Len (B.Attn_Norm.Gamma);
            C := C + Len (B.Attn_Norm.Beta);
            C := C + Len (B.MLP_Norm.Gamma);
            C := C + Len (B.MLP_Norm.Beta);
            C := C + Len (B.MLP.FC1.W);
            C := C + Len (B.MLP.FC1.B);
            C := C + Len (B.MLP.FC2.W);
            C := C + Len (B.MLP.FC2.B);
         end;
      end loop;
      return C;
   end Param_Count;

   --------------------------------------------------------------------
   -- Forward pass
   --------------------------------------------------------------------

   function Forward (M : GPT_Model; Token_Ids : Tensor) return Tensor is
      Dim   : constant Integer := M.Config.Dim;
      Seq_Len : constant Integer := Numel (Token_Ids);

      -- Embedding lookup: pool all token embeddings into one vector
      -- For simplicity: average the embeddings of all input tokens
      H : Tensor := New_Tensor ([1, Dim]);

      -- Sum token + position embeddings
   begin
      if Seq_Len < 1 then
         return New_Tensor ([1, M.Config.Vocab_Size]);
      end if;

      -- Average token embeddings
      for Pos in 1 .. Seq_Len loop
         declare
            Tid : constant Integer := Integer (Get_Flat (Token_Ids, Pos));
            Tok_Emb : constant Tensor := Data (LLM_Layer.Forward (M.Token_Emb, Tid));
            Pos_Emb : constant Tensor := Data (LLM_Layer.Forward (M.Pos_Emb, Pos));
         begin
            H := H + Tok_Emb + Pos_Emb;
         end;
      end loop;

      -- Average over sequence
      declare
         Scale : constant Float := 1.0 / Float (Seq_Len);
      begin
         for I in 1 .. Numel (H) loop
            Set_Flat (H, I, Get_Flat (H, I) * Scale);
         end loop;
      end;

      -- Pass through transformer blocks
      declare
         X : LLM_Autograd.Var := LLM_Autograd.New_Var (H);
      begin
         for I in 1 .. M.Blocks'Length loop
            X := LLM_Block.Forward (M.Blocks (I), X);
         end loop;

         -- Final layer norm
         X := LLM_Layer.Forward (M.Final_Norm, X);

         -- LM head projection: dim → vocab
         X := LLM_Layer.Forward (M.LM_Head, X);

         return Data (X);
      end;
   end Forward;

   --------------------------------------------------------------------
   -- Predict next token: argmax over logits
   --------------------------------------------------------------------

   function Predict_Next (M : GPT_Model; Token_Ids : Tensor) return Integer is
      Logits : constant Tensor := Forward (M, Token_Ids);
      Best_Token : Integer := 1;
      Best_Score : Float := Float'First;
      N : constant Integer := Numel (Logits);
   begin
      for I in 1 .. N loop
         declare
            Score : constant Float := Get_Flat (Logits, I);
         begin
            if Score > Best_Score then
               Best_Score := Score;
               Best_Token := I;
            end if;
         end;
      end loop;
      return Best_Token;
   end Predict_Next;

   --------------------------------------------------------------------
   -- Generate text
   --------------------------------------------------------------------

   function Generate (M : GPT_Model; Prompt : String; Max_New_Tokens : Integer := 50) return String is
      Result : String (1 .. 1024);
      Result_Len : Integer := 0;
      Max_Len : constant Integer := 1024;
      Context : Tensor := New_Tensor ([1, 1]);
   begin
      -- Copy prompt into result
      for I in Prompt'Range loop
         if Result_Len < Max_Len then
            Result_Len := Result_Len + 1;
            Result (Result_Len) := Prompt (I);
         end if;
      end loop;

      -- Build initial context from prompt (character-level tokenization)
      if Prompt'Length > 0 then
         Context := New_Tensor ([1, Prompt'Length]);
         for I in 1 .. Prompt'Length loop
            Set_Flat (Context, I, Float (Character'Pos (Prompt (I))));
         end loop;
      else
         Context := New_Tensor ([1, 1]);
         Set_Flat (Context, 1, 0.0);
      end if;

      -- Autoregressive generation
      for Step in 1 .. Max_New_Tokens loop
         declare
            Logits : constant Tensor := Forward (M, Context);
            Next_Tok : Integer := 1;
            Best_Score : Float := Float'First;
         begin
            -- Argmax
            for I in 1 .. Numel (Logits) loop
               declare
                  Score : constant Float := Get_Flat (Logits, I);
               begin
                  if Score > Best_Score then
                     Best_Score := Score;
                     Next_Tok := I;
                  end if;
               end;
            end loop;

            -- Detokenize: token id → character
            if Next_Tok >= 32 and Next_Tok <= 126 then
               if Result_Len < Max_Len then
                  Result_Len := Result_Len + 1;
                  Result (Result_Len) := Character'Val (Next_Tok);
               end if;
            elsif Next_Tok = 0 or Next_Tok = 50256 then
               -- EOS token (GPT-2 special)
               exit;
            else
               -- Non-printable: skip (but continue)
               null;
            end if;

            -- Append to context
            declare
               Old_Len : constant Integer := Numel (Context);
               New_Ctx : Tensor := New_Tensor ([1, Old_Len + 1]);
            begin
               for I in 1 .. Old_Len loop
                  Set_Flat (New_Ctx, I, Get_Flat (Context, I));
               end loop;
               Set_Flat (New_Ctx, Old_Len + 1, Float (Next_Tok));
               Context := New_Ctx;
            end;
         end;
      end loop;

      return Result (1 .. Result_Len);
   end Generate;

end LLM_Model;
