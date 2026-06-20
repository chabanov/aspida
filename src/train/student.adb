---------------------------------------------------------------------
-- Student body — multi-layer forward (with activation cache) + reverse pass.
---------------------------------------------------------------------

with Ada.Streams.Stream_IO; use Ada.Streams.Stream_IO;
with LLM_Quant;
with LLM_Tensor;

package body Student is

   Magic    : constant Integer := 16#A59DA01#;   -- v1: weights only
   Magic_V2 : constant Integer := 16#A59DA02#;   -- v2: weights + AdamW state

   procedure Wr (S : Stream_Access; Mx : Matrix) is
   begin
      for I in Mx'Range (1) loop
         for J in Mx'Range (2) loop Real'Write (S, Mx (I, J)); end loop;
      end loop;
   end Wr;

   procedure Rd (S : Stream_Access; Mx : out Matrix) is
   begin
      for I in Mx'Range (1) loop
         for J in Mx'Range (2) loop Real'Read (S, Mx (I, J)); end loop;
      end loop;
   end Rd;

   --  Every optimizer state in a fixed order; shared by Save and Load so the
   --  layout cannot drift between the two.
   procedure Wr_Opt (S : Stream_Access; M : Model) is
   begin
      Write_Adam (S, M.AE); Write_Adam (S, M.APos);
      Write_Adam (S, M.AGf); Write_Adam (S, M.AWout);
      for L in 1 .. Lyr loop
         Write_Adam (S, M.B (L).AG1); Write_Adam (S, M.B (L).AG2);
         Write_Adam (S, M.B (L).AWq); Write_Adam (S, M.B (L).AWk);
         Write_Adam (S, M.B (L).AWv); Write_Adam (S, M.B (L).AWo);
         Write_Adam (S, M.B (L).AWg); Write_Adam (S, M.B (L).AWu);
         Write_Adam (S, M.B (L).AWd);
      end loop;
   end Wr_Opt;

   procedure Rd_Opt (S : Stream_Access; M : in out Model) is
   begin
      Read_Adam (S, M.AE); Read_Adam (S, M.APos);
      Read_Adam (S, M.AGf); Read_Adam (S, M.AWout);
      for L in 1 .. Lyr loop
         Read_Adam (S, M.B (L).AG1); Read_Adam (S, M.B (L).AG2);
         Read_Adam (S, M.B (L).AWq); Read_Adam (S, M.B (L).AWk);
         Read_Adam (S, M.B (L).AWv); Read_Adam (S, M.B (L).AWo);
         Read_Adam (S, M.B (L).AWg); Read_Adam (S, M.B (L).AWu);
         Read_Adam (S, M.B (L).AWd);
      end loop;
   end Rd_Opt;

   procedure Init (M : in out Model; Seed : Long_Float) is
   begin
      M.RG := Seeded (Seed);
      Init_Glorot (M.E, M.RG);
      Init_Glorot (M.Wout, M.RG);
      Init_Glorot (M.Pos, M.RG);
      M.Gf := [others => [others => 1.0]];
      for L in 1 .. Lyr loop
         M.B (L).G1 := [others => [others => 1.0]];
         M.B (L).G2 := [others => [others => 1.0]];
         Init_Glorot (M.B (L).Wq, M.RG);
         Init_Glorot (M.B (L).Wk, M.RG);
         Init_Glorot (M.B (L).Wv, M.RG);
         Init_Glorot (M.B (L).Wo, M.RG);
         Init_Glorot (M.B (L).Wg, M.RG);
         Init_Glorot (M.B (L).Wu, M.RG);
         Init_Glorot (M.B (L).Wd, M.RG);
      end loop;
   end Init;

   procedure Forward
     (M : in out Model; Tokens : Train.Label_Array; Logits : out Logit_Mat)
   is
      Cur, Ao, Mo, Tmp : DMat;

      --  Weight-projection forward. With QAT on, the weight is fake-quantized
      --  for the forward (STE leaves the backward unchanged, so gradients still
      --  update the full-precision master weight). Use_QAT is a static generic,
      --  so the quant branch is compiled out entirely when off (no overhead).
      procedure Lin (Inp, W : Matrix; Out_M : out Matrix) is
      begin
         if Use_QAT then
            declare
               Wq : Matrix (W'Range (1), W'Range (2));
            begin
               if QAT_Block > 0 then
                  --  Per-block fake-quant: matches the K-quant super-block
                  --  layout the export quantizer applies, so QAT trains against
                  --  the same block-local scaling the deployed model sees.
                  Fake_Quant_Forward_Blocked (W, QAT_Bits, QAT_Block, Wq);
               else
                  Fake_Quant_Forward (W, QAT_Bits, Wq);
               end if;
               Linear_NB_Forward (Inp, Wq, Out_M);
            end;
         else
            Linear_NB_Forward (Inp, W, Out_M);
         end if;
      end Lin;
   begin
      for I in 1 .. Seq loop M.Toks (I) := Tokens (Tokens'First + I - 1); end loop;
      Embed_Forward (M.E, M.Toks, M.Xemb);
      if not Use_RoPE then
         for I in 1 .. Seq loop for J in 1 .. Dm loop   -- + learned position
            M.Xemb (I, J) := M.Xemb (I, J) + M.Pos (I, J);
         end loop; end loop;
      end if;
      Cur := M.Xemb;
      for L in 1 .. Lyr loop
         M.B (L).Inp := Cur;
         RMSNorm_Forward (Cur, M.B (L).G1, M.B (L).Xn1);
         Lin (M.B (L).Xn1, M.B (L).Wq, M.B (L).Q);
         Lin (M.B (L).Xn1, M.B (L).Wk, M.B (L).K);
         Lin (M.B (L).Xn1, M.B (L).Wv, M.B (L).V);
         if Use_RoPE then
            Tmp := M.B (L).Q; RoPE_Forward (Tmp, Heads, Real (Rope_Base), M.B (L).Q);
            Tmp := M.B (L).K; RoPE_Forward (Tmp, Heads, Real (Rope_Base), M.B (L).K);
         end if;
         MHA_Forward (M.B (L).Q, M.B (L).K, M.B (L).V, Heads, M.B (L).Oa, M.B (L).A);
         Lin (M.B (L).Oa, M.B (L).Wo, Ao);
         for I in 1 .. Seq loop for J in 1 .. Dm loop
            M.B (L).H2 (I, J) := Cur (I, J) + Ao (I, J);
         end loop; end loop;
         RMSNorm_Forward (M.B (L).H2, M.B (L).G2, M.B (L).Xn2);
         Lin (M.B (L).Xn2, M.B (L).Wg, M.B (L).Gpre);
         SiLU_Forward (M.B (L).Gpre, M.B (L).Gate);
         Lin (M.B (L).Xn2, M.B (L).Wu, M.B (L).Up);
         for I in 1 .. Seq loop for J in 1 .. Ff loop
            M.B (L).Hid (I, J) := M.B (L).Gate (I, J) * M.B (L).Up (I, J);
         end loop; end loop;
         Lin (M.B (L).Hid, M.B (L).Wd, Mo);
         for I in 1 .. Seq loop for J in 1 .. Dm loop
            Cur (I, J) := M.B (L).H2 (I, J) + Mo (I, J);
         end loop; end loop;
      end loop;
      M.Hf := Cur;
      RMSNorm_Forward (M.Hf, M.Gf, M.Xf);
      Lin (M.Xf, M.Wout, M.Logits);
      Logits := M.Logits;
   end Forward;

   function Backward (M : in out Model; Target : Logit_Mat) return Train.Real is
      Loss : constant Real := KL_Loss (M.Logits, Target);
      DLog : Logit_Mat;
      DXf, DHf, DOut, DIn : DMat;
      DHid, DGate, DUp, DGpre : FMat;
      DXn2, DXn2a, DXn2b, DH2, DH2b, DOat, DQ, DK, DV,
      DXn1, DXn1q, DXn1k, DXn1v, DInNorm : DMat;
   begin
      KL_Backward (M.Logits, Target, DLog);
      Linear_NB_Backward (M.Xf, M.Wout, DLog, DXf, M.dWout);
      RMSNorm_Backward (M.Hf, M.Gf, DXf, DHf, M.dGf);
      DOut := DHf;
      for L in reverse 1 .. Lyr loop
         --  H_out = H2 + Mo
         Linear_NB_Backward (M.B (L).Hid, M.B (L).Wd, DOut, DHid, M.B (L).dWd);
         for I in 1 .. Seq loop for J in 1 .. Ff loop
            DGate (I, J) := DHid (I, J) * M.B (L).Up (I, J);
            DUp   (I, J) := DHid (I, J) * M.B (L).Gate (I, J);
         end loop; end loop;
         Linear_NB_Backward (M.B (L).Xn2, M.B (L).Wu, DUp, DXn2a, M.B (L).dWu);
         SiLU_Backward (M.B (L).Gpre, DGate, DGpre);
         Linear_NB_Backward (M.B (L).Xn2, M.B (L).Wg, DGpre, DXn2b, M.B (L).dWg);
         for I in 1 .. Seq loop for J in 1 .. Dm loop
            DXn2 (I, J) := DXn2a (I, J) + DXn2b (I, J);
         end loop; end loop;
         RMSNorm_Backward (M.B (L).H2, M.B (L).G2, DXn2, DH2b, M.B (L).dG2);
         for I in 1 .. Seq loop for J in 1 .. Dm loop
            DH2 (I, J) := DOut (I, J) + DH2b (I, J);
         end loop; end loop;
         Linear_NB_Backward (M.B (L).Oa, M.B (L).Wo, DH2, DOat, M.B (L).dWo);
         MHA_Backward (M.B (L).Q, M.B (L).K, M.B (L).V, M.B (L).A, Heads,
                       DOat, DQ, DK, DV);
         if Use_RoPE then                  -- rotate the Q/K grads back
            DXn1q := DQ; RoPE_Backward (DXn1q, Heads, Real (Rope_Base), DQ);
            DXn1k := DK; RoPE_Backward (DXn1k, Heads, Real (Rope_Base), DK);
         end if;
         Linear_NB_Backward (M.B (L).Xn1, M.B (L).Wq, DQ, DXn1q, M.B (L).dWq);
         Linear_NB_Backward (M.B (L).Xn1, M.B (L).Wk, DK, DXn1k, M.B (L).dWk);
         Linear_NB_Backward (M.B (L).Xn1, M.B (L).Wv, DV, DXn1v, M.B (L).dWv);
         for I in 1 .. Seq loop for J in 1 .. Dm loop
            DXn1 (I, J) := DXn1q (I, J) + DXn1k (I, J) + DXn1v (I, J);
         end loop; end loop;
         RMSNorm_Backward (M.B (L).Inp, M.B (L).G1, DXn1, DInNorm, M.B (L).dG1);
         for I in 1 .. Seq loop for J in 1 .. Dm loop
            DIn (I, J) := DH2 (I, J) + DInNorm (I, J);   -- residual + norm
         end loop; end loop;
         DOut := DIn;     -- becomes the output-gradient of the earlier block
      end loop;
      Embed_Backward (M.Toks, DOut, M.dE);
      if not Use_RoPE then
         M.dPos := DOut;  -- Xemb = token_embd + Pos -> both get dXemb
      end if;
      return Loss;
   end Backward;

   procedure Step (M : in out Model; LR : Train.Real := 5.0E-3; Clip : Train.Real := 0.0) is
   begin
      Adam_Step (M.E, M.dE, M.AE, LR, Clip => Clip);
      if not Use_RoPE then
         Adam_Step (M.Pos, M.dPos, M.APos, LR, Clip => Clip);
      end if;
      Adam_Step (M.Gf, M.dGf, M.AGf, LR, Clip => Clip);
      Adam_Step (M.Wout, M.dWout, M.AWout, LR, Clip => Clip);
      for L in 1 .. Lyr loop
         Adam_Step (M.B (L).G1, M.B (L).dG1, M.B (L).AG1, LR, Clip => Clip);
         Adam_Step (M.B (L).G2, M.B (L).dG2, M.B (L).AG2, LR, Clip => Clip);
         Adam_Step (M.B (L).Wq, M.B (L).dWq, M.B (L).AWq, LR, Clip => Clip);
         Adam_Step (M.B (L).Wk, M.B (L).dWk, M.B (L).AWk, LR, Clip => Clip);
         Adam_Step (M.B (L).Wv, M.B (L).dWv, M.B (L).AWv, LR, Clip => Clip);
         Adam_Step (M.B (L).Wo, M.B (L).dWo, M.B (L).AWo, LR, Clip => Clip);
         Adam_Step (M.B (L).Wg, M.B (L).dWg, M.B (L).AWg, LR, Clip => Clip);
         Adam_Step (M.B (L).Wu, M.B (L).dWu, M.B (L).AWu, LR, Clip => Clip);
         Adam_Step (M.B (L).Wd, M.B (L).dWd, M.B (L).AWd, LR, Clip => Clip);
      end loop;
   end Step;

   procedure Save (M : Model; Path : String) is
      F : File_Type;
      S : Stream_Access;
   begin
      Create (F, Out_File, Path);
      S := Stream (F);
      Integer'Write (S, Magic_V2);
      Integer'Write (S, Voc); Integer'Write (S, Dm); Integer'Write (S, Ff);
      Integer'Write (S, Seq); Integer'Write (S, Lyr); Integer'Write (S, Heads);
      Wr (S, M.E); Wr (S, M.Pos); Wr (S, M.Gf); Wr (S, M.Wout);
      for L in 1 .. Lyr loop
         Wr (S, M.B (L).G1); Wr (S, M.B (L).G2);
         Wr (S, M.B (L).Wq); Wr (S, M.B (L).Wk);
         Wr (S, M.B (L).Wv); Wr (S, M.B (L).Wo);
         Wr (S, M.B (L).Wg); Wr (S, M.B (L).Wu); Wr (S, M.B (L).Wd);
      end loop;
      Wr_Opt (S, M);            -- AdamW state (resume support)
      Close (F);
   end Save;

   procedure Load (M : in out Model; Path : String) is
      F : File_Type;
      S : Stream_Access;
      V : Integer;
      Has_Opt : Boolean := False;     -- v2 checkpoints carry AdamW state
      procedure Expect (Got, Want : Integer) is
      begin
         if Got /= Want then Close (F); raise Bad_Checkpoint; end if;
      end Expect;
   begin
      Open (F, In_File, Path);
      S := Stream (F);
      --  Accept both v1 (weights only) and v2 (weights + optimizer state).
      Integer'Read (S, V);
      if V = Magic_V2 then Has_Opt := True;
      elsif V /= Magic then Close (F); raise Bad_Checkpoint;
      end if;
      Integer'Read (S, V); Expect (V, Voc);
      Integer'Read (S, V); Expect (V, Dm);
      Integer'Read (S, V); Expect (V, Ff);
      Integer'Read (S, V); Expect (V, Seq);
      Integer'Read (S, V); Expect (V, Lyr);
      Integer'Read (S, V); Expect (V, Heads);
      Rd (S, M.E); Rd (S, M.Pos); Rd (S, M.Gf); Rd (S, M.Wout);
      for L in 1 .. Lyr loop
         Rd (S, M.B (L).G1); Rd (S, M.B (L).G2);
         Rd (S, M.B (L).Wq); Rd (S, M.B (L).Wk);
         Rd (S, M.B (L).Wv); Rd (S, M.B (L).Wo);
         Rd (S, M.B (L).Wg); Rd (S, M.B (L).Wu); Rd (S, M.B (L).Wd);
      end loop;
      if Has_Opt then Rd_Opt (S, M); end if;   -- restore Adam moments / step
      Close (F);
   end Load;

   --------------------------------------------------------------------
   --  GGUF export (multi-layer Llama arch; weights transposed to GGUF
   --  [ne0=in, ne1=out], Rows=out).
   --------------------------------------------------------------------
   function To_GGUF (W : Matrix) return GGUF_Write.Float_Array is
      Inn  : constant Positive := W'Length (1);
      Outd : constant Positive := W'Length (2);
      R    : GGUF_Write.Float_Array (1 .. Inn * Outd);
      Idx  : Positive := 1;
   begin
      for O in 1 .. Outd loop
         for I in 1 .. Inn loop R (Idx) := Float (W (I, O)); Idx := Idx + 1; end loop;
      end loop;
      return R;
   end To_GGUF;

   function Emb_Flat (Em : Matrix) return GGUF_Write.Float_Array is
      R : GGUF_Write.Float_Array (1 .. Em'Length (1) * Em'Length (2));
      Idx : Positive := 1;
   begin
      for V in 1 .. Em'Length (1) loop
         for J in 1 .. Em'Length (2) loop R (Idx) := Float (Em (V, J)); Idx := Idx + 1; end loop;
      end loop;
      return R;
   end Emb_Flat;

   function Norm_Flat (Ga : Matrix) return GGUF_Write.Float_Array is
      R : GGUF_Write.Float_Array (1 .. Ga'Length (2));
   begin
      for J in 1 .. Ga'Length (2) loop R (J) := Float (Ga (1, J)); end loop;
      return R;
   end Norm_Flat;

   procedure Export_GGUF
     (M : Model; Path : String; Tokens : GGUF_Write.Str_List;
      Bos, Eos : Natural := 0; Ctx : Natural := 256;
      Fmt : Quant_Format := Q_None)
   is
      use GGUF_Write;
      B  : Builder;
      HD : constant Natural := Dm / Heads;
      function Img (N : Natural) return String is
         S : constant String := Natural'Image (N);
      begin return S (S'First + 1 .. S'Last); end Img;

      --  Weight matrix: quantized per Fmt (when block-aligned), else F32.
      --  The engine dequantizes per ROW (ne0 elements), so the row length —
      --  not the total — must be a whole number of blocks: Q4_K uses
      --  256-element super-blocks, Q8_0/Q4_0 use 32-element blocks. ne0 is the
      --  fastest-varying GGUF dim, Dims (Dims'First). Rows that don't align
      --  fall back to F32 (so tiny demo dims still export cleanly).
      procedure Add_W (Name : String; Dims : Dims_Array; Data : Float_Array) is
         Ne0     : constant Natural := Dims (Dims'First);
         Aligned : constant Boolean :=
           (case Fmt is
               when Q_Q4_K | Q_Q5_K | Q_Q6_K   => Ne0 mod 256 = 0,
               when Q_Q8_0 | Q_Q4_0 | Q_Q5_0   => Ne0 mod 32 = 0,
               when Q_None                     => False);
      begin
         if Aligned then
            declare
               T : LLM_Tensor.Tensor := LLM_Tensor.New_Tensor ([1, Data'Length]);
            begin
               for I in Data'Range loop
                  LLM_Tensor.Set_Flat (T, I - Data'First + 1, Data (I));
               end loop;
               case Fmt is
                  when Q_Q8_0 =>
                     Add_Tensor_Q8_0 (B, Name, Dims, LLM_Quant.Quantize_Q8_0 (T));
                  when Q_Q4_0 =>
                     Add_Tensor_Q4_0 (B, Name, Dims, LLM_Quant.Quantize_Q4_0 (T));
                  when Q_Q5_0 =>
                     Add_Tensor_Q5_0 (B, Name, Dims, LLM_Quant.Quantize_Q5_0 (T));
                  when Q_Q4_K =>
                     Add_Tensor_Q4_K (B, Name, Dims, LLM_Quant.Quantize_Q4_K (T));
                  when Q_Q5_K =>
                     Add_Tensor_Q5_K (B, Name, Dims, LLM_Quant.Quantize_Q5_K (T));
                  when Q_Q6_K =>
                     Add_Tensor_Q6_K (B, Name, Dims, LLM_Quant.Quantize_Q6_K (T));
                  when Q_None => null;
               end case;
            end;
         else
            Add_Tensor_F32 (B, Name, Dims, Data);
         end if;
      end Add_W;
   begin
      Meta_Str (B, "general.architecture", "llama");
      Meta_Str (B, "general.name", "aspida");
      Meta_U32 (B, "llama.embedding_length", Dm);
      Meta_U32 (B, "llama.block_count", Lyr);
      Meta_U32 (B, "llama.attention.head_count", Heads);
      Meta_U32 (B, "llama.attention.head_count_kv", Heads);
      Meta_U32 (B, "llama.feed_forward_length", Ff);
      Meta_U32 (B, "llama.context_length", Ctx);
      Meta_U32 (B, "llama.rope.dimension_count", HD);
      Meta_F32 (B, "llama.rope.freq_base", Float (Rope_Base));
      Meta_F32 (B, "llama.attention.layer_norm_rms_epsilon", 1.0E-6);
      Meta_Str (B, "tokenizer.ggml.model", "gpt2");
      Meta_U32 (B, "tokenizer.ggml.bos_token_id", Bos);
      Meta_U32 (B, "tokenizer.ggml.eos_token_id", Eos);
      Meta_U32 (B, "tokenizer.ggml.unknown_token_id", 0);
      Meta_Str_Array (B, "tokenizer.ggml.tokens", Tokens);

      Add_W (         "token_embd.weight",  [Dm, Voc], Emb_Flat (M.E));
      Add_Tensor_F32 (B, "output_norm.weight", [Dm],      Norm_Flat (M.Gf));
      Add_W (         "output.weight",      [Dm, Voc], To_GGUF (M.Wout));
      for L in 1 .. Lyr loop
         declare
            P : constant String := "blk." & Img (L - 1) & ".";
         begin
            Add_Tensor_F32 (B, P & "attn_norm.weight", [Dm], Norm_Flat (M.B (L).G1));
            Add_Tensor_F32 (B, P & "ffn_norm.weight",  [Dm], Norm_Flat (M.B (L).G2));
            Add_W (P & "attn_q.weight",      [Dm, Dm], To_GGUF (M.B (L).Wq));
            Add_W (P & "attn_k.weight",      [Dm, Dm], To_GGUF (M.B (L).Wk));
            Add_W (P & "attn_v.weight",      [Dm, Dm], To_GGUF (M.B (L).Wv));
            Add_W (P & "attn_output.weight", [Dm, Dm], To_GGUF (M.B (L).Wo));
            Add_W (P & "ffn_gate.weight", [Dm, Ff], To_GGUF (M.B (L).Wg));
            Add_W (P & "ffn_up.weight",   [Dm, Ff], To_GGUF (M.B (L).Wu));
            Add_W (P & "ffn_down.weight", [Ff, Dm], To_GGUF (M.B (L).Wd));
         end;
      end loop;
      GGUF_Write.Save (B, Path);
   end Export_GGUF;

end Student;
