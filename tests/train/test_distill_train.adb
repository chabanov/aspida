------------------------------------------------------------------------
-- test_distill_train — the full teacher->student distillation loop, end to
-- end, on CPU:
--   * a synthetic teacher produces a top-K distillation dataset (Distill);
--   * a real tiny student  (embedding -> RMSNorm -> causal attention ->
--     SwiGLU MLP -> RMSNorm -> output head)  is trained with AdamW to match
--     the teacher's top-K distribution at every position (sparse KL).
-- If any backward in the chain were wrong, the loss would not converge — so
-- a large drop validates the whole pipeline (embedding + block + head).
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Train;       use Train;
with Distill;

procedure Test_Distill_Train is
   V  : constant := 32;     -- vocab
   D  : constant := 8;      -- model dim
   F  : constant := 16;     -- ffn dim
   Tn : constant := 4;      -- sequence length
   K  : constant := 6;      -- teacher top-K
   M  : constant := 6;      -- number of training sequences
   Steps : constant := 1500;

   G : RNG := Seeded (11.0);

   --  synthetic teacher (peaked, deterministic) — same idea as test_distill
   type Synth is new Distill.Teacher with null record;
   overriding function Vocab (T : Synth) return Positive is (V);
   overriding procedure Forward
     (T : in out Synth; Tokens : Distill.Token_Array;
      Out_Logits : out Distill.Logit_Matrix)
   is
      Peak : Integer;
   begin
      for R in 1 .. Tokens'Length loop
         Peak := Integer (Tokens (Tokens'First + R - 1)) mod V + 1;
         for C in 1 .. V loop
            Out_Logits (R, C) :=
              Distill.Logit (-0.35 * Float (abs (C - Peak)) - 0.0001 * Float (C));
         end loop;
      end loop;
   end Forward;

   --  student parameters
   E    : Matrix (1 .. V, 1 .. D);
   G1, G2, Gf : Matrix (1 .. 1, 1 .. D);
   Wq, Wk, Wv, Wo : Matrix (1 .. D, 1 .. D);
   Bq, Bk, Bv, Bo : Matrix (1 .. 1, 1 .. D);
   Wg, Wu : Matrix (1 .. D, 1 .. F);
   Bg, Bu : Matrix (1 .. 1, 1 .. F);
   Wd : Matrix (1 .. F, 1 .. D); Bd : Matrix (1 .. 1, 1 .. D);
   Wout : Matrix (1 .. D, 1 .. V); Bout : Matrix (1 .. 1, 1 .. V);

   --  Adam states
   AE  : Adam := New_Adam (V, D);
   AG1 : Adam := New_Adam (1, D); AG2 : Adam := New_Adam (1, D); AGf : Adam := New_Adam (1, D);
   AWq : Adam := New_Adam (D, D); AWk : Adam := New_Adam (D, D);
   AWv : Adam := New_Adam (D, D); AWo : Adam := New_Adam (D, D);
   ABq : Adam := New_Adam (1, D); ABk : Adam := New_Adam (1, D);
   ABv : Adam := New_Adam (1, D); ABo : Adam := New_Adam (1, D);
   AWg : Adam := New_Adam (D, F); AWu : Adam := New_Adam (D, F);
   ABg : Adam := New_Adam (1, F); ABu : Adam := New_Adam (1, F);
   AWd : Adam := New_Adam (F, D); ABd : Adam := New_Adam (1, D);
   AWout : Adam := New_Adam (D, V); ABout : Adam := New_Adam (1, V);

   --  activations
   Xa, Xn1, Qp, Kp, Vp, Oatt, Ao, H2, Xn2, Hblk, Xf : Matrix (1 .. Tn, 1 .. D);
   Aatt : Matrix (1 .. Tn, 1 .. Tn);
   Gpre, Gate, Up, Hid : Matrix (1 .. Tn, 1 .. F);
   Logits : Matrix (1 .. Tn, 1 .. V);

   --  gradients
   dE : Matrix (1 .. V, 1 .. D);
   dG1, dG2, dGf, dBq, dBk, dBv, dBo, dBd : Matrix (1 .. 1, 1 .. D);
   dWq, dWk, dWv, dWo : Matrix (1 .. D, 1 .. D);
   dWg, dWu : Matrix (1 .. D, 1 .. F); dBg, dBu : Matrix (1 .. 1, 1 .. F);
   dWd : Matrix (1 .. F, 1 .. D); dWout : Matrix (1 .. D, 1 .. V);
   dBout : Matrix (1 .. 1, 1 .. V);
   dLogits : Matrix (1 .. Tn, 1 .. V);
   dXf, dHblk, dH2, dH2b, dOatt, dXn2, dXn2a, dXn2b : Matrix (1 .. Tn, 1 .. D);
   dQ, dK, dV, dXn1, dXn1q, dXn1k, dXn1v, dXn, dXa : Matrix (1 .. Tn, 1 .. D);
   dHid, dGate, dUp, dGpre : Matrix (1 .. Tn, 1 .. F);

   --  dataset
   ST   : Synth;
   Toks : array (1 .. M) of Label_Array (1 .. Tn);
   Tgts : array (1 .. M) of Matrix (1 .. Tn, 1 .. V);

   procedure Build_Dataset is
   begin
      for Mi in 1 .. M loop
         declare
            DT : Distill.Token_Array (1 .. Tn);
         begin
            for T in 1 .. Tn loop
               Toks (Mi)(T) := (Mi * 7 + T * 5) mod V;        -- 0-based id
               DT (T) := Distill.Token (Toks (Mi)(T));
            end loop;
            declare
               S : constant Distill.Sample := Distill.Capture (ST, DT, K);
            begin
               Tgts (Mi) := [others => [others => 0.0]];      -- scatter top-K
               for R in 1 .. Tn loop
                  declare
                     P : constant Distill.Prob_Vector := Distill.Teacher_Prob (S, R);
                  begin
                     for J in 1 .. K loop
                        Tgts (Mi)(R, Integer (S.Top_Ids (R, J)) + 1) := Real (P (J));
                     end loop;
                  end;
               end loop;
            end;
         end;
      end loop;
   end Build_Dataset;

   procedure Forward_Pass (Tok : Label_Array) is
   begin
      Embed_Forward (E, Tok, Xa);
      RMSNorm_Forward (Xa, G1, Xn1);
      Linear_Forward (Xn1, Wq, Bq, Qp);
      Linear_Forward (Xn1, Wk, Bk, Kp);
      Linear_Forward (Xn1, Wv, Bv, Vp);
      Attention_Forward (Qp, Kp, Vp, Oatt, Aatt);
      Linear_Forward (Oatt, Wo, Bo, Ao);
      for I in 1 .. Tn loop for J in 1 .. D loop H2 (I, J) := Xa (I, J) + Ao (I, J); end loop; end loop;
      RMSNorm_Forward (H2, G2, Xn2);
      Linear_Forward (Xn2, Wg, Bg, Gpre);
      SiLU_Forward (Gpre, Gate);
      Linear_Forward (Xn2, Wu, Bu, Up);
      for I in 1 .. Tn loop for J in 1 .. F loop Hid (I, J) := Gate (I, J) * Up (I, J); end loop; end loop;
      Linear_Forward (Hid, Wd, Bd, Hblk);   -- reuse Hblk for Mo then add residual
      for I in 1 .. Tn loop for J in 1 .. D loop Hblk (I, J) := H2 (I, J) + Hblk (I, J); end loop; end loop;
      RMSNorm_Forward (Hblk, Gf, Xf);
      Linear_Forward (Xf, Wout, Bout, Logits);
   end Forward_Pass;

   procedure Backward_Pass (Tok : Label_Array; Tgt : Matrix) is
   begin
      KL_Backward (Logits, Tgt, dLogits);
      Linear_Backward (Xf, Wout, dLogits, dXf, dWout, dBout);
      RMSNorm_Backward (Hblk, Gf, dXf, dHblk, dGf);
      --  Hblk = H2 + Mo
      Linear_Backward (Hid, Wd, dHblk, dHid, dWd, dBd);
      for I in 1 .. Tn loop for J in 1 .. F loop
         dGate (I, J) := dHid (I, J) * Up (I, J);
         dUp   (I, J) := dHid (I, J) * Gate (I, J);
      end loop; end loop;
      Linear_Backward (Xn2, Wu, dUp, dXn2a, dWu, dBu);
      SiLU_Backward (Gpre, dGate, dGpre);
      Linear_Backward (Xn2, Wg, dGpre, dXn2b, dWg, dBg);
      for I in 1 .. Tn loop for J in 1 .. D loop dXn2 (I, J) := dXn2a (I, J) + dXn2b (I, J); end loop; end loop;
      RMSNorm_Backward (H2, G2, dXn2, dH2b, dG2);
      for I in 1 .. Tn loop for J in 1 .. D loop dH2 (I, J) := dHblk (I, J) + dH2b (I, J); end loop; end loop;
      Linear_Backward (Oatt, Wo, dH2, dOatt, dWo, dBo);
      Attention_Backward (Qp, Kp, Vp, Aatt, dOatt, dQ, dK, dV);
      Linear_Backward (Xn1, Wq, dQ, dXn1q, dWq, dBq);
      Linear_Backward (Xn1, Wk, dK, dXn1k, dWk, dBk);
      Linear_Backward (Xn1, Wv, dV, dXn1v, dWv, dBv);
      for I in 1 .. Tn loop for J in 1 .. D loop
         dXn1 (I, J) := dXn1q (I, J) + dXn1k (I, J) + dXn1v (I, J);
      end loop; end loop;
      RMSNorm_Backward (Xa, G1, dXn1, dXn, dG1);
      for I in 1 .. Tn loop for J in 1 .. D loop dXa (I, J) := dH2 (I, J) + dXn (I, J); end loop; end loop;
      Embed_Backward (Tok, dXa, dE);
   end Backward_Pass;

   procedure Step (LR : Real) is
   begin
      Adam_Step (E, dE, AE, LR);
      Adam_Step (G1, dG1, AG1, LR); Adam_Step (G2, dG2, AG2, LR); Adam_Step (Gf, dGf, AGf, LR);
      Adam_Step (Wq, dWq, AWq, LR); Adam_Step (Wk, dWk, AWk, LR);
      Adam_Step (Wv, dWv, AWv, LR); Adam_Step (Wo, dWo, AWo, LR);
      Adam_Step (Bq, dBq, ABq, LR); Adam_Step (Bk, dBk, ABk, LR);
      Adam_Step (Bv, dBv, ABv, LR); Adam_Step (Bo, dBo, ABo, LR);
      Adam_Step (Wg, dWg, AWg, LR); Adam_Step (Wu, dWu, AWu, LR);
      Adam_Step (Bg, dBg, ABg, LR); Adam_Step (Bu, dBu, ABu, LR);
      Adam_Step (Wd, dWd, AWd, LR); Adam_Step (Bd, dBd, ABd, LR);
      Adam_Step (Wout, dWout, AWout, LR); Adam_Step (Bout, dBout, ABout, LR);
   end Step;

   procedure Init_W (W : out Matrix) is begin Init_Glorot (W, G); end Init_W;
   L0, Lf : Real := 0.0;
begin
   Put_Line ("=== Aspida distillation loop (embedding -> block -> head) ===");
   Init_W (E);
   G1 := [others => [others => 1.0]]; G2 := [others => [others => 1.0]]; Gf := [others => [others => 1.0]];
   Init_W (Wq); Init_W (Wk); Init_W (Wv); Init_W (Wo);
   Init_W (Wg); Init_W (Wu); Init_W (Wd); Init_W (Wout);
   Bq := [others => [others => 0.0]]; Bk := [others => [others => 0.0]];
   Bv := [others => [others => 0.0]]; Bo := [others => [others => 0.0]];
   Bg := [others => [others => 0.0]]; Bu := [others => [others => 0.0]];
   Bd := [others => [others => 0.0]]; Bout := [others => [others => 0.0]];

   Build_Dataset;

   for S in 1 .. Steps loop
      declare
         Total : Real := 0.0;
      begin
         for Mi in 1 .. M loop
            Forward_Pass (Toks (Mi));
            Total := Total + KL_Loss (Logits, Tgts (Mi));
            Backward_Pass (Toks (Mi), Tgts (Mi));
            Step (5.0E-3);
         end loop;
         if S = 1 then L0 := Total / Real (M); end if;
         Lf := Total / Real (M);
         if S mod 300 = 0 then
            Put_Line ("  step" & S'Image & "   mean KL =" & Lf'Image);
         end if;
      end;
   end loop;

   Put_Line ("  initial mean KL =" & L0'Image & "   final mean KL =" & Lf'Image);
   New_Line;
   if Lf < 0.25 * L0 then
      Put_Line ("RESULT: PASS  (student learned the teacher distribution)");
   else
      Put_Line ("RESULT: FAIL");
   end if;
end Test_Distill_Train;
