------------------------------------------------------------------------
-- test_block — gradient-check a complete tiny Llama-style decoder block
-- assembled from the training primitives:
--
--   xn1 = RMSNorm(h, g1)
--   Q,K,V = xn1.Wq, xn1.Wk, xn1.Wv          (single-head causal attention)
--   h2  = h + Attention(Q,K,V).Wo
--   xn2 = RMSNorm(h2, g2)
--   y   = h2 + ( SiLU(xn2.Wg) (*) xn2.Wu ).Wd     (SwiGLU MLP)
--
-- Every parameter AND the block input gradient are verified against finite
-- differences. The input gradient deliberately combines the residual path
-- with the norm path (a classic place to drop a term) — if our backward is
-- wrong, this check fails.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Train;       use Train;

procedure Test_Block is
   T : constant := 3;     -- sequence length
   D : constant := 4;     -- model dim
   F : constant := 8;     -- ffn dim
   Pass : Boolean := True;
   G : RNG := Seeded (3.0);

   --  parameters
   G1 : Matrix (1 .. 1, 1 .. D); G2 : Matrix (1 .. 1, 1 .. D);
   Wq : Matrix (1 .. D, 1 .. D); Bq : Matrix (1 .. 1, 1 .. D);
   Wk : Matrix (1 .. D, 1 .. D); Bk : Matrix (1 .. 1, 1 .. D);
   Wv : Matrix (1 .. D, 1 .. D); Bv : Matrix (1 .. 1, 1 .. D);
   Wo : Matrix (1 .. D, 1 .. D); Bo : Matrix (1 .. 1, 1 .. D);
   Wg : Matrix (1 .. D, 1 .. F); Bg : Matrix (1 .. 1, 1 .. F);
   Wu : Matrix (1 .. D, 1 .. F); Bu : Matrix (1 .. 1, 1 .. F);
   Wd : Matrix (1 .. F, 1 .. D); Bd : Matrix (1 .. 1, 1 .. D);

   --  block input + teacher target
   H     : Matrix (1 .. T, 1 .. D);
   Teach : Matrix (1 .. T, 1 .. D);

   --  forward intermediates (shared by Loss and Backward)
   Xn1, Qp, Kp, Vp, Oatt, Ao, H2, Xn2, Yo : Matrix (1 .. T, 1 .. D);
   Aatt : Matrix (1 .. T, 1 .. T);
   Gpre, Gate, Up, Hid : Matrix (1 .. T, 1 .. F);

   --  parameter gradients
   dG1, dG2, dBq, dBk, dBv, dBo, dBd : Matrix (1 .. 1, 1 .. D);
   dWq, dWk, dWv, dWo : Matrix (1 .. D, 1 .. D);
   dWg, dWu : Matrix (1 .. D, 1 .. F);
   dWd : Matrix (1 .. F, 1 .. D);
   dBg, dBu : Matrix (1 .. 1, 1 .. F);
   dH : Matrix (1 .. T, 1 .. D);

   procedure Forward is
   begin
      RMSNorm_Forward (H, G1, Xn1);
      Linear_Forward (Xn1, Wq, Bq, Qp);
      Linear_Forward (Xn1, Wk, Bk, Kp);
      Linear_Forward (Xn1, Wv, Bv, Vp);
      Attention_Forward (Qp, Kp, Vp, Oatt, Aatt);
      Linear_Forward (Oatt, Wo, Bo, Ao);
      for I in 1 .. T loop for J in 1 .. D loop
         H2 (I, J) := H (I, J) + Ao (I, J);
      end loop; end loop;
      RMSNorm_Forward (H2, G2, Xn2);
      Linear_Forward (Xn2, Wg, Bg, Gpre);
      SiLU_Forward (Gpre, Gate);
      Linear_Forward (Xn2, Wu, Bu, Up);
      for I in 1 .. T loop for J in 1 .. F loop
         Hid (I, J) := Gate (I, J) * Up (I, J);
      end loop; end loop;
      Linear_Forward (Hid, Wd, Bd, Yo);  -- reuse Yo storage [T,D]
      for I in 1 .. T loop for J in 1 .. D loop
         Yo (I, J) := H2 (I, J) + Yo (I, J);
      end loop; end loop;
   end Forward;

   function Loss return Real is
   begin
      Forward;
      return KL_Loss (Yo, Teach);
   end Loss;

   procedure Backward is
      dYo, dMo, dH2, dH2b, dHn, dOatt : Matrix (1 .. T, 1 .. D);
      dXn2, dXn2a, dXn2b : Matrix (1 .. T, 1 .. D);
      dQ, dK, dV, dXn1, dXn1q, dXn1k, dXn1v : Matrix (1 .. T, 1 .. D);
      dHid, dGate, dUp, dGpre : Matrix (1 .. T, 1 .. F);
   begin
      Forward;
      KL_Backward (Yo, Teach, dYo);
      --  Yo = H2 + Mo
      dMo := dYo;
      Linear_Backward (Hid, Wd, dMo, dHid, dWd, dBd);
      --  Hid = Gate (*) Up
      for I in 1 .. T loop for J in 1 .. F loop
         dGate (I, J) := dHid (I, J) * Up (I, J);
         dUp   (I, J) := dHid (I, J) * Gate (I, J);
      end loop; end loop;
      Linear_Backward (Xn2, Wu, dUp, dXn2a, dWu, dBu);
      SiLU_Backward (Gpre, dGate, dGpre);
      Linear_Backward (Xn2, Wg, dGpre, dXn2b, dWg, dBg);
      for I in 1 .. T loop for J in 1 .. D loop
         dXn2 (I, J) := dXn2a (I, J) + dXn2b (I, J);
      end loop; end loop;
      RMSNorm_Backward (H2, G2, dXn2, dH2b, dG2);
      --  H2 feeds both Yo (residual) and Xn2 (norm)
      for I in 1 .. T loop for J in 1 .. D loop
         dH2 (I, J) := dYo (I, J) + dH2b (I, J);
      end loop; end loop;
      --  Ao = Oatt.Wo
      Linear_Backward (Oatt, Wo, dH2, dOatt, dWo, dBo);
      Attention_Backward (Qp, Kp, Vp, Aatt, dOatt, dQ, dK, dV);
      Linear_Backward (Xn1, Wq, dQ, dXn1q, dWq, dBq);
      Linear_Backward (Xn1, Wk, dK, dXn1k, dWk, dBk);
      Linear_Backward (Xn1, Wv, dV, dXn1v, dWv, dBv);
      for I in 1 .. T loop for J in 1 .. D loop
         dXn1 (I, J) := dXn1q (I, J) + dXn1k (I, J) + dXn1v (I, J);
      end loop; end loop;
      RMSNorm_Backward (H, G1, dXn1, dHn, dG1);
      --  H feeds both H2 (residual) and Xn1 (norm)
      for I in 1 .. T loop for J in 1 .. D loop
         dH (I, J) := dH2 (I, J) + dHn (I, J);
      end loop; end loop;
   end Backward;

   procedure Check (W : in out Matrix; Ga : Matrix; Name : String) is
      Eps : constant Real := 1.0E-6;
      Save, Lp, Lm, Num, Ana, Rel : Real;
      Max : Real := 0.0;
   begin
      for I in W'Range (1) loop
         for J in W'Range (2) loop
            Save := W (I, J);
            W (I, J) := Save + Eps;  Lp := Loss;
            W (I, J) := Save - Eps;  Lm := Loss;
            W (I, J) := Save;
            Num := (Lp - Lm) / (2.0 * Eps);
            Ana := Ga (I, J);
            Rel := abs (Num - Ana) / (abs (Num) + abs (Ana) + 1.0E-12);
            if Rel > Max then Max := Rel; end if;
         end loop;
      end loop;
      Put_Line ("  " & Name & ": max rel err =" & Max'Image);
      if Max >= 1.0E-4 then Pass := False; end if;
   end Check;

   procedure Init (W : out Matrix) is begin Init_Glorot (W, G); end Init;
   procedure Ones (W : out Matrix) is begin W := [others => [others => 1.0]]; end Ones;

begin
   Put_Line ("=== Aspida transformer-block gradient check ===");
   Ones (G1); Ones (G2);
   Init (Wq); Init (Wk); Init (Wv); Init (Wo);
   Init (Wg); Init (Wu); Init (Wd);
   Init (Bq); Init (Bk); Init (Bv); Init (Bo);
   Init (Bg); Init (Bu); Init (Bd);
   Init (H);
   declare
      TL : Matrix (1 .. T, 1 .. D);
   begin
      Init (TL); Softmax_Rows (TL, Teach);
   end;

   Backward;     -- analytic grads at the base point
   Check (G1, dG1, "RMSNorm g1");
   Check (G2, dG2, "RMSNorm g2");
   Check (Wq, dWq, "Wq");
   Check (Wk, dWk, "Wk");
   Check (Wv, dWv, "Wv");
   Check (Wo, dWo, "Wo (attn out)");
   Check (Wg, dWg, "Wg (gate)");
   Check (Wu, dWu, "Wu (up)");
   Check (Wd, dWd, "Wd (down)");
   Check (Bq, dBq, "bq");
   Check (Bd, dBd, "bd");
   Check (H,  dH,  "input H (residual+norm)");

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Block;
