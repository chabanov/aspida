------------------------------------------------------------------------
-- test_train — proves the from-scratch training core is correct:
--   1) finite-difference gradient check of the full Linear->SiLU->Linear->KL
--      pipeline (every parameter), and
--   2) a tiny MLP actually learns to match a synthetic teacher distribution
--      via AdamW on the KL loss (loss must collapse toward zero).
-- This is the bedrock for the teacher->student distillation engine.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Train;       use Train;

procedure Test_Train is
   Pass : Boolean := True;

   --------------------------------------------------------------------
   --  Part 1 — gradient check (small dims for speed & clarity)
   --------------------------------------------------------------------
   procedure Gradient_Check is
      Inp : constant := 3;
      H   : constant := 4;
      Ou  : constant := 3;
      N   : constant := 2;
      G   : RNG := Seeded (1.0);

      W1 : Matrix (1 .. Inp, 1 .. H);  B1 : Matrix (1 .. 1, 1 .. H);
      W2 : Matrix (1 .. H,  1 .. Ou);  B2 : Matrix (1 .. 1, 1 .. Ou);
      X  : Matrix (1 .. N,  1 .. Inp);
      T  : Matrix (1 .. N,  1 .. Ou);

      dW1 : Matrix (1 .. Inp, 1 .. H);  dB1 : Matrix (1 .. 1, 1 .. H);
      dW2 : Matrix (1 .. H,  1 .. Ou);  dB2 : Matrix (1 .. 1, 1 .. Ou);

      function Loss return Real is
         H1 : Matrix (1 .. N, 1 .. H);
         A1 : Matrix (1 .. N, 1 .. H);
         Lg : Matrix (1 .. N, 1 .. Ou);
      begin
         Linear_Forward (X, W1, B1, H1);
         SiLU_Forward (H1, A1);
         Linear_Forward (A1, W2, B2, Lg);
         return KL_Loss (Lg, T);
      end Loss;

      procedure Backward is
         H1 : Matrix (1 .. N, 1 .. H);
         A1 : Matrix (1 .. N, 1 .. H);
         Lg : Matrix (1 .. N, 1 .. Ou);
         dLg : Matrix (1 .. N, 1 .. Ou);
         dA1 : Matrix (1 .. N, 1 .. H);
         dH1 : Matrix (1 .. N, 1 .. H);
      begin
         Linear_Forward (X, W1, B1, H1);
         SiLU_Forward (H1, A1);
         Linear_Forward (A1, W2, B2, Lg);
         KL_Backward (Lg, T, dLg);
         Linear_Backward (A1, W2, dLg, dA1, dW2, dB2);
         SiLU_Backward (H1, dA1, dH1);
         Linear_Backward_NoDX (X, W1, dH1, dW1, dB1);
      end Backward;

      procedure Check (W : in out Matrix; Ga : Matrix; Name : String) is
         Eps  : constant Real := 1.0E-6;
         Save, Lp, Lm, Num, Ana, Rel : Real;
         Max  : Real := 0.0;
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
         Put_Line ("  grad-check " & Name & ": max rel err =" & Max'Image);
         if Max >= 1.0E-4 then Pass := False; end if;
      end Check;
   begin
      Init_Glorot (W1, G);  Init_Glorot (W2, G);  Init_Glorot (X, G);
      for J in 1 .. H  loop B1 (1, J) := 0.1 * Uniform (G); end loop;
      for J in 1 .. Ou loop B2 (1, J) := 0.1 * Uniform (G); end loop;
      declare
         TL : Matrix (1 .. N, 1 .. Ou);
      begin
         Init_Glorot (TL, G);
         Softmax_Rows (TL, T);            -- a valid teacher distribution
      end;

      Backward;
      Put_Line ("[1] gradient check (Linear->SiLU->Linear->KL):");
      Check (W1, dW1, "W1");
      Check (B1, dB1, "B1");
      Check (W2, dW2, "W2");
      Check (B2, dB2, "B2");
   end Gradient_Check;

   --------------------------------------------------------------------
   --  Part 2 — a tiny MLP learns to mimic a synthetic teacher (KL)
   --------------------------------------------------------------------
   procedure Train_Distill is
      Inp : constant := 8;
      H   : constant := 16;
      Ou  : constant := 5;
      N   : constant := 32;
      G   : RNG := Seeded (7.0);

      W1 : Matrix (1 .. Inp, 1 .. H);  B1 : Matrix (1 .. 1, 1 .. H) := [others => [others => 0.0]];
      W2 : Matrix (1 .. H,  1 .. Ou);  B2 : Matrix (1 .. 1, 1 .. Ou) := [others => [others => 0.0]];
      X  : Matrix (1 .. N,  1 .. Inp);
      T  : Matrix (1 .. N,  1 .. Ou);

      AW1 : Adam := New_Adam (Inp, H);  AB1 : Adam := New_Adam (1, H);
      AW2 : Adam := New_Adam (H, Ou);   AB2 : Adam := New_Adam (1, Ou);

      H1 : Matrix (1 .. N, 1 .. H);   A1 : Matrix (1 .. N, 1 .. H);
      Lg : Matrix (1 .. N, 1 .. Ou);
      dLg : Matrix (1 .. N, 1 .. Ou); dA1 : Matrix (1 .. N, 1 .. H);
      dH1 : Matrix (1 .. N, 1 .. H);
      dW1 : Matrix (1 .. Inp, 1 .. H); dB1 : Matrix (1 .. 1, 1 .. H);
      dW2 : Matrix (1 .. H, 1 .. Ou);  dB2 : Matrix (1 .. 1, 1 .. Ou);
      L, L0 : Real := 0.0;
   begin
      Init_Glorot (W1, G);  Init_Glorot (W2, G);  Init_Glorot (X, G);
      declare
         TL : Matrix (1 .. N, 1 .. Ou);
      begin
         Init_Glorot (TL, G);
         Softmax_Rows (TL, T);
      end;

      Put_Line ("[2] distillation training (student MLP -> teacher dist):");
      for Step in 1 .. 1500 loop
         Linear_Forward (X, W1, B1, H1);
         SiLU_Forward (H1, A1);
         Linear_Forward (A1, W2, B2, Lg);
         L := KL_Loss (Lg, T);
         if Step = 1 then L0 := L; end if;

         KL_Backward (Lg, T, dLg);
         Linear_Backward (A1, W2, dLg, dA1, dW2, dB2);
         SiLU_Backward (H1, dA1, dH1);
         Linear_Backward_NoDX (X, W1, dH1, dW1, dB1);

         Adam_Step (W1, dW1, AW1, LR => 1.0E-2);
         Adam_Step (B1, dB1, AB1, LR => 1.0E-2);
         Adam_Step (W2, dW2, AW2, LR => 1.0E-2);
         Adam_Step (B2, dB2, AB2, LR => 1.0E-2);

         if Step mod 300 = 0 then
            Put_Line ("  step" & Step'Image & "   KL =" & L'Image);
         end if;
      end loop;
      Put_Line ("  initial KL =" & L0'Image & "   final KL =" & L'Image);
      if L > 0.02 then Pass := False; end if;
   end Train_Distill;

begin
   Put_Line ("=== Aspida training core self-test ===");
   Gradient_Check;
   New_Line;
   Train_Distill;
   New_Line;
   if Pass then
      Put_Line ("RESULT: PASS");
   else
      Put_Line ("RESULT: FAIL");
   end if;
end Test_Train;
