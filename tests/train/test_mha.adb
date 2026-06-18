------------------------------------------------------------------------
-- test_mha — finite-difference gradient check of multi-head attention
-- (H heads). Scalar readout L = sum(O .* Grd); compares MHA_Backward's
-- dQ/dK/dV against numerical gradients.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Train;       use Train;

procedure Test_MHA is
   T : constant := 4;
   D : constant := 6;
   H : constant := 2;          -- head_dim = 3
   Pass : Boolean := True;
   G : RNG := Seeded (4.0);

   Q, K, V, Grd : Matrix (1 .. T, 1 .. D);
   dQ, dK, dV   : Matrix (1 .. T, 1 .. D);

   function F return Real is
      O : Matrix (1 .. T, 1 .. D);
      A : Matrix (1 .. H * T, 1 .. T);
      S : Real := 0.0;
   begin
      MHA_Forward (Q, K, V, H, O, A);
      for I in 1 .. T loop
         for J in 1 .. D loop S := S + O (I, J) * Grd (I, J); end loop;
      end loop;
      return S;
   end F;

   procedure Check (W : in out Matrix; Ga : Matrix; Name : String) is
      Eps : constant Real := 1.0E-6;
      Save, Lp, Lm, Num, Rel, Maxr : Real := 0.0;
   begin
      Maxr := 0.0;
      for I in W'Range (1) loop
         for J in W'Range (2) loop
            Save := W (I, J);
            W (I, J) := Save + Eps; Lp := F;
            W (I, J) := Save - Eps; Lm := F;
            W (I, J) := Save;
            Num := (Lp - Lm) / (2.0 * Eps);
            Rel := abs (Num - Ga (I, J)) / (abs (Num) + abs (Ga (I, J)) + 1.0E-12);
            if Rel > Maxr then Maxr := Rel; end if;
         end loop;
      end loop;
      Put_Line ("  d" & Name & " max rel err =" & Maxr'Image);
      if Maxr >= 1.0E-4 then Pass := False; end if;
   end Check;

   O : Matrix (1 .. T, 1 .. D);
   A : Matrix (1 .. H * T, 1 .. T);
begin
   Put_Line ("=== Multi-head attention gradient check (H=2) ===");
   Init_Glorot (Q, G); Init_Glorot (K, G); Init_Glorot (V, G); Init_Glorot (Grd, G);
   MHA_Forward (Q, K, V, H, O, A);
   MHA_Backward (Q, K, V, A, H, Grd, dQ, dK, dV);   -- dO = Grd
   Check (Q, dQ, "Q");
   Check (K, dK, "K");
   Check (V, dV, "V");

   --  RoPE gradient check
   declare
      Xr, Gr, dXr : Matrix (1 .. T, 1 .. D);
      function FR return Real is
         Y : Matrix (1 .. T, 1 .. D);
         Sm : Real := 0.0;
      begin
         RoPE_Forward (Xr, H, 10000.0, Y);
         for I in 1 .. T loop
            for J in 1 .. D loop Sm := Sm + Y (I, J) * Gr (I, J); end loop;
         end loop;
         return Sm;
      end FR;
      Eps : constant Real := 1.0E-6;
      Save, Lp, Lm, Num, Rel, Maxr : Real := 0.0;
   begin
      Init_Glorot (Xr, G); Init_Glorot (Gr, G);
      RoPE_Backward (Gr, H, 10000.0, dXr);
      for I in 1 .. T loop
         for J in 1 .. D loop
            Save := Xr (I, J);
            Xr (I, J) := Save + Eps; Lp := FR;
            Xr (I, J) := Save - Eps; Lm := FR;
            Xr (I, J) := Save;
            Num := (Lp - Lm) / (2.0 * Eps);
            Rel := abs (Num - dXr (I, J)) / (abs (Num) + abs (dXr (I, J)) + 1.0E-12);
            if Rel > Maxr then Maxr := Rel; end if;
         end loop;
      end loop;
      Put_Line ("  dRoPE_X max rel err =" & Maxr'Image);
      if Maxr >= 1.0E-4 then Pass := False; end if;
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_MHA;
