------------------------------------------------------------------------
-- test_qat — fake quantization for quantization-aware training:
--  * forward error is bounded by half the quant step (round-to-nearest),
--  * more bits => smaller error,
--  * quantized values lie on the grid,
--  * straight-through backward is the identity,
--  * all-zero input is handled (no divide-by-zero).
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Train;       use Train;

procedure Test_QAT is
   Pass : Boolean := True;
   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   X : constant Matrix (1 .. 2, 1 .. 3) :=
     [[0.5, -0.3, 0.1], [0.9, -0.7, 0.2]];   -- max |x| = 0.9

   function Max_Err (A, B : Matrix) return Real is
      M : Real := 0.0;
   begin
      for I in A'Range (1) loop
         for J in A'Range (2) loop
            if abs (A (I, J) - B (I, J)) > M then M := abs (A (I, J) - B (I, J)); end if;
         end loop;
      end loop;
      return M;
   end Max_Err;

   Y8, Y2 : Matrix (1 .. 2, 1 .. 3);
begin
   Put_Line ("=== QAT fake-quant ===");

   Fake_Quant_Forward (X, 8, Y8);
   Fake_Quant_Forward (X, 2, Y2);

   --  8-bit: scale = 0.9/127; error <= scale/2.
   declare
      Step8 : constant Real := 0.9 / 127.0;
   begin
      Check ("8-bit error <= half a step", Max_Err (X, Y8) <= Step8 / 2.0 + 1.0e-12);
   end;

   --  2-bit: scale = 0.9/1; values land on {-0.9, 0, 0.9}.
   declare
      Step2 : constant Real := 0.9 / 1.0;
      On_Grid : Boolean := True;
   begin
      Check ("2-bit error <= half a step", Max_Err (X, Y2) <= Step2 / 2.0 + 1.0e-12);
      for I in Y2'Range (1) loop
         for J in Y2'Range (2) loop
            declare R : constant Real := Y2 (I, J) / Step2;
            begin
               if abs (R - Real'Rounding (R)) > 1.0e-9 then On_Grid := False; end if;
            end;
         end loop;
      end loop;
      Check ("2-bit values lie on the quant grid", On_Grid);
   end;

   Check ("more bits => smaller error", Max_Err (X, Y8) < Max_Err (X, Y2));

   --  Straight-through backward: DX = DY exactly.
   declare
      DY : constant Matrix (1 .. 2, 1 .. 3) :=
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
      DX : Matrix (1 .. 2, 1 .. 3);
   begin
      Fake_Quant_Backward (DY, DX);
      Check ("STE backward is identity", Max_Err (DX, DY) = 0.0);
   end;

   --  Degenerate all-zero input must not divide by zero.
   declare
      Z  : constant Matrix (1 .. 1, 1 .. 2) := [[0.0, 0.0]];
      YZ : Matrix (1 .. 1, 1 .. 2);
   begin
      Fake_Quant_Forward (Z, 8, YZ);
      Check ("all-zero input handled", YZ (1, 1) = 0.0 and then YZ (1, 2) = 0.0);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_QAT;
