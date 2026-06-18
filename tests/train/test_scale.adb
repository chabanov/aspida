------------------------------------------------------------------------
-- test_scale — depth generalization: a MULTI-LAYER student (Lyr blocks) is
-- trained by distillation against a synthetic teacher. If the multi-layer
-- forward/backward (residual + norm chaining across blocks) is correct, the
-- KL loss collapses. Proves the engine composes to real transformer depth.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Train;       use Train;
with Student;

procedure Test_Scale is
   Voc : constant := 24;
   Dm  : constant := 16;
   Ff  : constant := 32;
   Seq : constant := 4;
   Lyr : constant := 2;
   NSeq  : constant := 8;
   Steps : constant := 1200;

   package S is new Student (Voc => Voc, Dm => Dm, Ff => Ff, Seq => Seq, Lyr => Lyr);

   Mdl    : S.Model;
   Toks   : array (1 .. NSeq) of Label_Array (1 .. Seq);
   Tgts   : array (1 .. NSeq) of Matrix (1 .. Seq, 1 .. Voc);
   Logits : S.Logit_Mat;
   L0, Lf : Real := 0.0;
begin
   Put_Line ("=== Aspida scale: " & Integer'Image (Lyr)
             & "-layer student distillation ===");

   --  build a deterministic synthetic teacher dataset
   for M in 1 .. NSeq loop
      for T in 1 .. Seq loop
         Toks (M)(T) := (M * 5 + T * 3) mod Voc;
      end loop;
      declare
         TL : Matrix (1 .. Seq, 1 .. Voc);
      begin
         for T in 1 .. Seq loop
            declare
               Peak : constant Integer := (Toks (M)(T) * 7) mod Voc + 1;
            begin
               for C in 1 .. Voc loop
                  TL (T, C) := -0.3 * Real (abs (C - Peak));
               end loop;
            end;
         end loop;
         Softmax_Rows (TL, Tgts (M));
      end;
   end loop;

   S.Init (Mdl, 13.0);
   for Step in 1 .. Steps loop
      declare
         Total : Real := 0.0;
      begin
         for M in 1 .. NSeq loop
            S.Forward (Mdl, Toks (M), Logits);
            Total := Total + S.Backward (Mdl, Tgts (M));
            S.Step (Mdl, 5.0E-3);
         end loop;
         if Step = 1 then L0 := Total / Real (NSeq); end if;
         Lf := Total / Real (NSeq);
         if Step mod 300 = 0 then
            Put_Line ("  step" & Step'Image & "   mean KL =" & Lf'Image);
         end if;
      end;
   end loop;

   --  a final forward (uses Logits) — show the argmax at position 1
   S.Forward (Mdl, Toks (1), Logits);
   declare
      Best : Integer := 0;
      BV   : Real := Real'First;
   begin
      for C in 1 .. Voc loop
         if Logits (1, C) > BV then BV := Logits (1, C); Best := C - 1; end if;
      end loop;
      Put_Line ("  (" & Integer'Image (Lyr) & " layers) initial KL =" & L0'Image
                & "   final KL =" & Lf'Image);
      Put_Line ("  sample argmax @pos1 = token" & Best'Image);
   end;

   New_Line;
   if Lf < 0.3 * L0 then
      Put_Line ("RESULT: PASS  (deep student learned the teacher)");
   else
      Put_Line ("RESULT: FAIL");
   end if;
end Test_Scale;
