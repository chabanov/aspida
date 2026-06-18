------------------------------------------------------------------------
-- test_ckpt — Student checkpoint round-trip including AdamW optimizer state.
-- Train a few steps (so Adam moments M/V and step T are non-trivial), Save,
-- Load into a DIFFERENTLY-initialized model, re-Save, and assert the two
-- files are byte-identical. If optimizer state were dropped (or weights not
-- fully restored), the re-saved file would differ.
------------------------------------------------------------------------

with Ada.Text_IO;            use Ada.Text_IO;
with Ada.Streams;           use Ada.Streams;
with Ada.Streams.Stream_IO;
with Train;                  use Train;
with Student;

procedure Test_Ckpt is
   package SIO renames Ada.Streams.Stream_IO;
   package S is new Student
     (Voc => 8, Dm => 16, Ff => 32, Seq => 4, Lyr => 2, Heads => 2,
      Use_RoPE => True);

   A    : S.Model;
   B    : S.Model;
   Toks : constant Label_Array (1 .. 4) := [1, 2, 3, 4];
   L    : S.Logit_Mat;
   P    : Matrix (1 .. 4, 1 .. 8);
   Tgt  : S.Logit_Mat;
   Loss : Real := 0.0;

   F1 : constant String := "/tmp/aspida_ckpt_a.bin";
   F2 : constant String := "/tmp/aspida_ckpt_b.bin";

   function Files_Equal (P1, P2 : String) return Boolean is
      Fa, Fb : SIO.File_Type;
      Ba, Bb : Stream_Element_Array (1 .. 4096);
      La, Lb : Stream_Element_Offset;
      Eq     : Boolean := True;
   begin
      SIO.Open (Fa, SIO.In_File, P1);
      SIO.Open (Fb, SIO.In_File, P2);
      loop
         SIO.Read (Fa, Ba, La);
         SIO.Read (Fb, Bb, Lb);
         if La /= Lb then Eq := False; exit; end if;
         exit when La < Ba'First;                         -- both at EOF
         if Ba (Ba'First .. La) /= Bb (Bb'First .. Lb) then
            Eq := False; exit;
         end if;
      end loop;
      SIO.Close (Fa);
      SIO.Close (Fb);
      return Eq;
   end Files_Equal;

   Pass : Boolean := True;
begin
   Put_Line ("=== Checkpoint optimizer-state round-trip ===");
   S.Init (A, 7.0);
   for It in 1 .. 5 loop
      S.Forward (A, Toks, L);
      Softmax_Rows (L, P);
      Tgt := P;
      for R in 1 .. 4 loop
         for C in 1 .. 8 loop Tgt (R, C) := 0.0; end loop;
         Tgt (R, (R mod 8) + 1) := 1.0;
      end loop;
      Loss := S.Backward (A, Tgt);
      S.Step (A, 1.0E-2);
   end loop;
   Put_Line ("  trained 5 steps, loss=" & Loss'Image);

   S.Save (A, F1);
   S.Init (B, 999.0);          -- different init: Load must overwrite everything
   S.Load (B, F1);
   S.Save (B, F2);

   if Files_Equal (F1, F2) then
      Put_Line ("  PASS: re-saved checkpoint byte-identical (weights + AdamW)");
   else
      Put_Line ("  FAIL: checkpoints differ -> optimizer state not round-tripped");
      Pass := False;
   end if;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Ckpt;
