------------------------------------------------------------------------
-- test_bpe — from-scratch BPE tokenizer trainer.
--  * learns merges from a corpus (vocab grows past the 256-byte base),
--  * Decode (Encode (X)) = X for seen, unseen, empty and edge inputs,
--  * a frequent string encodes to far fewer tokens than its raw bytes.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with BPE_Train;             use BPE_Train;

procedure Test_BPE is
   Pass : Boolean := True;

   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then
         Put_Line ("  PASS: " & Name);
      else
         Put_Line ("  FAIL: " & Name);
         Pass := False;
      end if;
   end Check;

   T : Trainer;

   --  A small corpus with strong repetition so common pieces get merged.
   function Corpus return String is
      One : constant String := "the cat sat on the mat and the dog ran to the cat. ";
      R   : Unbounded_String;
   begin
      for I in 1 .. 40 loop Append (R, One); end loop;
      return To_String (R);
   end Corpus;

   procedure Round_Trip (Label, Text : String) is
      Ids : constant Id_Array := Encode (T, Text);
      Back : constant String  := Decode (T, Ids);
   begin
      Check ("round-trip " & Label, Back = Text);
   end Round_Trip;

begin
   Put_Line ("=== BPE tokenizer trainer ===");
   Train (T, Corpus, Target_Vocab => 320);

   Put_Line ("  vocab size =" & Vocab_Size (T)'Image
             & "   merges =" & Num_Merges (T)'Image);
   Check ("base alphabet present (>=256)", Vocab_Size (T) >= 256);
   Check ("merges were learned",          Num_Merges (T) > 0);
   Check ("vocab honours target",         Vocab_Size (T) <= 320);

   --  Lossless on training text, unseen text, and edge cases.
   Round_Trip ("seen",        "the cat sat on the mat");
   Round_Trip ("unseen",      "ZEBRA quok 42!@# zzz");
   Round_Trip ("empty",       "");
   Round_Trip ("spaces only", "     ");
   Round_Trip ("punctuation", ".,;:!?()[]{}");
   Round_Trip ("newlines",    "a" & Character'Val (10) & "b" & Character'Val (9) & "c");

   --  Compression: a frequent string should need far fewer tokens than bytes.
   declare
      S   : constant String := "the cat sat on the mat";
      Ids : constant Id_Array := Encode (T, S);
   begin
      Put_Line ("  ""the cat sat on the mat"":" & Ids'Length'Image
                & " tokens vs" & S'Length'Image & " bytes");
      Check ("merges compress frequent text", Ids'Length < S'Length);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_BPE;
