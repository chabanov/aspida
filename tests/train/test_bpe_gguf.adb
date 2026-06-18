------------------------------------------------------------------------
-- test_bpe_gguf — embed a learned BPE vocabulary into a GGUF (byte-level /
-- gpt2 tokenizer) and prove our own engine's LLM_Tokenizer loads it and
-- reproduces the trainer's tokenization EXACTLY (same ids) and decodes
-- losslessly. This closes: learn vocab -> export to GGUF -> serve with our
-- tokenizer.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GGUF_Write;
with BPE_Train;
with BPE_GGUF;
with LLM_GGUF;
with LLM_Tokenizer;

procedure Test_BPE_GGUF is
   Pass : Boolean := True;

   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   function Corpus return String is
      One : constant String := "the cat sat on the mat and the dog ran to the cat. ";
      R   : Unbounded_String;
   begin
      for I in 1 .. 40 loop Append (R, One); end loop;
      return To_String (R);
   end Corpus;

   Path : constant String := "/tmp/aspida_bpe.gguf";
   T    : BPE_Train.Trainer;
   Tok  : LLM_Tokenizer.Tokenizer := LLM_Tokenizer.Create;
   G    : LLM_GGUF.GGUF_File;

   procedure Round (Label, S : String) is
      IdsT : constant BPE_Train.Id_Array := BPE_Train.Encode (T, S);
      IdsK : constant LLM_Tokenizer.Token_Array := LLM_Tokenizer.Encode (Tok, S);
      OK   : Boolean := IdsT'Length = IdsK'Length;
   begin
      if OK then
         for I in IdsT'Range loop
            if Natural (IdsK (IdsK'First + (I - IdsT'First))) /= IdsT (I) then
               OK := False;
            end if;
         end loop;
      end if;
      Check ("engine encode == trainer encode: " & Label, OK);
      Check ("engine decode lossless: " & Label,
             LLM_Tokenizer.Decode (Tok, IdsK) = S);
   end Round;

begin
   Put_Line ("=== BPE vocab -> GGUF -> engine tokenizer ===");
   BPE_Train.Train (T, Corpus, Target_Vocab => 320);

   --  Build a GGUF carrying the learned tokenizer (+ one dummy tensor so the
   --  file is well-formed) and write it.
   declare
      B : GGUF_Write.Builder;
   begin
      BPE_GGUF.Write_Tokenizer (B, T);
      GGUF_Write.Add_Tensor_F32 (B, "x", [1], [0.0]);
      GGUF_Write.Save (B, Path);
   end;

   LLM_GGUF.Open (G, Path);
   Check ("GGUF opened", LLM_GGUF.Is_Open (G));
   Check ("token count matches vocab",
          LLM_GGUF.Token_Count (G) = BPE_Train.Vocab_Size (T));
   Check ("merge count matches",
          LLM_GGUF.Merge_Count (G) = BPE_Train.Num_Merges (T));

   LLM_Tokenizer.Load_From_GGUF (Tok, G);
   Check ("tokenizer loaded", LLM_Tokenizer.Is_Loaded (Tok));

   Round ("seen",        "the cat sat on the mat");
   Round ("unseen",      "ZEBRA quok 42!@# zzz");
   Round ("empty",       "");
   Round ("spaces only", "     ");
   Round ("punctuation", ".,;:!?()[]{}");

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_BPE_GGUF;
