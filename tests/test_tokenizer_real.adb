---------------------------------------------------------------------
-- Test LLM_Tokenizer against the REAL Qwen GGUF vocabulary.
--
-- Loads tokenizer.ggml.tokens/.merges (248k/247k) from the model and
-- verifies GPT-2 byte-level round-trip: Decode(Encode(s)) = s, plus
-- that BPE merging actually fires on common words. Skips cleanly if the
-- model file is not present.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Environment_Variables;
with LLM_GGUF;
with LLM_Tokenizer; use LLM_Tokenizer;

procedure Test_Tokenizer_Real is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Condition : Boolean) is
   begin
      if Condition then
         Put_Line ("  PASS: " & Name);
         Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);
         Failed := Failed + 1;
      end if;
   end Assert;

   function Model_Path return String is
      Var : constant String := "QWEN_MODEL_PATH";
   begin
      if Ada.Environment_Variables.Exists (Var) then
         return Ada.Environment_Variables.Value (Var);
      end if;
      return "/Users/ceo/.lmstudio/models/HauhauCS/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf";
   end Model_Path;

   G   : LLM_GGUF.GGUF_File;
   Tok : Tokenizer := Create;

   procedure Round_Trip (S : String) is
      Ids : constant Token_Array := Encode (Tok, S);
      Out_S : constant String := Decode (Tok, Ids);
   begin
      Assert ("round-trip [" & S & "] (" & Integer'Image (Ids'Length) & " toks)",
        Out_S = S);
   end Round_Trip;

begin
   Put_Line ("=== Tokenizer Real-Vocab Test Suite ===");
   New_Line;

   LLM_GGUF.Open (G, Model_Path);
   if not LLM_GGUF.Is_Open (G) or else LLM_GGUF.Token_Count (G) = 0 then
      Put_Line ("  SKIP: model not found at " & Model_Path);
      return;
   end if;

   LLM_Tokenizer.Load_From_GGUF (Tok, G);
   LLM_GGUF.Close (G);

   New_Line;
   Put_Line ("  vocab size:" & Integer'Image (Vocab_Size (Tok))
             & "   loaded: " & Boolean'Image (Is_Loaded (Tok)));
   New_Line;

   --  A real LLM vocab is tens of thousands of tokens (Llama 128k, Qwen 152k–
   --  248k, Gemma 262k) — far above the 256 byte-fallback of a synthetic one.
   Assert ("loaded a real vocab (>10k tokens)", Vocab_Size (Tok) > 10_000);

   Round_Trip ("Hello, world!");
   Round_Trip ("The quick brown fox jumps over the lazy dog.");
   Round_Trip ("def foo(x): return x + 1");
   Round_Trip ("   leading and  internal   spaces");
   Round_Trip ("Numbers 12345 and symbols @#$%^&*()");
   Round_Trip ("");

   --  BPE must actually merge: a common word is far fewer tokens than bytes.
   declare
      Ids : constant Token_Array := Encode (Tok, " transformer");
   begin
      Assert ("BPE merges (' transformer' < 12 byte-tokens)", Ids'Length < 12);
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Tokenizer_Real;
