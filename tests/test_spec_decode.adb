------------------------------------------------------------------------
-- test_spec_decode — proves the speculative-decoding invariant on the real
-- hura model pair: for greedy decoding, the tokens produced by draft+target
-- speculation are BYTE-IDENTICAL to what the target produces greedily alone.
-- Speculation changes speed, never output.
--
-- Needs both GGUFs. Point at them with:
--   HURA_MODEL=/path/to/hura.gguf  HURA_DRAFT=/path/to/hura-9b.gguf
-- Absent either, the test SKIPS (documented) rather than failing, so the
-- model-free suite stays green on a machine without the weights.
------------------------------------------------------------------------
with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Command_Line;        use Ada.Command_Line;
with Ada.Environment_Variables;
with LLM_Qwen;
with LLM_Tokenizer;           use LLM_Tokenizer;
with LLM_Spec_Decode;

procedure Test_Spec_Decode is

   Pass : Boolean := True;
   procedure Chk (Name : String; Cond : Boolean) is
   begin
      Put_Line ("  " & (if Cond then "PASS" else "FAIL") & ": " & Name);
      if not Cond then Pass := False; end if;
   end Chk;

   function Env (Name : String) return String is
     (if Ada.Environment_Variables.Exists (Name)
      then Ada.Environment_Variables.Value (Name) else "");

   T_Path : constant String := Env ("HURA_MODEL");
   D_Path : constant String := Env ("HURA_DRAFT");

   --  Pure greedy from the target alone — the oracle the speculation must match.
   function Greedy_Baseline
     (M : LLM_Qwen.Qwen_Model; Prompt : Token_Array;
      Max_New : Positive; Stop_Id : Integer) return Token_Array
   is
      Vocab : constant Natural := LLM_Qwen.Vocab_Size (M);
      Seq   : Token_Array (1 .. Prompt'Length + Max_New);
      Len   : Natural := Prompt'Length;
      Out_A : Token_Array (1 .. Max_New);
      NO    : Natural := 0;
   begin
      Seq (1 .. Len) := Prompt;
      while NO < Max_New loop
         declare
            L : constant LLM_Qwen.Logits_Flat :=
              LLM_Qwen.Forward_Logits (M, Seq (1 .. Len));
            Base : constant Natural := (Len - 1) * Vocab;
            Best : Natural := 0;
            Bv   : Float := L (Base);
         begin
            for K in 1 .. Vocab - 1 loop
               if L (Base + K) > Bv then Bv := L (Base + K); Best := K; end if;
            end loop;
            NO := NO + 1; Out_A (NO) := Best;
            Len := Len + 1; Seq (Len) := Best;
            exit when Best = Stop_Id;
         end;
      end loop;
      return Out_A (1 .. NO);
   end Greedy_Baseline;

begin
   Put_Line ("=== speculative decoding: greedy == target-alone (byte-identical) ===");

   if T_Path = "" or else D_Path = "" then
      Put_Line ("SKIP: set HURA_MODEL and HURA_DRAFT to the two GGUFs");
      Put_Line ("RESULT: PASS");
      return;
   end if;

   declare
      Target : LLM_Qwen.Qwen_Model := LLM_Qwen.Load (T_Path);
      Draft  : LLM_Qwen.Qwen_Model := LLM_Qwen.Load (D_Path);
      --  The model-draft phase runs the 35B twice per step and is very slow on
      --  CPU (~40 min for 24 tokens). SPEC_LOOKUP_ONLY skips it to iterate fast
      --  on the lookup path (verified vs the 9B in seconds), since the fix under
      --  active development — Verify_Round's bounds-agnostic indexing — is
      --  model-independent. Unset, both phases run.
      Skip_Model_Draft : constant Boolean :=
        Ada.Environment_Variables.Exists ("SPEC_LOOKUP_ONLY");
   begin
      Chk ("vocab sizes match",
           LLM_Qwen.Vocab_Size (Target) = LLM_Qwen.Vocab_Size (Draft));

      if not Skip_Model_Draft then
      declare
         --  A short prompt with a determinate continuation keeps the run fast
         --  on CPU while still exercising several accept/reject rounds.
         Prompt  : constant Token_Array :=
           LLM_Qwen.Encode (Target, "The capital of France is");
         Max_New : constant Positive := 24;
         Stop    : constant Integer := -1;   -- no early stop; compare full run
         St      : aliased LLM_Spec_Decode.Stats;

         Baseline : constant Token_Array :=
           Greedy_Baseline (Target, Prompt, Max_New, Stop);
         Spec     : constant Token_Array :=
           LLM_Spec_Decode.Generate
             (Draft => Draft, Target => Target, Prompt_Ids => Prompt,
              Max_New_Tokens => Max_New, Stop_Id => Stop, Gamma => 4,
              Result_Stats => St'Access);

         Ident : Boolean := Baseline'Length = Spec'Length;
      begin
         if Ident then
            for I in Baseline'Range loop
               if Baseline (I) /= Spec (Baseline'First + (I - Baseline'First))
               then Ident := False; end if;
            end loop;
         end if;

         Put_Line ("  baseline: " & LLM_Qwen.Decode (Target, Baseline));
         Put_Line ("  spec    : " & LLM_Qwen.Decode (Target, Spec));
         Chk ("same length", Baseline'Length = Spec'Length);
         Chk ("BYTE-IDENTICAL to target-alone greedy", Ident);

         --  Report the numbers that decide whether GPU work is worth it.
         Put_Line ("  --- stats (Gamma=4) ---");
         Put_Line ("  proposed=" & St.Proposed'Image
                   & " accepted=" & St.Accepted'Image
                   & " target_forwards=" & St.Target_Forwards'Image
                   & " emitted=" & St.Emitted'Image);
         if St.Proposed > 0 then
            Put_Line ("  acceptance rate ="
                      & Integer'Image (St.Accepted * 100 / St.Proposed) & "%");
         end if;
         if St.Target_Forwards > 0 then
            Put_Line ("  tokens per target forward ="
                      & Integer'Image (St.Emitted * 100 / St.Target_Forwards)
                      & "/100  (>100 = speculation is winning)");
         end if;
      end;
      end if;  -- Skip_Model_Draft

      --  PROMPT-LOOKUP path: no draft model, target only. Same invariant —
      --  byte-identical to target-alone greedy — and it should actually WIN
      --  here, because the prompt asks for a verbatim repeat, which is exactly
      --  what lookup drafts for free. Verified against the 9B (already loaded)
      --  rather than the 35B: the invariant holds for ANY Qwen target, and the
      --  fix under test (Verify_Round indexing a non-1-based lookup slice) is
      --  model-independent, so the 9B is a faster but equally valid witness.
      New_Line;
      Put_Line ("  --- prompt-lookup (no draft model, verified vs 9B) ---");
      declare
         LTgt : LLM_Qwen.Qwen_Model renames Draft;   -- 9B, ~4x faster to run
         Prompt : constant Token_Array :=
           LLM_Qwen.Encode
             (LTgt,
              "abc abc abc abc abc abc");
         --  Small on purpose: the CPU reference re-projects the whole vocab at
         --  every position each step (O(n^2) x 248k vocab), so keep it short.
         --  6 tokens on a repeating "abc" still exercises multiple lookup
         --  hits, the bounds-agnostic Verify_Round indexing, and the G=0
         --  fallback. It is a correctness witness, not a benchmark.
         Max_New : constant Positive := 6;
         Stop    : constant Integer := -1;
         St      : aliased LLM_Spec_Decode.Stats;

         Baseline : constant Token_Array :=
           Greedy_Baseline (LTgt, Prompt, Max_New, Stop);
         Look     : constant Token_Array :=
           LLM_Spec_Decode.Generate_Lookup
             (Target => LTgt, Prompt_Ids => Prompt, Max_New_Tokens => Max_New,
              Stop_Id => Stop, Ngram => 3, Gamma => 8, Result_Stats => St'Access);

         Ident : Boolean := Baseline'Length = Look'Length;
      begin
         if Ident then
            for I in Baseline'Range loop
               if Baseline (I) /= Look (Baseline'First + (I - Baseline'First))
               then Ident := False; end if;
            end loop;
         end if;
         Put_Line ("  baseline: " & LLM_Qwen.Decode (LTgt, Baseline));
         Put_Line ("  lookup  : " & LLM_Qwen.Decode (LTgt, Look));
         Chk ("lookup: same length", Baseline'Length = Look'Length);
         Chk ("lookup: BYTE-IDENTICAL to target-alone greedy", Ident);
         Put_Line ("  proposed=" & St.Proposed'Image
                   & " accepted=" & St.Accepted'Image
                   & " target_forwards=" & St.Target_Forwards'Image
                   & " emitted=" & St.Emitted'Image);
         if St.Target_Forwards > 0 then
            Put_Line ("  tokens per target forward ="
                      & Integer'Image (St.Emitted * 100 / St.Target_Forwards)
                      & "/100  (draft is FREE here, so this is the raw win)");
         end if;
      end;

      LLM_Qwen.Free (Target);
      LLM_Qwen.Free (Draft);
   end;

   New_Line;
   if Pass then
      Put_Line ("RESULT: PASS");
   else
      Put_Line ("RESULT: FAIL");
      Set_Exit_Status (Failure);
   end if;
end Test_Spec_Decode;
