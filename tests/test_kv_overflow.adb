------------------------------------------------------------------------
-- test_kv_overflow — regression for the Batch 1.4 KV-cache prefill overflow.
--
-- Run_Request prefills K(Pos+1) for Pos in 0..PLen-1 into Tensor_Array
-- (1 .. Ctx_Cap) with no per-step cap guard, so a prompt longer than Ctx_Cap
-- would write past the array and crash the scheduler. The fix clamps the
-- prompt (BOS pinned) before enqueue. The clamp math lives in the pure,
-- model-free LLM_Llama.KV_Prompt_Trim, so this test exercises the overflow
-- path without loading a model.
------------------------------------------------------------------------

with Ada.Text_IO;   use Ada.Text_IO;
with LLM_Llama;
with LLM_Tokenizer; use LLM_Tokenizer;

procedure Test_KV_Overflow is
   Pass : Boolean := True;

   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   function Make (N : Positive) return Token_Array is
      T : Token_Array (1 .. N);
   begin
      for I in T'Range loop T (I) := 100 + I; end loop;  -- 101,102,... distinct
      return T;
   end Make;

   Cap : constant := 10;
   --  Cap=10, Max=1 -> Cap_For_Prompt = 9, Room = 7.
begin
   Put_Line ("=== KV_Prompt_Trim (Batch 1.4 overflow) ===");

   --  1) Prompt well inside the window -> unchanged.
   declare
      P : constant Token_Array := Make (50);
      R : constant Token_Array := LLM_Llama.KV_Prompt_Trim (P, 100, 1);
   begin
      Check ("small prompt unchanged", R = P);
   end;

   --  2) HARD: prompt longer than the cache -> clipped to Cap_For_Prompt,
   --     BOS (first token) pinned, most-recent tokens kept.
   declare
      P : constant Token_Array := Make (20);           -- 101 .. 120
      R : constant Token_Array := LLM_Llama.KV_Prompt_Trim (P, Cap, 1);
      Expected : constant Token_Array :=
        [P (1), P (13), P (14), P (15), P (16), P (17), P (18), P (19), P (20)];
   begin
      Check ("HARD clip length = Cap-1", R'Length = Cap - 1);
      Check ("HARD BOS pinned", R (R'First) = P (P'First));
      Check ("HARD keeps most-recent tokens", R = Expected);
   end;

   --  3) SOFT: prompt fits the cache but not alongside Max -> clipped to Room,
   --     BOS pinned. Cap=20, Max=5 -> Cap_For_Prompt=19, Room=13.
   declare
      P : constant Token_Array := Make (15);           -- 101 .. 115
      R : constant Token_Array := LLM_Llama.KV_Prompt_Trim (P, 20, 5);
      Expected : constant Token_Array :=
        [P (1), P (4), P (5), P (6), P (7), P (8), P (9),
         P (10), P (11), P (12), P (13), P (14), P (15)];
   begin
      Check ("SOFT clip length = Room", R'Length = 13);
      Check ("SOFT BOS pinned", R (R'First) = P (P'First));
      Check ("SOFT keeps most-recent tokens", R = Expected);
   end;

   --  4) Empty prompt -> empty result (no crash on null slices).
   declare
      P : constant Token_Array (1 .. 0) := [];
      R : constant Token_Array := LLM_Llama.KV_Prompt_Trim (P, Cap, 1);
   begin
      Check ("empty prompt -> empty result", R'Length = 0);
   end;

   --  5) Single token -> unchanged (BOS only, nothing to trim).
   declare
      P : constant Token_Array := Make (1);
      R : constant Token_Array := LLM_Llama.KV_Prompt_Trim (P, Cap, 1);
   begin
      Check ("single token unchanged", R = P);
   end;

   --  6) Degenerate tiny cache (Ctx_Cap=1) -> HARD clip keeps just BOS.
   declare
      P : constant Token_Array := Make (8);
      R : constant Token_Array := LLM_Llama.KV_Prompt_Trim (P, 1, 1);
   begin
      Check ("Ctx_Cap=1 keeps only BOS", R'Length = 1 and then R (1) = P (1));
   end;

   --  7) Exhaustive sweep: the result NEVER exceeds the cache and BOS is
   --     always pinned (the actual Batch 1.4 invariant). If this ever
   --     regresses, a long prompt would write past Tensor_Array (1..Ctx_Cap).
   declare
      Violations : Natural := 0;
   begin
      for Ctx in 1 .. 32 loop
         for Mx in 0 .. 32 loop
            for PL in 0 .. 80 loop
               declare
                  P : constant Token_Array :=
                    (if PL = 0 then [] else Make (PL));
                  R : constant Token_Array :=
                    LLM_Llama.KV_Prompt_Trim (P, Ctx, Mx);
                  Cap_For_Prompt : constant Integer := Integer'Max (1, Ctx - 1);
               begin
                  if R'Length > Cap_For_Prompt
                    or else R'Length > P'Length
                    or else (P'Length > 0 and then R (R'First) /= P (P'First))
                  then
                     Violations := Violations + 1;
                  end if;
               end;
            end loop;
         end loop;
      end loop;
      Check ("sweep: result never exceeds cache, BOS always pinned",
             Violations = 0);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_KV_Overflow;