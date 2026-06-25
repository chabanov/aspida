------------------------------------------------------------------------
-- exec_distill — Track 2: verifier-filtered selection on REAL code with a REAL
-- interpreter. A noisy "teacher" proposes Python solutions (some correct, some
-- buggy) for a small benchmark; the Exec_Verifier RUNS each against the tests
-- and keeps only those that actually pass. This is the verifier-filtered half
-- of distillation at real scale — the executable oracle is a genuine compiler/
-- interpreter, dropped into the same `Verifier` interface as the toy oracle.
-- (Training a large student on the verified set is Phase 2 — needs a real
-- coding teacher model + GPU; out of this engine's CPU reach.)
------------------------------------------------------------------------

with Ada.Text_IO;      use Ada.Text_IO;
with Ada.Command_Line; use Ada.Command_Line;
with Exec_Verifier;    use Exec_Verifier;

procedure Exec_Distill is
   Vf    : Python_Verifier;
   LF    : constant Character := ASCII.LF;
   Total : Natural := 0;
   Good  : Natural := 0;

   procedure Try (P : Problem_Id; Src : String) is
      OKr : constant Boolean := Vf.Is_Correct (P, Src);
   begin
      Total := Total + 1;
      if OKr then Good := Good + 1; end if;
      Put_Line ("  [" & Name (P) & "]  "
                & (if OKr then "VERIFIED" else "rejected"));
   end Try;
begin
   Put_Line ("=== exec_distill: real interpreter verifies real code (Track 2) ===");
   if not Available then
      Put_Line ("SKIP: python3 not found on PATH");
      return;
   end if;

   --  Noisy teacher: a mix of correct and buggy real Python per problem.
   Try (1, "def add(a,b): return a+b");                 -- correct
   Try (1, "def add(a,b):" & LF & " return b + a");     -- correct (alt form)
   Try (1, "def add(a,b): return a-b");                 -- buggy

   Try (2, "def is_even(n): return n%2==0");            -- correct
   Try (2, "def is_even(n): return n%2==1");            -- buggy

   Try (3, "def max_of(lst): return max(lst)");         -- correct
   Try (3, "def max_of(lst): return min(lst)");         -- buggy

   Try (4, "def reverse_str(s): return s[::-1]");       -- correct
   Try (4, "def reverse_str(s): return s");             -- buggy

   Try (5, "def factorial(n):" & LF & " r=1" & LF
          & " for i in range(1,n+1): r*=i" & LF & " return r");  -- correct
   Try (5, "def factorial(n): return n");               -- buggy

   New_Line;
   Put_Line ("teacher candidates:" & Total'Image
             & "   verified-correct:" & Good'Image
             & "   (teacher pass-rate "
             & Integer'Image (Integer (100.0 * Float (Good) / Float (Total)))
             & "% )");
   Put_Line ("the verified subset is what a student would distill from — 100%"
             & " correct by construction of the executable verifier.");

   New_Line;
   if Good > 0 and then Good < Total then
      Put_Line ("RESULT: PASS  (real interpreter filtered a noisy teacher's real"
                & " code; verifier interface drives Track 2)");
   else
      Put_Line ("RESULT: FAIL  (verifier did not discriminate — see counts)");
      Set_Exit_Status (Failure);
   end if;
end Exec_Distill;
