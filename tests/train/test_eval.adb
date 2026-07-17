------------------------------------------------------------------------
-- test_eval — evaluation rigor: scoring must use HELD-OUT (hidden) tests,
-- separate from the VISIBLE tests used for best-of-N selection. This is what
-- makes "the student beats its teacher" an honest claim: a solution that
-- overfits / hard-codes the visible answers passes selection but FAILS the
-- hidden eval, so it cannot game the result. Model-free (uses the real python3
-- verifier, no LLM).
------------------------------------------------------------------------

with Ada.Text_IO;      use Ada.Text_IO;
with Ada.Command_Line; use Ada.Command_Line;
with Exec_Verifier;    use Exec_Verifier;

procedure Test_Eval is
   Vf   : Python_Verifier;
   Pass : Boolean := True;
   procedure Chk (Name : String; Cond : Boolean) is
   begin
      Put_Line ("  " & (if Cond then "PASS" else "FAIL") & ": " & Name);
      if not Cond then Pass := False; end if;
   end Chk;

   --  A genuinely correct add.
   Good : constant String := "def add(a,b): return a+b";
   --  Overfit: hard-codes the VISIBLE test answers, wrong elsewhere.
   Overfit : constant String :=
     "def add(a,b): return 5 if (a,b)==(2,3) else (0 if (a,b)==(-1,1) else -999)";
   --  Plainly wrong.
   Wrong : constant String := "def add(a,b): return a-b";
begin
   Put_Line ("=== Step 2 rigor: select-on-visible / score-on-hidden ===");
   if not Available then
      Put_Line ("SKIP: python3 not found"); return;
   end if;

   --  correct solution passes BOTH selection and held-out eval
   Chk ("correct: passes selection (visible)", Vf.Is_Correct (1, Good));
   Chk ("correct: passes held-out eval (hidden)", Eval_Correct (1, Good));

   --  THE point: overfit passes selection but FAILS held-out eval
   Chk ("overfit: passes selection (visible)", Vf.Is_Correct (1, Overfit));
   Chk ("overfit: FAILS held-out eval (caught!)", not Eval_Correct (1, Overfit));

   --  plainly wrong fails both
   Chk ("wrong: fails selection", not Vf.Is_Correct (1, Wrong));
   Chk ("wrong: fails held-out eval", not Eval_Correct (1, Wrong));

   New_Line;
   if Pass then
      Put_Line ("RESULT: PASS  (held-out eval catches the overfit)");
   else
      Put_Line ("RESULT: FAIL");
      Set_Exit_Status (Failure);
   end if;
end Test_Eval;
