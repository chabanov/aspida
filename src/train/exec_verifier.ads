------------------------------------------------------------------------
-- Exec_Verifier — a REAL executable correctness oracle (Track 2): it writes a
-- candidate solution plus the spec's test asserts to a file and RUNS it through
-- a real interpreter (python3), accepting only if every assert passes. This is
-- the genuine "compiler/test-runner" verifier the `Verifier` interface was
-- designed to host; it drops in wherever the toy `Code_DSL.DSL_Verifier` did.
--
-- A small built-in benchmark (5 classic functions) supplies the per-spec tests.
------------------------------------------------------------------------

with Verifier;

package Exec_Verifier is

   N_Problems : constant := 5;
   subtype Problem_Id is Integer range 1 .. N_Problems;

   --  Short name / required function for each problem.
   function Name (P : Problem_Id) return String;

   --  Natural-language task for a real model generator (Track 2 Phase 2).
   function Prompt (P : Problem_Id) return String;

   --  True iff a usable python3 was located at elaboration.
   function Available return Boolean;

   --  Source-text oracle. Is_Correct runs Source against the problem's VISIBLE
   --  tests (used for best-of-N SELECTION — the model may see these).
   type Python_Verifier is new Verifier.Source_Instance with null record;
   overriding function Is_Correct
     (V : Python_Verifier; Spec : Natural; Source : String) return Boolean;

   --  Eval_Correct runs the HELD-OUT (hidden) tests with DIFFERENT inputs,
   --  used only for the final EVALUATION. select-on-visible / score-on-hidden
   --  is what catches overfitting / reward-hacking (a solution that hard-codes
   --  the visible answers passes Is_Correct but fails Eval_Correct).
   function Eval_Correct (Spec : Problem_Id; Source : String) return Boolean;

end Exec_Verifier;
