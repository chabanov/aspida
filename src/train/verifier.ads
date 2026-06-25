------------------------------------------------------------------------
-- Verifier — a pluggable EXECUTABLE correctness oracle for distillation.
--
-- An implementation decides whether a candidate program (a token sequence) is
-- a correct solution for a given spec. This is the extra signal — absent from
-- any teacher's distribution — that lets a student EXCEED its teacher (see
-- ARCHITECTURE.md §5). The program-synthesis oracle is `Code_DSL.DSL_Verifier`;
-- a future compiler / test-runner oracle implements this very same interface,
-- so the distillation engines (tools/code_distill, tools/code_iterate) stay
-- verifier-agnostic.
------------------------------------------------------------------------

package Verifier is

   type Token_Array is array (Positive range <>) of Integer;

   type Instance is limited interface;

   --  True iff Program is a correct solution for Spec. Implementations are
   --  expected to *execute* / check the candidate, not merely pattern-match.
   function Is_Correct
     (V : Instance; Spec : Natural; Program : Token_Array) return Boolean
     is abstract;

   --  Source-text oracle: the candidate is real source code (Track 2 — a real
   --  compiler / interpreter runs it against the spec's tests). Distinct from
   --  the token-sequence Instance above, which fits the discrete synthesis toy.
   type Source_Instance is limited interface;
   function Is_Correct
     (V : Source_Instance; Spec : Natural; Source : String) return Boolean
     is abstract;

end Verifier;
