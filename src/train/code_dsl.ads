------------------------------------------------------------------------
-- Code_DSL — a tiny program-synthesis task with an EXECUTABLE correctness
-- verifier, for verifier-filtered distillation experiments.
--
-- A spec selects a target function of two inputs (a+b, a-b, a*b, min, max).
-- A "program" is three tokens — operand, operand, op — evaluated as `x OP y`.
-- Verify RUNS the program on several test inputs and accepts it only if it
-- matches the target on ALL of them (so operand order and op choice must be
-- right — this is functional correctness, not memorization). This executable
-- check is the new signal that lets a student exceed a noisy teacher.
--
-- Vocabulary (0-based token ids):
--   0       = pad
--   1 .. 5  = spec selectors (F1..F5)
--   6 = a,  7 = b
--   8 = '+', 9 = '-', 10 = '*', 11 = min, 12 = max
------------------------------------------------------------------------

with Verifier;

package Code_DSL is

   N_Specs : constant := 5;
   Vocab   : constant := 13;

   subtype Spec_Id is Integer range 1 .. N_Specs;

   --  operand, operand, op — each a vocabulary id.
   type Program is array (1 .. 3) of Integer;

   --  Vocabulary id of a spec selector (also the first token of a sequence).
   function Spec_Token (S : Spec_Id) return Integer;

   --  Reference value of spec S on inputs (A, B).
   function Target (S : Spec_Id; A, B : Integer) return Integer;

   --  Execute P on (A,B): Val is the result, Ok is False if P is malformed.
   procedure Run (P : Program; A, B : Integer; Val : out Integer; Ok : out Boolean);

   --  Correct iff well-formed AND matches Target on every test pair.
   function Verify (S : Spec_Id; P : Program) return Boolean;

   --  A canonical correct program for S, and a systematic-error program.
   function Golden     (S : Spec_Id) return Program;
   function Distractor (S : Spec_Id) return Program;

   --  Pluggable-oracle view: wraps Verify behind the Verifier interface so the
   --  distillation engines can treat this (and any future oracle) uniformly.
   type DSL_Verifier is new Verifier.Instance with null record;
   overriding function Is_Correct
     (V : DSL_Verifier; Spec : Natural; Program : Verifier.Token_Array)
      return Boolean;

end Code_DSL;
