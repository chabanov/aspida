------------------------------------------------------------------------
-- Turnkey — the MVP platform job orchestrator (Step 6). Sequences a training
-- job through the validated control-plane primitives:
--
--   Admit (legal/persona/budget gate)
--     -> train  (the GPU engine, Step 5 — abstracted behind a callback)
--     -> quality-gate (held-out beats-teachers eval — abstracted behind a callback)
--     -> Final_Charge (Delivered = metered; Failed_Gate = provider-cost only)
--
-- The two GPU/IO-heavy steps are callbacks so the orchestration is testable
-- model-free (stubs) and wired to the real engine (Student_GPU + Exec_Verifier)
-- for production runs. Delivery (export GGUF + E2EE serve) plugs in after a
-- Delivered outcome.
------------------------------------------------------------------------

with Platform;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Turnkey is

   --  Trains a student for the job; returns True on success.
   type Trainer is access function (J : Platform.Job_Spec) return Boolean;

   --  Runs the held-out evaluation, yielding the gate inputs.
   type Evaluator is access procedure
     (J               : Platform.Job_Spec;
      Domain_Verified : out Boolean;
      Eval_N          : out Natural;
      Teacher_Pass    : out Float;
      Student_Pass    : out Float);

   --  Delivery artifact produced ONLY on a Delivered outcome: the exported GGUF
   --  and the E2EE serving endpoint handed to the engineer.
   type Delivery is record
      Served    : Boolean := False;
      GGUF_Path : Unbounded_String;
      Endpoint  : Unbounded_String;   -- host:port + pinned server key
   end record;

   --  Exports the trained student to GGUF and brings it up on the encrypted
   --  inference path; returns the delivery artifact. Run calls it only when the
   --  quality gate passed (Delivered). May be null (no delivery wired).
   type Deliverer is access function (J : Platform.Job_Spec) return Delivery;

   type Outcome is record
      Admitted : Platform.Admit_Result;   -- why a job was/was not admitted
      State    : Platform.Job_State;       -- final lifecycle state
      Report   : Platform.Job_Report;      -- the beats-teachers verdict
      Charge   : Platform.Money;           -- what the engineer is charged
      Deliver  : Delivery;                 -- the served artifact (if Delivered)
   end record;

   --  Run the job end-to-end. Hours_Used is the metered GPU-hours (from the
   --  trainer/meter). Pure orchestration over Platform primitives. Deliver is
   --  invoked only on a Delivered outcome (export GGUF + serve).
   function Run
     (J          : Platform.Job_Spec;
      Q          : Platform.Quote_T;
      Train      : Trainer;
      Eval       : Evaluator;
      Deliver    : Deliverer;
      Hours_Used : Natural) return Outcome;

end Turnkey;
