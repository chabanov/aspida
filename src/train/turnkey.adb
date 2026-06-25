------------------------------------------------------------------------
-- Turnkey body — orchestration over the Platform primitives. No GPU/IO here;
-- training and evaluation are the injected callbacks.
------------------------------------------------------------------------

package body Turnkey is

   use Platform;

   function Run
     (J          : Platform.Job_Spec;
      Q          : Platform.Quote_T;
      Train      : Trainer;
      Eval       : Evaluator;
      Deliver    : Deliverer;
      Hours_Used : Natural) return Outcome
   is
      A : constant Admit_Result := Admit (J, Q);
      O : Outcome :=
        (Admitted => A,
         State    => Quoted,
         Report   => Make_Report (False, 0, 0.0, 0.0),
         Charge   => 0.00,
         Deliver  => (Served => False, others => <>));
   begin
      --  1. Admission gate (legal attestation + persona + budget).
      if A /= Allow then
         return O;                       -- not funded, nothing charged
      end if;

      --  2. Train (the GPU engine). Failure aborts at the spend cap.
      if Train = null or else not Train (J) then
         O.State  := Aborted_Cap;
         O.Charge := Final_Charge (J, Aborted_Cap, Hours_Used);
         return O;
      end if;

      --  3. Quality gate — held-out beats-teachers eval.
      declare
         DV : Boolean;
         EN : Natural;
         TP, SP : Float;
      begin
         Eval (J, DV, EN, TP, SP);
         O.Report := Make_Report (DV, EN, TP, SP);
      end;

      --  4. Charge by outcome: Delivered = metered (capped); Failed_Gate =
      --     provider cost only (no margin — we don't profit on a missed promise).
      if O.Report.Beats_Teachers then
         O.State  := Delivered;
         O.Charge := Final_Charge (J, Delivered, Hours_Used);
         if Deliver /= null then          -- export GGUF + serve over E2EE
            O.Deliver := Deliver (J);
         end if;
      else
         O.State  := Failed_Gate;
         O.Charge := Final_Charge (J, Failed_Gate, Hours_Used);
      end if;
      return O;
   end Run;

end Turnkey;
