------------------------------------------------------------------------
-- Job_Store — control-plane state spine (P6). A registry of training jobs with a
-- validated lifecycle: Submit (Quoted) -> Fund -> Start (Running) -> Finish
-- (Delivered / Failed_Gate / Aborted_Cap). Invalid transitions raise. The
-- control-plane API/service (submit/status/run) is a thin layer over this; the
-- Turnkey orchestrator produces the Outcome stored at Finish.
------------------------------------------------------------------------

with Platform;
with Turnkey;

package Job_Store is

   type Job_Id is new Positive;
   Max_Jobs : constant := 4096;

   Not_Found       : exception;
   Bad_Transition  : exception;
   Store_Full      : exception;

   --  Register a quoted job; returns its id (state Quoted, nothing charged yet).
   function Submit (Spec : Platform.Job_Spec) return Job_Id;

   procedure Fund  (Id : Job_Id);   -- Quoted  -> Funded   (deposit received)
   procedure Start (Id : Job_Id);   -- Funded  -> Running  (provisioned, training)
   procedure Finish (Id : Job_Id; O : Turnkey.Outcome);  -- Running -> O.State

   function State    (Id : Job_Id) return Platform.Job_State;
   function Quote_Of (Id : Job_Id) return Platform.Quote_T;
   function Spec_Of  (Id : Job_Id) return Platform.Job_Spec;
   function Outcome_Of (Id : Job_Id) return Turnkey.Outcome;  -- valid after Finish
   function Count return Natural;

end Job_Store;
