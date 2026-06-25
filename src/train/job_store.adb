------------------------------------------------------------------------
-- Job_Store body — in-memory registry + lifecycle guards. (Disk persistence
-- uses the at_rest sealing pattern; in-memory spine here.)
------------------------------------------------------------------------

package body Job_Store is

   use type Platform.Job_State;

   type Rec is record
      Spec  : Platform.Job_Spec;
      Q     : Platform.Quote_T;
      St    : Platform.Job_State;
      Out_R : Turnkey.Outcome;
      Used  : Boolean := False;
   end record;

   Jobs : array (1 .. Max_Jobs) of Rec;
   N    : Natural := 0;

   procedure Check (Id : Job_Id) is
   begin
      if Natural (Id) > N or else not Jobs (Integer (Id)).Used then
         raise Not_Found;
      end if;
   end Check;

   function Submit (Spec : Platform.Job_Spec) return Job_Id is
   begin
      if N >= Max_Jobs then
         raise Store_Full;
      end if;
      N := N + 1;
      Jobs (N) := (Spec  => Spec,
                   Q     => Platform.Quote (Spec),
                   St    => Platform.Quoted,
                   Out_R => <>,
                   Used  => True);
      return Job_Id (N);
   end Submit;

   procedure Require (Id : Job_Id; Want : Platform.Job_State) is
   begin
      Check (Id);
      if Jobs (Integer (Id)).St /= Want then
         raise Bad_Transition;
      end if;
   end Require;

   procedure Fund (Id : Job_Id) is
   begin
      Require (Id, Platform.Quoted);
      Jobs (Integer (Id)).St := Platform.Funded;
   end Fund;

   procedure Start (Id : Job_Id) is
   begin
      Require (Id, Platform.Funded);
      Jobs (Integer (Id)).St := Platform.Running;
   end Start;

   procedure Finish (Id : Job_Id; O : Turnkey.Outcome) is
   begin
      Require (Id, Platform.Running);
      Jobs (Integer (Id)).St    := O.State;
      Jobs (Integer (Id)).Out_R := O;
   end Finish;

   function State (Id : Job_Id) return Platform.Job_State is
   begin
      Check (Id); return Jobs (Integer (Id)).St;
   end State;

   function Quote_Of (Id : Job_Id) return Platform.Quote_T is
   begin
      Check (Id); return Jobs (Integer (Id)).Q;
   end Quote_Of;

   function Spec_Of (Id : Job_Id) return Platform.Job_Spec is
   begin
      Check (Id); return Jobs (Integer (Id)).Spec;
   end Spec_Of;

   function Outcome_Of (Id : Job_Id) return Turnkey.Outcome is
   begin
      Check (Id); return Jobs (Integer (Id)).Out_R;
   end Outcome_Of;

   function Count return Natural is (N);

end Job_Store;
