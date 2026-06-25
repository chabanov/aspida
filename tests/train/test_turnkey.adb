------------------------------------------------------------------------
-- test_turnkey — the MVP orchestrator end-to-end, model-free: every branch of
-- the turnkey loop (admit → train → gate → charge) with stub train/eval.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Platform;              use Platform;
with Turnkey;               use Turnkey;

procedure Test_Turnkey is
   Pass : Boolean := True;
   procedure Chk (Name : String; Cond : Boolean) is
   begin
      Put_Line ("  " & (if Cond then "PASS" else "FAIL") & ": " & Name);
      if not Cond then Pass := False; end if;
   end Chk;

   --  configurable stub behaviour
   Train_OK : Boolean := True;
   E_DV     : constant Boolean := True;
   E_N      : constant Natural := 60;
   E_TP     : Float   := 0.40;
   E_SP     : Float   := 0.92;

   function Stub_Train (J : Job_Spec) return Boolean is
      pragma Unreferenced (J);
   begin
      return Train_OK;
   end Stub_Train;
   procedure Stub_Eval (J : Job_Spec; Domain_Verified : out Boolean;
                        Eval_N : out Natural; Teacher_Pass, Student_Pass : out Float) is
      pragma Unreferenced (J);
   begin
      Domain_Verified := E_DV; Eval_N := E_N; Teacher_Pass := E_TP; Student_Pass := E_SP;
   end Stub_Eval;
   function Stub_Deliver (J : Job_Spec) return Delivery is
      pragma Unreferenced (J);
   begin
      return (Served => True, GGUF_Path => To_Unbounded_String ("student.gguf"),
              Endpoint => To_Unbounded_String ("127.0.0.1:8765#pinnedkey"));
   end Stub_Deliver;

   J : constant Job_Spec :=
     (Domain => Code, Tier => Medium, Droplets => 4, Hours_Per_Drop => 10,
      Max_Spend => 200.00,
      Persona_Name     => To_Unbounded_String ("AdaCoder"),
      Persona_System   => To_Unbounded_String ("precise Ada coder"),
      Teacher_Attested => True);
   Q : constant Quote_T := Quote (J);
   TR : constant Trainer   := Stub_Train'Unrestricted_Access;
   EV : constant Evaluator := Stub_Eval'Unrestricted_Access;
   DL : constant Deliverer := Stub_Deliver'Unrestricted_Access;
begin
   Put_Line ("=== Turnkey orchestrator (admit -> train -> gate -> charge) ===");

   --  1. happy path: delivered, metered charge, beats teachers, SERVED
   declare O : constant Outcome := Run (J, Q, TR, EV, DL, Hours_Used => 40); begin
      Chk ("delivered: admitted",      O.Admitted = Allow);
      Chk ("delivered: state Delivered", O.State = Delivered);
      Chk ("delivered: beats teachers",  O.Report.Beats_Teachers);
      Chk ("delivered: charged metered 130.00", O.Charge = 130.00);
      Chk ("delivered: exported + served", O.Deliver.Served
           and then Length (O.Deliver.Endpoint) > 0);
   end;

   --  2. gate fail (sub-margin): provider cost only, NOT served
   E_TP := 0.40; E_SP := 0.41;
   declare O : constant Outcome := Run (J, Q, TR, EV, DL, Hours_Used => 40); begin
      Chk ("gate-fail: state Failed_Gate", O.State = Failed_Gate);
      Chk ("gate-fail: not beats",         not O.Report.Beats_Teachers);
      Chk ("gate-fail: provider cost only 100.00", O.Charge = 100.00);
      Chk ("gate-fail: NOT served",        not O.Deliver.Served);
   end;
   E_SP := 0.92;  -- restore

   --  3. not attested: rejected before any cost, not served
   declare
      Jna : Job_Spec := J;
   begin
      Jna.Teacher_Attested := False;
      declare O : constant Outcome := Run (Jna, Quote (Jna), TR, EV, DL, 40); begin
         Chk ("reject: not attested", O.Admitted = Reject_Not_Attested);
         Chk ("reject: state Quoted (unfunded)", O.State = Quoted);
         Chk ("reject: charged nothing", O.Charge = 0.00);
         Chk ("reject: NOT served", not O.Deliver.Served);
      end;
   end;

   --  4. training fails: aborted at the cap, not served
   Train_OK := False;
   declare O : constant Outcome := Run (J, Q, TR, EV, DL, Hours_Used => 40); begin
      Chk ("train-fail: state Aborted_Cap", O.State = Aborted_Cap);
      Chk ("train-fail: charged the cap 200.00", O.Charge = 200.00);
      Chk ("train-fail: NOT served", not O.Deliver.Served);
   end;
   Train_OK := True;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Turnkey;
