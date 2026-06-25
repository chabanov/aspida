------------------------------------------------------------------------
-- test_job_store — control-plane lifecycle: submit -> fund -> start -> finish,
-- with invalid-transition and not-found guards. Model-free.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Platform;              use Platform;
with Turnkey;
with Job_Store;             use Job_Store;

procedure Test_Job_Store is
   Pass : Boolean := True;
   procedure Chk (Name : String; Cond : Boolean) is
   begin
      Put_Line ("  " & (if Cond then "PASS" else "FAIL") & ": " & Name);
      if not Cond then Pass := False; end if;
   end Chk;

   J : constant Job_Spec :=
     (Domain => Code, Tier => Medium, Droplets => 4, Hours_Per_Drop => 10,
      Max_Spend => 200.00,
      Persona_Name     => To_Unbounded_String ("AdaCoder"),
      Persona_System   => To_Unbounded_String ("precise"),
      Teacher_Attested => True);

   O_Deliv : constant Turnkey.Outcome :=
     (Admitted => Allow, State => Delivered,
      Report   => Make_Report (True, 60, 0.40, 0.92),
      Charge   => 130.00,
      Deliver  => (Served => True, GGUF_Path => To_Unbounded_String ("s.gguf"),
                   Endpoint => To_Unbounded_String ("127.0.0.1:8765")));

   Id : Job_Id;
begin
   Put_Line ("=== Job_Store lifecycle ===");
   Id := Submit (J);
   Chk ("submit -> Quoted", State (Id) = Quoted);
   Chk ("quote stored (130.00)", Quote_Of (Id).Platform_Price = 130.00);
   Chk ("count = 1", Job_Store.Count = 1);

   Fund (Id);  Chk ("fund -> Funded", State (Id) = Funded);
   Start (Id); Chk ("start -> Running", State (Id) = Running);
   Finish (Id, O_Deliv);
   Chk ("finish -> Delivered", State (Id) = Delivered);
   Chk ("outcome charge stored", Outcome_Of (Id).Charge = 130.00);
   Chk ("outcome served", Outcome_Of (Id).Deliver.Served);

   --  invalid transition: a fresh job cannot Start before Fund
   declare
      Id2 : constant Job_Id := Submit (J);
      Raised : Boolean := False;
   begin
      begin Start (Id2); exception when Bad_Transition => Raised := True; end;
      Chk ("Start before Fund raises Bad_Transition", Raised);
   end;

   --  not found
   declare Raised : Boolean := False;
   begin
      begin declare S : constant Job_State := State (Job_Id'Last); pragma Unreferenced (S);
            begin null; end;
      exception when Not_Found => Raised := True; end;
      Chk ("unknown id raises Not_Found", Raised);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Job_Store;
