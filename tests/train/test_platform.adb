------------------------------------------------------------------------
-- test_platform — control-plane contract: EXACT money (no float drift),
-- spend cap, failed-job charge policy, and the rigorous beats-teachers gate.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Platform;              use Platform;

procedure Test_Platform is
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
      Persona_System   => To_Unbounded_String ("You are a precise Ada coder."),
      Teacher_Attested => True);
begin
   Put_Line ("=== Platform control-plane (exact money / cap / gate) ===");

   --  4 x 10 = 40 GPU-h; cost 40*2.50 = 100.00; price *1.30 = 130.00 (EXACT)
   declare
      Q : constant Quote_T := Quote (J);
   begin
      Chk ("GPU-hours = 40",          Q.GPU_Hours = 40);
      Chk ("provider cost = 100.00",  Q.Provider_Cost = 100.00);
      Chk ("platform price = 130.00 (exact, no float)", Q.Platform_Price = 130.00);
      Chk ("deposit = min(price,cap) = 130.00", Q.Deposit = 130.00);
      Chk ("within budget (cap 200)", Q.Within_Budget);
   end;

   --  cap below price -> not within budget, deposit clamped to cap
   declare
      Jc : Job_Spec := J;
   begin
      Jc.Max_Spend := 50.00;
      declare Q : constant Quote_T := Quote (Jc);
      begin
         Chk ("over-cap flagged", not Q.Within_Budget);
         Chk ("deposit clamped to cap 50.00", Q.Deposit = 50.00);
      end;
   end;

   --  failed-job charge policy
   Chk ("Delivered (30h) charges metered 97.50",
        Final_Charge (J, Delivered, 30) = 97.50);
   Chk ("Delivered capped at Max_Spend",
        Final_Charge (J, Delivered, 1000) = J.Max_Spend);
   Chk ("Failed_Gate charges provider cost only (no margin), 40h = 100.00",
        Final_Charge (J, Failed_Gate, 40) = 100.00);
   Chk ("Aborted_Cap charges the cap",
        Final_Charge (J, Aborted_Cap, 999) = J.Max_Spend);
   Chk ("Running charges nothing", Final_Charge (J, Running, 5) = 0.00);

   --  tier config sanity
   declare
      S : constant Student_Config := Config_Of (Small);
      L : constant Student_Config := Config_Of (Large);
   begin
      Chk ("tiers scale (Large.Dim > Small.Dim)", L.Dim > S.Dim);
      Chk ("heads divide dim", S.Dim mod S.Heads = 0 and then L.Dim mod L.Heads = 0);
   end;

   --  rigorous delivery gate
   Chk ("gate PASS: verified domain, N>=50, margin met",
        Make_Report (True, 100, 0.40, 0.92).Beats_Teachers);
   Chk ("gate FAIL: margin not met (0.01 < 0.02)",
        not Make_Report (True, 100, 0.80, 0.81).Beats_Teachers);
   Chk ("gate FAIL: eval too small (N=10)",
        not Make_Report (True, 10, 0.40, 0.99).Beats_Teachers);
   Chk ("gate FAIL: unverified domain",
        not Make_Report (False, 100, 0.40, 0.99).Beats_Teachers);

   --  pre-provision admission gate (legal attestation + persona + budget)
   Chk ("admit: valid job", Admit (J, Quote (J)) = Allow);
   declare Jx : Job_Spec := J;
   begin Jx.Teacher_Attested := False;
      Chk ("reject: teacher not attested", Admit (Jx, Quote (Jx)) = Reject_Not_Attested);
   end;
   declare Jx : Job_Spec := J;
   begin Jx.Persona_Name := Null_Unbounded_String;
      Chk ("reject: no persona", Admit (Jx, Quote (Jx)) = Reject_No_Persona);
   end;
   declare Jx : Job_Spec := J;
   begin Jx.Max_Spend := 50.00;
      Chk ("reject: over budget", Admit (Jx, Quote (Jx)) = Reject_Over_Budget);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Platform;
