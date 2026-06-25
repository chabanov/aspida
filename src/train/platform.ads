------------------------------------------------------------------------
-- Platform — control-plane contract for the Aspida training platform (see
-- PLATFORM.md). Pure, deterministic, host-only logic. Revised after the
-- engineering review:
--   * money is EXACT fixed-point throughout — no Float in the billing path;
--   * overflow-guarded job sizing;
--   * a hard spend CAP + job lifecycle states + an explicit failed-job charge
--     policy (the platform does not profit on a job that fails the guarantee);
--   * a RIGOROUS delivery gate: the student must beat the teachers by a margin
--     on a held-out eval of adequate size, in a domain with a real verifier.
------------------------------------------------------------------------

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Platform is

   --  Currency as fixed-point (exact cents). ALL arithmetic stays fixed-point
   --  or fixed×integer / fixed÷integer — never Float (review finding).
   type Money is delta 0.01 digits 14;

   type Domain_Kind  is (Code, SVG, Web_Layout, Multi_Lang);
   type Student_Tier is (Small, Medium, Large);

   --  Abuse / overflow guards on job sizing.
   Max_Droplets : constant := 4_096;
   Max_Hours    : constant := 100_000;    -- per droplet

   type Job_Spec is record
      Domain         : Domain_Kind;
      Tier           : Student_Tier;
      Droplets       : Positive;
      Hours_Per_Drop : Positive;
      Max_Spend      : Money;             -- hard ceiling; the runner kills at this
      --  Step 4 — turnkey identity + legal gate:
      Persona_Name   : Unbounded_String;  -- the student's identity / copyright name
      Persona_System : Unbounded_String;  -- system-behaviour prompt applied at serve
      Teacher_Attested : Boolean;         -- engineer attests rights to distil teachers
   end record;

   --  Pricing config (real values from the provider/business).
   Provider_Rate : constant Money := 2.50;   -- $ per GPU-droplet-hour
   Markup_Pct    : constant       := 30;     -- platform margin, percent

   type Quote_T is record
      GPU_Hours      : Natural;
      Provider_Cost  : Money;
      Platform_Price : Money;
      Deposit        : Money;             -- prepaid escrow = min(price, cap)
      Within_Budget  : Boolean;           -- Platform_Price <= Max_Spend
   end record;

   --  Quote for a job (pure, exact). Bounded inputs => no overflow.
   function Quote (J : Job_Spec) return Quote_T
     with Pre  => J.Droplets <= Max_Droplets and then J.Hours_Per_Drop <= Max_Hours,
          Post => Quote'Result.GPU_Hours = J.Droplets * J.Hours_Per_Drop;

   --  Pre-provision admission gate: a job may only run if the engineer has
   --  attested teacher-distillation rights, named a persona, and the price fits
   --  the cap. (Legal/identity gate before any GPU is rented — review finding.)
   type Admit_Result is
     (Allow, Reject_Not_Attested, Reject_No_Persona, Reject_Over_Budget);
   function Admit (J : Job_Spec; Q : Quote_T) return Admit_Result;

   --  Job lifecycle.
   type Job_State is (Quoted, Funded, Running, Delivered, Failed_Gate, Aborted_Cap);

   --  Final charge by outcome (fair policy):
   --    Delivered   -> metered price, capped at Max_Spend
   --    Failed_Gate -> provider cost ONLY (no margin — we don't profit on failure)
   --    Aborted_Cap -> the cap (Max_Spend)
   --    Quoted/Funded/Running -> 0 (nothing delivered/charged yet)
   function Final_Charge
     (J : Job_Spec; State : Job_State; Hours_Used : Natural) return Money;

   --  Engine instantiation target per tier.
   type Student_Config is record
      Voc, Dim, Ff, Seq, Lyr, Heads : Positive;
   end record;
   function Config_Of (T : Student_Tier) return Student_Config;

   --  Delivery gate — the platform's guarantee, made rigorous:
   --  the student must beat the teachers by Win_Margin on a held-out eval of
   --  >= Min_Eval items, in a domain that actually has a validated verifier.
   Min_Eval   : constant := 50;
   Win_Margin : constant Float := 0.02;

   type Job_Report is record
      Domain_Verified : Boolean;   -- domain ships a real, validated verifier
      Eval_N          : Natural;   -- held-out eval size (disjoint from training)
      Teacher_Pass    : Float;     -- teacher ensemble, fairly prompted, same eval
      Student_Pass    : Float;
      Beats_Teachers  : Boolean;   -- the gate (see Make_Report)
   end record;

   function Make_Report
     (Domain_Verified : Boolean; Eval_N : Natural;
      Teacher_Pass, Student_Pass : Float) return Job_Report
     with Post => Make_Report'Result.Beats_Teachers =
            (Domain_Verified
             and then Eval_N >= Min_Eval
             and then Student_Pass >= Teacher_Pass + Win_Margin);

end Platform;
