------------------------------------------------------------------------
-- Platform body — exact fixed-point pricing (no Float), capped settlement,
-- failed-job policy, tier config, rigorous delivery gate.
------------------------------------------------------------------------

package body Platform is

   --  price = cost + cost*markup%   — all fixed×integer / fixed÷integer (exact).
   function Priced (Cost : Money) return Money is
     (Cost + (Cost * Markup_Pct) / 100);

   function Cost_Of (Hours : Natural) return Money is
     (Provider_Rate * Hours);            -- Money * Integer = Money (exact)

   function Quote (J : Job_Spec) return Quote_T is
      Hours : constant Natural := J.Droplets * J.Hours_Per_Drop;
      Cost  : constant Money   := Cost_Of (Hours);
      Price : constant Money   := Priced (Cost);
   begin
      return (GPU_Hours      => Hours,
              Provider_Cost  => Cost,
              Platform_Price => Price,
              Deposit        => Money'Min (Price, J.Max_Spend),
              Within_Budget  => Price <= J.Max_Spend);
   end Quote;

   function Final_Charge
     (J : Job_Spec; State : Job_State; Hours_Used : Natural) return Money
   is
      Cost  : constant Money := Cost_Of (Hours_Used);
      Price : constant Money := Priced (Cost);
   begin
      case State is
         when Delivered   => return Money'Min (Price, J.Max_Spend);
         when Failed_Gate => return Cost;            -- provider cost only
         when Aborted_Cap => return J.Max_Spend;
         when others      => return 0.00;            -- nothing delivered yet
      end case;
   end Final_Charge;

   function Admit (J : Job_Spec; Q : Quote_T) return Admit_Result is
   begin
      if not J.Teacher_Attested then
         return Reject_Not_Attested;          -- legal gate first
      elsif Length (J.Persona_Name) = 0 then
         return Reject_No_Persona;            -- a turnkey student needs an identity
      elsif not Q.Within_Budget then
         return Reject_Over_Budget;
      else
         return Allow;
      end if;
   end Admit;

   function Config_Of (T : Student_Tier) return Student_Config is
   begin
      case T is
         when Small  =>
            return (Voc => 8_192,  Dim => 512,   Ff => 1_376,
                    Seq => 1_024,  Lyr => 6,      Heads => 8);
         when Medium =>
            return (Voc => 16_384, Dim => 1_024,  Ff => 2_752,
                    Seq => 2_048,  Lyr => 12,     Heads => 16);
         when Large  =>
            return (Voc => 32_768, Dim => 2_048,  Ff => 5_504,
                    Seq => 4_096,  Lyr => 24,     Heads => 16);
      end case;
   end Config_Of;

   function Make_Report
     (Domain_Verified : Boolean; Eval_N : Natural;
      Teacher_Pass, Student_Pass : Float) return Job_Report is
   begin
      return (Domain_Verified => Domain_Verified,
              Eval_N          => Eval_N,
              Teacher_Pass    => Teacher_Pass,
              Student_Pass    => Student_Pass,
              Beats_Teachers  =>
                Domain_Verified
                and then Eval_N >= Min_Eval
                and then Student_Pass >= Teacher_Pass + Win_Margin);
   end Make_Report;

end Platform;
