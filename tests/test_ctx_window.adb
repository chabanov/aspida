------------------------------------------------------------------------
-- test_ctx_window — turn-aware context fitting (Ctx_Window.Select_Messages).
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Ctx_Window;  use Ctx_Window;

procedure Test_Ctx_Window is
   Pass : Boolean := True;

   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   function Img (K : Keep_Array) return String is
      R : String (1 .. K'Length); I : Natural := 0;
   begin
      for B of K loop I := I + 1; R (I) := (if B then '1' else '0'); end loop;
      return R;
   end Img;
begin
   Put_Line ("=== Ctx_Window.Select_Messages ===");

   --  1) Everything fits -> keep all, no overflow.
   declare
      L : constant Len_Array := [10, 10, 10, 10, 10];
      K : Keep_Array (L'Range); Ovf : Boolean;
   begin
      Select_Messages (L, System_First => True, Overhead => 4,
                       Budget => 100, Keep => K, Overflow => Ovf);
      Check ("all fit -> keep all", Img (K) = "11111" and then not Ovf);
   end;

   --  2) Budget forces dropping the OLDEST middle turns; system + recent kept.
   declare
      L : constant Len_Array := [10, 10, 10, 10, 10];
      K : Keep_Array (L'Range); Ovf : Boolean;
   begin
      Select_Messages (L, System_First => True, Overhead => 4,
                       Budget => 34, Keep => K, Overflow => Ovf);
      --  essentials 4+10(sys)+10(newest)=24; +msg4(10)=34 ok; msg3 would be 44 -> drop.
      Check ("drop oldest middle, pin system+newest", Img (K) = "10011"
             and then not Ovf);
   end;

   --  3) Overflow: even system + newest + overhead exceed the budget.
   declare
      L : constant Len_Array := [10, 10, 10, 10, 10];
      K : Keep_Array (L'Range); Ovf : Boolean;
   begin
      Select_Messages (L, System_First => True, Overhead => 4,
                       Budget => 20, Keep => K, Overflow => Ovf);
      Check ("overflow flagged, essentials still pinned",
             Ovf and then K (1) and then K (5));
   end;

   --  4) No system prompt -> oldest message is a normal, eligible turn.
   declare
      L : constant Len_Array := [10, 10, 10];
      K : Keep_Array (L'Range); Ovf : Boolean;
   begin
      Select_Messages (L, System_First => False, Overhead => 4,
                       Budget => 100, Keep => K, Overflow => Ovf);
      Check ("no system: all eligible kept", Img (K) = "111" and then not Ovf);
   end;

   --  5) No system, tight budget -> keep newest contiguous, drop oldest.
   declare
      L : constant Len_Array := [10, 10, 10];
      K : Keep_Array (L'Range); Ovf : Boolean;
   begin
      Select_Messages (L, System_First => False, Overhead => 4,
                       Budget => 26, Keep => K, Overflow => Ovf);
      --  newest(3)=10 +4 =14; msg2(10)=24<=26 keep; msg1(10)=34>26 drop.
      Check ("no system: keep recent, drop oldest", Img (K) = "011"
             and then not Ovf);
   end;

   --  6) Single message -> kept.
   declare
      L : constant Len_Array := [1 => 7];
      K : Keep_Array (L'Range); Ovf : Boolean;
   begin
      Select_Messages (L, System_First => False, Overhead => 2,
                       Budget => 100, Keep => K, Overflow => Ovf);
      Check ("single message kept", K (1) and then not Ovf);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Ctx_Window;
