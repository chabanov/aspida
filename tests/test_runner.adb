---------------------------------------------------------------------
-- Aspida — Test Runner (AUnit-based)
--
-- Build with: gprbuild -P aspida.gpr -XBUILD=coverage tests/test_runner.adb
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Aspida;
use Aspida;

procedure Test_Runner is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Condition : Boolean) is
   begin
      if Condition then
         Put_Line ("  PASS: " & Name);
         Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);
         Failed := Failed + 1;
      end if;
   end Assert;

begin
   Put_Line ("=== Aspida Test Suite ===");
   New_Line;

   -- Test: empty spec is not valid
   Put_Line ("[Unit] Api_Spec validation");
   declare
      Empty_Spec : constant Aspida.Api_Spec (1 .. 0) := (others => <>);
   begin
      Assert ("Empty spec is not valid", not Aspida.Is_Valid (Empty_Spec));
   end;

   -- Test: endpoint type construction
   Put_Line ("[Unit] Endpoint record");
   declare
      Ep : Aspida.Endpoint;
   begin
      Ep.Method := Aspida.GET;
      Assert ("Default method is GET", Ep.Method = Aspida.GET);

      Ep.Method := Aspida.POST;
      Assert ("Method can be changed to POST", Ep.Method = Aspida.POST);
   end;

   New_Line;
   Put_Line ("Results: " & Passed'Image & " passed, " & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Runner;
