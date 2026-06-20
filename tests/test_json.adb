---------------------------------------------------------------------
-- test_json — regression guard for the JSON parser depth cap.
-- A deeply-nested "[[[[…]]]]" payload must raise Parse_Error (not overflow
-- the stack and crash the process); a normal payload must still parse.
---------------------------------------------------------------------
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with JSON;

procedure Test_Json is
   Deep : constant String := [1 .. 4000 => '['] & [1 .. 4000 => ']'];
   Flat : constant String := "[1,2,3,{""k"":42}]";
   OK   : Boolean := True;
begin
   begin
      declare
         V : constant JSON.Value_Ref := JSON.Parse (Deep);
         pragma Unreferenced (V);
      begin
         Put_Line ("FAIL: deep nested input did NOT raise (depth cap missing)");
         OK := False;
      end;
   exception
      when JSON.Parse_Error =>
         Put_Line ("PASS: deep nested input raised Parse_Error (depth cap works)");
      when E : others =>
         Put_Line ("FAIL: wrong exception " & Ada.Exceptions.Exception_Name (E));
         OK := False;
   end;
   begin
      declare
         V : constant JSON.Value_Ref := JSON.Parse (Flat);
         pragma Unreferenced (V);
      begin
         Put_Line ("PASS: normal input parsed ok");
      end;
   exception
      when others =>
         Put_Line ("FAIL: normal input rejected");
         OK := False;
   end;
   if OK then
      Put_Line ("Results: 2 passed, 0 failed.");
   else
      Put_Line ("Results: 0 passed, 2 failed.");
      raise Program_Error;
   end if;
end Test_Json;
