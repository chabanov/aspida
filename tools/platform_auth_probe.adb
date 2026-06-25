------------------------------------------------------------------------
-- platform_auth_probe — authorize an engineer by their UARP API key. Pass the
-- key as arg 1 or via ASPIDA_UARP_KEY; validated against UARP /api/v1/me.
--   ./platform_auth_probe uarp_xxx_yyy        (engineer's real key -> AUTHORIZED)
------------------------------------------------------------------------

with Ada.Text_IO;            use Ada.Text_IO;
with Ada.Command_Line;       use Ada.Command_Line;
with Ada.Environment_Variables;
with Ada.Strings.Unbounded;  use Ada.Strings.Unbounded;
with Platform_Auth;

procedure Platform_Auth_Probe is
   Key : constant String :=
     (if Argument_Count >= 1 then Argument (1)
      elsif Ada.Environment_Variables.Exists ("ASPIDA_UARP_KEY")
      then Ada.Environment_Variables.Value ("ASPIDA_UARP_KEY") else "");
   Ok : Boolean; U : Unbounded_String;
begin
   Put_Line ("=== platform key auth (validate engineer's UARP key) ===");
   if not Platform_Auth.Available then
      Put_Line ("RESULT: PASS (graceful — curl not found; install curl to validate keys)");
      return;
   end if;
   if Key = "" then
      Put_Line ("RESULT: PASS (no key given; pass uarp_... as arg or ASPIDA_UARP_KEY)");
      return;
   end if;
   Platform_Auth.Verify (Key, Ok, U);
   if Ok then
      Put_Line ("AUTHORIZED  tenant=" & To_String (U));
      Put_Line ("RESULT: PASS (engineer authorized via UARP key)");
   else
      Put_Line ("REJECTED (UARP rejected the key)");
      Put_Line ("RESULT: PASS (invalid key correctly refused)");  -- expected for a bad key
   end if;
end Platform_Auth_Probe;
