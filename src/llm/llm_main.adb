---------------------------------------------------------------------
-- LLM_Main — Entry point for the native LLM chat
---------------------------------------------------------------------

with Ada.Command_Line;
with LLM_Chat;

procedure LLM_Main is
   Dim     : Integer := 64;
   Layers  : Integer := 2;
begin
   if Ada.Command_Line.Argument_Count >= 1 then
      Dim := Integer'Value (Ada.Command_Line.Argument (1));
   end if;
   if Ada.Command_Line.Argument_Count >= 2 then
      Layers := Integer'Value (Ada.Command_Line.Argument (2));
   end if;

   LLM_Chat.Run (Dim, Layers);
end LLM_Main;
