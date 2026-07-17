--  Reproduce the 65th-tool-call bound bug: N_Calls is incremented before the
--  cap check and never rolled back, so Tool_Calls_Of slices 1 .. 65 out of a
--  1 .. 64 array. Model-free: drives LLM_Chat_Parser directly.
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with LLM_Chat_Parser;
with LLM_Qwen;

procedure Test_Toolcap is

   type Null_Sink is new LLM_Qwen.Chat_Sink with null record;
   Sink : aliased Null_Sink;

   P : LLM_Chat_Parser.Parser;

   One_Call : constant String :=
     "<tool_call><function=f><parameter=k>v</parameter></function></tool_call>";

   N_Emitted : constant := 65;
begin
   Put_Line ("feeding" & Integer'Image (N_Emitted) & " tool calls"
             & " (Calls array is 1 .. 64)");

   for I in 1 .. N_Emitted loop
      LLM_Chat_Parser.Feed (P, One_Call, Sink'Access);
   end loop;
   LLM_Chat_Parser.Finalize (P, Sink'Access);

   Put_Line ("Finalize survived; now reading Tool_Calls_Of ...");

   declare
      TC : constant LLM_Chat_Parser.Tool_Call_Array :=
        LLM_Chat_Parser.Tool_Calls_Of (P);
   begin
      Put_Line ("NO CRASH: Tool_Calls_Of returned"
                & Integer'Image (TC'Length) & " calls");
   end;
exception
   when E : others =>
      Put_Line ("*** RAISED: "
                & Ada.Exceptions.Exception_Name (E)
                & " / " & Ada.Exceptions.Exception_Message (E));
end Test_Toolcap;
