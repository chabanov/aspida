---------------------------------------------------------------------
-- Main entry point for the Aspida CLI
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Aspida;

procedure Main is
   use Ada.Text_IO;
begin
   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line ("Usage: aspida <command> [options]");
      Put_Line ("");
      Put_Line ("Commands:");
      Put_Line ("  build       Generate TypeScript client from API specification");
      Put_Line ("  validate    Check API specification for errors");
      Put_Line ("  help        Show this help message");
      return;
   end if;

   declare
      Command : constant String := Ada.Command_Line.Argument (1);
   begin
      if Command = "build" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line ("aspida: missing spec path");
            Put_Line ("Usage: aspida build <spec-file>");
            return;
         end if;

         declare
            Spec_Path : constant String := Ada.Command_Line.Argument (2);
         begin
            Put_Line ("aspida: loading spec from " & Spec_Path & " ...");
            Aspida.Initialize (Spec_Path);
            Aspida.Generate;
            Put_Line ("aspida: client generated successfully.");
         end;

      elsif Command = "validate" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line ("aspida: missing spec path");
            Put_Line ("Usage: aspida validate <spec-file>");
            return;
         end if;

         -- TODO: load spec, run Aspida.Is_Valid, report result
         Put_Line ("aspida: validation not yet implemented.");

      elsif Command = "help" then
         Put_Line ("Usage: aspida <command> [options]");
         Put_Line ("");
         Put_Line ("Commands:");
         Put_Line ("  build <spec>    Generate TypeScript client");
         Put_Line ("  validate <spec>  Validate specification");
         Put_Line ("  help            Show this help");

      else
         Put_Line ("aspida: unknown command '" & Command & "'");
         Put_Line ("Run 'aspida help' for usage.");
      end if;
   end;
end Main;
