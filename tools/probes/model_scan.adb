------------------------------------------------------------------------
-- model_scan — list every GGUF model present on this system (metadata only,
-- no weights loaded). Mirrors what the inference server discovers at startup.
--
--   gprbuild -P probe.gpr && ./obj/model_scan
--   ASPIDA_MODELS_DIR=/path/a:/path/b ./obj/model_scan   # extra roots
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with LLM_Catalog;

procedure Model_Scan is
   use LLM_Catalog;
   Models : constant Entry_Vectors.Vector := Discover;
   N_Sup, N_Uns, N_Proj, N_Bad : Natural := 0;
begin
   Put_Line ("Aspida — models available on this system");
   Put_Line ("search roots: " & Roots_Description);
   New_Line;

   for E of Models loop
      case E.Status is
         when Supported   => N_Sup  := N_Sup  + 1;
         when Unsupported => N_Uns  := N_Uns  + 1;
         when Projector   => N_Proj := N_Proj + 1;
         when Invalid     => N_Bad  := N_Bad  + 1;
      end case;
      Put_Line (Describe (E));
   end loop;

   New_Line;
   Put_Line
     ("found" & Natural'Image (Natural (Models.Length)) & " gguf file(s):"
      & N_Sup'Image & " runnable,"
      & N_Uns'Image & " other-architecture,"
      & N_Proj'Image & " projector(s),"
      & N_Bad'Image & " unreadable");
end Model_Scan;
