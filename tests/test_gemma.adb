------------------------------------------------------------------------
-- test_gemma — functional smoke test for the Gemma backend on a real GGUF:
-- loads the model, checks sane metadata, runs a short completion, and
-- verifies non-empty valid output with no crash. Skips (PASS) if no model.
-- Set ASPIDA_GEMMA_MODEL to override the path.
------------------------------------------------------------------------

with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Directories;
with Ada.Environment_Variables;
with Ada.Exceptions;          use Ada.Exceptions;
with LLM_Gemma;

procedure Test_Gemma is
   Pass : Boolean := True;

   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   function Model_Path return String is
   begin
      if Ada.Environment_Variables.Exists ("ASPIDA_GEMMA_MODEL") then
         return Ada.Environment_Variables.Value ("ASPIDA_GEMMA_MODEL");
      end if;
      return "/Users/ceo/.lmstudio/models/lmstudio-community/"
        & "gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf";
   end Model_Path;
begin
   Put_Line ("=== Gemma backend smoke test ===");
   if not Ada.Directories.Exists (Model_Path) then
      Put_Line ("  SKIP: gemma model not found at " & Model_Path);
      Put_Line ("RESULT: PASS");
      return;
   end if;

   declare
      M : constant LLM_Gemma.Gemma_Model := LLM_Gemma.Load (Model_Path);
   begin
      Check ("vocabulary loaded (sane size)", LLM_Gemma.Vocab_Size (M) > 1000);
      Check ("model dim loaded",              LLM_Gemma.Dim (M) > 0);

      declare
         Reply : constant String :=
           LLM_Gemma.Complete (M, "The capital of France is", 4);
      begin
         Put_Line ("   completion: """ & Reply & """");
         Check ("non-empty completion, no crash", Reply'Length > 0);
      end;
   exception
      when E : others =>
         Put_Line ("  load/generate raised: " & Exception_Message (E));
         Pass := False;
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Gemma;
