---------------------------------------------------------------------
-- Test loading + greedy generation against the REAL dense qwen35 GGUF
-- (DeepReinforce Hura 9B). Gated on HURA_DRAFT pointing at the file
-- (skips cleanly if unset/missing). Loads via LLM_Engine (so the registry
-- + arch dispatch + dense-FFN path are all exercised), then runs a short
-- greedy Chat and prints the verbatim output. Success = loads without
-- error AND produces coherent text.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Exceptions;
with Ada.Environment_Variables;
with Ada.Directories;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with LLM_Engine;
with LLM_Qwen;
with LLM_Sampler;

procedure Test_Qwen35_Real is
   use Ada.Text_IO;

   function Model_Path return String is
      Var : constant String := "HURA_DRAFT";
   begin
      if Ada.Environment_Variables.Exists (Var) then
         return Ada.Environment_Variables.Value (Var);
      end if;
      return "/Users/ceo/models/hura/hura-9b.gguf";
   end Model_Path;

   Path : constant String := Model_Path;
begin
   Put_Line ("=== test_qwen35_real (dense qwen35) ===");

   if not Ada.Directories.Exists (Path) then
      Put_Line ("SKIP: model not found at " & Path
                & " (set HURA_DRAFT to override)");
      return;
   end if;

   declare
      E : LLM_Engine.Engine := LLM_Engine.Load (Path);
   begin
      Put_Line ("Loaded: arch=" & LLM_Engine.Arch_Name (E)
                & " vocab=" & Integer'Image (LLM_Engine.Vocab_Size (E)));

      declare
         Conv : constant LLM_Qwen.Message_Array :=
           [1 => (LLM_Qwen.Role_User,
                  To_Unbounded_String
                    ("What is 2+2? Answer with just the number."))];
         Out_Text : constant String :=
           LLM_Engine.Chat (E, Conv,
                            Max_New_Tokens => 32,
                            Params         => LLM_Sampler.Greedy);
      begin
         Put_Line ("--- generated (verbatim) ---");
         Put_Line (Out_Text);
         Put_Line ("--- end ---");
      end;

      LLM_Engine.Unload (E);
   end;

   Put_Line ("OK");
exception
   when Err : others =>
      Put_Line ("FAIL: exception during load/generate");
      Put_Line (Ada.Exceptions.Exception_Information (Err));
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
end Test_Qwen35_Real;
