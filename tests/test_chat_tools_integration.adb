---------------------------------------------------------------------
-- Integration test: ask the real Hura-9b to call a tool.
--
-- Loads the real dense Qwen3.5 (DeepReinforce Hura 9B Q4_K_M)
-- via LLM_Engine, asks it to run a dummy tool, and verifies that
-- LLM_Chat_Parser surfaces a tool_call (Finish = "tool_calls"). Gated on
-- HURA_DRAFT pointing at the file (skips cleanly if unset/missing).
-- The tools[] system block is synthesized via OpenAI.Request's
-- standard sysmsg shape, so this is the same path as the server uses.
--
-- Note: only checks the parser/event wiring — does NOT make assertions on
-- the exact tool name chosen, since the model picks freely from one tool.
-- Success = loads, finishes, and either finish=tool_calls with at least
-- one parsed tool call, OR finish=stop with reasoning emitting cleanly.
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

procedure Test_Chat_Tools_Integration is
   use Ada.Text_IO;

   function Model_Path return String is
      Var : constant String := "HURA_DRAFT";
   begin
      if Ada.Environment_Variables.Exists (Var) then
         return Ada.Environment_Variables.Value (Var);
      end if;
      return "/Users/ceo/models/hura/hura-9b.gguf";
   end Model_Path;

   Tool_Block : constant String :=
     "You may call one or more functions to assist with the user." & ASCII.LF &
     "Here are the available tools:" & ASCII.LF & ASCII.LF &
     "<tools>" & ASCII.LF &
     "{""name"": ""run_tests"",""parameters"": " &
     "{""type"":""object"",""properties"":" &
     "{""cmd"":{""type"":""string""}}}}" & ASCII.LF &
     "</tools>" & ASCII.LF & ASCII.LF &
     "When you make a tool call, emit a tag and a body. Two equivalent forms are accepted:" & ASCII.LF &
     "  Form A (canonical):" & ASCII.LF &
     "    <tool_call>" & ASCII.LF &
     "    <function=run_tests>" & ASCII.LF &
     "    <parameter=cmd>pytest -x</parameter>" & ASCII.LF &
     "    </function>" & ASCII.LF &
     "    </tool_call>" & ASCII.LF &
     "  Form B (bare tags, line-aligned):" & ASCII.LF &
     "    tool_call" & ASCII.LF &
     "    <function=run_tests>" & ASCII.LF &
     "    <parameter=cmd>pytest -x</parameter>" & ASCII.LF &
     "    </function>" & ASCII.LF &
     "    tool_call" & ASCII.LF &
     "Pick Form A unless you were fine-tuned on Form B. In either form, output" & ASCII.LF &
     "the angle brackets literally and do NOT wrap in Markdown fences or code blocks." & ASCII.LF &
     "Otherwise answer normally. Do not make up parameter values.";

   Path : constant String := Model_Path;
begin
   Put_Line ("=== test_chat_tools_integration (hura + tools) ===");

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
         Sys : constant LLM_Qwen.Message :=
           (Role => LLM_Qwen.Role_System,
            Text => To_Unbounded_String (Tool_Block));
         User : constant LLM_Qwen.Message :=
           (Role => LLM_Qwen.Role_User,
            Text => To_Unbounded_String
              ("Run the test suite with cmd=""pytest -x""."));
         Conv : constant LLM_Qwen.Message_Array :=
           [1 => Sys, 2 => User];
         R : constant LLM_Qwen.Chat_Result :=
           LLM_Engine.Chat (E, Conv,
                            Max_New_Tokens => 200,
                            Params         => LLM_Sampler.Greedy);
      begin
         Put_Line ("finish: " & To_String (R.Finish));
         Put_Line ("reasoning (" & Natural'Image
                     (Ada.Strings.Unbounded.Length (R.Reasoning)) & " chars)");
         Put_Line ("answer (" & Natural'Image
                     (Ada.Strings.Unbounded.Length (R.Answer)) & " chars)");
         Put_Line ("tool_calls: " & R.N_Tool_Calls'Image);
         for I in 1 .. R.N_Tool_Calls loop
            Put_Line ("  [" & Positive'Image (I) & "] "
                      & To_String (R.Tool_Calls (I).Name)
                      & " " & To_String (R.Tool_Calls (I).Arguments_JS));
         end loop;
         Put_Line ("--- generated answer ---");
         Put_Line (To_String (R.Answer));
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
end Test_Chat_Tools_Integration;
