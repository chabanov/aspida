------------------------------------------------------------------------
-- chat_probe — load any supported GGUF and run one short chat turn through
-- the real LLM_Engine (the same path the server uses). Used to smoke-test
-- every discovered model locally.
--
--   ./obj/chat_probe <model.gguf> [max_tokens] [prompt words...]
------------------------------------------------------------------------

with Ada.Command_Line;       use Ada.Command_Line;
with Ada.Text_IO;            use Ada.Text_IO;
with Ada.Real_Time;          use Ada.Real_Time;
with Ada.Exceptions;         use Ada.Exceptions;
with Ada.Strings.Unbounded;  use Ada.Strings.Unbounded;
with LLM_Engine;
with LLM_Qwen;

procedure Chat_Probe is
begin
   if Argument_Count < 1 then
      Put_Line ("usage: chat_probe <model.gguf> [max_tokens] [prompt...]");
      return;
   end if;

   declare
      Path   : constant String  := Argument (1);
      MaxTok : Integer := 32;
      Prompt : Unbounded_String;
   begin
      if Argument_Count >= 2 then
         begin
            MaxTok := Integer'Value (Argument (2));
         exception
            when others => MaxTok := 32;
         end;
      end if;
      if Argument_Count >= 3 then
         for I in 3 .. Argument_Count loop
            Append (Prompt, Argument (I));
            if I < Argument_Count then Append (Prompt, " "); end if;
         end loop;
      else
         Prompt := To_Unbounded_String
           ("Answer in one short sentence: what is the capital of France?");
      end if;

      Put_Line ("loading: " & Path);
      declare
         E    : constant LLM_Engine.Engine := LLM_Engine.Load (Path);
         Conv : constant LLM_Qwen.Message_Array :=
           [1 => (LLM_Qwen.Role_User, Prompt)];
         T0   : constant Time := Clock;
         R    : constant String := LLM_Engine.Chat (E, Conv, MaxTok);
         DT   : constant Duration := To_Duration (Clock - T0);
      begin
         Put_Line ("arch: " & LLM_Engine.Arch_Name (E)
                   & "   vocab:" & Integer'Image (LLM_Engine.Vocab_Size (E)));
         Put_Line ("=== reply (" & Duration'Image (DT) & "s,"
                   & Integer'Image (MaxTok) & " tok cap) ===");
         Put_Line (R);
         Put_Line ("=== end ===");
      end;
   exception
      when X : others =>
         Put_Line ("ERROR: " & Exception_Name (X) & " — " & Exception_Message (X));
   end;
end Chat_Probe;
