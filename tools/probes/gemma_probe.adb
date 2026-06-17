--  Gemma_Probe — greedy completion for validating LLM_Gemma against
--  llama.cpp (e.g. "The capital of France is" -> "Paris").
--  With a 4th argument "chat", wraps the prompt in the gemma turn template
--  and uses LLM_Gemma.Chat (single user turn) instead of raw Complete.
with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with LLM_Gemma;
with LLM_Qwen;

procedure Gemma_Probe is
   N : constant Integer :=
     (if Ada.Command_Line.Argument_Count >= 3
      then Integer'Value (Ada.Command_Line.Argument (3)) else 4);
   Prompt : constant String :=
     (if Ada.Command_Line.Argument_Count >= 2
      then Ada.Command_Line.Argument (2) else "The capital of France is");
   Mode : constant String :=
     (if Ada.Command_Line.Argument_Count >= 4
      then Ada.Command_Line.Argument (4) else "raw");
   --  Optional 5th arg = a system message (exercises the harmony system turn).
   Sys : constant String :=
     (if Ada.Command_Line.Argument_Count >= 5
      then Ada.Command_Line.Argument (5) else "");
   M : constant LLM_Gemma.Gemma_Model :=
     LLM_Gemma.Load (Ada.Command_Line.Argument (1));
begin
   Ada.Text_IO.Put_Line ("=== prompt: '" & Prompt & "' (mode=" & Mode
     & (if Sys /= "" then ", sys='" & Sys & "'" else "") & ") ===");
   if Mode = "chat" then
      declare
         use type LLM_Qwen.Message_Array;
         U : constant LLM_Qwen.Message_Array :=
           (1 => (Role => LLM_Qwen.Role_User,
                  Text => Ada.Strings.Unbounded.To_Unbounded_String (Prompt)));
         Conv : constant LLM_Qwen.Message_Array :=
           (if Sys = "" then U
            else LLM_Qwen.Message_Array'
                   (1 => (Role => LLM_Qwen.Role_System,
                          Text => Ada.Strings.Unbounded.To_Unbounded_String (Sys)))
                 & U);
      begin
         Ada.Text_IO.Put_Line
           ("completion: '" & LLM_Gemma.Chat (M, Conv, N) & "'");
      end;
   else
      Ada.Text_IO.Put_Line
        ("completion: '" & LLM_Gemma.Complete (M, Prompt, N) & "'");
   end if;
end Gemma_Probe;
