--  Llama_Probe — raw greedy completion for validating LLM_Llama against
--  llama.cpp (e.g. "The capital of France is" -> " Paris").
with Ada.Command_Line;
with Ada.Text_IO;
with LLM_Llama;

procedure Llama_Probe is
   N : constant Integer :=
     (if Ada.Command_Line.Argument_Count >= 3
      then Integer'Value (Ada.Command_Line.Argument (3)) else 4);
   Prompt : constant String :=
     (if Ada.Command_Line.Argument_Count >= 2
      then Ada.Command_Line.Argument (2) else "The capital of France is");
   M : constant LLM_Llama.Llama_Model :=
     LLM_Llama.Load (Ada.Command_Line.Argument (1));
begin
   Ada.Text_IO.Put_Line ("=== prompt: '" & Prompt & "' ===");
   Ada.Text_IO.Put_Line ("completion: '" & LLM_Llama.Complete (M, Prompt, N) & "'");
end Llama_Probe;
