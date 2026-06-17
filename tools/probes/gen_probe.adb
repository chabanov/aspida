---------------------------------------------------------------------
-- Gen_Probe — load the real model and greedily generate a few tokens
-- from a prompt, to check the whole stack produces coherent text.
---------------------------------------------------------------------

with Ada.Text_IO;       use Ada.Text_IO;
with Ada.Calendar;      use Ada.Calendar;
with LLM_Qwen;

procedure Gen_Probe is
   M : constant LLM_Qwen.Qwen_Model := LLM_Qwen.Load
     ("/Users/ceo/.lmstudio/models/HauhauCS/"
      & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/"
      & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf");

   Prompt   : constant String := "The capital of France is";
   N_Tokens : constant := 6;
   T0       : Time;
begin
   New_Line;
   Put_Line ("=== prompt ===");
   Put_Line ("'" & Prompt & "'");
   Put_Line ("generating" & Integer'Image (N_Tokens) & " tokens (greedy)...");
   Flush;
   T0 := Clock;
   declare
      Out_S   : constant String := LLM_Qwen.Generate (M, Prompt, N_Tokens);
      Elapsed : constant Duration := Clock - T0;
   begin
      New_Line;
      Put_Line ("=== output ===");
      Put_Line ("'" & Out_S & "'");
      New_Line;
      Put_Line ("time =" & Duration'Image (Elapsed) & " s"
                & "  (" & Duration'Image (Elapsed / N_Tokens) & " s/token avg)");
   end;
end Gen_Probe;
