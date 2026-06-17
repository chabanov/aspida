---------------------------------------------------------------------
-- Forward_Probe — load the real Qwen model and run ONE forward pass,
-- checking the logits are finite and reporting argmax + timing.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Calendar; use Ada.Calendar;
with LLM_Qwen;
with LLM_Tensor; use LLM_Tensor;

procedure Forward_Probe is
   use Ada.Text_IO;
   M : constant LLM_Qwen.Qwen_Model := LLM_Qwen.Load
     ("/Users/ceo/.lmstudio/models/HauhauCS/"
      & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/"
      & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf");

   Ctx : Tensor := New_Tensor ([1, 1]);   -- single token (1-based embed row)
   T0  : Time;
begin
   Set_Flat (Ctx, 1, 761.0);  -- token "The" (id 760, 1-based row)
   Put_Line ("Running one forward pass...");
   Flush;
   T0 := Clock;
   declare
      Logits   : constant Tensor := LLM_Qwen.Forward (M, Ctx);
      Elapsed  : constant Duration := Clock - T0;
      N        : constant Integer := Numel (Logits);
      Best     : Integer := 1;
      Best_V   : Float := Float'First;
      Min_V    : Float := Float'Last;
      Finite   : Boolean := True;
   begin
      for I in 1 .. N loop
         declare
            V : constant Float := Get_Flat (Logits, I);
         begin
            if not (V = V) or else abs V > Float'Last then
               Finite := False;
            else
               if V > Best_V then Best_V := V; Best := I; end if;
               if V < Min_V then Min_V := V; end if;
            end if;
         end;
      end loop;
      Put_Line ("vocab logits =" & Integer'Image (N));
      Put_Line ("all finite    = " & Boolean'Image (Finite));
      Put_Line ("argmax id     =" & Integer'Image (Best - 1)
                & "  (logit=" & Float'Image (Best_V) & ")");
      Put_Line ("min logit     =" & Float'Image (Min_V));
      Put_Line ("forward time  =" & Duration'Image (Elapsed) & " s");
   end;
end Forward_Probe;
