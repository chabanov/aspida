------------------------------------------------------------------------
-- teacher_probe — runtime check of the real-model distillation teachers.
-- Loads a real GGUF through the chosen backend, wraps it as a Distill.Teacher,
-- captures a top-K sample over a short token sequence, and validates the
-- per-position logits pipeline (finite, top-K descending, ids in range). This
-- exercises LLM_{Llama,Qwen,Gemma}.Forward_Logits + Teacher_{Llama,Qwen,Gemma}
-- on a real model. Model-dependent; NOT part of the model-free test suite.
--
--   QWEN_MODEL_PATH=/path/to/model.gguf  ./obj/teacher_probe <llama|qwen|gemma>
------------------------------------------------------------------------

with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Command_Line;
with Ada.Environment_Variables;
with Distill;
with Teacher_Llama; with LLM_Llama;
with Teacher_Qwen;  with LLM_Qwen;
with Teacher_Gemma; with LLM_Gemma;

procedure Teacher_Probe is
   use type Distill.Logit;

   Arch : constant String :=
     (if Ada.Command_Line.Argument_Count >= 1
      then Ada.Command_Line.Argument (1) else "");
   Path : constant String :=
     Ada.Environment_Variables.Value ("QWEN_MODEL_PATH", "");
   K      : constant := 8;
   Prompt : constant Distill.Token_Array := [1, 5, 7, 3, 9, 11, 2, 13];

   procedure Validate (S : Distill.Sample; Vocab : Positive) is
      Ok_Fin, Ok_Desc, Ok_Range : Boolean := True;
   begin
      for R in 1 .. S.N loop
         for J in 1 .. S.K loop
            if abs (Float (S.Top_Logit (R, J))) > 1.0E30 then
               Ok_Fin := False;
            end if;
            if Integer (S.Top_Ids (R, J)) not in 0 .. Vocab - 1 then
               Ok_Range := False;
            end if;
         end loop;
         for J in 1 .. S.K - 1 loop
            if S.Top_Logit (R, J) < S.Top_Logit (R, J + 1) then
               Ok_Desc := False;
            end if;
         end loop;
      end loop;
      Put_Line ("  vocab =" & Vocab'Image & "  N =" & S.N'Image
                & "  K =" & S.K'Image);
      Put_Line ("  finite=" & Ok_Fin'Image & "  top-K descending="
                & Ok_Desc'Image & "  ids-in-range=" & Ok_Range'Image);
      if Ok_Fin and then Ok_Desc and then Ok_Range then
         Put_Line ("  RESULT: PASS  (" & Arch & " teacher produced a sample)");
      else
         Put_Line ("  RESULT: FAIL");
      end if;
   end Validate;

begin
   if Path = "" then
      Put_Line ("set QWEN_MODEL_PATH=/path/to/model.gguf");
      return;
   end if;
   Put_Line ("=== teacher probe: " & Arch & " <- " & Path & " ===");

   if Arch = "llama" then
      declare
         M : constant LLM_Llama.Llama_Model := LLM_Llama.Load (Path);
         T : Teacher_Llama.LM_Teacher := Teacher_Llama.Make (M);
         S : constant Distill.Sample := Distill.Capture (T, Prompt, K);
      begin
         Validate (S, Positive (LLM_Llama.Vocab_Size (M)));
      end;
   elsif Arch = "qwen" then
      declare
         M : constant LLM_Qwen.Qwen_Model := LLM_Qwen.Load (Path);
         T : Teacher_Qwen.LM_Teacher := Teacher_Qwen.Make (M);
         S : constant Distill.Sample := Distill.Capture (T, Prompt, K);
      begin
         Validate (S, Positive (LLM_Qwen.Vocab_Size (M)));
      end;
   elsif Arch = "gemma" then
      declare
         M : constant LLM_Gemma.Gemma_Model := LLM_Gemma.Load (Path);
         T : Teacher_Gemma.LM_Teacher := Teacher_Gemma.Make (M);
         S : constant Distill.Sample := Distill.Capture (T, Prompt, K);
      begin
         Validate (S, Positive (LLM_Gemma.Vocab_Size (M)));
      end;
   else
      Put_Line ("usage: teacher_probe <llama|qwen|gemma>  (QWEN_MODEL_PATH=...)");
   end if;
end Teacher_Probe;
