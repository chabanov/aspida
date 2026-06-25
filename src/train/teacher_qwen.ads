---------------------------------------------------------------------
-- Teacher_Qwen — adapt a loaded Qwen model into a Distill.Teacher, so a real
-- Qwen-MoE/SSM model can produce the distillation dataset for a new student.
-- Mirrors Teacher_Llama; the student MUST share the Qwen tokenizer/vocabulary.
---------------------------------------------------------------------

with Distill;
with LLM_Qwen;

package Teacher_Qwen is

   type LM_Teacher is new Distill.Teacher with private;

   --  Wrap an already-loaded Qwen model as a distillation teacher.
   function Make (Model : LLM_Qwen.Qwen_Model) return LM_Teacher;

   overriding function Vocab (T : LM_Teacher) return Positive;
   overriding procedure Forward
     (T : in out LM_Teacher; Tokens : Distill.Token_Array;
      Out_Logits : out Distill.Logit_Matrix);

private

   type LM_Teacher is new Distill.Teacher with record
      Model : LLM_Qwen.Qwen_Model;
   end record;

end Teacher_Qwen;
