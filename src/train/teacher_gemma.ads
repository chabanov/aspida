---------------------------------------------------------------------
-- Teacher_Gemma — adapt a loaded Gemma model into a Distill.Teacher, so a real
-- Gemma dense model can produce the distillation dataset for a new student.
-- Mirrors Teacher_Llama; the student MUST share the Gemma tokenizer/vocabulary.
---------------------------------------------------------------------

with Distill;
with LLM_Gemma;

package Teacher_Gemma is

   type LM_Teacher is new Distill.Teacher with private;

   --  Wrap an already-loaded Gemma model as a distillation teacher.
   function Make (Model : LLM_Gemma.Gemma_Model) return LM_Teacher;

   overriding function Vocab (T : LM_Teacher) return Positive;
   overriding procedure Forward
     (T : in out LM_Teacher; Tokens : Distill.Token_Array;
      Out_Logits : out Distill.Logit_Matrix);

private

   type LM_Teacher is new Distill.Teacher with record
      Model : LLM_Gemma.Gemma_Model;
   end record;

end Teacher_Gemma;
