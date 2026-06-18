---------------------------------------------------------------------
-- Teacher_Llama — adapt a loaded llama model into a Distill.Teacher, so a
-- real existing model can produce the distillation dataset for a new student.
-- This is the bridge that makes "new models are taught by existing models"
-- concrete: the same engine that serves a model now also teaches with it.
---------------------------------------------------------------------

with Distill;
with LLM_Llama;

package Teacher_Llama is

   type LM_Teacher is new Distill.Teacher with private;

   --  Wrap an already-loaded llama model as a distillation teacher.
   function Make (Model : LLM_Llama.Llama_Model) return LM_Teacher;

   overriding function Vocab (T : LM_Teacher) return Positive;
   overriding procedure Forward
     (T : in out LM_Teacher; Tokens : Distill.Token_Array;
      Out_Logits : out Distill.Logit_Matrix);

private

   type LM_Teacher is new Distill.Teacher with record
      Model : LLM_Llama.Llama_Model;
   end record;

end Teacher_Llama;
