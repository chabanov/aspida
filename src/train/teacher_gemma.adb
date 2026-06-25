---------------------------------------------------------------------
-- Teacher_Gemma body.
---------------------------------------------------------------------

with LLM_Tokenizer;

package body Teacher_Gemma is

   function Make (Model : LLM_Gemma.Gemma_Model) return LM_Teacher is
   begin
      return (Distill.Teacher with Model => Model);
   end Make;

   function Vocab (T : LM_Teacher) return Positive is
     (Positive (LLM_Gemma.Vocab_Size (T.Model)));

   procedure Forward
     (T : in out LM_Teacher; Tokens : Distill.Token_Array;
      Out_Logits : out Distill.Logit_Matrix)
   is
      N   : constant Integer := Tokens'Length;
      Vc  : constant Integer := LLM_Gemma.Vocab_Size (T.Model);
      Ids : LLM_Tokenizer.Token_Array (1 .. N);
   begin
      for I in 1 .. N loop
         Ids (I) := Integer (Tokens (Tokens'First + I - 1));
      end loop;
      declare
         Flat : constant LLM_Gemma.Logits_Flat :=
           LLM_Gemma.Forward_Logits (T.Model, Ids);
      begin
         for P in 1 .. N loop
            for K in 1 .. Vc loop
               Out_Logits (P, K) := Distill.Logit (Flat ((P - 1) * Vc + (K - 1)));
            end loop;
         end loop;
      end;
   end Forward;

end Teacher_Gemma;
