---------------------------------------------------------------------
-- BPE_GGUF body.
---------------------------------------------------------------------

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with LLM_Tokenizer;

package body BPE_GGUF is

   procedure Write_Tokenizer
     (B : in out GGUF_Write.Builder; T : BPE_Train.Trainer)
   is
      V    : constant Natural := BPE_Train.Vocab_Size (T);
      M    : constant Natural := BPE_Train.Num_Merges (T);
      Toks : GGUF_Write.Str_List (1 .. V);
      Mrgs : GGUF_Write.Str_List (1 .. M);

      function BL (Raw : String) return String
        renames LLM_Tokenizer.Byte_Level_Piece;
   begin
      for Id in 0 .. V - 1 loop
         Toks (Id + 1) :=
           To_Unbounded_String (BL (BPE_Train.Token_Piece (T, Id)));
      end loop;

      GGUF_Write.Meta_Str (B, "tokenizer.ggml.model", "gpt2");
      GGUF_Write.Meta_Str_Array (B, "tokenizer.ggml.tokens", Toks);

      if M > 0 then
         for I in 1 .. M loop
            Mrgs (I) := To_Unbounded_String
              (BL (BPE_Train.Token_Piece (T, BPE_Train.Merge_Left_Id  (T, I)))
               & " "
               & BL (BPE_Train.Token_Piece (T, BPE_Train.Merge_Right_Id (T, I))));
         end loop;
         GGUF_Write.Meta_Str_Array (B, "tokenizer.ggml.merges", Mrgs);
      end if;
   end Write_Tokenizer;

end BPE_GGUF;
