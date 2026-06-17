--  gemma_tok — load a GGUF tokenizer and encode/decode a test string, to
--  validate the SentencePiece path against llama-tokenize.
with Ada.Command_Line;
with Ada.Text_IO; use Ada.Text_IO;
with LLM_GGUF;
with LLM_Tokenizer;

procedure Gemma_Tok is
   G   : LLM_GGUF.GGUF_File;
   Tok : LLM_Tokenizer.Tokenizer := LLM_Tokenizer.Create;
   Txt : constant String :=
     (if Ada.Command_Line.Argument_Count >= 2
      then Ada.Command_Line.Argument (2) else "The capital of France is");
begin
   LLM_GGUF.Open (G, Ada.Command_Line.Argument (1));
   LLM_Tokenizer.Load_From_GGUF (Tok, G);
   Put_Line ("vocab=" & Integer'Image (LLM_Tokenizer.Vocab_Size (Tok))
     & " unk=" & Integer'Image (LLM_Tokenizer.Unk_Id (Tok)));
   declare
      Ids : constant LLM_Tokenizer.Token_Array := LLM_Tokenizer.Encode (Tok, Txt);
   begin
      Put ("encode '" & Txt & "' ->");
      for I in Ids'Range loop Put (Integer'Image (Ids (I))); end loop;
      New_Line;
      Put ("decode -> '");
      for I in Ids'Range loop Put (LLM_Tokenizer.Decode_One (Tok, Ids (I))); end loop;
      Put_Line ("'");
   end;
   LLM_GGUF.Close (G);
end Gemma_Tok;
