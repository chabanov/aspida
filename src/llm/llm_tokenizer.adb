---------------------------------------------------------------------
-- LLM_Tokenizer body — GPT-2 BPE tokenizer for Qwen
---------------------------------------------------------------------

with Ada.Containers.Indefinite_Ordered_Maps;
with Ada.Strings.Fixed;

package body LLM_Tokenizer is

   package String_Maps is new Ada.Containers.Indefinite_Ordered_Maps
     (Key_Type => String, Element_Type => Integer);
   Vocab : String_Maps.Map;
   Vocab_Loaded : Boolean := False;

   procedure Load_Vocab (G : LLM_GGUF.GGUF_File) is
      use LLM_GGUF;
      N_Tok : constant Integer := Integer'Value (Metadata (G, "tokenizer.ggml.tokens"));
   begin
      for I in 1 .. N_Tok loop
         declare
            Token_Key : constant String := "tokenizer.ggml.tokens." & Trim (Integer'Image (I), Ada.Strings.Left);
            Token_Str : constant String := Metadata (G, Token_Key);
         begin
            if Token_Str /= "" then
               Vocab.Insert (Token_Str, I - 1);
            end if;
         end;
      end loop;
      Vocab_Loaded := True;
   end Load_Vocab;

   function Encode (Text : String) return String is
      Result : String (1 .. Text'Length * 8);
      Pos : Integer := 1;
      Cursor : Integer := Text'First;
   begin
      -- Simple: one token per character (placeholder — real BPE merge later)
      while Cursor <= Text'Last loop
         declare
            Code : constant Integer := Character'Pos (Text (Cursor));
         begin
            if Code >= 32 and Code <= 127 then
               -- Print token ID as text for now (model expects float tensor from IDs)
               declare
                  Id_Str : constant String := Trim (Integer'Image (Code), Ada.Strings.Left);
               begin
                  for J in Id_Str'Range loop
                     Result (Pos + J - Id_Str'First) := Id_Str (J);
                  end loop;
                  Pos := Pos + Id_Str'Length;
               end;
            end if;
            Cursor := Cursor + 1;
         end;
      end loop;
      return Result (1 .. Pos - 1);
   end Encode;

   function Decode (Tokens : String) return String is
      Result : String (1 .. 4096);
      Pos : Integer := 1;
   begin
      -- Simple: copy printable characters through
      for I in Tokens'Range loop
         declare
            C : constant Character := Tokens (I);
         begin
            if Character'Pos (C) >= 32 then
               Result (Pos) := C;
               Pos := Pos + 1;
            end if;
         end;
      end loop;
      return Result (1 .. Pos - 1);
   end Decode;

end LLM_Tokenizer;
