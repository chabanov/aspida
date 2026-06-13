---------------------------------------------------------------------
-- LLM_Tokenizer — Simple BPE (Byte Pair Encoding) tokenizer
-- GPT-2 compatible vocabulary
---------------------------------------------------------------------

package LLM_Tokenizer is

   -- Encode string → token ids (GPT-2 BPE)
   function Encode (Text : String) return String;
   -- Decode token ids → string
   function Decode (Tokens : String) return String;

   -- Vocab size
   Vocab_Size : constant Integer := 50257;

end LLM_Tokenizer;
