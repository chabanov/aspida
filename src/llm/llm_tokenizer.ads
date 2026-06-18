---------------------------------------------------------------------
-- LLM_Tokenizer — byte-level BPE tokenizer
--
-- Integer-id interface (ids are integers, not characters). When loaded
-- with a vocabulary + merge table (from GGUF) it performs greedy
-- byte-pair encoding by merge rank; with no vocabulary it falls back to
-- a 1-id-per-byte encoding so callers always get a usable result.
--
-- Real GGUF vocabularies use the GPT-2 byte->unicode remapping; that
-- bijection is implemented (see Byte_To_Piece / Unmap_Bytes in the body),
-- so raw input bytes map onto the UTF-8 vocab pieces and back on decode.
---------------------------------------------------------------------

with LLM_GGUF;

package LLM_Tokenizer is

   type Token_Array is array (Positive range <>) of Integer;

   type Tokenizer is private;

   -- Empty tokenizer (byte-level fallback until a vocab is added).
   function Create return Tokenizer;

   -- Populate from a parsed GGUF file (tokenizer.ggml.tokens / .merges).
   procedure Load_From_GGUF (T : in out Tokenizer; G : LLM_GGUF.GGUF_File);

   -- Manual construction (used by tests).
   procedure Add_Token (T : in out Tokenizer; Piece : String; Id : Integer);
   --  Pair is the GGUF "left right" form (single space separator).
   procedure Add_Merge (T : in out Tokenizer; Pair : String; Rank : Integer);
   procedure Mark_Loaded (T : in out Tokenizer);

   -- True once a vocabulary has been loaded (otherwise byte-level mode).
   function Is_Loaded (T : Tokenizer) return Boolean;
   function Vocab_Size (T : Tokenizer) return Integer;
   --  The model's unknown-token id, or -1 if the vocab defines none.
   function Unk_Id (T : Tokenizer) return Integer;

   -- Text -> token ids.
   function Encode (T : Tokenizer; Text : String) return Token_Array;
   -- Token ids -> text.
   function Decode (T : Tokenizer; Ids : Token_Array) return String;
   -- Single id -> its piece (for streaming generation).
   function Decode_One (T : Tokenizer; Id : Integer) return String;

   -- Exact piece -> id, or -1 if absent. Resolves special/control tokens
   -- (e.g. "<|im_start|>", "<|im_end|>") that byte-level Encode would split.
   function Token_To_Id (T : Tokenizer; Piece : String) return Integer;

   -- GPT-2 byte-level representation of a raw byte string: each byte mapped
   -- through the byte<->unicode bijection and concatenated. Use this to EXPORT
   -- a learned vocabulary in exactly the form byte-level Load_From_GGUF expects
   -- (tokenizer.ggml.model = "gpt2"), so the trained tokenizer round-trips.
   function Byte_Level_Piece (Raw : String) return String;

private

   type Tokenizer_Data;
   type Tokenizer is access Tokenizer_Data;

end LLM_Tokenizer;
