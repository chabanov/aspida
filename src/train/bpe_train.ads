---------------------------------------------------------------------
-- BPE_Train — a from-scratch byte-level Byte-Pair-Encoding tokenizer TRAINER.
--
-- LLM_Tokenizer can LOAD a vocabulary (from GGUF) and encode/decode; it cannot
-- LEARN one. This learns merges from a corpus so a model trained from scratch
-- has its own vocabulary — the missing piece between toy fixed-token tasks and
-- training on real text. Byte-level base (256 tokens), so any input round-trips
-- losslessly regardless of the learned merges.
--
-- Algorithm: pre-tokenize on spaces (a space attaches to the following word,
-- so merges never cross word boundaries), then greedily merge the most
-- frequent adjacent symbol pair until the target vocabulary size is reached.
---------------------------------------------------------------------

package BPE_Train is

   type Trainer is private;

   --  Learn a vocabulary of up to Target_Vocab tokens (>= 256; the first 256
   --  are the byte alphabet) from Corpus.
   procedure Train
     (T : out Trainer; Corpus : String; Target_Vocab : Positive);

   function Vocab_Size (T : Trainer) return Natural;
   function Num_Merges (T : Trainer) return Natural;

   --  The literal byte string token Id expands to (Id in 0 .. Vocab_Size - 1).
   function Token_Piece (T : Trainer; Id : Natural) return String;

   --  The two parent token ids of the Index-th merge (1 .. Num_Merges), in the
   --  rank order they were learned. Lets an exporter emit the GGUF merges list.
   function Merge_Left_Id  (T : Trainer; Index : Positive) return Natural;
   function Merge_Right_Id (T : Trainer; Index : Positive) return Natural;

   type Id_Array is array (Positive range <>) of Natural;

   --  Greedy BPE encode (applies learned merges by ascending rank) and its
   --  exact inverse: Decode (Encode (X)) = X for every input.
   function Encode (T : Trainer; Text : String) return Id_Array;
   function Decode (T : Trainer; Ids : Id_Array) return String;

private

   type Trainer_Data;
   type Trainer is access Trainer_Data;

end BPE_Train;
