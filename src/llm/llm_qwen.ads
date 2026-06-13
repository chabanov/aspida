---------------------------------------------------------------------
-- LLM_Qwen — Qwen 3.5 MoE model loader and inference
--
-- Loads a Qwen 3.5 GGUF file, creates the model graph, and provides
-- forward pass + generation.
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_Qwen_Blk;
with LLM_Tokenizer;

package LLM_Qwen is

   type Qwen_Model is private;

   -- Load model from GGUF file
   function Load (Path : String) return Qwen_Model;

   -- Forward pass: token_ids [seq_len] → logits [vocab_size]
   function Forward (M : Qwen_Model; Token_Ids : LLM_Tensor.Tensor) return LLM_Tensor.Tensor;

   -- Generate text from prompt
   function Generate (M : Qwen_Model; Prompt : String; Max_New_Tokens : Integer := 128) return String;

   -- Model params count (total, not activated)
   function Param_Count (M : Qwen_Model) return Long_Long_Integer;

   -- Get model metadata
   function Vocab_Size (M : Qwen_Model) return Integer;
   function Context_Len (M : Qwen_Model) return Integer;
   function Dim (M : Qwen_Model) return Integer;
   function Block_Count (M : Qwen_Model) return Integer;

private

   type Block_Access is access LLM_Qwen_Blk.Qwen_Block;
   type Block_Array is array (Positive range <>) of Block_Access;
   type Block_Array_Ptr is access Block_Array;
   type Qwen_Model is record
      Token_Emb  : LLM_Tensor.Tensor;
      Blocks     : Block_Array_Ptr;
      Final_Norm : LLM_Tensor.Tensor;
      LM_Head    : LLM_Tensor.Tensor;
      N_Blocks   : Integer;
      Vocab_Sz   : Integer;
      Model_Dim  : Integer;
      N_Heads    : Integer;
      N_KV_Heads : Integer := 2;
      Ctx_Len    : Integer;
      Tok        : LLM_Tokenizer.Tokenizer;
   end record;

   function "=" (Left, Right : Qwen_Model) return Boolean;

end LLM_Qwen;
