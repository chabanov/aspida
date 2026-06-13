---------------------------------------------------------------------
-- LLM_Model — Full GPT-style transformer for inference
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_Layer;
with LLM_Block;

package LLM_Model is

   -- GPT-like model configuration
   type Model_Config is record
      Vocab_Size : Integer := 50257;   -- GPT-2 vocab
      Dim        : Integer := 768;     -- hidden size
      N_Layers   : Integer := 12;      -- transformer blocks
      N_Heads    : Integer := 12;      -- attention heads
      Max_Seq_Len : Integer := 1024;   -- context window
   end record;

   type GPT_Model is private;

   -- Default: GPT-2 Small (124M params)
   function New_GPT2_Small return GPT_Model;

   -- Tiny model for testing / low-resource
   function New_Tiny (Dim, N_Layers : Integer) return GPT_Model;

   -- Forward pass: (batch=1, seq_len, vocab_size) → logits for next token
   function Forward (M : GPT_Model; Token_Ids : LLM_Tensor.Tensor) return LLM_Tensor.Tensor;

   -- Predict next token id from logits (argmax)
   function Predict_Next (M : GPT_Model; Token_Ids : LLM_Tensor.Tensor) return Integer;

   -- Generate up to Max_New_Tokens
   function Generate (M : GPT_Model; Prompt : String; Max_New_Tokens : Integer := 50) return String;

   -- Load pretrained GPT-2 weights from directory (reads config.txt first)
   function Load_GPT2 (Dir : String) return GPT_Model;

   -- Count params
   function Param_Count (M : GPT_Model) return Integer;

private

   type Block_Array is array (Positive range <>) of LLM_Block.Transformer_Block;
   type Block_Array_Ptr is access Block_Array;

   type GPT_Model is record
      Config     : Model_Config;
      Token_Emb  : LLM_Layer.Embedding_Layer;
      Pos_Emb    : LLM_Layer.Embedding_Layer;
      Blocks     : Block_Array_Ptr;
      Final_Norm : LLM_Layer.LayerNorm_Layer;
      LM_Head    : LLM_Layer.Linear_Layer;
   end record;

end LLM_Model;
