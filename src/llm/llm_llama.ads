---------------------------------------------------------------------
-- LLM_Llama — loader + forward pass for the standard dense transformer
-- family (general.architecture = "llama"): Llama 3.x, and by extension any
-- dense GQA + RMSNorm + SwiGLU + RoPE model (Mistral, Qwen2-dense, ...).
--
-- This is the plain decoder: pre-attention RMSNorm, GQA attention with NeoX
-- RoPE (optional rope_freqs proportional scaling), SwiGLU FFN, post norms via
-- residuals, and an untied (or tied) output projection. No MoE, no per-layer
-- embeddings, no sliding window, no logit soft-cap.
--
-- Decoding keeps an incremental K/V cache (one forward per new token).
---------------------------------------------------------------------

with LLM_Qwen;
with LLM_Tokenizer;
with LLM_Sampler;

package LLM_Llama is

   type Llama_Model is private;

   Model_Load_Error : exception;

   function Load (Path : String) return Llama_Model;

   --  Decode core: greedy generation from a ready token-id sequence, stopping
   --  at Stop_A or Stop_B (token ids; -1 disables).  Shared by Chat/Complete
   --  and used by the unified engine-level chat layer.
   function Generate
     (M : Llama_Model; Ids : LLM_Tokenizer.Token_Array;
      Stop_A, Stop_B : Integer := -1;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy) return String;

   --  Multi-turn chat using the Llama-3 header template; same streaming sink
   --  contract as the other backends.
   function Chat
     (M : Llama_Model; Conversation : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy) return String;

   --  Raw greedy completion (BOS + prompt, no chat template) — for validation.
   function Complete
     (M : Llama_Model; Prompt : String; Max_New_Tokens : Integer := 8)
      return String;

   function Vocab_Size  (M : Llama_Model) return Integer;
   function Dim         (M : Llama_Model) return Integer;
   function Block_Count (M : Llama_Model) return Integer;

private

   type Llama_Model_Rec;
   type Llama_Model is access Llama_Model_Rec;

end LLM_Llama;
