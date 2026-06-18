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

   --  Decode core: single-stream sampling from a ready token-id sequence
   --  (sampler Params; greedy by default), stopping at Stop_A or Stop_B
   --  (token ids; -1 disables). Used by Complete and direct callers;
   --  concurrent Chat sessions go through the batched scheduler (Run_Request).
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
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats : access LLM_Qwen.Gen_Stats := null) return String;

   --  Raw greedy completion (BOS + prompt, no chat template) — for validation.
   function Complete
     (M : Llama_Model; Prompt : String; Max_New_Tokens : Integer := 8)
      return String;

   --  Teacher-forcing forward for knowledge distillation: the full-vocab
   --  next-token logits at every position of Ids, row-major and flat
   --  ([Ids'Length * Vocab_Size]; row p, 0-based, at offset p*Vocab_Size).
   --  Single-threaded; allocates a temporary KV cache.
   type Logits_Flat is array (Natural range <>) of Float;
   function Forward_Logits
     (M : Llama_Model; Ids : LLM_Tokenizer.Token_Array) return Logits_Flat;

   function Vocab_Size  (M : Llama_Model) return Integer;
   function Dim         (M : Llama_Model) return Integer;
   function Block_Count (M : Llama_Model) return Integer;

   --  The context window actually served (prompt + generation): the model's
   --  trained context bounded by the ASPIDA_CTX budget. Used for turn-aware
   --  prompt fitting and honest reporting.
   function Effective_Context (M : Llama_Model) return Integer;

   --  Validate the batched forward (continuous-batching primitive): runs two
   --  equal-length sequences both single-step and batched, returns the max
   --  abs logit difference. ~0 (FP noise) means Forward_Batch is correct.
   function Batch_Self_Test (M : Llama_Model) return Float;

   --  Validate the continuous-batch scheduler: generate two prompts both
   --  one-at-a-time and batched-together (greedy), report whether each
   --  sequence's completion is identical. Returns a human-readable summary.
   function Batch_Gen_Self_Test (M : Llama_Model; Max_New : Integer) return String;

private

   type Llama_Model_Rec;
   type Llama_Model is access Llama_Model_Rec;

end LLM_Llama;
