---------------------------------------------------------------------
-- LLM_Gemma — loader + forward pass for the gemma4 (Gemma 3n E4B) GGUF.
--
-- Status: FOUNDATION + honest forward. The standard Gemma transformer is
-- implemented faithfully (RMSNorm 1+w, embedding scaling, QK-norm attention
-- with dual RoPE + sliding-window masking + logit soft-capping, GeGLU FFN
-- with sandwich norms, tied output + final soft-cap). The Gemma-3n-specific
-- PER-LAYER-EMBEDDING (inp_gate / proj / layer_output_scale) and the
-- SHARED-KV-LAYER mechanisms are implemented BEST-EFFORT and are NOT yet
-- validated against a reference; until that validation, generated text is
-- expected to be poor. Every uncertain step is marked with `-- TODO(gemma3n)`.
--
-- Reuses LLM_Tensor / LLM_Weight / LLM_RMSNorm / LLM_RoPE / LLM_Tokenizer and
-- LLM_Qwen's streaming Token_Sink + Message types (no cyclic dependency:
-- LLM_Qwen does not depend on LLM_Gemma).
---------------------------------------------------------------------

with LLM_Qwen;
with LLM_Sampler;

package LLM_Gemma is

   type Gemma_Model is private;

   Model_Load_Error : exception;

   function Load (Path : String) return Gemma_Model;

   --  Multi-turn chat using the Gemma turn template; greedy decode with the
   --  same streaming sink contract as the Qwen backend.
   function Chat
     (M : Gemma_Model; Conversation : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy) return String;

   --  Raw greedy completion (BOS + prompt, no turn template) — for validation.
   function Complete
     (M : Gemma_Model; Prompt : String; Max_New_Tokens : Integer := 8)
      return String;

   function Vocab_Size  (M : Gemma_Model) return Integer;
   function Dim         (M : Gemma_Model) return Integer;
   function Block_Count (M : Gemma_Model) return Integer;

private

   type Gemma_Model_Rec;
   type Gemma_Model is access Gemma_Model_Rec;

end LLM_Gemma;
