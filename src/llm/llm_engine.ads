---------------------------------------------------------------------
-- LLM_Engine — architecture dispatcher.
--
-- Detects the GGUF's general.architecture and routes to a backend that
-- implements the unified LLM_Backend.Model_Backend protocol:
--   qwen35moe / qwen2 -> LLM_Qwen    (MoE + gated delta-net hybrid)
--   gemma4            -> LLM_Gemma   (PLE + shared-KV + dual RoPE)
--   llama             -> LLM_Llama   (dense GQA + SwiGLU; Llama 3.x, Mistral)
-- Dispatch is class-wide over Model_Backend; adding an architecture is one
-- registry row (see the body) — no case statements here.
---------------------------------------------------------------------

with LLM_Qwen;
with LLM_Backend;
with LLM_Sampler;

package LLM_Engine is

   type Engine is private;

   Model_Load_Error : exception;

   function Load (Path : String) return Engine;

   --  Sampling parameters (re-exported for callers); default is greedy.
   subtype Sampling is LLM_Sampler.Params;
   Greedy : LLM_Sampler.Params renames LLM_Sampler.Greedy;

   --  Per-generation accounting (token counts + truncated flag) for
   --  OpenAI-standard usage/finish_reason; pass an access to receive it.
   subtype Gen_Stats is LLM_Qwen.Gen_Stats;

   function Chat
     (E : Engine; Conversation : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats : access Gen_Stats := null) return String;

   function Vocab_Size (E : Engine) return Integer;
   function Arch_Name  (E : Engine) return String;

   --  Discovery helpers (no model load). Peek a GGUF's general.architecture
   --  ("" if the file cannot be opened), and test whether the engine has a
   --  backend for that architecture. Used by LLM_Catalog to enumerate the
   --  models present on the system without loading any weights.
   function Detect_Arch (Path : String) return String;
   function Supports (Arch : String) return Boolean;

private

   type Engine is record
      Impl : LLM_Backend.Backend_Access;
   end record;

end LLM_Engine;
