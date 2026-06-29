---------------------------------------------------------------------
-- LLM_Backend — the unified protocol every model architecture implements.
--
-- A single dispatching contract so the engine (and thus the encrypted chat
-- server) works with ANY model the same way: build a conversation, generate a
-- reply, query basic metadata. Adding a new architecture means implementing
-- this interface and adding one row to the engine's registry — no case
-- statements, no edits to the engine's internals.
--
-- The interface is `limited` so a backend may carry non-copyable state; it is
-- always held by reference through Backend_Access (Model_Backend'Class).
-- Shared conversation types (Message_Array, Token_Sink, Role_Kind) live in
-- LLM_Qwen and are reused here, exactly as LLM_Gemma/LLM_Llama already do.
---------------------------------------------------------------------

with LLM_Qwen;
with LLM_Sampler;

package LLM_Backend is

   type Model_Backend is limited interface;

   type Backend_Access is access all Model_Backend'Class;

   --  Generate the assistant reply for a full conversation (system/user/
   --  assistant turns), streaming each piece via Sink (a Chat_Sink so
   --  reasoning + tool calls + text are all routable to dedicated
   --  callbacks), sampled with Params. When Stats /= null it is filled with
   --  token counts and the stop reason (for OpenAI-standard usage +
   --  finish_reason). Returns the full Chat_Result so the caller can render
   --  the final message without consuming the stream.
   function Chat
     (M              : Model_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Chat_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access LLM_Qwen.Gen_Stats := null)
      return LLM_Qwen.Chat_Result is abstract;

   function Vocab_Size  (M : Model_Backend) return Integer is abstract;
   function Arch_Name   (M : Model_Backend) return String  is abstract;
   function Dim         (M : Model_Backend) return Integer is abstract;
   function Block_Count (M : Model_Backend) return Integer is abstract;

   --  Release every resource the backend owns — the model's quantized weight
   --  bytes, any GPU-side mirror of them, per-layer/per-block heap structures,
   --  and (where held) the streaming GGUF file handle. Called by
   --  LLM_Engine.Unload when a model is evicted (Phase 1b LRU). Must be
   --  idempotent (a second call is a no-op) and must only run when the backend
   --  is not in use by any in-flight Chat — the registry guarantees this by
   --  evicting only a slot with refcount = 0. The default is a null procedure
   --  so a backend that owns nothing reclaimable needs no override.
   procedure Release (M : in out Model_Backend) is null;

end LLM_Backend;
