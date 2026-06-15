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
   --  assistant turns), streaming each piece via Sink, sampled with Params.
   function Chat
     (M              : Model_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Token_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy)
      return String is abstract;

   function Vocab_Size  (M : Model_Backend) return Integer is abstract;
   function Arch_Name   (M : Model_Backend) return String  is abstract;
   function Dim         (M : Model_Backend) return Integer is abstract;
   function Block_Count (M : Model_Backend) return Integer is abstract;

end LLM_Backend;
