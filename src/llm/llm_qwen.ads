---------------------------------------------------------------------
-- LLM_Qwen — Qwen 3.5 MoE model loader and inference
--
-- Loads a Qwen 3.5 GGUF file, creates the model graph, and provides
-- forward pass + generation.
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_Qwen_Blk;
with LLM_Tokenizer;
with LLM_Sampler;
with Ada.Strings.Unbounded;

package LLM_Qwen is

   type Qwen_Model is private;

   --  Raised by Load when the GGUF cannot be opened or a critical tensor is
   --  missing/unreadable. Loading must fail loudly rather than return a model
   --  that silently emits garbage.
   Model_Load_Error : exception;

   -- Load model from GGUF file
   function Load (Path : String) return Qwen_Model;

   --  Release everything the model owns and deallocate it (Phase 1b eviction):
   --  every block's projection/expert weights' quantized host bytes (and any
   --  GPU mirror), the per-block records and block array, then the model
   --  record. Idempotent — M is set to null. Must not be in use.
   procedure Free (M : in out Qwen_Model);

   -- Forward pass: token_ids [seq_len] → logits [vocab_size]
   function Forward (M : Qwen_Model; Token_Ids : LLM_Tensor.Tensor) return LLM_Tensor.Tensor;

   --  Per-position logits over a token sequence, returned flat as
   --  [seq*vocab] with row index (Pos-1)*vocab + (k-1). Unlike Forward (which
   --  projects only the last position), this projects EVERY position, so a real
   --  Qwen model can act as a distillation teacher (see Teacher_Qwen): the
   --  student matches the teacher's distribution at each position. Ids are
   --  0-based token ids (as the tokenizer produces).
   type Logits_Flat is array (Natural range <>) of Float;
   function Forward_Logits
     (M : Qwen_Model; Ids : LLM_Tokenizer.Token_Array) return Logits_Flat;

   -- Streaming sink for real-time output. Override Emit to receive each
   -- generated token's text as soon as it is produced; Tick fires once per
   -- prompt token during prefill (so a UI can show progress before the first
   -- token). Pass an access to a concrete sink to Generate/Chat; null = no
   -- streaming (the full string is still returned either way).
   type Token_Sink is abstract tagged null record;
   procedure Emit (Sink : in out Token_Sink; Piece : String) is abstract;
   procedure Tick (Sink : in out Token_Sink) is null;

   --  Per-generation accounting for OpenAI-standard usage + finish_reason.
   --  Truncated = generation stopped on the token/context cap, not a natural
   --  end-of-turn token (=> finish_reason "length" rather than "stop"). Lives
   --  here so every backend and the engine can share it (as with the other
   --  conversation types) without a circular dependency on LLM_Backend.
   type Gen_Stats is record
      Prompt_Tokens     : Natural := 0;
      Completion_Tokens : Natural := 0;
      Truncated         : Boolean := False;   -- stopped on the token cap
      Overflow          : Boolean := False;   -- request refused: prompt > window
   end record;

   -- Generate text from prompt (raw completion; no chat template, no stop).
   function Generate
     (M : Qwen_Model; Prompt : String; Max_New_Tokens : Integer := 128;
      Sink : access Token_Sink'Class := null) return String;

   -- Single-turn chat: wraps User in the Qwen ChatML template, generates the
   -- assistant reply (stopping at <|im_end|>/EOS), and strips the model's
   -- <think>...</think> reasoning. Returns just the assistant's answer.
   function Chat
     (M : Qwen_Model; User : String; Max_New_Tokens : Integer := 256;
      Sink : access Token_Sink'Class := null) return String;

   --  Multi-turn chat. Conversation is the full message history with the
   --  current user message LAST (roles alternate user/assistant). Builds the
   --  ChatML transcript so the model has the prior turns as context.
   type Role_Kind is (Role_System, Role_User, Role_Assistant);
   type Message is record
      Role : Role_Kind;
      Text : Ada.Strings.Unbounded.Unbounded_String;
   end record;
   type Message_Array is array (Positive range <>) of Message;

   function Chat
     (M : Qwen_Model; Conversation : Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access Token_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats : access Gen_Stats := null) return String;

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
      --  Resolved special-token ids for ChatML (-1 if absent).
      Eos_Id       : Integer := -1;
      Im_Start_Id  : Integer := -1;
      Im_End_Id    : Integer := -1;
   end record;

   function "=" (Left, Right : Qwen_Model) return Boolean;

end LLM_Qwen;
