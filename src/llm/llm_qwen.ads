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

   --  Structured chat sink: extends Token_Sink with the events Chat can emit
   --  when its output contains reasoning blocks (OpenAI `reasoning_content`)
   --  and/or tool-call XML blocks (`tool_c...tool_care_close` format).
   --  Defaults are null so existing Token_Sink callers keep working; the
   --  chat layer routes reasoning/tools via their dedicated callbacks,
   --  never Emit.
   --
   --  Event ordering: Tick (prefill) → optional reasoning → optional tool
   --  calls (in source order) → text → On_Finish_Reason. Each Tool_Call_Id
   --  is unique within a single Chat call. On_Finish_Reason fires exactly
   --  once at the end with "stop" / "length" / "tool_calls".
   type Chat_Sink is abstract new Token_Sink with null record;
   procedure On_Reasoning    (S : in out Chat_Sink; Piece : String) is null;
   procedure On_Text         (S : in out Chat_Sink; Piece : String) is null;
   procedure On_Tool_Call
     (S : in out Chat_Sink;
      Id            : String;
      Name          : String;
      Arguments_JS  : String) is null;
   procedure On_Finish_Reason
     (S : in out Chat_Sink; Reason : String) is null;
   --  The default Emit forwards to On_Text so a sink that overrides only
   --  On_Text still receives text pieces (legacy Token_Sink-style use).
   overriding procedure Emit  (S : in out Chat_Sink; Piece : String);
   overriding procedure Tick  (S : in out Chat_Sink);

   --  Concrete empty sink used by the non-streaming Chat variant so the
   --  FSM parser has a non-null `Sink` pointer to feed (the parser writes
   --  no events because every callback is null — but it cannot dereference
   --  a null access).  Reusable, never emits anything.
   type Null_Sink is new Chat_Sink with null record;
   type Null_Sink_Access is access all Null_Sink;
   function New_Null_Sink return Null_Sink_Access;

   --  A single assembled tool invocation the model asked for.
   type Tool_Call is record
      Id            : Ada.Strings.Unbounded.Unbounded_String;
      Name          : Ada.Strings.Unbounded.Unbounded_String;
      Arguments_JS  : Ada.Strings.Unbounded.Unbounded_String;
   end record;
   type Tool_Call_Array is array (Positive range <>) of Tool_Call;

   --  Per-generation accounting for OpenAI-standard usage + finish_reason.
   --  Truncated = generation stopped on the token/context cap, not a natural
   --  end-of-turn token (=> finish_reason "length" rather than "stop").
   type Gen_Stats is record
      Prompt_Tokens     : Natural := 0;
      Completion_Tokens : Natural := 0;
      Truncated         : Boolean := False;   -- stopped on the token cap
      Overflow          : Boolean := False;   -- request refused: prompt > window
   end record;

   --  Result of a Chat call: the model can produce reasoning (exposed
   --  separately), a final answer, and zero-or-more tool calls. In streaming
   --  mode callers use the Chat_Sink events instead; the return value still
   --  carries the full structured result for convenience.
   type Chat_Result (N_Tool_Calls : Natural) is record
      Reasoning  : Ada.Strings.Unbounded.Unbounded_String;
      Answer     : Ada.Strings.Unbounded.Unbounded_String;
      Finish     : Ada.Strings.Unbounded.Unbounded_String;
      Tool_Calls : Tool_Call_Array (1 .. N_Tool_Calls);
   end record;

   --  Multi-turn chat. Conversation is the full message history with the
   --  current user message LAST (roles alternate user/assistant).
   type Role_Kind is (Role_System, Role_User, Role_Assistant);
   type Message is record
      Role : Role_Kind;
      Text : Ada.Strings.Unbounded.Unbounded_String;
   end record;
   type Message_Array is array (Positive range <>) of Message;

   -- Generate text from prompt (raw completion; no chat template, no stop).
   function Generate
     (M : Qwen_Model; Prompt : String; Max_New_Tokens : Integer := 128;
      Sink : access Token_Sink'Class := null) return String;

   -- Structured chat returning Chat_Result (non-streaming). When a chat
   -- contains tool calls but no text, Finish = "tool_calls".
   function Chat
     (M : Qwen_Model; Conversation : Message_Array;
      Max_New_Tokens : Integer := 256;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats : access Gen_Stats := null) return Chat_Result;

   -- Streaming variant: Chat_Sink callbacks fire as the model emits each
   -- piece. Returns the same Chat_Result.
   function Chat
     (M : Qwen_Model; Conversation : Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access Chat_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats : access Gen_Stats := null) return Chat_Result;

   -- Model params count (total, not activated)
   function Param_Count (M : Qwen_Model) return Long_Long_Integer;

   -- Get model metadata
   function Vocab_Size (M : Qwen_Model) return Integer;
   function Context_Len (M : Qwen_Model) return Integer;
   function Dim (M : Qwen_Model) return Integer;
   function Block_Count (M : Qwen_Model) return Integer;
   --  general.architecture this model was loaded as (qwen35moe / qwen35 / qwen2).
   function Arch_Name (M : Qwen_Model) return String;

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
      Arch         : Ada.Strings.Unbounded.Unbounded_String;
   end record;

   function "=" (Left, Right : Qwen_Model) return Boolean;

end LLM_Qwen;
