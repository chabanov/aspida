---------------------------------------------------------------------
-- LLM_Chat_Parser — streaming FSM for ChatML output with thinking blocks
-- (`<think>...</think>`) and tool-call XML blocks
-- (`<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>`).
--
-- Hura (and the broader Qwen3.5 family) emits:
--
--    <think>
--    reasoning ...
--    </think>
--    <tool_call>
--    <function=run_tests>
--    <parameter=cmd>pytest -x</parameter>
--    </function>
--    <tool_call>
--    final answer text
--
-- A chat layer needs to (a) split thinking vs answer for the
-- `reasoning_content` / `content` fields, (b) extract each tool-call block
-- into {id, name, arguments JSON}, and (c) feed text-pieces and assembled
-- tool-calls to a Chat_Sink as the model emits them (so the proxy can
-- stream them in real time).
--
-- Design:
--   * Stateful: the parser keeps a small buffer because the XML tags
--     cross token boundaries — a `<tool_call>` opener may arrive in piece
--     N, the function=NAME in piece N+1, the closing tag in piece N+k.
--   * Strict XML: matches `<function=NAME>`, `<parameter=KEY>VALUE</parameter>`
--     with no whitespace inside the open tag; malformed blocks fall through
--     to text (so a malformed model output never gets silently dropped).
--   * Pure (no I/O, no logging) — the chat layer drives the FSM and emits
--     events to its Chat_Sink.
---------------------------------------------------------------------

with LLM_Qwen;
with Ada.Strings.Unbounded;

package LLM_Chat_Parser is

   --  A parsed tool call (same shape as LLM_Qwen.Tool_Call but kept as
   --  primitives so the parser can be unit-tested without LLM_Qwen).
   type Tool_Call is record
      Id            : Ada.Strings.Unbounded.Unbounded_String;
      Name          : Ada.Strings.Unbounded.Unbounded_String;
      Arguments_JS  : Ada.Strings.Unbounded.Unbounded_String;
   end record;
   type Tool_Call_Array is array (Positive range <>) of Tool_Call;

   --  Final parsed result of a complete ChatML stream. Reasoning and Answer
   --  are concatenated text (with XML stripped). Finish mirrors the
   --  On_Finish_Reason event the sink received.
   type Result (N_Tool_Calls : Natural) is record
      Reasoning  : Ada.Strings.Unbounded.Unbounded_String;
      Answer     : Ada.Strings.Unbounded.Unbounded_String;
      Finish     : Ada.Strings.Unbounded.Unbounded_String;
      Tool_Calls : Tool_Call_Array (1 .. N_Tool_Calls);
   end record;

   --  FSM: holds the in-progress text buffer, current state, and the
   --  accumulated Result. Feed pieces one at a time; Finalize at the end
   --  of the stream to flush any trailing state and finalize Finish.
   type Parser is tagged private;

   --  Initial state (Finish not yet determined; parser doesn't emit
   --  On_Finish_Reason until you call Finalize).
   --
   --  Start_In_Reasoning seeds the FSM as if a `<think>` opener was already
   --  consumed. Use it when the PROMPT opens a thinking block (the assistant
   --  generation prompt ends with `<think>`, as this fine-tune's always-think
   --  template does): the model then emits its chain-of-thought and a closing
   --  `</think>` in the stream, so the opener is NOT in the generated text and
   --  the parser must already be in S_In_Reasoning to route it to
   --  reasoning_content rather than mis-classifying it as the answer.
   function New_Parser (Start_In_Reasoning : Boolean := False) return Parser;

   --  Append a piece of model output. Drives the FSM:
   --   * emits On_Reasoning(...) for any text inside a <think> block,
   --   * emits On_Text(...) for any text outside thinking/tools,
   --   * emits On_Tool_Call(Id, Name, Arguments) once a complete
   --     <tool_call>...</tool_call> block has been seen (the Arguments
   --     are reconstructed as a JSON object from the parameter pairs).
   procedure Feed
     (P     : in out Parser;
      Piece : String;
      Sink  : access LLM_Qwen.Chat_Sink'Class);

   --  Close the FSM: flushes any pending state, sets Finish to "stop"
   --  (or "tool_calls" if at least one tool was seen and no text came
   --  after the last tool — heuristically the "primary intent" case).
   procedure Finalize (P : in out Parser;
                       Sink : access LLM_Qwen.Chat_Sink'Class);

   --  Read accessors used by tests / debug.
   function Reasoning_Of (P : Parser) return String;
   function Answer_Of    (P : Parser) return String;
   function Finish_Of    (P : Parser) return String;
   function Tool_Calls_Of (P : Parser) return Tool_Call_Array;
   function N_Tool_Calls (P : Parser) return Natural;

private

   --  Strict pattern positions in the buffer:
   --    S_Idle         — outside any tag, looking for <think> or <tool_call>
   --    S_In_Reasoning — inside <think>...</think>
   --    S_In_Text      — saw </think> (or are in plain text mode), looking
   --                      for <tool_call>
   --    S_In_Tool      — inside <tool_call>, parsing <function=NAME> or
   --                      <parameter=KEY>VALUE</parameter> until </tool_call>
   type State_Kinds is (S_Idle, S_In_Reasoning, S_In_Text, S_In_Tool);

   --  Internal state for parsing one tool block.
   type Tool_Parse_State is record
      Name         : Ada.Strings.Unbounded.Unbounded_String;
      -- Params_Seen removed: not needed; Build_Args derives JSON directly from Buf
      In_Param     : Boolean := False;
      Cur_Key      : Ada.Strings.Unbounded.Unbounded_String;
      Cur_Val      : Ada.Strings.Unbounded.Unbounded_String;
      Id_Counter   : Natural := 0;
   end record;

   type Parser is tagged record
      State    : State_Kinds := S_Idle;
      Buf      : Ada.Strings.Unbounded.Unbounded_String; -- rolling buffer
      Max_Buf  : Natural := 64;  -- rolling buffer cap; if a single piece is
                                 -- larger, we still process what we can.
      Reasoning: Ada.Strings.Unbounded.Unbounded_String;
      Answer   : Ada.Strings.Unbounded.Unbounded_String;
      Finish   : Ada.Strings.Unbounded.Unbounded_String :=
                   Ada.Strings.Unbounded.Null_Unbounded_String;
      Calls    : Tool_Call_Array (1 .. 64);  -- max tool calls per Chat
      N_Calls  : Natural := 0;
      Tp       : Tool_Parse_State;
      --  Track of whether text came after the last tool call (used by
      --  Finalize to pick Finish = "stop" vs "tool_calls" heuristically).
      Text_After_Last_Tool : Boolean := False;
      Tool_Cap_Reached     : Boolean := False;
      --  Nested-balance counter for "bare" tags. Both "think" and
      --  "tool_call" have ambiguous close/open (the fine-tune uses the
      --  same token for both). We treat the first occurrence inside a
      --  region as open and the next as close. Reset by Canonical
      --  close ("</...>").
      Think_Depth  : Natural := 0;
      Tool_Depth   : Natural := 0;
   end record;

end LLM_Chat_Parser;