---------------------------------------------------------------------
-- OpenAI — map the OpenAI /v1 chat schema to/from the engine types.
--
-- Parses a /v1/chat/completions request body into the engine's Message_Array +
-- sampling params, and builds the chat.completion / chat.completion.chunk /
-- models / error response JSON. Uses the in-house JSON package.
--
-- Tools[] parsing: when the request carries `tools`, we synthesize one
-- `Role_System` message at the FRONT of the conversation that teaches the
-- model the available functions and the exact XML block format
-- (`<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>`).
-- If the model wants to call a tool, the parser (LLM_Chat_Parser) will split
-- it out of the stream and the chat layer returns it as a structured
-- tool_calls[] field on the OpenAI response (see Chat_Response).
---------------------------------------------------------------------

with LLM_Qwen;
with LLM_Sampler;
with Ada.Strings.Unbounded;

package OpenAI is

   type Request (N : Natural) is record
      Messages   : LLM_Qwen.Message_Array (1 .. N);
      Params     : LLM_Sampler.Params;
      Max_Tokens : Integer := 256;
      Stream     : Boolean := False;
      Model      : Ada.Strings.Unbounded.Unbounded_String;
      --  Tools_Sysmsg (if non-empty) is prepended to Messages as a
      --  Role_System entry. Set by Parse_Chat when the request carries a
      --  non-empty `tools` array.
      Tools_Sysmsg : Ada.Strings.Unbounded.Unbounded_String;
   end record;

   --  Parse a request body. May raise JSON.Parse_Error on malformed input.
   function Parse_Chat (Body_JSON : String) return Request;

   --  Non-streaming chat.completion response. Reasoning and tool calls are
   --  optional: when absent the response is the legacy {role, content} shape
   --  (existing clients see no change). When a reasoning block is present
   --  we add reasoning_content (o1-style field). When tool_calls exist we
   --  add a tool_calls[] array, set finish_reason="tool_calls" if no text
   --  followed the tools, and still carry usage + finish_reason on the root.
   function Chat_Response
     (Model, Content : String;
      Reasoning       : String := "";
      Tool_Calls_JSON : String := "";   -- raw JSON array string; empty = none
      Prompt_Tokens, Completion_Tokens : Natural := 0;
      Finish : String := "stop") return String;

   --  Streaming: one chat.completion.chunk per token piece, then a final
   --  chunk carrying finish_reason + usage. The Reason flag switches the
   --  delta's reasoning_content vs content field (proxy maps to OpenAI
   --  o1-style reasoning_content). The proxy wraps each in an SSE
   --  "data: ...\n\n" line.
   function Chat_Chunk
     (Model, Piece : String; First, Reason : Boolean) return String;

   --  Tool-call delta chunk: emits a single
   --  choices[0].delta.tool_calls=[{index, id, type, function:{name, arguments}}]
   --  object. The proxy forwards it as-is.
   function Tool_Call_Chunk
     (Model, Id, Name, Arguments : String) return String;

   function Chat_Done_Chunk
     (Model : String;
      Prompt_Tokens, Completion_Tokens : Natural := 0;
      Finish : String := "stop") return String;

   --  /v1/models list response (single active model).
   function Models_Response (Model_Id : String) return String;

   --  /v1/models list of EVERY model discovered on this system (via
   --  LLM_Catalog), each with extra aspida fields: name, arch, quant, params,
   --  size, supported, active. Active_Path marks the currently-loaded model;
   --  Switchable says whether this server can change models at runtime.
   function Catalog_Response
     (Active_Path : String; Switchable : Boolean) return String;

   --  Result of a model-selection request.
   function Select_Result
     (OK : Boolean; Reload : Boolean; Message : String) return String;

   --  Error envelope. Code is the OpenAI machine-readable error code (e.g.
   --  "context_length_exceeded"); omitted => only the human message + type.
   function Error_Response (Message : String; Code : String := "") return String;

end OpenAI;
