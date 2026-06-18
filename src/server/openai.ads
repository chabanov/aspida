---------------------------------------------------------------------
-- OpenAI — map the OpenAI /v1 chat schema to/from the engine types.
--
-- Parses a /v1/chat/completions request body into the engine's Message_Array +
-- sampling params, and builds the chat.completion / chat.completion.chunk /
-- models / error response JSON. Uses the in-house JSON package.
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
   end record;

   --  Parse a request body. May raise JSON.Parse_Error on malformed input.
   function Parse_Chat (Body_JSON : String) return Request;

   --  Non-streaming chat.completion response, with OpenAI-standard usage and
   --  finish_reason ("stop" = natural end-of-turn, "length" = hit the token
   --  cap). Prompt/Completion are the real token counts.
   function Chat_Response
     (Model, Content : String;
      Prompt_Tokens, Completion_Tokens : Natural := 0;
      Finish : String := "stop") return String;

   --  Streaming: one chat.completion.chunk per token piece, then a final
   --  chunk carrying finish_reason + usage. (The proxy wraps each in an SSE
   --  "data: ...\n\n" line.)
   function Chat_Chunk      (Model, Piece : String; First : Boolean) return String;
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
