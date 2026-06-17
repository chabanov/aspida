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

   --  Non-streaming chat.completion response.
   function Chat_Response (Model, Content : String) return String;

   --  Streaming: one chat.completion.chunk per token piece, then a final
   --  stop chunk. (The proxy wraps each in an SSE "data: ...\n\n" line.)
   function Chat_Chunk      (Model, Piece : String; First : Boolean) return String;
   function Chat_Done_Chunk (Model : String) return String;

   --  /v1/models list response.
   function Models_Response (Model_Id : String) return String;

   --  Error envelope.
   function Error_Response (Message : String) return String;

end OpenAI;
