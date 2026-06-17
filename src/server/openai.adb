---------------------------------------------------------------------
-- OpenAI body.
---------------------------------------------------------------------

with JSON;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body OpenAI is

   use type LLM_Qwen.Role_Kind;

   ID_Str : constant String := "chatcmpl-aspida";

   function Role_Of (S : String) return LLM_Qwen.Role_Kind is
     (if S = "system" then LLM_Qwen.Role_System
      elsif S = "assistant" then LLM_Qwen.Role_Assistant
      else LLM_Qwen.Role_User);

   function Parse_Chat (Body_JSON : String) return Request is
      V : constant JSON.Value_Ref := JSON.Parse (Body_JSON);
      M : constant JSON.Value_Ref := JSON.Get (V, "messages");
      N : constant Natural := JSON.Length (M);
   begin
      return R : Request (N) do
         for I in 1 .. N loop
            declare
               Mi : constant JSON.Value_Ref := JSON.Item (M, I);
            begin
               R.Messages (I) :=
                 (Role => Role_Of (JSON.As_String (JSON.Get (Mi, "role"), "user")),
                  Text => To_Unbounded_String (JSON.As_String (JSON.Get (Mi, "content"))));
            end;
         end loop;
         R.Model      := To_Unbounded_String (JSON.As_String (JSON.Get (V, "model"), "aspida"));
         R.Max_Tokens := JSON.As_Int (JSON.Get (V, "max_tokens"), 256);
         R.Stream     := JSON.As_Bool (JSON.Get (V, "stream"), False);
         --  Sampling: honour OpenAI fields (temperature default 1.0, top_p 1.0).
         R.Params.Temperature := JSON.As_Float (JSON.Get (V, "temperature"), 1.0);
         R.Params.Top_P       := JSON.As_Float (JSON.Get (V, "top_p"), 1.0);
         R.Params.Top_K       := JSON.As_Int   (JSON.Get (V, "top_k"), 0);
         R.Params.Seed        :=
           Long_Long_Integer (JSON.As_Int (JSON.Get (V, "seed"), 0));
         if JSON.Exists (JSON.Get (V, "frequency_penalty")) then
            R.Params.Repeat_Penalty :=
              1.0 + JSON.As_Float (JSON.Get (V, "frequency_penalty"), 0.0);
         end if;
      end return;
   end Parse_Chat;

   function Chat_Response (Model, Content : String) return String is
      Root    : constant JSON.Value_Ref := JSON.New_Object;
      Choices : constant JSON.Value_Ref := JSON.New_Array;
      Choice  : constant JSON.Value_Ref := JSON.New_Object;
      Msg     : constant JSON.Value_Ref := JSON.New_Object;
      Usage   : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (Msg, "role", JSON.Str ("assistant"));
      JSON.Set (Msg, "content", JSON.Str (Content));
      JSON.Set (Choice, "index", JSON.Int (0));
      JSON.Set (Choice, "message", Msg);
      JSON.Set (Choice, "finish_reason", JSON.Str ("stop"));
      JSON.Append (Choices, Choice);
      JSON.Set (Usage, "prompt_tokens", JSON.Int (0));
      JSON.Set (Usage, "completion_tokens", JSON.Int (0));
      JSON.Set (Usage, "total_tokens", JSON.Int (0));
      JSON.Set (Root, "id", JSON.Str (ID_Str));
      JSON.Set (Root, "object", JSON.Str ("chat.completion"));
      JSON.Set (Root, "created", JSON.Int (0));
      JSON.Set (Root, "model", JSON.Str (Model));
      JSON.Set (Root, "choices", Choices);
      JSON.Set (Root, "usage", Usage);
      return JSON.To_String (Root);
   end Chat_Response;

   function Chat_Chunk (Model, Piece : String; First : Boolean) return String is
      Root    : constant JSON.Value_Ref := JSON.New_Object;
      Choices : constant JSON.Value_Ref := JSON.New_Array;
      Choice  : constant JSON.Value_Ref := JSON.New_Object;
      Delta_O : constant JSON.Value_Ref := JSON.New_Object;
   begin
      if First then
         JSON.Set (Delta_O, "role", JSON.Str ("assistant"));
      end if;
      JSON.Set (Delta_O, "content", JSON.Str (Piece));
      JSON.Set (Choice, "index", JSON.Int (0));
      JSON.Set (Choice, "delta", Delta_O);
      JSON.Set (Choice, "finish_reason", JSON.Null_Value);
      JSON.Append (Choices, Choice);
      JSON.Set (Root, "id", JSON.Str (ID_Str));
      JSON.Set (Root, "object", JSON.Str ("chat.completion.chunk"));
      JSON.Set (Root, "model", JSON.Str (Model));
      JSON.Set (Root, "choices", Choices);
      return JSON.To_String (Root);
   end Chat_Chunk;

   function Chat_Done_Chunk (Model : String) return String is
      Root    : constant JSON.Value_Ref := JSON.New_Object;
      Choices : constant JSON.Value_Ref := JSON.New_Array;
      Choice  : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (Choice, "index", JSON.Int (0));
      JSON.Set (Choice, "delta", JSON.New_Object);
      JSON.Set (Choice, "finish_reason", JSON.Str ("stop"));
      JSON.Append (Choices, Choice);
      JSON.Set (Root, "id", JSON.Str (ID_Str));
      JSON.Set (Root, "object", JSON.Str ("chat.completion.chunk"));
      JSON.Set (Root, "model", JSON.Str (Model));
      JSON.Set (Root, "choices", Choices);
      return JSON.To_String (Root);
   end Chat_Done_Chunk;

   function Models_Response (Model_Id : String) return String is
      Root : constant JSON.Value_Ref := JSON.New_Object;
      Data : constant JSON.Value_Ref := JSON.New_Array;
      M    : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (M, "id", JSON.Str (Model_Id));
      JSON.Set (M, "object", JSON.Str ("model"));
      JSON.Set (M, "created", JSON.Int (0));
      JSON.Set (M, "owned_by", JSON.Str ("aspida"));
      JSON.Append (Data, M);
      JSON.Set (Root, "object", JSON.Str ("list"));
      JSON.Set (Root, "data", Data);
      return JSON.To_String (Root);
   end Models_Response;

   function Error_Response (Message : String) return String is
      Root : constant JSON.Value_Ref := JSON.New_Object;
      Err  : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (Err, "message", JSON.Str (Message));
      JSON.Set (Err, "type", JSON.Str ("invalid_request_error"));
      JSON.Set (Root, "error", Err);
      return JSON.To_String (Root);
   end Error_Response;

end OpenAI;
