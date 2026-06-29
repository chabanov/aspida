---------------------------------------------------------------------
-- OpenAI body.
---------------------------------------------------------------------

with JSON;
with Ada.Directories;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with LLM_Catalog;

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
      T : constant JSON.Value_Ref := JSON.Get (V, "tools");
      N : constant Natural := JSON.Length (M);
      Has_Tools : constant Boolean := JSON.Is_Array (T) and then JSON.Length (T) > 0;
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
         --  Omitted max_tokens means "as much as allowed" (OpenAI semantics):
         --  default high and let the server cap bound it, instead of a low 256
         --  that silently truncates normal answers.
         R.Max_Tokens := JSON.As_Int (JSON.Get (V, "max_tokens"), 1_000_000);
         R.Stream     := JSON.As_Bool (JSON.Get (V, "stream"), False);
         --  Sampling: honour OpenAI fields (temperature default 1.0, top_p 1.0).
         R.Params.Temperature := JSON.As_Float (JSON.Get (V, "temperature"), 1.0);
         R.Params.Top_P       := JSON.As_Float (JSON.Get (V, "top_p"), 1.0);
         R.Params.Top_K       := JSON.As_Int   (JSON.Get (V, "top_k"), 0);
         R.Params.Min_P       := JSON.As_Float (JSON.Get (V, "min_p"), 0.0);
         R.Params.Seed        :=
           Long_Long_Integer (JSON.As_Int (JSON.Get (V, "seed"), 0));
         if JSON.Exists (JSON.Get (V, "frequency_penalty")) then
            R.Params.Repeat_Penalty :=
              1.0 + JSON.As_Float (JSON.Get (V, "frequency_penalty"), 0.0);
         end if;
         if Has_Tools then
            --  Synthesize a system prompt that lists every available function
            --  and the exact XML block format Ornith emits. The chat layer
            --  then prepends this to the conversation. The format mirrors
            --  Qwen3.5 / ChatML tool spec:
            --
            --    <tool_call>
            --    <function=NAME>
            --    <parameter=KEY>VALUE</parameter>
            --    </function>
            --    <tool_call>
            --
            declare
               Acc : Unbounded_String :=
                 Null_Unbounded_String & "# Tools" & ASCII.LF
                 & "You may call one or more functions to assist with the user"
                 & " query." & ASCII.LF & "Here are the available tools:" & ASCII.LF
                 & ASCII.LF & "<tools>" & ASCII.LF;
            begin
               for I in 1 .. JSON.Length (T) loop
                  declare
                     Ti : constant JSON.Value_Ref := JSON.Item (T, I);
                     F  : constant JSON.Value_Ref := JSON.Get (Ti, "function");
                     Nm : constant String :=
                       JSON.As_String (JSON.Get (F, "name"), "");
                     Desc : constant String :=
                       JSON.As_String (JSON.Get (F, "description"), "");
                     Params : constant JSON.Value_Ref := JSON.Get (F, "parameters");
                  begin
                     if Nm /= "" then
                        Acc := Acc & "{" & ASCII.LF
                          & "  ""name"": """ & Nm & """,";
                        if Desc /= "" then
                           Acc := Acc & ASCII.LF
                             & "  ""description"": """ & Desc & """,";
                        end if;
                        Acc := Acc & ASCII.LF
                          & "  ""parameters"": "
                          & JSON.To_String (Params) & ASCII.LF
                          & "}" & ASCII.LF;
                     end if;
                  end;
               end loop;
               Acc := Acc & "</tools>" & ASCII.LF & ASCII.LF
                 & "When you make a tool call, emit a tag and a body. Two equivalent forms are accepted:" & ASCII.LF
                 & "  Form A (canonical):" & ASCII.LF
                 & "    <tool_call>" & ASCII.LF
                 & "    <function=NAME>" & ASCII.LF
                 & "    <parameter=KEY>VALUE</parameter>" & ASCII.LF
                 & "    </function>" & ASCII.LF
                 & "    </tool_call>" & ASCII.LF
                 & "  Form B (bare tags, line-aligned):" & ASCII.LF
                 & "    tool_call" & ASCII.LF
                 & "    <function=NAME>" & ASCII.LF
                 & "    <parameter=KEY>VALUE</parameter>" & ASCII.LF
                 & "    </function>" & ASCII.LF
                 & "    tool_call" & ASCII.LF
                 & "Pick Form A unless you were fine-tuned on Form B. In either form, output" & ASCII.LF
                 & "the angle brackets literally and do NOT wrap in Markdown fences or code blocks." & ASCII.LF
                 & "Otherwise answer normally. Do not make up parameter values." & ASCII.LF;
               R.Tools_Sysmsg := Acc;
            end;
         end if;
      end return;
   end Parse_Chat;

   function Chat_Response
     (Model, Content : String;
      Reasoning       : String := "";
      Tool_Calls_JSON : String := "";   -- raw JSON array string; empty = none
      Prompt_Tokens, Completion_Tokens : Natural := 0;
      Finish : String := "stop") return String
   is
      Root    : constant JSON.Value_Ref := JSON.New_Object;
      Choices : constant JSON.Value_Ref := JSON.New_Array;
      Choice  : constant JSON.Value_Ref := JSON.New_Object;
      Msg     : constant JSON.Value_Ref := JSON.New_Object;
      Usage   : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (Msg, "role", JSON.Str ("assistant"));
      if Reasoning /= "" then
         JSON.Set (Msg, "reasoning_content", JSON.Str (Reasoning));
      end if;
      if Tool_Calls_JSON /= "" then
         --  When tools are emitted the assistant may have no `content` (the
         --  model chose to act instead of talk). OpenAI convention: omit
         --  the field rather than emit an empty string.
         if Content /= "" then
            JSON.Set (Msg, "content", JSON.Str (Content));
         end if;
         declare
            Arr : constant JSON.Value_Ref := JSON.Parse (Tool_Calls_JSON);
         begin
            if JSON.Is_Array (Arr) then
               JSON.Set (Msg, "tool_calls", Arr);
            end if;
         end;
      else
         JSON.Set (Msg, "content", JSON.Str (Content));
      end if;
      JSON.Set (Choice, "index", JSON.Int (0));
      JSON.Set (Choice, "message", Msg);
      JSON.Set (Choice, "finish_reason", JSON.Str (Finish));
      JSON.Append (Choices, Choice);
      JSON.Set (Usage, "prompt_tokens", JSON.Int (Prompt_Tokens));
      JSON.Set (Usage, "completion_tokens", JSON.Int (Completion_Tokens));
      JSON.Set (Usage, "total_tokens", JSON.Int (Prompt_Tokens + Completion_Tokens));
      JSON.Set (Root, "id", JSON.Str (ID_Str));
      JSON.Set (Root, "object", JSON.Str ("chat.completion"));
      JSON.Set (Root, "created", JSON.Int (0));
      JSON.Set (Root, "model", JSON.Str (Model));
      JSON.Set (Root, "choices", Choices);
      JSON.Set (Root, "usage", Usage);
      return JSON.To_String (Root);
   end Chat_Response;

   function Chat_Chunk
     (Model, Piece : String; First, Reason : Boolean) return String
   is
      Root    : constant JSON.Value_Ref := JSON.New_Object;
      Choices : constant JSON.Value_Ref := JSON.New_Array;
      Choice  : constant JSON.Value_Ref := JSON.New_Object;
      Delta_O : constant JSON.Value_Ref := JSON.New_Object;
   begin
      if First then
         JSON.Set (Delta_O, "role", JSON.Str ("assistant"));
      end if;
      if Reason then
         JSON.Set (Delta_O, "reasoning_content", JSON.Str (Piece));
      else
         JSON.Set (Delta_O, "content", JSON.Str (Piece));
      end if;
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

   function Tool_Call_Chunk
     (Model, Id, Name, Arguments : String) return String
   is
      Root    : constant JSON.Value_Ref := JSON.New_Object;
      Choices : constant JSON.Value_Ref := JSON.New_Array;
      Choice  : constant JSON.Value_Ref := JSON.New_Object;
      Delta_O : constant JSON.Value_Ref := JSON.New_Object;
      TC      : constant JSON.Value_Ref := JSON.New_Object;
      Fn      : constant JSON.Value_Ref := JSON.New_Object;
      Arr     : constant JSON.Value_Ref := JSON.New_Array;
   begin
      JSON.Set (Fn, "name", JSON.Str (Name));
      JSON.Set (Fn, "arguments", JSON.Str (Arguments));
      JSON.Set (TC, "index", JSON.Int (0));
      JSON.Set (TC, "id", JSON.Str (Id));
      JSON.Set (TC, "type", JSON.Str ("function"));
      JSON.Set (TC, "function", Fn);
      JSON.Append (Arr, TC);
      JSON.Set (Delta_O, "role", JSON.Str ("assistant"));
      JSON.Set (Delta_O, "tool_calls", Arr);
      JSON.Set (Choice, "index", JSON.Int (0));
      JSON.Set (Choice, "delta", Delta_O);
      JSON.Set (Choice, "finish_reason", JSON.Null_Value);
      JSON.Append (Choices, Choice);
      JSON.Set (Root, "id", JSON.Str (ID_Str));
      JSON.Set (Root, "object", JSON.Str ("chat.completion.chunk"));
      JSON.Set (Root, "model", JSON.Str (Model));
      JSON.Set (Root, "choices", Choices);
      return JSON.To_String (Root);
   end Tool_Call_Chunk;

   function Chat_Done_Chunk
     (Model : String;
      Prompt_Tokens, Completion_Tokens : Natural := 0;
      Finish : String := "stop") return String
   is
      Root    : constant JSON.Value_Ref := JSON.New_Object;
      Choices : constant JSON.Value_Ref := JSON.New_Array;
      Choice  : constant JSON.Value_Ref := JSON.New_Object;
      Usage   : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (Choice, "index", JSON.Int (0));
      JSON.Set (Choice, "delta", JSON.New_Object);
      JSON.Set (Choice, "finish_reason", JSON.Str (Finish));
      JSON.Append (Choices, Choice);
      JSON.Set (Root, "id", JSON.Str (ID_Str));
      JSON.Set (Root, "object", JSON.Str ("chat.completion.chunk"));
      JSON.Set (Root, "model", JSON.Str (Model));
      JSON.Set (Root, "choices", Choices);
      JSON.Set (Usage, "prompt_tokens", JSON.Int (Prompt_Tokens));
      JSON.Set (Usage, "completion_tokens", JSON.Int (Completion_Tokens));
      JSON.Set (Usage, "total_tokens", JSON.Int (Prompt_Tokens + Completion_Tokens));
      JSON.Set (Root, "usage", Usage);
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

   function Catalog_Response
     (Active_Path : String; Switchable : Boolean) return String
   is
      use type LLM_Catalog.Model_Status;
      Cat  : constant LLM_Catalog.Entry_Vectors.Vector := LLM_Catalog.Discover;
      Root : constant JSON.Value_Ref := JSON.New_Object;
      Data : constant JSON.Value_Ref := JSON.New_Array;
   begin
      for E of Cat loop
         --  Only list things a user could pick: runnable models, plus
         --  valid-but-unsupported architectures (greyed out by the client).
         if E.Status = LLM_Catalog.Supported
           or else E.Status = LLM_Catalog.Unsupported
         then
            declare
               M    : constant JSON.Value_Ref := JSON.New_Object;
               Path : constant String := To_String (E.Path);
               --  Opaque, client-facing id: the file basename, never the
               --  absolute path (which would leak the server's username /
               --  directory layout). Selection accepts the basename or the
               --  full path server-side.
               Id   : constant String := Ada.Directories.Simple_Name (Path);
               MiB  : constant Long_Long_Integer := 1024 * 1024;
            begin
               JSON.Set (M, "id", JSON.Str (Id));
               JSON.Set (M, "object", JSON.Str ("model"));
               JSON.Set (M, "created", JSON.Int (0));
               JSON.Set (M, "owned_by", JSON.Str ("aspida"));
               JSON.Set (M, "name", JSON.Str (To_String (E.Name)));
               JSON.Set (M, "arch", JSON.Str (To_String (E.Arch)));
               JSON.Set (M, "quant", JSON.Str (To_String (E.Quant)));
               JSON.Set (M, "params", JSON.Str (To_String (E.Params)));
               JSON.Set (M, "size", JSON.Str (LLM_Catalog.Human_Size (E.Size)));
               JSON.Set (M, "size_mb", JSON.Int (Integer (E.Size / MiB)));
               JSON.Set (M, "supported",
                 JSON.Bool (E.Status = LLM_Catalog.Supported));
               JSON.Set (M, "active", JSON.Bool (Path = Active_Path));
               JSON.Append (Data, M);
            end;
         end if;
      end loop;
      JSON.Set (Root, "object", JSON.Str ("list"));
      JSON.Set (Root, "switchable", JSON.Bool (Switchable));
      JSON.Set (Root, "data", Data);
      return JSON.To_String (Root);
   end Catalog_Response;

   function Select_Result
     (OK : Boolean; Reload : Boolean; Message : String) return String
   is
      Root : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (Root, "ok", JSON.Bool (OK));
      JSON.Set (Root, "reload", JSON.Bool (Reload));
      JSON.Set (Root, "message", JSON.Str (Message));
      return JSON.To_String (Root);
   end Select_Result;

   function Error_Response (Message : String; Code : String := "") return String is
      Root : constant JSON.Value_Ref := JSON.New_Object;
      Err  : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (Err, "message", JSON.Str (Message));
      JSON.Set (Err, "type", JSON.Str ("invalid_request_error"));
      if Code /= "" then
         JSON.Set (Err, "code", JSON.Str (Code));
      end if;
      JSON.Set (Root, "error", Err);
      return JSON.To_String (Root);
   end Error_Response;

end OpenAI;
