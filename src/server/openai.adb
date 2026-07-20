---------------------------------------------------------------------
-- OpenAI body.
---------------------------------------------------------------------

with JSON;
with Ada.Directories;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with LLM_Catalog;

package body OpenAI is

   use type LLM_Qwen.Role_Kind;
   use type JSON.Value_Ref;

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
         --  Sampling defaults MATCH the LIVE hura Ollama config (verified via
         --  `ollama show hura --parameters` 2026-07-14) so a client that omits
         --  a field gets byte-for-byte Ollama behaviour: temp 0.15, top_p 0.8,
         --  top_k 20, min_p 0.05, repeat_penalty 1.0 (i.e. NO repeat penalty),
         --  presence_penalty 0.0. The platform sends ONLY temperature/max_tokens
         --  to the model (openai-compat #buildRequestBody / #buildOllamaBody
         --  forward neither top_p/top_k/min_p nor repeat_penalty), so these
         --  server-side defaults ARE the effective sampler — they must equal the
         --  Modelfile exactly. (The 2026-07-13 "settings-parity" audit used
         --  guessed values temp 0.2 / top_p 0.9 / top_k 40 / min_p 0 / repeat
         --  1.1 that never matched the real Modelfile; that looser candidate set
         --  — esp. min_p 0 — let the model ramble/hallucinate where Ollama, with
         --  min_p 0.05 + top_k 20 + top_p 0.8, stays focused.)
         --  Prod default = RELIABILITY (operator decision 2026-07-20). The
         --  official eval settings (temp=1.0/top_p=1.0, pure sampling) maximise
         --  the trained distribution and coding-exploration diversity, but the
         --  prod-metrics run measured occasional factual slips under pure
         --  sampling (e.g. "Sydney" for Australia's capital) that temp=0.2
         --  fixes deterministically. We serve general chat, so default to the
         --  low-variance config; a caller wanting the benchmark distribution
         --  passes temperature=1.0/top_p=1.0 explicitly per request.
         R.Params.Temperature      := JSON.As_Float (JSON.Get (V, "temperature"), 0.2);
         R.Params.Top_P            := JSON.As_Float (JSON.Get (V, "top_p"), 0.95);
         R.Params.Top_K            := JSON.As_Int   (JSON.Get (V, "top_k"), 20);
         R.Params.Min_P            := JSON.As_Float (JSON.Get (V, "min_p"), 0.0);
         R.Params.Presence_Penalty := JSON.As_Float (JSON.Get (V, "presence_penalty"), 0.0);
         --  Ollama-native `think` maps to this on the /api/chat bridge; default
         --  thinking ON (the model reasons unless the caller disables it).
         R.Params.Enable_Thinking  := JSON.As_Bool (JSON.Get (V, "enable_thinking"), True);
         R.Params.Seed        :=
           Long_Long_Integer (JSON.As_Int (JSON.Get (V, "seed"), 0));
         --  Repetition penalty (Ollama `repeat_penalty`, multiplicative over the
         --  last repeat_last_n=64 tokens). Default 1.0 = hura's live Modelfile,
         --  i.e. NO repeat penalty — Ollama relies on min_p 0.05 + top_k 20 for
         --  anti-looping, not a penalty. An OpenAI `frequency_penalty` (additive
         --  semantics) still maps in as an override when the caller sends it.
         R.Params.Repeat_Penalty := JSON.As_Float (JSON.Get (V, "repeat_penalty"), 1.0);
         if JSON.Exists (JSON.Get (V, "frequency_penalty")) then
            R.Params.Repeat_Penalty :=
              1.0 + JSON.As_Float (JSON.Get (V, "frequency_penalty"), 0.0);
         end if;
         if Has_Tools then
            --  Synthesize a system prompt that lists every available function
            --  and the exact XML block format Hura emits. The chat layer
            --  then prepends this to the conversation. The format mirrors
            --  Qwen3.5 / ChatML tool spec:
            --
            --    <tool_call>
            --    <function=NAME>
            --    <parameter=KEY>VALUE</parameter>
            --    </function>
            --    <tool_call>
            --
            --  VERBATIM mirror of the official Ornith-1.0 chat template's tools
            --  block (deepreinforce-ai/Ornith-1.0-35B chat_template.jinja):
            --  header text, one-line `tojson` of each WHOLE tool object,
            --  the exact example format (parameter values on their own lines),
            --  and the <IMPORTANT> reminder. The previous synthesized prompt
            --  (different wording, pretty-printed JSON, an invented "Form B")
            --  deviated from the train-time distribution and measurably hurt
            --  tool-call reliability.
            declare
               Acc : Unbounded_String :=
                 Null_Unbounded_String & "# Tools" & ASCII.LF & ASCII.LF
                 & "You have access to the following functions:" & ASCII.LF
                 & ASCII.LF & "<tools>";
            begin
               for I in 1 .. JSON.Length (T) loop
                  declare
                     Ti : constant JSON.Value_Ref := JSON.Item (T, I);
                  begin
                     Acc := Acc & ASCII.LF & JSON.To_String (Ti);
                  end;
               end loop;
               Acc := Acc & ASCII.LF & "</tools>" & ASCII.LF & ASCII.LF
                 & "If you choose to call a function ONLY reply in the following format with NO suffix:" & ASCII.LF
                 & ASCII.LF
                 & "<tool_call>" & ASCII.LF
                 & "<function=example_function_name>" & ASCII.LF
                 & "<parameter=example_parameter_1>" & ASCII.LF
                 & "value_1" & ASCII.LF
                 & "</parameter>" & ASCII.LF
                 & "<parameter=example_parameter_2>" & ASCII.LF
                 & "This is the value for the second parameter" & ASCII.LF
                 & "that can span" & ASCII.LF
                 & "multiple lines" & ASCII.LF
                 & "</parameter>" & ASCII.LF
                 & "</function>" & ASCII.LF
                 & "</tool_call>" & ASCII.LF
                 & ASCII.LF
                 & "<IMPORTANT>" & ASCII.LF
                 & "Reminder:" & ASCII.LF
                 & "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags" & ASCII.LF
                 & "- Required parameters MUST be specified" & ASCII.LF
                 & "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after" & ASCII.LF
                 & "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls" & ASCII.LF
                 & "</IMPORTANT>";
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
     (Model, Id, Name, Arguments : String; Index : Natural := 0) return String
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
      JSON.Set (TC, "index", JSON.Int (Index));
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

   function Ollama_Body_To_OpenAI (Raw : String) return String is
      V     : JSON.Value_Ref;
      Out_O : constant JSON.Value_Ref := JSON.New_Object;
      Opts  : JSON.Value_Ref;
   begin
      begin
         V := JSON.Parse (Raw);
      exception when others => return Raw;
      end;
      if V = null then return Raw; end if;
      if JSON.Exists (JSON.Get (V, "model")) then
         JSON.Set (Out_O, "model", JSON.Get (V, "model"));
      end if;
      if JSON.Exists (JSON.Get (V, "messages")) then
         JSON.Set (Out_O, "messages", JSON.Get (V, "messages"));
      end if;
      if JSON.Exists (JSON.Get (V, "tools")) then
         JSON.Set (Out_O, "tools", JSON.Get (V, "tools"));
      end if;
      JSON.Set (Out_O, "stream", JSON.Bool (JSON.As_Bool (JSON.Get (V, "stream"), True)));
      JSON.Set (Out_O, "enable_thinking",
                JSON.Bool (JSON.As_Bool (JSON.Get (V, "think"), True)));
      Opts := JSON.Get (V, "options");
      if Opts /= null then
         if JSON.Exists (JSON.Get (Opts, "temperature")) then
            JSON.Set (Out_O, "temperature", JSON.Get (Opts, "temperature"));
         end if;
         if JSON.Exists (JSON.Get (Opts, "num_predict")) then
            JSON.Set (Out_O, "max_tokens", JSON.Get (Opts, "num_predict"));
         end if;
         if JSON.Exists (JSON.Get (Opts, "top_p")) then
            JSON.Set (Out_O, "top_p", JSON.Get (Opts, "top_p"));
         end if;
         if JSON.Exists (JSON.Get (Opts, "top_k")) then
            JSON.Set (Out_O, "top_k", JSON.Get (Opts, "top_k"));
         end if;
      end if;
      return JSON.To_String (Out_O);
   end Ollama_Body_To_OpenAI;

   function Ollama_Chunk (Model, Piece : String; Reason : Boolean) return String is
      Root : constant JSON.Value_Ref := JSON.New_Object;
      Msg  : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (Msg, "role", JSON.Str ("assistant"));
      if Reason then
         JSON.Set (Msg, "thinking", JSON.Str (Piece));
      else
         JSON.Set (Msg, "content", JSON.Str (Piece));
      end if;
      JSON.Set (Root, "model", JSON.Str (Model));
      JSON.Set (Root, "message", Msg);
      JSON.Set (Root, "done", JSON.Bool (False));
      return JSON.To_String (Root);
   end Ollama_Chunk;

   function Ollama_Tool_Chunk (Model, Id, Name, Arguments : String) return String is
      Root : constant JSON.Value_Ref := JSON.New_Object;
      Msg  : constant JSON.Value_Ref := JSON.New_Object;
      Arr  : constant JSON.Value_Ref := JSON.New_Array;
      TC   : constant JSON.Value_Ref := JSON.New_Object;
      Fn   : constant JSON.Value_Ref := JSON.New_Object;
      Args : JSON.Value_Ref;
   begin
      JSON.Set (Fn, "name", JSON.Str (Name));
      --  Ollama tool arguments are an OBJECT; parse when the model gave valid
      --  JSON, else fall back to the raw string (the platform accepts both).
      begin
         Args := (if Arguments'Length > 0 then JSON.Parse (Arguments) else JSON.New_Object);
      exception when others => Args := JSON.Str (Arguments);
      end;
      JSON.Set (Fn, "arguments", Args);
      JSON.Set (TC, "function", Fn);
      if Id'Length > 0 then JSON.Set (TC, "id", JSON.Str (Id)); end if;
      JSON.Append (Arr, TC);
      JSON.Set (Msg, "role", JSON.Str ("assistant"));
      JSON.Set (Msg, "tool_calls", Arr);
      JSON.Set (Root, "model", JSON.Str (Model));
      JSON.Set (Root, "message", Msg);
      JSON.Set (Root, "done", JSON.Bool (False));
      return JSON.To_String (Root);
   end Ollama_Tool_Chunk;

   function Ollama_Response_From_OpenAI (Raw, Model : String) return String is
      V    : JSON.Value_Ref;
      Root : constant JSON.Value_Ref := JSON.New_Object;
      Msg  : constant JSON.Value_Ref := JSON.New_Object;
   begin
      begin V := JSON.Parse (Raw); exception when others => return Raw; end;
      if V = null then return Raw; end if;
      declare
         Choices : constant JSON.Value_Ref := JSON.Get (V, "choices");
         C0   : constant JSON.Value_Ref :=
           (if Choices /= null and then JSON.Length (Choices) >= 1
            then JSON.Item (Choices, 1) else null);
         OMsg : constant JSON.Value_Ref :=
           (if C0 /= null then JSON.Get (C0, "message") else null);
         Usage : constant JSON.Value_Ref := JSON.Get (V, "usage");
      begin
         JSON.Set (Msg, "role", JSON.Str ("assistant"));
         if OMsg /= null then
            if JSON.Exists (JSON.Get (OMsg, "content")) then
               JSON.Set (Msg, "content", JSON.Get (OMsg, "content"));
            else
               JSON.Set (Msg, "content", JSON.Str (""));
            end if;
            if JSON.Exists (JSON.Get (OMsg, "reasoning_content")) then
               JSON.Set (Msg, "thinking", JSON.Get (OMsg, "reasoning_content"));
            end if;
            if JSON.Exists (JSON.Get (OMsg, "tool_calls")) then
               JSON.Set (Msg, "tool_calls", JSON.Get (OMsg, "tool_calls"));
            end if;
         else
            JSON.Set (Msg, "content", JSON.Str (""));
         end if;
         JSON.Set (Root, "model", JSON.Str (Model));
         JSON.Set (Root, "message", Msg);
         JSON.Set (Root, "done", JSON.Bool (True));
         JSON.Set (Root, "done_reason",
           (if C0 /= null and then JSON.Exists (JSON.Get (C0, "finish_reason"))
            then JSON.Get (C0, "finish_reason") else JSON.Str ("stop")));
         if Usage /= null then
            JSON.Set (Root, "prompt_eval_count",
              (if JSON.Exists (JSON.Get (Usage, "prompt_tokens"))
               then JSON.Get (Usage, "prompt_tokens") else JSON.Int (0)));
            JSON.Set (Root, "eval_count",
              (if JSON.Exists (JSON.Get (Usage, "completion_tokens"))
               then JSON.Get (Usage, "completion_tokens") else JSON.Int (0)));
         end if;
         return JSON.To_String (Root);
      end;
   end Ollama_Response_From_OpenAI;

   function Ollama_Done_Chunk
     (Model : String;
      Prompt_Tokens, Completion_Tokens : Natural := 0;
      Finish : String := "stop") return String
   is
      Root : constant JSON.Value_Ref := JSON.New_Object;
      Msg  : constant JSON.Value_Ref := JSON.New_Object;
   begin
      JSON.Set (Msg, "role", JSON.Str ("assistant"));
      JSON.Set (Msg, "content", JSON.Str (""));
      JSON.Set (Root, "model", JSON.Str (Model));
      JSON.Set (Root, "message", Msg);
      JSON.Set (Root, "done", JSON.Bool (True));
      JSON.Set (Root, "done_reason", JSON.Str (Finish));
      JSON.Set (Root, "prompt_eval_count", JSON.Int (Prompt_Tokens));
      JSON.Set (Root, "eval_count", JSON.Int (Completion_Tokens));
      return JSON.To_String (Root);
   end Ollama_Done_Chunk;

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
