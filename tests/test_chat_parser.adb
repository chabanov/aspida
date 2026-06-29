---------------------------------------------------------------------
-- test_chat_parser — unit tests for LLM_Chat_Parser FSM.
--
-- Covers the four shapes an Ornith-1.0 (Qwen3.5) chat reply can take:
--   1) plain text only,
--   2) thinking + answer,
--   3) single tool_call,
--   4) thinking + tool_call + final answer.
--
-- Each case feeds the parser a sequence of short pieces (simulating
-- token boundaries that fall mid-tag) and verifies the recovered
-- Reasoning / Answer / Tool_Calls after Finalize. We use a tiny
-- collecting sink to capture every callback the parser emits.
---------------------------------------------------------------------

with Ada.Text_IO;          use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with LLM_Qwen;
with LLM_Chat_Parser;

procedure Test_Chat_Parser is

   ----------------------------------------------------------------------
   --  Collecting sink: records every callback the parser fires.
   ----------------------------------------------------------------------
   type Collector is new LLM_Qwen.Chat_Sink with record
      Reasoning   : Unbounded_String := Null_Unbounded_String;
      Text        : Unbounded_String := Null_Unbounded_String;
      Finish      : Unbounded_String := Null_Unbounded_String;
      Tool_Count  : Natural := 0;
      Tool_Id     : Unbounded_String := Null_Unbounded_String;
      Tool_Name   : Unbounded_String := Null_Unbounded_String;
      Tool_Args   : Unbounded_String := Null_Unbounded_String;
      Tick_Count  : Natural := 0;
      Emit_Count  : Natural := 0;
   end record;
   procedure On_Reasoning    (S : in out Collector; Piece : String);
   procedure On_Text         (S : in out Collector; Piece : String);
   procedure On_Tool_Call
     (S : in out Collector; Id, Name, Arguments_JS : String);
   procedure On_Finish_Reason (S : in out Collector; Reason : String);
   procedure Tick            (S : in out Collector);
   procedure Emit            (S : in out Collector; Piece : String);

   procedure On_Reasoning (S : in out Collector; Piece : String) is
   begin
      S.Reasoning := S.Reasoning & Piece;
   end On_Reasoning;

   procedure On_Text (S : in out Collector; Piece : String) is
   begin
      S.Text := S.Text & Piece;
   end On_Text;

   procedure On_Tool_Call
     (S : in out Collector;
      Id, Name, Arguments_JS : String) is
   begin
      S.Tool_Count  := S.Tool_Count + 1;
      S.Tool_Id     := S.Tool_Id     & To_Unbounded_String (Id);
      S.Tool_Name   := S.Tool_Name   & To_Unbounded_String (Name);
      S.Tool_Args   := S.Tool_Args   & To_Unbounded_String (Arguments_JS);
   end On_Tool_Call;

   procedure On_Finish_Reason
     (S : in out Collector; Reason : String) is
   begin
      S.Finish := To_Unbounded_String (Reason);
   end On_Finish_Reason;

   procedure Tick (S : in out Collector) is
   begin
      S.Tick_Count := S.Tick_Count + 1;
   end Tick;

   procedure Emit (S : in out Collector; Piece : String) is
   begin
      S.Emit_Count := S.Emit_Count + 1;
      S.Text := S.Text & Piece;
   end Emit;

   procedure Run (Label : String; Pieces : String;
                  Expected_Reason, Expected_Text : String;
                  Expected_Tool_Name, Expected_Tool_Args : String;
                  Expected_Finish : String;
                  Expected_Tool_Count : Natural) is
      P : LLM_Chat_Parser.Parser := LLM_Chat_Parser.New_Parser;
      C : aliased Collector;
   begin
      --  Feed character-by-character to exercise the rolling buffer (a
      --  single Feed with the whole string would be too easy: every tag
      --  fits in Buf and never crosses the boundary).
      for I in Pieces'Range loop
         LLM_Chat_Parser.Feed (P, Pieces (I .. I), C'Unchecked_Access);
      end loop;
      LLM_Chat_Parser.Finalize (P, C'Unchecked_Access);

      Put ("[");
      Put (Label);
      Put ("] ");

      declare
         R : constant String := To_String (C.Reasoning);
         T : constant String := To_String (C.Text);
         F : constant String := To_String (C.Finish);
      begin
         if R /= Expected_Reason then
            Put_Line ("FAIL: reasoning got """ & R & """ expected """ & Expected_Reason & """");
            raise Program_Error;
         end if;
         if T /= Expected_Text then
            Put_Line ("FAIL: text got """ & T & """ expected """ & Expected_Text & """");
            raise Program_Error;
         end if;
         if F /= Expected_Finish then
            Put_Line ("FAIL: finish got """ & F & """ expected """ & Expected_Finish & """");
            raise Program_Error;
         end if;
         if C.Tool_Count /= Expected_Tool_Count then
            Put_Line ("FAIL: tool count got " & C.Tool_Count'Image &
                      " expected " & Expected_Tool_Count'Image);
            raise Program_Error;
         end if;
         if Expected_Tool_Count > 0 then
            declare
               Nm : constant String := To_String (C.Tool_Name);
               Ar : constant String := To_String (C.Tool_Args);
            begin
               if Nm /= Expected_Tool_Name then
                  Put_Line ("FAIL: tool name got """ & Nm & """");
                  raise Program_Error;
               end if;
               if Ar /= Expected_Tool_Args then
                  Put_Line ("FAIL: tool args got """ & Ar & """");
                  raise Program_Error;
               end if;
            end;
         end if;
      end;
      Put_Line ("OK");
   end Run;

begin
   Put_Line ("LLM_Chat_Parser tests");

   -- 1) plain text: just an answer, no reasoning, no tools.
   Run ("plain-text",
        "Hello, world!",
        "", "Hello, world!", "", "", "stop", 0);

   -- 2) thinking + text: ①-think ① reasoning ①/think ① answer.
   Run ("think+text",
        "<think>reasoning goes here</think>The answer.",
        "reasoning goes here", "The answer.", "", "", "stop", 0);

   -- 3) single tool_call: standard XML form.
   Run ("tool_call",
        "<tool_call><function=run_tests><parameter=cmd>pytest -x</parameter></function></tool_call>",
        "", "", "run_tests", "{""cmd"": ""pytest -x""}", "tool_calls", 1);

   -- 4) thinking + tool_call + final text (the "agentic" shape).
   Run ("think+tool+text",
        "<think>I should run the tests.</think>" &
        "<tool_call><function=run_tests><parameter=cmd>pytest -x</parameter></function></tool_call>" &
        "Done — tests passed.",
        "I should run the tests.",
        "Done — tests passed.",
        "run_tests", "{""cmd"": ""pytest -x""}", "stop", 1);

   -- 5) bare-form tool_call: Ornith-1.0 (Qwen3.5 9B) emits without angle
   --    brackets — same token opens and closes the block. The parser
   --    resolves the ambiguity with a balance counter: first "tool_call"
   --    in a region is the opener, second is the closer.
   Run ("bare-tool-call",
        "tool_call<function=run_tests><parameter=cmd>pytest -x</parameter></function>tool_call",
        "", "", "run_tests", "{""cmd"": ""pytest -x""}", "tool_calls", 1);

   -- 6) bare-form reasoning (line-start closer, real Ornith shape).
   Run ("bare-think+text",
        "thinkreasoning goes here" & ASCII.LF &
        "thinkThe answer.",
        "reasoning goes here" & ASCII.LF, "The answer.",
        "", "", "stop", 0);

   -- 7) bare-form agentic shape (full Ornith output): reasoning then
   --    tool_call then text, all bare, with line-start closers.
   Run ("bare-think+tool+text",
        "thinkI should run the tests." & ASCII.LF &
        "think" &
        "tool_call<function=run_tests><parameter=cmd>pytest -x</parameter></function>tool_call" &
        "Done — tests passed.",
        "I should run the tests." & ASCII.LF,
        "Done — tests passed.",
        "run_tests", "{""cmd"": ""pytest -x""}", "stop", 1);

   -- 8) bare-form closer on its own line. The parser only honours bare
   --    closer at line-start to avoid false-positives on the word
   --    "think" inside reasoning prose. The body newline is included in
   --    the reasoning content because it's part of the buffer leading up
   --    to the line-start closer.
   Run ("bare-think-line-start-closer",
        "thinkI think, therefore I am." & ASCII.LF &
        "thinkFinal.",
        "I think, therefore I am." & ASCII.LF,
        "Final.", "", "", "stop", 0);

   Put_Line ("all green");
end Test_Chat_Parser;