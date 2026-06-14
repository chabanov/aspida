---------------------------------------------------------------------
-- LLM_Chat body — interactive chat loop (Qwen 3.5 MoE, streaming)
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Calendar;
with Ada.Strings.Fixed;
with Ada.Strings;
with Ada.Exceptions;
with Ada.Environment_Variables;
with LLM_Qwen;

package body LLM_Chat is

   function Img (N : Integer) return String is
     (Ada.Strings.Fixed.Trim (Integer'Image (N), Ada.Strings.Left));

   --  Streams a model's tokens to the console in real time: a dot per prompt
   --  token while the prefill runs (so there is no silent wait), then the
   --  answer printed token-by-token as it is generated. Counts tokens so the
   --  caller can show a throughput footer.
   type Console_Sink is new LLM_Qwen.Token_Sink with record
      Started : Boolean := False;
      Count   : Natural := 0;
   end record;
   --  Specs immediately after the type (before it freezes); bodies below.
   overriding procedure Emit (S : in out Console_Sink; Piece : String);
   overriding procedure Tick (S : in out Console_Sink);

   overriding procedure Emit (S : in out Console_Sink; Piece : String) is
   begin
      if not S.Started then
         Ada.Text_IO.New_Line;          -- close the prefill "...." line
         S.Started := True;
      end if;
      Ada.Text_IO.Put (Piece);
      Ada.Text_IO.Flush;
      S.Count := S.Count + 1;
   end Emit;

   overriding procedure Tick (S : in out Console_Sink) is
   begin
      Ada.Text_IO.Put ('.');
      Ada.Text_IO.Flush;
   end Tick;

   procedure Run (Model_Dim : Integer := 64; N_Layers : Integer := 2) is
      use Ada.Text_IO;
      pragma Unreferenced (Model_Dim, N_Layers);  -- legacy tiny-model knobs

      Qwen_M : LLM_Qwen.Qwen_Model;               -- Qwen 3.5 model (from GGUF)
      Params : Long_Long_Integer := 0;
      Buffer : String (1 .. 4096);
      Last   : Integer;

      -- Model path: override with the QWEN_MODEL_PATH environment variable;
      -- otherwise fall back to the local LM Studio default.
      function Resolve_Model_Path return String is
         Var : constant String := "QWEN_MODEL_PATH";
      begin
         if Ada.Environment_Variables.Exists (Var) then
            return Ada.Environment_Variables.Value (Var);
         end if;
         return "/Users/ceo/.lmstudio/models/HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf";
      end Resolve_Model_Path;

      QWEN_PATH : constant String := Resolve_Model_Path;
   begin
      begin
         Qwen_M := LLM_Qwen.Load (QWEN_PATH);
         Put_Line ("=== Aspida LLM Chat (Qwen 3.5 MoE) ===");
         Params := LLM_Qwen.Param_Count (Qwen_M);
      exception
         when E : others =>
            Put_Line ("Qwen load failed: " & Ada.Exceptions.Exception_Message (E));
            return;
      end;

      Put_Line ("Params: " & Long_Long_Integer'Image (Params));
      Put_Line ("");
      Put_Line ("Commands: /quit, /clear, /model");
      Put_Line ("--------------------------");

      loop
         Put ("> ");
         Flush;
         Get_Line (Buffer, Last);

         if Last = 0 then
            goto Continue;
         end if;

         declare
            Input : constant String := Buffer (1 .. Last);
         begin
            if Input = "/quit" then
               Put_Line ("Bye.");
               exit;

            elsif Input = "/clear" then
               Put_Line ("Context cleared.");

            elsif Input = "/model" then
               Put_Line ("Qwen 3.5 MoE | " &
                 Integer'Image (LLM_Qwen.Block_Count (Qwen_M)) & " blocks, " &
                 Integer'Image (LLM_Qwen.Dim (Qwen_M)) & "d, " &
                 Long_Long_Integer'Image (Params) & " params");

            else
               --  Stream the reply token-by-token in real time (dots during
               --  prefill, then the answer as it is produced), then a footer.
               declare
                  Sink : aliased Console_Sink;
                  T0   : constant Ada.Calendar.Time := Ada.Calendar.Clock;
                  R    : constant String :=
                    LLM_Qwen.Chat (Qwen_M, Input, 256, Sink'Access);
                  Dt   : constant Duration :=
                    Ada.Calendar."-" (Ada.Calendar.Clock, T0);
                  pragma Unreferenced (R);   -- already streamed via Sink
                  Secs : constant Float := Float (Dt);
                  TPS10 : constant Integer :=
                    (if Secs > 0.0
                     then Integer (Float (Sink.Count) / Secs * 10.0) else 0);
               begin
                  if not Sink.Started then
                     New_Line;              -- model produced no tokens
                  end if;
                  New_Line;
                  Put_Line ("  ["  & Img (Sink.Count) & " токенів • "
                    & Img (Integer (Secs)) & " с • "
                    & Img (TPS10 / 10) & "." & Img (TPS10 mod 10) & " ток/с]");
               end;
            end if;
         end;

      <<Continue>>
         null;
      end loop;
   end Run;

end LLM_Chat;
