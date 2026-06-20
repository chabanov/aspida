--  Llama_Probe — completion for validating LLM_Llama against llama.cpp.
--  Modes: default = raw Complete; "chat" = single user turn via Chat;
--  "chat2" = a 2-turn conversation (to reproduce multi-turn degradation).
with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with LLM_Llama;
with LLM_Qwen;
with LLM_Tokenizer;
with LLM_GGUF;
with GNAT.OS_Lib;

procedure Llama_Probe is
   use type LLM_Qwen.Message_Array;
   function U (S : String) return Ada.Strings.Unbounded.Unbounded_String
     renames Ada.Strings.Unbounded.To_Unbounded_String;
   N : constant Integer :=
     (if Ada.Command_Line.Argument_Count >= 3
      then Integer'Value (Ada.Command_Line.Argument (3)) else 4);
   Prompt : constant String :=
     (if Ada.Command_Line.Argument_Count >= 2
      then Ada.Command_Line.Argument (2) else "The capital of France is");
   Mode : constant String :=
     (if Ada.Command_Line.Argument_Count >= 4 then Ada.Command_Line.Argument (4) else "raw");
begin
   Ada.Text_IO.Put_Line ("=== prompt: '" & Prompt & "' (mode=" & Mode & ") ===");
   if Mode = "tok" then
      --  Dump tokenization only (no model forward): compare to llama.cpp.
      declare
         G : LLM_GGUF.GGUF_File;
         Tok : LLM_Tokenizer.Tokenizer := LLM_Tokenizer.Create;
      begin
         LLM_GGUF.Open (G, Ada.Command_Line.Argument (1));
         LLM_Tokenizer.Load_From_GGUF (Tok, G);
         declare
            Ids : constant LLM_Tokenizer.Token_Array := LLM_Tokenizer.Encode (Tok, Prompt);
         begin
            Ada.Text_IO.Put ("tokens(" & Integer'Image (Ids'Length) & " ):");
            for I in Ids'Range loop
               Ada.Text_IO.Put (Integer'Image (Ids (I)) & "[" &
                 LLM_Tokenizer.Decode_One (Tok, Ids (I)) & "]");
            end loop;
            Ada.Text_IO.New_Line;
         end;
         LLM_GGUF.Close (G);
      end;
      return;
   end if;
   declare
      M : constant LLM_Llama.Llama_Model :=
        LLM_Llama.Load (Ada.Command_Line.Argument (1));
   begin
   if Mode = "batch" then
      Ada.Text_IO.Put_Line ("batch self-test: max |logit| diff (batched vs single) = "
        & Float'Image (LLM_Llama.Batch_Self_Test (M)));
   elsif Mode = "batchgen" then
      Ada.Text_IO.Put_Line ("batch-gen self-test (Max_New=" & N'Image & "):");
      Ada.Text_IO.Put_Line (LLM_Llama.Batch_Gen_Self_Test (M, N));
   elsif Mode = "sched" then
      --  Fire several concurrent Chat calls: they must all flow through the
      --  continuous-batch scheduler and complete (proves server concurrency).
      declare
         Ps : constant array (1 .. 4) of Ada.Strings.Unbounded.Unbounded_String :=
           [U ("Яка столиця Франції? Одне речення."),
            U ("Скільки буде 2+2? Коротко."),
            U ("Назви колір неба одним словом."),
            U ("Хто написав ""Кобзар""? Одне речення.")];
         task type W is entry Go (K : Integer); end W;
         task body W is
            KK : Integer;
         begin
            accept Go (K : Integer) do KK := K; end Go;
            declare
               R : constant String := LLM_Llama.Chat
                 (M, [1 => (LLM_Qwen.Role_User, Ps (KK))], N);
            begin
               Ada.Text_IO.Put_Line ("[seq" & KK'Image & "] " & R);
            end;
         end W;
         Ws : array (1 .. 4) of W;
      begin
         Ada.Text_IO.Put_Line ("launching 4 concurrent Chat requests...");
         for I in Ws'Range loop Ws (I).Go (I); end loop;
      end;
      Ada.Text_IO.Put_Line ("all 4 concurrent requests done");
      GNAT.OS_Lib.OS_Exit (0);   --  scheduler task runs forever; force clean exit
   elsif Mode = "chat" then
      Ada.Text_IO.Put_Line ("completion: '" & LLM_Llama.Chat
        (M, [1 => (LLM_Qwen.Role_User, U (Prompt))], N) & "'");
   elsif Mode = "chat2" then
      Ada.Text_IO.Put_Line ("completion: '" & LLM_Llama.Chat
        (M, (LLM_Qwen.Message'(LLM_Qwen.Role_User, U ("Привіт! Як справи?"))
           & LLM_Qwen.Message'(LLM_Qwen.Role_Assistant, U ("Все добре, дякую! Чим можу допомогти?"))
           & LLM_Qwen.Message'(LLM_Qwen.Role_User, U (Prompt))), N) & "'");
   else
      Ada.Text_IO.Put_Line ("completion: '" & LLM_Llama.Complete (M, Prompt, N) & "'");
   end if;
   end;
end Llama_Probe;
