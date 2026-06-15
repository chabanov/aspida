---------------------------------------------------------------------
-- LLM_Engine body — detect architecture, dispatch to the backend.
---------------------------------------------------------------------

with LLM_GGUF;

package body LLM_Engine is

   --  Peek general.architecture: open, read metadata, close. (One-time on
   --  load; the chosen backend reopens the file to stream tensor data.)
   function Detect (Path : String) return String is
      G : LLM_GGUF.GGUF_File;
   begin
      LLM_GGUF.Open (G, Path);
      if not LLM_GGUF.Is_Open (G) then
         return "";
      end if;
      return A : constant String :=
        LLM_GGUF.Metadata (G, "general.architecture")
      do
         LLM_GGUF.Close (G);
      end return;
   end Detect;

   function Load (Path : String) return Engine is
      Arch : constant String := Detect (Path);
   begin
      if Arch = "gemma4" then
         return (Kind => B_Gemma, Gm => LLM_Gemma.Load (Path), others => <>);
      elsif Arch = "qwen35moe" or else Arch = "qwen2" then
         return (Kind => B_Qwen, Q => LLM_Qwen.Load (Path), others => <>);
      elsif Arch = "" then
         raise Model_Load_Error with "cannot open GGUF file: " & Path;
      else
         raise Model_Load_Error with
           "unsupported architecture '" & Arch
           & "' (supported: qwen35moe, gemma4)";
      end if;
   end Load;

   function Chat
     (E : Engine; Conversation : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null) return String is
   begin
      case E.Kind is
         when B_Qwen  =>
            return LLM_Qwen.Chat (E.Q, Conversation, Max_New_Tokens, Sink);
         when B_Gemma =>
            return LLM_Gemma.Chat (E.Gm, Conversation, Max_New_Tokens, Sink);
      end case;
   end Chat;

   function Vocab_Size (E : Engine) return Integer is
     (case E.Kind is
         when B_Qwen  => LLM_Qwen.Vocab_Size (E.Q),
         when B_Gemma => LLM_Gemma.Vocab_Size (E.Gm));

   function Arch_Name (E : Engine) return String is
     (case E.Kind is when B_Qwen => "qwen35moe", when B_Gemma => "gemma4");

end LLM_Engine;
