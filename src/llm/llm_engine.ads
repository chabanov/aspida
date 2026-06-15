---------------------------------------------------------------------
-- LLM_Engine — architecture dispatcher.
--
-- Detects the GGUF's general.architecture and routes to the right backend:
--   qwen35moe / qwen2 -> LLM_Qwen   (bit-exact, validated)
--   gemma4            -> LLM_Gemma  (foundation; output not yet validated)
-- Presents one Chat/Vocab interface so the server is backend-agnostic.
---------------------------------------------------------------------

with LLM_Qwen;
with LLM_Gemma;

package LLM_Engine is

   type Engine is private;

   Model_Load_Error : exception;

   function Load (Path : String) return Engine;

   function Chat
     (E : Engine; Conversation : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Token_Sink'Class := null) return String;

   function Vocab_Size (E : Engine) return Integer;
   function Arch_Name  (E : Engine) return String;

private

   type Backend is (B_Qwen, B_Gemma);

   type Engine is record
      Kind : Backend := B_Qwen;
      Q    : LLM_Qwen.Qwen_Model;     -- valid when Kind = B_Qwen
      Gm   : LLM_Gemma.Gemma_Model;   -- valid when Kind = B_Gemma
   end record;

end LLM_Engine;
