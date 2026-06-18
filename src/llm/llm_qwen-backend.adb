package body LLM_Qwen.Backend is

   function Create (Path : String) return LLM_Backend.Backend_Access is
   begin
      return new Qwen_Backend'(Model => LLM_Qwen.Load (Path));
   end Create;

   overriding function Chat
     (M              : Qwen_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Token_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access LLM_Qwen.Gen_Stats := null) return String
   is (LLM_Qwen.Chat (M.Model, Conversation, Max_New_Tokens, Sink, Params, Stats));

   overriding function Vocab_Size  (M : Qwen_Backend) return Integer
     is (LLM_Qwen.Vocab_Size (M.Model));
   overriding function Arch_Name   (M : Qwen_Backend) return String
     is ("qwen35moe");
   overriding function Dim         (M : Qwen_Backend) return Integer
     is (LLM_Qwen.Dim (M.Model));
   overriding function Block_Count (M : Qwen_Backend) return Integer
     is (LLM_Qwen.Block_Count (M.Model));

end LLM_Qwen.Backend;
