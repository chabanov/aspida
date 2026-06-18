package body LLM_Llama.Backend is

   function Create (Path : String) return LLM_Backend.Backend_Access is
   begin
      return new Llama_Backend'(Model => LLM_Llama.Load (Path));
   end Create;

   overriding function Chat
     (M              : Llama_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Token_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access LLM_Qwen.Gen_Stats := null) return String
   is (LLM_Llama.Chat (M.Model, Conversation, Max_New_Tokens, Sink, Params, Stats));

   overriding function Vocab_Size  (M : Llama_Backend) return Integer
     is (LLM_Llama.Vocab_Size (M.Model));
   overriding function Arch_Name   (M : Llama_Backend) return String
     is ("llama");
   overriding function Dim         (M : Llama_Backend) return Integer
     is (LLM_Llama.Dim (M.Model));
   overriding function Block_Count (M : Llama_Backend) return Integer
     is (LLM_Llama.Block_Count (M.Model));

end LLM_Llama.Backend;
