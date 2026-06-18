--  LLM_Gemma.Backend — adapts the gemma4 backend to the unified
--  Model_Backend protocol (thin forwarding wrapper).
with LLM_Backend;
with LLM_Qwen;
with LLM_Sampler;

package LLM_Gemma.Backend is

   type Gemma_Backend is limited new LLM_Backend.Model_Backend with private;

   function Create (Path : String) return LLM_Backend.Backend_Access;

   overriding function Chat
     (M              : Gemma_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Token_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access LLM_Qwen.Gen_Stats := null) return String;

   overriding function Vocab_Size  (M : Gemma_Backend) return Integer;
   overriding function Arch_Name   (M : Gemma_Backend) return String;
   overriding function Dim         (M : Gemma_Backend) return Integer;
   overriding function Block_Count (M : Gemma_Backend) return Integer;

private
   type Gemma_Backend is limited new LLM_Backend.Model_Backend with record
      Model : LLM_Gemma.Gemma_Model;
   end record;
end LLM_Gemma.Backend;
