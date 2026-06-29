--  LLM_Qwen.Backend — adapts the Qwen 3.5 (MoE + delta-net) backend to the
--  unified Model_Backend protocol (thin forwarding wrapper).
with LLM_Backend;
with LLM_Sampler;

package LLM_Qwen.Backend is

   type Qwen_Backend is limited new LLM_Backend.Model_Backend with private;

   function Create (Path : String) return LLM_Backend.Backend_Access;

   overriding function Chat
     (M              : Qwen_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Chat_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access LLM_Qwen.Gen_Stats := null)
      return LLM_Qwen.Chat_Result;

   overriding function Vocab_Size  (M : Qwen_Backend) return Integer;
   overriding function Arch_Name   (M : Qwen_Backend) return String;
   overriding function Dim         (M : Qwen_Backend) return Integer;
   overriding function Block_Count (M : Qwen_Backend) return Integer;
   overriding procedure Release    (M : in out Qwen_Backend);

private
   type Qwen_Backend is limited new LLM_Backend.Model_Backend with record
      Model : LLM_Qwen.Qwen_Model;
   end record;
end LLM_Qwen.Backend;
