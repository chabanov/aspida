--  LLM_Llama.Backend — adapts the dense Llama backend to the unified
--  Model_Backend protocol (thin forwarding wrapper).
with LLM_Backend;
with LLM_Qwen;
with LLM_Sampler;

package LLM_Llama.Backend is

   type Llama_Backend is limited new LLM_Backend.Model_Backend with private;

   --  Load the model and return it as a class-wide backend handle.
   function Create (Path : String) return LLM_Backend.Backend_Access;

   overriding function Chat
     (M              : Llama_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Token_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access LLM_Qwen.Gen_Stats := null) return String;

   overriding function Vocab_Size  (M : Llama_Backend) return Integer;
   overriding function Arch_Name   (M : Llama_Backend) return String;
   overriding function Dim         (M : Llama_Backend) return Integer;
   overriding function Block_Count (M : Llama_Backend) return Integer;
   overriding procedure Release    (M : in out Llama_Backend);

private
   type Llama_Backend is limited new LLM_Backend.Model_Backend with record
      Model : LLM_Llama.Llama_Model;
   end record;
end LLM_Llama.Backend;
