--  LLM_Llama.Backend — adapts the dense Llama backend to the unified
--  Model_Backend protocol (thin forwarding wrapper).
with LLM_Backend;
with LLM_Qwen;
with LLM_GGUF;  --  H19: Create_From_File takes an already-open GGUF_File

package LLM_Llama.Backend is

   type Llama_Backend is limited new LLM_Backend.Model_Backend with private;

   --  Load the model and return it as a class-wide backend handle.
   function Create (Path : String) return LLM_Backend.Backend_Access;

   --  H19 (weight-streaming): build the backend from an ALREADY-OPEN GGUF_File
   --  whose byte source may be a Remote_AEAD_Source. LLM_Llama.Load_From_File
   --  reads the tensors and closes G (freeing the source); the backend then
   --  owns the model. Procedure (out result) because Ada forbids in-out
   --  function parameters.
   procedure Create_From_File
     (G      : in out LLM_GGUF.GGUF_File;
      Result : out LLM_Backend.Backend_Access);

   --  H19 Phase 7 partial-warm: build the backend from a heap-allocated
   --  GGUF_File, loading only the head + first K blocks eagerly and streaming
   --  the rest in the background. Takes ownership of G (the model keeps it
   --  alive for the fetcher; the caller must NOT close or free G). On failure
   --  G is freed and Model_Load_Error is raised.
   procedure Create_From_File_Partial
     (G      : LLM_GGUF.GGUF_Ptr;
      K      : Positive;
      Result : out LLM_Backend.Backend_Access);

   overriding function Chat
     (M              : Llama_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Chat_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access LLM_Qwen.Gen_Stats := null)
      return LLM_Qwen.Chat_Result;

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
