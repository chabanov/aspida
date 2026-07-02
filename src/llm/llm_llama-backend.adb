with LLM_Sampler;
with Ada.Strings.Unbounded;

package body LLM_Llama.Backend is

   --  Adapter sink: forwards every text piece to a Chat_Sink's On_Text
   --  callback so a llama chat streams text into the same surface the
   --  qwen backend uses. The On_Reasoning / On_Tool_Call callbacks are
   --  unreachable for llama (which never emits chat-template XML), so
   --  they are simply not called.
   type Text_Adapter is new LLM_Qwen.Chat_Sink with record
      Down : access LLM_Qwen.Chat_Sink'Class;
   end record;
   overriding procedure On_Text (S : in out Text_Adapter; Piece : String) is
   begin
      LLM_Qwen.On_Text (S.Down.all, Piece);
   end On_Text;

   function Create (Path : String) return LLM_Backend.Backend_Access is
   begin
      return new Llama_Backend'(Model => LLM_Llama.Load (Path));
   end Create;

   procedure Create_From_File
     (G      : in out LLM_GGUF.GGUF_File;
      Result : out LLM_Backend.Backend_Access) is
      M : LLM_Llama.Llama_Model;
   begin
      --  Load_From_File reads the tensors and closes G (freeing the byte
      --  source). On failure it raises Model_Load_Error with G already closed.
      LLM_Llama.Load_From_File (G, M);
      Result := new Llama_Backend'(Model => M);
   end Create_From_File;

   procedure Create_From_File_Partial
     (G      : LLM_GGUF.GGUF_Ptr;
      K      : Positive;
      Result : out LLM_Backend.Backend_Access) is
      M : LLM_Llama.Llama_Model;
   begin
      --  Takes ownership of G: Load_From_File_Partial keeps it alive (M.GGUF)
      --  for the background fetcher, which closes + frees it when done. On
      --  failure G is freed and Model_Load_Error propagates.
      LLM_Llama.Load_From_File_Partial (G, M, K);
      Result := new Llama_Backend'(Model => M);
   end Create_From_File_Partial;

   overriding function Chat
     (M              : Llama_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Chat_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access LLM_Qwen.Gen_Stats := null)
      return LLM_Qwen.Chat_Result
   is
      --  'Unchecked_Access is safe: LLM_Llama.Chat uses the sink only
      --  synchronously while generating and never retains the pointer, so
      --  the local Adapter outlives every use. 'Access fails Ada's runtime
      --  accessibility check (same as LLM_Gemma.Backend.Chat).
      Adapter : aliased Text_Adapter := (Down => Sink);
      Text    : constant String :=
        (if Sink /= null then
           LLM_Llama.Chat (M.Model, Conversation, Max_New_Tokens,
                           Adapter'Unchecked_Access, Params, Stats)
         else
           LLM_Llama.Chat (M.Model, Conversation, Max_New_Tokens,
                           null, Params, Stats));
   begin
      return R : LLM_Qwen.Chat_Result (0) do
         R.Answer := Ada.Strings.Unbounded.To_Unbounded_String (Text);
         R.Finish :=
           Ada.Strings.Unbounded.To_Unbounded_String
             ((if Stats /= null and then Stats.Truncated then "length"
               else "stop"));
      end return;
   end Chat;

   overriding function Vocab_Size  (M : Llama_Backend) return Integer
     is (LLM_Llama.Vocab_Size (M.Model));
   overriding function Arch_Name   (M : Llama_Backend) return String
     is ("llama");
   overriding function Dim         (M : Llama_Backend) return Integer
     is (LLM_Llama.Dim (M.Model));
   overriding function Block_Count (M : Llama_Backend) return Integer
     is (LLM_Llama.Block_Count (M.Model));

   overriding procedure Release (M : in out Llama_Backend) is
   begin
      LLM_Llama.Free (M.Model);   --  idempotent; nulls M.Model
   end Release;

end LLM_Llama.Backend;
