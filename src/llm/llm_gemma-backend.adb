with LLM_Sampler;
with Ada.Strings.Unbounded;

package body LLM_Gemma.Backend is

   --  Adapter sink: forwards text to the user-supplied Chat_Sink (see
   --  LLM_Llama.Backend for the full rationale; same shape).
   type Text_Adapter is new LLM_Qwen.Chat_Sink with record
      Down : access LLM_Qwen.Chat_Sink'Class;
   end record;
   overriding procedure On_Text (S : in out Text_Adapter; Piece : String) is
   begin
      LLM_Qwen.On_Text (S.Down.all, Piece);
   end On_Text;

   function Create (Path : String) return LLM_Backend.Backend_Access is
   begin
      return new Gemma_Backend'(Model => LLM_Gemma.Load (Path));
   end Create;

   overriding function Chat
     (M              : Gemma_Backend;
      Conversation   : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink           : access LLM_Qwen.Chat_Sink'Class := null;
      Params         : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats          : access LLM_Qwen.Gen_Stats := null)
      return LLM_Qwen.Chat_Result
   is
      --  'Unchecked_Access is safe here: LLM_Gemma.Chat uses the sink only
      --  synchronously while generating (it calls Emit on each piece and
      --  returns the assembled string) and never retains the pointer, so the
      --  local Adapter outlives every use. 'Access would fail Ada's runtime
      --  accessibility check (the local's level is shallower than the
      --  anonymous access-to-classwide formal expects) — observed as
      --  "accessibility check failed" at this line when a Gemma model is
      --  loaded through the C ABI.
      Adapter : aliased Text_Adapter := (Down => Sink);
      Text    : constant String :=
        (if Sink /= null then
           LLM_Gemma.Chat (M.Model, Conversation, Max_New_Tokens,
                           Adapter'Unchecked_Access, Params, Stats)
         else
           LLM_Gemma.Chat (M.Model, Conversation, Max_New_Tokens,
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

   overriding function Vocab_Size  (M : Gemma_Backend) return Integer
     is (LLM_Gemma.Vocab_Size (M.Model));
   overriding function Arch_Name   (M : Gemma_Backend) return String
     is ("gemma4");
   overriding function Dim         (M : Gemma_Backend) return Integer
     is (LLM_Gemma.Dim (M.Model));
   overriding function Block_Count (M : Gemma_Backend) return Integer
     is (LLM_Gemma.Block_Count (M.Model));

   overriding procedure Release (M : in out Gemma_Backend) is
   begin
      LLM_Gemma.Free (M.Model);   --  idempotent; nulls M.Model
   end Release;

end LLM_Gemma.Backend;
