---------------------------------------------------------------------
-- LLM_Engine body — detect architecture, construct the matching backend
-- (registry-driven), then dispatch over the unified Model_Backend protocol.
---------------------------------------------------------------------

with Ada.Unchecked_Deallocation;
with LLM_GGUF;
with LLM_Qwen.Backend;
with LLM_Gemma.Backend;
with LLM_Llama.Backend;

package body LLM_Engine is

   procedure Free_Backend is new Ada.Unchecked_Deallocation
     (LLM_Backend.Model_Backend'Class, LLM_Backend.Backend_Access);

   --  Architecture registry: a GGUF general.architecture string -> the backend
   --  constructor. Adding a model = add one row here; nothing else changes.
   type Constructor is access
     function (Path : String) return LLM_Backend.Backend_Access;

   type Registration is record
      Arch : access constant String;
      Make : Constructor;
   end record;

   A_Gemma  : aliased constant String := "gemma4";
   A_QMoe   : aliased constant String := "qwen35moe";
   A_Qwen35 : aliased constant String := "qwen35";
   A_Qwen2  : aliased constant String := "qwen2";
   A_Llama  : aliased constant String := "llama";

   Registry : constant array (Positive range <>) of Registration :=
     [1 => (A_Gemma'Access,  LLM_Gemma.Backend.Create'Access),
      2 => (A_QMoe'Access,   LLM_Qwen.Backend.Create'Access),
      3 => (A_Qwen35'Access, LLM_Qwen.Backend.Create'Access),
      4 => (A_Qwen2'Access,  LLM_Qwen.Backend.Create'Access),
      5 => (A_Llama'Access,  LLM_Llama.Backend.Create'Access)];

   --  Peek general.architecture: open, read metadata, close. (One-time on
   --  load; the chosen backend reopens the file to stream tensor data.)
   function Detect_Arch (Path : String) return String is
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
   exception
      when others =>
         return "";
   end Detect_Arch;

   function Supports (Arch : String) return Boolean is
   begin
      for R of Registry loop
         if R.Arch.all = Arch then
            return True;
         end if;
      end loop;
      return False;
   end Supports;

   function Load (Path : String) return Engine is
      Arch : constant String := Detect_Arch (Path);
   begin
      if Arch = "" then
         raise Model_Load_Error with "cannot open GGUF file: " & Path;
      end if;
      for R of Registry loop
         if R.Arch.all = Arch then
            return (Impl => R.Make (Path));
         end if;
      end loop;
      raise Model_Load_Error with
        "unsupported architecture '" & Arch
        & "' (supported: qwen35moe, qwen35, qwen2, gemma4, llama)";
   end Load;

   procedure Unload (E : in out Engine) is
      use type LLM_Backend.Backend_Access;
   begin
      if E.Impl /= null then
         E.Impl.Release;          --  free weights / GPU mirror / file handles
         Free_Backend (E.Impl);   --  deallocate the class-wide backend; nulls E.Impl
      end if;
   end Unload;

   function Chat
     (E : Engine; Conversation : LLM_Qwen.Message_Array;
      Max_New_Tokens : Integer := 256;
      Sink : access LLM_Qwen.Chat_Sink'Class := null;
      Params : LLM_Sampler.Params := LLM_Sampler.Greedy;
      Stats : access Gen_Stats := null) return LLM_Qwen.Chat_Result
   is (E.Impl.Chat (Conversation, Max_New_Tokens, Sink, Params, Stats));

   function Vocab_Size (E : Engine) return Integer is (E.Impl.Vocab_Size);
   function Arch_Name  (E : Engine) return String  is (E.Impl.Arch_Name);

end LLM_Engine;
