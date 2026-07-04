---------------------------------------------------------------------
-- LLM_Engine body — detect architecture, construct the matching backend
-- (registry-driven), then dispatch over the unified Model_Backend protocol.
---------------------------------------------------------------------

with Ada.Unchecked_Deallocation;
with LLM_GGUF; use type LLM_GGUF.GGUF_Ptr;
with LLM_Qwen.Backend;
with LLM_Gemma.Backend;
with LLM_Llama.Backend;

package body LLM_Engine is

   procedure Free_Backend is new Ada.Unchecked_Deallocation
     (LLM_Backend.Model_Backend'Class, LLM_Backend.Backend_Access);

   --  Architecture registry: a GGUF general.architecture string -> the backend
   --  constructor. Adding a model = add one row here; nothing else changes.
   --  Two constructors per row: Make (path-based, the original Load path) and
   --  Make_From_File (H19 source-based, taking an already-open GGUF_File).
   --  Make_From_File is null for architectures that have not yet gained a
   --  Create_From_File; Load_From_Source reports those clearly rather than
   --  silently dispatching to a null.
   type Constructor is access
     function (Path : String) return LLM_Backend.Backend_Access;
   type File_Constructor is access
     procedure (G      : in out LLM_GGUF.GGUF_File;
                Result : out LLM_Backend.Backend_Access);
   --  H19 Phase 7 partial-warm constructor: takes a HEAP-allocated GGUF_File
   --  (the model keeps it alive for the background fetcher) + a warm-layer
   --  count K. Ownership of G transfers to the backend on success.
   type File_Constructor_Partial is access
     procedure (G      : LLM_GGUF.GGUF_Ptr;
                K      : Positive;
                Result : out LLM_Backend.Backend_Access);

   type Registration is record
      Arch                  : access constant String;
      Make                  : Constructor;
      Make_From_File        : File_Constructor;
      Make_From_File_Partial : File_Constructor_Partial;
   end record;

   A_Gemma  : aliased constant String := "gemma4";
   A_QMoe   : aliased constant String := "qwen35moe";
   A_Qwen35 : aliased constant String := "qwen35";
   A_Qwen2  : aliased constant String := "qwen2";
   A_Llama  : aliased constant String := "llama";

   Registry : constant array (Positive range <>) of Registration :=
     [1 => (A_Gemma'Access,  LLM_Gemma.Backend.Create'Access,  null, null),
      2 => (A_QMoe'Access,   LLM_Qwen.Backend.Create'Access,   null, null),
      3 => (A_Qwen35'Access, LLM_Qwen.Backend.Create'Access,   null, null),
      4 => (A_Qwen2'Access,  LLM_Qwen.Backend.Create'Access,   null, null),
      5 => (A_Llama'Access,  LLM_Llama.Backend.Create'Access,
            LLM_Llama.Backend.Create_From_File'Access,
            LLM_Llama.Backend.Create_From_File_Partial'Access)];

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

   function Load_From_Source
     (Src : LLM_Byte_Source.Byte_Source_Access) return Engine
   is
      G    : LLM_GGUF.GGUF_File;
      Impl : LLM_Backend.Backend_Access;
   begin
      --  Parse the GGUF header on the source (takes ownership of Src).
      LLM_GGUF.Open_From_Source (G, Src);
      if not LLM_GGUF.Is_Open (G) then
         --  Open_From_Source freed Src on the bad-magic / absurd-count path.
         raise Model_Load_Error with "cannot parse GGUF from source";
      end if;

      --  Detect the architecture from the already-parsed metadata (no second
      --  open — the source is consumed once) and dispatch to the backend's
      --  Create_From_File, which reads the tensors and closes G (freeing Src).
      declare
         Arch : constant String := LLM_GGUF.Metadata (G, "general.architecture");
      begin
         for R of Registry loop
            if R.Arch.all = Arch then
               if R.Make_From_File = null then
                  LLM_GGUF.Close (G);
                  raise Model_Load_Error with
                    "Load_From_Source not yet implemented for architecture '"
                    & Arch & "' (needs a Create_From_File in its backend)";
               end if;
               R.Make_From_File (G, Impl);   --  consumes + closes G (frees Src)
               return (Impl => Impl);
            end if;
         end loop;
         LLM_GGUF.Close (G);
         raise Model_Load_Error with
           "unsupported architecture '" & Arch
           & "' (supported: qwen35moe, qwen35, qwen2, gemma4, llama)";
      end;
   exception
      --  If a backend Create_From_File fails mid-load, ensure the (still-open)
      --  GGUF and its byte source are freed before propagating. The success
      --  path already closed G; a backend that raised before closing would
      --  otherwise orphan the source.
      when others =>
         if LLM_GGUF.Is_Open (G) then
            LLM_GGUF.Close (G);
         end if;
         raise;
   end Load_From_Source;

   function Load_From_Source_Partial
     (Src : LLM_Byte_Source.Byte_Source_Access; K : Positive) return Engine
   is
      --  Heap-allocate the GGUF_File so the model can keep it alive (M.GGUF)
      --  for the background fetcher. The eager Load_From_Source uses a stack
      --  G (it closes it before returning); the partial path transfers
      --  ownership to the backend, which frees it when the fetcher finishes.
      G    : LLM_GGUF.GGUF_Ptr := new LLM_GGUF.GGUF_File;
      Impl : LLM_Backend.Backend_Access;
      procedure Free_G is new Ada.Unchecked_Deallocation
        (LLM_GGUF.GGUF_File, LLM_GGUF.GGUF_Ptr);
      procedure Cleanup is
      begin
         if G /= null then
            if LLM_GGUF.Is_Open (G.all) then
               LLM_GGUF.Close (G.all);
            end if;
            Free_G (G);
         end if;
      end Cleanup;
   begin
      --  Parse the GGUF header on the source (takes ownership of Src).
      LLM_GGUF.Open_From_Source (G.all, Src);
      if not LLM_GGUF.Is_Open (G.all) then
         --  Open_From_Source freed Src on the bad-magic path; free the record.
         Free_G (G);
         raise Model_Load_Error with "cannot parse GGUF from source";
      end if;

      declare
         Arch : constant String := LLM_GGUF.Metadata (G.all, "general.architecture");
      begin
         for R of Registry loop
            if R.Arch.all = Arch then
               if R.Make_From_File_Partial = null then
                  Cleanup;
                  raise Model_Load_Error with
                    "Load_From_Source_Partial not yet implemented for architecture '"
                    & Arch & "' (needs a Create_From_File_Partial in its backend)";
               end if;
               --  Ownership of G transfers to the backend the instant we
               --  dispatch: on success the model keeps it alive (M.GGUF) for
               --  the fetcher; on a backend failure Load_From_File_Partial's
               --  handler frees M.GGUF (== this same pointer) and re-raises.
               --  Either way the engine must NOT touch G afterwards, so null
               --  our handle FIRST and dispatch on a local copy. If we nulled
               --  only after the call (the old code), a backend exception
               --  would skip the assignment and leave G dangling for the
               --  `when others` Cleanup to double-free / use-after-free.
               declare
                  Owned : constant LLM_GGUF.GGUF_Ptr := G;
               begin
                  G := null;   --  ownership handed off; Cleanup is now a no-op
                  R.Make_From_File_Partial (Owned, K, Impl);
               end;
               return (Impl => Impl);
            end if;
         end loop;
         Cleanup;
         raise Model_Load_Error with
           "unsupported architecture '" & Arch
           & "' (supported: qwen35moe, qwen35, qwen2, gemma4, llama)";
      end;
   exception
      when others =>
         --  A failure before the dispatch (or an unsupported arch) leaves us
         --  owning G — free it. Once we dispatch, G is already null (we hand
         --  ownership off before the call), so a backend failure that frees
         --  M.GGUF and re-raises lands here with Cleanup as a safe no-op.
         Cleanup;
         raise;
   end Load_From_Source_Partial;

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
