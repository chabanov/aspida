---------------------------------------------------------------------
-- LLM_Byte_Source — random-access byte source abstraction under the
-- GGUF loader.
--
-- The GGUF parser accesses tensor bytes on demand: the header/metadata
-- are read sequentially from a moving cursor, and each tensor's data is
-- read by absolute offset. Today the only implementation is a local file
-- (POSIX fd + lseek + read), moved here verbatim from the old LLM_GGUF
-- body. This interface lets a future Remote_AEAD_Source (H19
-- weight-streaming) fetch byte ranges over the Secure_Channel with a
-- local write-through cache — without the parser or any backend changing.
--
-- The abstraction mirrors how the parser uses a file: a single cursor
-- advances on every read, so Seek + Read_Seq is equivalent to Read_At.
-- Byte_Length is known at Open (probed once for a file; advertised by an
-- attested manifest for a streaming source) and is the bound the parser
-- validates every tensor offset against — a hostile descriptor that
-- reaches past the end fails loud (Malformed_Source), never an OOB read.
--
-- This is Phase 0 of docs/H19_WEIGHT_STREAM_ROADMAP.md: a clean seam with
-- one implementation, so the engine can later gain a remote one.
---------------------------------------------------------------------

with Interfaces;
with System;

package LLM_Byte_Source is

   --  Raised on a short read (EOF reached before the requested Count bytes,
   --  or an I/O error) and on a Seek past the end. The GGUF parser maps this
   --  to Malformed_GGUF so a hostile or truncated source fails loud, never
   --  garbage.
   Malformed_Source : exception;

   ------------------------------------------------------------------
   -- Byte_Source — the interface
   ------------------------------------------------------------------

   --  A random-access byte source with a cursor. Limited so an
   --  implementation carrying a real resource (fd / connection) can never be
   --  accidentally copied; the engine holds it through a Byte_Source_Access.
   type Byte_Source is limited interface;
   type Byte_Source_Access is access all Byte_Source'Class;

   --  Read Count bytes at the current cursor into Addr; advance the cursor by
   --  Count. Raises Malformed_Source on a short read.
   procedure Read_Seq
     (S     : in out Byte_Source;
      Addr  : System.Address;
      Count : Natural) is abstract;

   --  Absolute byte length of the source. Stable for a file; for a streaming
   --  source it is the length advertised by the (attested) manifest and is the
   --  bound the GGUF parser validates tensor offsets against. Does not move
   --  the cursor.
   function Byte_Length
     (S : Byte_Source) return Interfaces.Unsigned_64 is abstract;

   --  Current cursor position (bytes from the start).
   function Cursor
     (S : Byte_Source) return Interfaces.Unsigned_64 is abstract;

   --  Set the cursor to Off (absolute). Raises Malformed_Source if Off exceeds
   --  Byte_Length.
   procedure Seek
     (S   : in out Byte_Source;
      Off : Interfaces.Unsigned_64) is abstract;

   --  Release the underlying resource (fd / connection). Idempotent: safe to
   --  call on an already-closed source. Does NOT deallocate the object — use
   --  Free_Source for that.
   procedure Close
     (S : in out Byte_Source) is abstract;

   --  Warm the source into local storage so later Read_Seq calls need no
   --  outbound fetch. H19 Phase 5 (oblivious warm-fetch): a Remote_AEAD_Source
   --  overrides this to pull every chunk in a FIXED, prompt-independent order
   --  (chunk index 0, 1, 2, ...), so the cold access pattern reveals only
   --  "loaded model X" (the total volume), never which tensor is touched first.
   --  After it returns, every byte is a tier-1 (in-memory) hit -> zero outbound
   --  fetches during inference. A Local_File_Source is already local, so the
   --  default is a no-op (inherited unchanged). Does not move the cursor.
   procedure Prefetch_All
     (S : in out Byte_Source) is null;

   ------------------------------------------------------------------
   -- Concrete class-wide helpers (non-dispatching, call the primitives)
   ------------------------------------------------------------------

   --  Read Count bytes at absolute offset Off, leaving the cursor at
   --  Off + Count. Equivalent to Seek (Off); Read_Seq (Addr, Count).
   procedure Read_At
     (S     : in out Byte_Source'Class;
      Off   : Interfaces.Unsigned_64;
      Addr  : System.Address;
      Count : Natural);

   --  Release the resource (dispatching Close) and deallocate the object. S
   --  becomes null. Idempotent (no-op on null). This is how the engine frees a
   --  source it owns: one call closes the fd/connection AND frees the access.
   procedure Free_Source (S : in out Byte_Source_Access);

   ------------------------------------------------------------------
   -- Local_File_Source — POSIX fd + lseek + read (today's path)
   ------------------------------------------------------------------

   type Local_File_Source is new Byte_Source with private;

   --  Open Path for reading and return a heap-allocated source (the engine
   --  holds it via a Byte_Source_Access). Returns null on failure (missing or
   --  unreadable file, or an un-probeable length) — this mirrors the existing
   --  LLM_GGUF.Open semantics of NOT raising on a bad path; an actual read
   --  error during parse raises Malformed_Source.
   function Open_Source (Path : String) return Byte_Source_Access;

   overriding procedure Read_Seq
     (S     : in out Local_File_Source;
      Addr  : System.Address;
      Count : Natural);

   overriding function Byte_Length
     (S : Local_File_Source) return Interfaces.Unsigned_64;

   overriding function Cursor
     (S : Local_File_Source) return Interfaces.Unsigned_64;

   overriding procedure Seek
     (S   : in out Local_File_Source;
      Off : Interfaces.Unsigned_64);

   overriding procedure Close (S : in out Local_File_Source);

private

   type Local_File_Source is new Byte_Source with record
      FD  : Integer := -1;            --  POSIX file descriptor (-1 = closed)
      Len : Interfaces.Unsigned_64 := 0;  --  file length, probed once at open
      Pos : Interfaces.Unsigned_64 := 0;  --  current cursor (bytes from start)
   end record;

end LLM_Byte_Source;