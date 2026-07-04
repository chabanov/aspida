---------------------------------------------------------------------
-- LLM_Weight_Source — Byte_Source backed by encrypted byte-range fetches
-- over a Secure_Channel, plus the matching server-side responder.
--
-- H19 (docs/H19_WEIGHT_STREAM_ROADMAP.md Phases 1+2): the operator's host
-- holds only encrypted weight bytes and serves ranges of them; the client
-- streams the ranges it needs and runs the forward pass locally, so the
-- prompt, activations, and generated tokens never exist on the server. This
-- package implements both ends of the byte-range protocol defined in
-- LLM_Weight_Proto / Protocol:
--
--   * Remote_AEAD_Source is an LLM_Byte_Source.Byte_Source. The GGUF parser
--     reads through it exactly as through a local file — the abstraction
--     from Phase 0 is the whole point. It keeps an in-memory chunk cache so
--     repeated reads (the header is parsed sequentially, tensor data by
--     absolute offset) do not re-fetch, and (Phase 3) an opt-in on-disk
--     AEAD-sealed cache (LLM_Weight_Cache) so a later session reads chunks
--     locally with zero outbound fetches. Fetch order is memory -> disk ->
--     channel, with write-through: a channel-fetched chunk is sealed to disk
--     and held in memory. The channel + transport are caller-owned; this
--     source only borrows them.
--
--   * Serve_Weight_Requests is the thin server: loop Recv -> decode WReq ->
--     Read the range from a local model source -> Send WData. Per-request
--     errors (malformed request, out-of-range) are returned as Tag_WErr
--     without dropping the channel; a transport/handshake exception ends the
--     loop (peer closed / tamper).
--
-- Integrity of the artifact itself (signed manifest, pinned hash) is Phase
-- 4; here we only move bytes through the AEAD channel, which already defeats
-- a network MITM and tamper with the weight bytes.
---------------------------------------------------------------------

with Ada.Containers.Vectors;
with Ada.Containers.Hashed_Maps;
with Ada.Strings.Unbounded;
with Interfaces;
with System;
with Secure_Channel;
with LLM_Byte_Source;
with LLM_Weight_Cache;

package LLM_Weight_Source is

   --  Chunk size for the client cache and the unit of a single range request.
   --  64 KiB keeps every response well under the 1 MiB channel frame cap.
   Chunk_Size : constant Interfaces.Unsigned_64 := 65_536;

   --  Raised when the server reports an error for a requested range, or when
   --  a response is structurally wrong (wrong tag / short data). Maps to
   --  Malformed_Source semantics for the GGUF parser.
   Weight_Fetch_Error : exception;

   ------------------------------------------------------------------
   -- Client side: a streaming Byte_Source over an established channel
   ------------------------------------------------------------------

   type Remote_AEAD_Source is new LLM_Byte_Source.Byte_Source with private;

   --  Open a streaming source over an already-handshaken channel for a model
   --  of advertised byte length Len (the bound the GGUF parser validates
   --  tensor offsets against). Borrows Ch / Trans (caller owns them); the
   --  source does not close them — Close only releases the chunk cache and
   --  the disk cache. Model_ID is sent with every range request so a multi-
   --  model server can route; today the server serves one model and the id
   --  is a sanity check. Model_ID is also the on-disk cache key (Phase 4 will
   --  replace it with the attested model hash).
   --
   --  Persistence is opt-in via the ASPIDA_WEIGHT_CACHE_DIR and
   --  ASPIDA_WEIGHT_CACHE_PASS environment variables; if either is unset the
   --  on-disk cache is disabled and the source is in-memory only (Phase 1+2
   --  behavior, unchanged).
   function Open_Remote
     (Ch       : access Secure_Channel.Channel;
      Trans    : access Secure_Channel.Byte_Transport'Class;
      Model_ID : String;
      Len      : Interfaces.Unsigned_64) return LLM_Byte_Source.Byte_Source_Access;

   overriding procedure Read_Seq
     (S     : in out Remote_AEAD_Source;
      Addr  : System.Address;
      Count : Natural);

   overriding procedure Read_At_Pos
     (S     : in out Remote_AEAD_Source;
      Off   : Interfaces.Unsigned_64;
      Addr  : System.Address;
      Count : Natural);

   overriding function Byte_Length
     (S : Remote_AEAD_Source) return Interfaces.Unsigned_64;

   overriding function Cursor
     (S : Remote_AEAD_Source) return Interfaces.Unsigned_64;

   overriding procedure Seek
     (S   : in out Remote_AEAD_Source;
      Off : Interfaces.Unsigned_64);

   overriding procedure Close (S : in out Remote_AEAD_Source);

   --  H19 Phase 5: prefetch every chunk in a fixed, prompt-independent order
   --  (chunk index 0, 1, ..., N-1) into the in-memory cache so the engine's
   --  later Read_Seq calls are all tier-1 hits (zero outbound fetches during
   --  inference). This is the oblivious warm-fetch: the cold access pattern is
   --  "every chunk ascending" for ANY prompt, so the operator learns only that
   --  model X was loaded, never its content. Does not move the cursor; the
   --  chunks are owned by the in-memory cache and freed on Close. Disk cache
   --  (if enabled) is write-through populated as a side effect.
   overriding procedure Prefetch_All (S : in out Remote_AEAD_Source);

   --  Fetch-log observability (concrete to the remote source — a local file
   --  has no outbound fetches, so the hook lives here, not on the interface).
   --  The log records the chunk index of every successful outbound (tier-3)
   --  fetch, in order. With Prefetch_All it is [0 .. N-1] ascending; the test
   --  asserts this to prove the cold pattern is prompt-independent. Access
   --  these via a tagged downcast from Byte_Source'Class (the source opened by
   --  Open_Remote is always a Remote_AEAD_Source).
   procedure Enable_Fetch_Log (S : in out Remote_AEAD_Source);
   function Fetch_Log_Length (S : Remote_AEAD_Source) return Natural;
   function Fetch_Log_At
     (S : Remote_AEAD_Source;
      I : Natural) return Interfaces.Unsigned_64;

   ------------------------------------------------------------------
   -- Server side: serve encrypted byte-ranges of a local model source
   ------------------------------------------------------------------

   --  Serve Tag_WReq records addressed to Model (a local Byte_Source — e.g. a
   --  Local_File_Source opened on the GGUF). Each request reads its range with
   --  a positional read (Model.Read_At_Pos, backed by pread) and replies
   --  Tag_WData; per-request failures reply Tag_WErr and the loop continues.
   --  The positional read carries the offset in the syscall and never touches
   --  a shared cursor, so serving concurrent clients from one Model is safe.
   --
   --  Max_Requests bounds the run: 0 = serve until a transport/handshake
   --  exception (the production mode — a socket close ends the loop); >0 =
   --  serve that many requests then return (the in-memory test mode, where
   --  the loopback transport has no "close"). A transport exception
   --  (Handshake_Error / Auth_Error) ends the loop quietly.
   procedure Serve_Weight_Requests
     (Ch           : in out Secure_Channel.Channel;
      Trans        : access Secure_Channel.Byte_Transport'Class;
      Model        : access LLM_Byte_Source.Byte_Source'Class;
      Model_ID     : String;
      Max_Requests : Natural := 0);

private

   --  Heap-access to a chunk buffer, shared with LLM_Weight_Cache (the disk
   --  cache returns the same access type, so a disk-loaded chunk drops straight
   --  into the in-memory vector with no copy).
   subtype Byte_Array_Access is LLM_Weight_Cache.Byte_Array_Access;
   --  Make the predefined "=" on the access type directly visible so the
   --  Hashed_Maps instantiation below (which needs "=" on Element_Type for
   --  its default Element equality) resolves it.
   use type LLM_Weight_Cache.Byte_Array_Access;

   --  In-memory chunk cache keyed by chunk index. A hashed map gives O(1)
   --  lookup on the hot path: during inference the GGUF parser drives one
   --  Fetch_Chunk per chunk read, and after Prefetch_All the whole model is
   --  resident (tens of thousands of chunks for a multi-GB model). A linear
   --  vector scan per Fetch_Chunk would make the streamed load O(N^2); the map
   --  keeps it O(N). The map OWNS each Byte_Array_Access value and frees it on
   --  Close. Memory footprint is unchanged from the vector (one buffer per
   --  resident chunk) — this fixes only the lookup cost, not residency, since
   --  the oblivious Prefetch_All intentionally keeps every chunk in RAM.
   function Hash_U64
     (K : Interfaces.Unsigned_64) return Ada.Containers.Hash_Type;

   package Chunk_Maps is new Ada.Containers.Hashed_Maps
     (Key_Type        => Interfaces.Unsigned_64,
      Element_Type    => Byte_Array_Access,
      Hash            => Hash_U64,
      Equivalent_Keys => Interfaces."=");

   --  Outbound-fetch log: the chunk index of every successful tier-3 (channel)
   --  fetch, in order. Off by default (Log_On = False); Enable_Fetch_Log flips
   --  it on. Used to assert the oblivious cold pattern (ascending 0..N-1).
   --  `use type` makes predefined "=" on Unsigned_64 directly visible at the
   --  generic instantiation (Ada.Containers.Vectors defaults "=" to it).
   use type Interfaces.Unsigned_64;
   package Chunk_Log_Vectors is new Ada.Containers.Vectors
     (Positive, Interfaces.Unsigned_64);

   type Remote_AEAD_Source is new LLM_Byte_Source.Byte_Source with record
      --  Anonymous access components imply "access to variable" (the `all` is
      --  implicit; Ada forbids writing it explicitly on anonymous access
      --  types). The channel + transport are caller-owned; this source only
      --  borrows them for the lifetime of the caller's session.
      Chan     : access Secure_Channel.Channel := null;
      Trans    : access Secure_Channel.Byte_Transport'Class := null;
      Model_ID : Ada.Strings.Unbounded.Unbounded_String;
      Len      : Interfaces.Unsigned_64 := 0;
      Pos      : Interfaces.Unsigned_64 := 0;
      Cache    : Chunk_Maps.Map;
      --  Opt-in on-disk AEAD-sealed chunk cache. Disabled by default (Open
      --  leaves it disabled unless ASPIDA_WEIGHT_CACHE_DIR/PASS are set).
      Disk     : LLM_Weight_Cache.Weight_Cache;
      --  Fetch-log instrumentation (H19 Phase 5). Off by default so the hot
      --  path pays only one branch per channel fetch.
      Log_On : Boolean := False;
      Log    : Chunk_Log_Vectors.Vector;
   end record;

end LLM_Weight_Source;