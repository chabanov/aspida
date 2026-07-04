---------------------------------------------------------------------
-- LLM_Weight_Source body — chunk cache + AEAD fetch (client) and the
-- byte-range responder (server).
--
-- The client side fetches whole chunks (Chunk_Size bytes) over the channel
-- and caches them by chunk index; Read_Seq serves sub-ranges from the cache,
-- fetching any missing chunk on demand. The server side reads the requested
-- range from a local Byte_Source via absolute offset and returns the bytes.
---------------------------------------------------------------------

with Ada.Containers;
with Ada.Environment_Variables;
with Ada.Unchecked_Deallocation;
with Crypto;
with System.Storage_Elements; use System.Storage_Elements;
with Protocol;
with LLM_Weight_Proto;

package body LLM_Weight_Source is

   use Ada.Strings.Unbounded;         --  spec withs Unbounded
   use Crypto;                         --  spec withs Crypto
   use type Interfaces.Unsigned_8;     --  '=' on record tags (U8)
   use type Interfaces.Unsigned_32;    --  '=' / '>' on range counts (U32)
   --  Note: `use type Interfaces.Unsigned_64` lives in the spec (needed at the
   --  Chunk_Log_Vectors instantiation), so it is already visible here; the
   --  arithmetic on offsets / lengths resolves through that.

   ------------------------------------------------------------------
   -- Client side
   ------------------------------------------------------------------

   --  Hash a chunk index for the in-memory Chunk_Maps cache. Chunk indices are
   --  dense (0, 1, 2, ...), so the identity hash (mod the map's bucket count)
   --  distributes them perfectly with no clustering.
   function Hash_U64
     (K : Interfaces.Unsigned_64) return Ada.Containers.Hash_Type is
   begin
      return Ada.Containers.Hash_Type
        (K mod Interfaces.Unsigned_64 (Ada.Containers.Hash_Type'Last));
   end Hash_U64;

   --  Read an environment variable, returning Default when unset. Used for
   --  the opt-in cache knobs so an unset var leaves the on-disk cache off.
   function Env (Name : String; Default : String := "") return String is
   begin
      if Ada.Environment_Variables.Exists (Name) then
         return Ada.Environment_Variables.Value (Name);
      end if;
      return Default;
   end Env;

   --  String -> Byte_Array (one byte per char). An empty string yields an
   --  empty array, which LLM_Weight_Cache.Open treats as "no password" (cache
   --  disabled), matching the opt-in intent.
   function To_Bytes (S : String) return Crypto.Byte_Array is
      R : Crypto.Byte_Array (0 .. S'Length - 1);
   begin
      for I in S'Range loop
         R (I - S'First) := Crypto.U8 (Character'Pos (S (I)));
      end loop;
      return R;
   end To_Bytes;

   function Open_Remote
     (Ch       : access Secure_Channel.Channel;
      Trans    : access Secure_Channel.Byte_Transport'Class;
      Model_ID : String;
      Len      : Interfaces.Unsigned_64) return LLM_Byte_Source.Byte_Source_Access
   is
   begin
      --  Aggregate-initialize inside the return so the access parameters
      --  (Ch / Trans) transfer into the heap object under the return
      --  statement's relaxed accessibility rule — a post-hoc field assignment
      --  (R.Chan := Ch), or assigning the allocator to a local access object,
      --  would instead fail the accessibility check (an `access` parameter is
      --  local-level and the heap object is library-level). The disk cache is
      --  opened via the Open function so its result drops straight into the
      --  value-typed Disk component of the same aggregate.
      return new Remote_AEAD_Source'
        (Chan     => Ch,
         Trans    => Trans,
         Model_ID => To_Unbounded_String (Model_ID),
         Len      => Len,
         Pos      => 0,
         Cache    => <>,
         Disk     => LLM_Weight_Cache.Open
                       (Env ("ASPIDA_WEIGHT_CACHE_DIR"),
                        To_Bytes (Env ("ASPIDA_WEIGHT_CACHE_PASS")),
                        Model_ID),
         Log_On   => False,
         Log      => <>);
   end Open_Remote;

   --  Fetch chunk Index, returning an access to its 0-based byte buffer. The
   --  lookup order is memory -> disk -> channel:
   --    * memory: the in-session vector (repeated reads, header + tensor data).
   --    * disk:   the opt-in AEAD-sealed chunk cache (zero outbound fetches on a
   --               warm session). A Cache_Miss falls through to the channel; a
   --               Cache_Error (tamper/format) is raised loud, never silently
   --               refetched.
   --    * channel: a WReq/WData round trip; the result is write-through stored
   --               to the disk cache (best-effort) and held in memory.
   --  The chunk length is min (Chunk_Size, Len - Index*Chunk_Size) and is
   --  verified against the reply length so a short server answer fails loud
   --  rather than yielding a partial chunk.
   function Fetch_Chunk
     (S     : in out Remote_AEAD_Source;
      Index : Interfaces.Unsigned_64) return Byte_Array_Access
   is
      Existing : constant Chunk_Maps.Cursor := S.Cache.Find (Index);
   begin
      --  Tier 1: in-memory cache (O(1) hashed lookup).
      if Chunk_Maps.Has_Element (Existing) then
         return Chunk_Maps.Element (Existing);
      end if;

      --  Tier 2: on-disk AEAD-sealed cache. D is an access type (default null),
      --  so the Load call lives in the statement part — that is what makes the
      --  Cache_Miss / Cache_Error handlers below actually cover it (a block's
      --  handlers do not cover its own declarative part). On a miss we fall
      --  through to the channel; the loaded chunk is promoted into memory so
      --  later reads in this session are a tier-1 hit.
      if LLM_Weight_Cache.Enabled (S.Disk)
        and then LLM_Weight_Cache.Has (S.Disk, Index)
      then
         declare
            D : Byte_Array_Access;
         begin
            D := LLM_Weight_Cache.Load (S.Disk, Index);
            S.Cache.Insert (Index, D);
            return D;
         exception
            when LLM_Weight_Cache.Cache_Miss =>
               null;   --  absent under this key -> fetch from the channel
            when LLM_Weight_Cache.Cache_Error =>
               raise Weight_Fetch_Error with "sealed disk chunk is corrupt";
         end;
      end if;

      --  Tier 3: not cached anywhere — fetch over the channel.
      declare
         Off       : constant Interfaces.Unsigned_64 := Index * Chunk_Size;
         Chunk_Rem : constant Interfaces.Unsigned_64 := S.Len - Off;
         Count     : constant Crypto.U32 :=
           Crypto.U32 (Interfaces.Unsigned_64'Min (Chunk_Size, Chunk_Rem));
         Req   : constant Crypto.Byte_Array :=
           LLM_Weight_Proto.Encode_WReq (Off, Count, To_String (S.Model_ID));
      begin
         Secure_Channel.Send_Message (S.Chan.all, S.Trans, Req);
         declare
            Resp : constant Crypto.Byte_Array :=
              Secure_Channel.Recv_Message (S.Chan.all, S.Trans);
            Tag  : constant Crypto.U8 := LLM_Weight_Proto.Tag_Of (Resp);
         begin
            if Tag = Protocol.Tag_WErr then
               raise Weight_Fetch_Error with "server refused range";
            elsif Tag /= Protocol.Tag_WData then
               raise Weight_Fetch_Error with "unexpected response tag";
            end if;

            --  body is Resp (Resp'First+1 .. Resp'Last); must equal Count
            if Resp'Length - 1 /= Natural (Count) then
               raise Weight_Fetch_Error with "short weight data";
            end if;

            declare
               D : constant Byte_Array_Access :=
                 new Crypto.Byte_Array (0 .. Natural (Count) - 1);
            begin
               for I in 0 .. Natural (Count) - 1 loop
                  D (I) := Resp (Resp'First + 1 + I);
               end loop;
               S.Cache.Insert (Index, D);

               --  Record this outbound (tier-3) fetch in the opt-in log. This
               --  is the only place a genuine channel fetch happens, so the log
               --  is exactly the cold access pattern. Prefetch_All produces
               --  [0, 1, ..., N-1] here, independent of the prompt.
               if S.Log_On then
                  S.Log.Append (Index);
               end if;

               --  Write-through to the on-disk cache so a later session reads
               --  this chunk locally. Best-effort: a Store I/O failure (disk
               --  full, permission) must not kill the fetch — the in-memory
               --  chunk is already valid for this session.
               if LLM_Weight_Cache.Enabled (S.Disk) then
                  begin
                     LLM_Weight_Cache.Store (S.Disk, Index, D.all);
                  exception
                     when others => null;
                  end;
               end if;

               return D;
            end;
         end;
      end;
   end Fetch_Chunk;

   --  Positional read from the chunk cache: serve Count bytes starting at the
   --  absolute Off, fetching any missing chunk on demand. Does NOT touch S.Pos,
   --  so it composes for both the cursor-based Read_Seq and any positional
   --  caller. (The remote source is not shared across tasks — the client owns
   --  it for its session — but implementing the positional primitive keeps the
   --  interface uniform and lets a caller read a range without a seek.)
   overriding procedure Read_At_Pos
     (S     : in out Remote_AEAD_Source;
      Off   : Interfaces.Unsigned_64;
      Addr  : System.Address;
      Count : Natural)
   is
      Remaining : Natural          := Count;
      Cur       : System.Address   := Addr;
      Cur_Pos   : Interfaces.Unsigned_64 := Off;
   begin
      --  Fail loud on an out-of-range request (honouring the interface
      --  contract), matching Local_File_Source, rather than surfacing it as a
      --  Fetch_Chunk failure or a Constraint_Error on the tail chunk.
      if Off > S.Len
        or else Interfaces.Unsigned_64 (Count) > S.Len - Off
      then
         raise LLM_Byte_Source.Malformed_Source
           with "positional read past end of stream";
      end if;
      while Remaining > 0 loop
         declare
            Chunk_Index : constant Interfaces.Unsigned_64 := Cur_Pos / Chunk_Size;
            Chunk_Off   : constant Interfaces.Unsigned_64 := Cur_Pos mod Chunk_Size;
            Data        : constant Byte_Array_Access := Fetch_Chunk (S, Chunk_Index);
            --  bytes present in this chunk from Chunk_Off onward (the tail
            --  chunk may be shorter than Chunk_Size)
            Have        : constant Interfaces.Unsigned_64 :=
              Interfaces.Unsigned_64 (Data.all'Length) - Chunk_Off;
            Want        : constant Interfaces.Unsigned_64 :=
              Interfaces.Unsigned_64'Min
                (Interfaces.Unsigned_64 (Remaining),
                 Interfaces.Unsigned_64'Min (Chunk_Size - Chunk_Off, Have));
            Take        : constant Natural := Natural (Want);
         begin
            --  overlay the destination and copy Take bytes from the chunk
            declare
               Dst : Crypto.Byte_Array (0 .. Take - 1);
               for Dst'Address use Cur;
               pragma Import (Ada, Dst);
            begin
               for J in 0 .. Take - 1 loop
                  Dst (J) := Data (Data.all'First + Natural (Chunk_Off) + J);
               end loop;
            end;
            Cur       := Cur + Storage_Offset (Take);
            Cur_Pos   := Cur_Pos + Want;
            Remaining := Remaining - Take;
         end;
      end loop;
   end Read_At_Pos;

   overriding procedure Read_Seq
     (S     : in out Remote_AEAD_Source;
      Addr  : System.Address;
      Count : Natural)
   is
   begin
      Read_At_Pos (S, S.Pos, Addr, Count);
      S.Pos := S.Pos + Interfaces.Unsigned_64 (Count);
   end Read_Seq;

   overriding function Byte_Length
     (S : Remote_AEAD_Source) return Interfaces.Unsigned_64 is
   begin
      return S.Len;
   end Byte_Length;

   overriding function Cursor
     (S : Remote_AEAD_Source) return Interfaces.Unsigned_64 is
   begin
      return S.Pos;
   end Cursor;

   overriding procedure Seek
     (S   : in out Remote_AEAD_Source;
      Off : Interfaces.Unsigned_64) is
   begin
      --  A seek past the advertised length is rejected here (matching
      --  Local_File_Source) so a hostile/miscomputed tensor offset fails loud
      --  at the seek, not as a fetch of a non-existent range.
      if Off > S.Len then
         raise LLM_Byte_Source.Malformed_Source with "seek past end of stream";
      end if;
      S.Pos := Off;
   end Seek;

   overriding procedure Close (S : in out Remote_AEAD_Source) is
      procedure Free is new Ada.Unchecked_Deallocation
        (Crypto.Byte_Array, Byte_Array_Access);
      D : Byte_Array_Access;
   begin
      --  Release only the owned chunk buffers; the channel + transport are
      --  caller-owned and left open. The disk cache is closed (password
      --  wiped) but its sealed files persist for the next session. Idempotent.
      for C in S.Cache.Iterate loop
         D := Chunk_Maps.Element (C);
         Free (D);
      end loop;
      S.Cache.Clear;
      LLM_Weight_Cache.Close (S.Disk);
   end Close;

   ------------------------------------------------------------------
   -- H19 Phase 5: oblivious warm-fetch + fetch-log observability
   ------------------------------------------------------------------

   --  Number of whole chunks covering the model (the last may be short).
   --  (Len + Chunk_Size - 1) / Chunk_Size, computed without overflow: Len is a
   --  GGUF length (fits in U64 with headroom); Chunk_Size is 65536.
   function Chunk_Count (S : Remote_AEAD_Source) return Interfaces.Unsigned_64 is
     ((S.Len + Chunk_Size - 1) / Chunk_Size);

   overriding procedure Prefetch_All (S : in out Remote_AEAD_Source) is
      N : constant Interfaces.Unsigned_64 := Chunk_Count (S);
   begin
      --  Nothing to prefetch for a zero-length source (and it avoids the
      --  0 .. N-1 = 0 .. U64'Last wrap that an empty model would otherwise
      --  produce). A real GGUF never has Len = 0, but this stays correct.
      if N = 0 then
         return;
      end if;

      --  Fetch every chunk in a FIXED ascending order, independent of the
      --  prompt. Fetch_Chunk populates the in-memory cache (and write-through
      --  to disk if enabled); the returned access is owned by the cache, so we
      --  drop it without freeing. After this loop every byte is a tier-1 hit.
      --  The cursor is untouched (Fetch_Chunk never moves S.Pos).
      for Index in 0 .. N - 1 loop
         declare
            Discard : constant Byte_Array_Access := Fetch_Chunk (S, Index);
            pragma Unreferenced (Discard);
         begin
            null;
         end;
      end loop;
   end Prefetch_All;

   procedure Enable_Fetch_Log (S : in out Remote_AEAD_Source) is
   begin
      S.Log_On := True;
   end Enable_Fetch_Log;

   function Fetch_Log_Length (S : Remote_AEAD_Source) return Natural is
     (Natural (S.Log.Length));

   function Fetch_Log_At
     (S : Remote_AEAD_Source;
      I : Natural) return Interfaces.Unsigned_64
   is
   begin
      --  1-based internal index (Chunk_Log_Vectors is keyed by Positive); the
      --  public I is 0-based, matching the LLM_Byte_Source contract.
      return S.Log.Element (I + 1);
   end Fetch_Log_At;

   ------------------------------------------------------------------
   -- Server side
   ------------------------------------------------------------------

   procedure Serve_Weight_Requests
     (Ch           : in out Secure_Channel.Channel;
      Trans        : access Secure_Channel.Byte_Transport'Class;
      Model        : access LLM_Byte_Source.Byte_Source'Class;
      Model_ID     : String;
      Max_Requests : Natural := 0)
   is
      Served    : Natural := 0;
      Model_Len : constant Interfaces.Unsigned_64 := Model.Byte_Length;
   begin
      loop
         if Max_Requests > 0 and then Served >= Max_Requests then
            exit;
         end if;

         declare
            Req : constant Crypto.Byte_Array :=
              Secure_Channel.Recv_Message (Ch, Trans);
            Tag : constant Crypto.U8 := LLM_Weight_Proto.Tag_Of (Req);
         begin
            if Tag /= Protocol.Tag_WReq then
               Secure_Channel.Send_Message
                 (Ch, Trans, LLM_Weight_Proto.Encode_WErr ("expected weight request"));
            else
               declare
                  Off      : Crypto.U64;
                  Count    : Crypto.U32;
                  ID_Req   : Ada.Strings.Unbounded.Unbounded_String;
                  OK       : Boolean;
               begin
                  LLM_Weight_Proto.Decode_WReq (Req, Off, Count, ID_Req, OK);
                  if not OK then
                     Secure_Channel.Send_Message
                       (Ch, Trans, LLM_Weight_Proto.Encode_WErr ("malformed weight request"));
                  elsif To_String (ID_Req) /= Model_ID then
                     --  The request names a different model than this server
                     --  serves. Reject (don't serve the wrong file's bytes) so a
                     --  multi-model deployment can route by id without a client
                     --  silently reading another model's weights.
                     Secure_Channel.Send_Message
                       (Ch, Trans, LLM_Weight_Proto.Encode_WErr ("model id mismatch"));
                  elsif Count = 0 or else Count > LLM_Weight_Proto.Max_Range_Count then
                     Secure_Channel.Send_Message
                       (Ch, Trans, LLM_Weight_Proto.Encode_WErr ("range size out of bounds"));
                  elsif Off > Model_Len then
                     Secure_Channel.Send_Message
                       (Ch, Trans, LLM_Weight_Proto.Encode_WErr ("offset out of range"));
                  elsif Interfaces.Unsigned_64 (Count) > Model_Len - Off then
                     Secure_Channel.Send_Message
                       (Ch, Trans, LLM_Weight_Proto.Encode_WErr ("count out of range"));
                  else
                     --  Read the range from the local model and reply WData.
                     --  Positional read (Read_At_Pos), NOT Seek + Read_Seq:
                     --  serving concurrent clients from ONE shared Model source
                     --  must not race on a single mutable cursor. pread on the
                     --  underlying fd is atomic w.r.t. the offset, so N clients
                     --  read disjoint ranges of the immutable GGUF in parallel.
                     declare
                        Buf : Crypto.Byte_Array (0 .. Natural (Count) - 1);
                     begin
                        Model.Read_At_Pos (Off, Buf'Address, Natural (Count));
                        Secure_Channel.Send_Message
                          (Ch, Trans, LLM_Weight_Proto.Encode_WData (Buf));
                     end;
                  end if;
               end;
            end if;
         end;

         Served := Served + 1;
      end loop;
   exception
      --  Peer closed the transport or sent a tampered frame: stop serving
      --  quietly. The channel's AEAD has already defeated any in-flight
      --  tamper; a closed socket is a normal end of session.
      when Secure_Channel.Handshake_Error | Secure_Channel.Auth_Error =>
         null;
   end Serve_Weight_Requests;

end LLM_Weight_Source;