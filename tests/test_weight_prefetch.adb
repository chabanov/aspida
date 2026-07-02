---------------------------------------------------------------------
-- Test_Weight_Prefetch — H19 Phase 5 integration test: the oblivious
-- warm-fetch invariant.
--
-- One session over a real in-memory loopback channel, NO on-disk cache
-- (Phase 5 warms the in-memory cache for THIS session — disk persistence
-- is Phase 3's concern, tested separately):
--
--   * Cold prefetch: the client calls Prefetch_All, which fetches every
--     chunk in a FIXED ascending order (chunk 0, 1, ..., N-1). The opt-in
--     fetch log records each outbound (tier-3) fetch's chunk index; we
--     assert the log is exactly [0, 1, ..., N-1] — i.e. the cold access
--     pattern is prompt-independent (the oblivious property that is the
--     whole point of Phase 5 vs. on-demand, prompt-driven fetches).
--
--   * Warm read: after prefetch, the transport is "armed" so ANY weight-
--     fetch Send raises. The client then reads the whole model via Read_Seq.
--     Every byte is a tier-1 (in-memory) hit, so no Send fires — proving
--     the engine reads only local bytes after warm. Asserts byte-identical
--     to a local-file read, no exception, and the fetch log is unchanged
--     (zero outbound fetches during inference).
--
-- This is the property that makes H19 leak-free under a cold cache: the
-- operator serving weights sees "model X was loaded, in full, once" —
-- never which tensors the prompt touched or in what order.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Exceptions;    use Ada.Exceptions;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with Crypto.X25519;
with Secure_Channel;
with LLM_Byte_Source;
use type LLM_Byte_Source.Byte_Source_Access;
with LLM_Weight_Source;

procedure Test_Weight_Prefetch is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Cond : Boolean) is
   begin
      if Cond then
         Put_Line ("  PASS: " & Name); Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name); Failed := Failed + 1;
      end if;
   end Assert;

   function Eq (A, B : Byte_Array) return Boolean is
   begin
      if A'Length /= B'Length then
         return False;
      end if;
      for I in 0 .. A'Length - 1 loop
         if A (A'First + I) /= B (B'First + I) then
            return False;
         end if;
      end loop;
      return True;
   end Eq;

   type Byte_Array_Access is access all Byte_Array;

   --  Blocking single-byte FIFO. Cap exceeds a full weight-stream frame
   --  (4 + AEAD(1 + 65536 + 16) = 65557) so the writer cannot lap the reader.
   Cap : constant := 262_144;
   protected type Pipe is
      procedure Put (B : U8);
      entry Get (B : out U8);
   private
      Buf  : Byte_Array (0 .. Cap - 1);
      Head : Natural := 0;
      Tail : Natural := 0;
      Cnt  : Natural := 0;
   end Pipe;

   protected body Pipe is
      procedure Put (B : U8) is
      begin
         Buf (Tail) := B; Tail := (Tail + 1) mod Cap; Cnt := Cnt + 1;
      end Put;
      entry Get (B : out U8) when Cnt > 0 is
      begin
         B := Buf (Head); Head := (Head + 1) mod Cap; Cnt := Cnt - 1;
      end Get;
   end Pipe;

   --  Plain loopback for the handshake + prefetch phase.
   type Loopback is limited new Secure_Channel.Byte_Transport with record
      In_P, Out_P : access Pipe;
   end record;
   overriding procedure Write (T : in out Loopback; Data : Byte_Array);
   overriding procedure Read  (T : in out Loopback; Data : out Byte_Array);

   overriding procedure Write (T : in out Loopback; Data : Byte_Array) is
   begin
      for B of Data loop T.Out_P.Put (B); end loop;
   end Write;
   overriding procedure Read (T : in out Loopback; Data : out Byte_Array) is
      B : U8;
   begin
      for I in Data'Range loop T.In_P.Get (B); Data (I) := B; end loop;
   end Read;

   --  Armed loopback: once Armed is set (after the prefetch), any Write raises,
   --  so a weight-fetch Send during the warm read fails loud instead of
   --  silently hitting the channel. Read is left alone (no reads happen post-
   --  Arm if every chunk is in the in-memory cache).
   type Armed_Loopback is limited new Loopback with record
      Armed : Boolean := False;
   end record;
   overriding procedure Write (T : in out Armed_Loopback; Data : Byte_Array);

   overriding procedure Write (T : in out Armed_Loopback; Data : Byte_Array) is
   begin
      if T.Armed then
         raise Secure_Channel.Handshake_Error
           with "warm read attempted a channel fetch (expected in-memory only)";
      end if;
      for B of Data loop T.Out_P.Put (B); end loop;
   end Write;

   Model_Path : constant String := "svgdata/student.gguf";

   Server_Secret : constant Crypto.X25519.Key_256 :=
     [16#01#, 16#23#, 16#45#, 16#67#, 16#89#, 16#ab#, 16#cd#, 16#ef#,
      others => 16#5a#];
   Server_Public : constant Crypto.X25519.Key_256 :=
     Crypto.X25519.Public_Key (Server_Secret);

   --  Library-level state (see test_weight_stream for why 'Access needs this).
   Ref       : LLM_Byte_Source.Byte_Source_Access;
   Model_Len : Unsigned_64 := 0;
   N_Chunks  : Natural := 0;
   Ref_Bytes : Byte_Array_Access;

   C2S : aliased Pipe;
   S2C : aliased Pipe;

   --  Server: opens the model, handshakes, serves exactly N_Chunks requests
   --  (the prefetch), then returns. The warm read sends nothing, so a bounded
   --  count lets the task exit cleanly without a blocked Recv to abort.
   task Server_Task is
      entry Launch;
   end Server_Task;

   task body Server_Task is
      ST : aliased Loopback;
      Ch : Secure_Channel.Channel;
      M  : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      M := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S'Access; ST.Out_P := S2C'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      LLM_Weight_Source.Serve_Weight_Requests
        (Ch, ST'Access, M, Model_Path, N_Chunks);
      LLM_Byte_Source.Free_Source (M);
   exception
      when others => null;
   end Server_Task;

   CT : aliased Armed_Loopback;
   Ch : aliased Secure_Channel.Channel;

begin
   Put_Line ("=== H19 Oblivious Prefetch (Warm In-Memory Read) Test ===");
   New_Line;

   --  Ground truth: local-file read of the fixture.
   Ref := LLM_Byte_Source.Open_Source (Model_Path);
   if Ref = null then
      Put_Line ("  SKIP: model fixture " & Model_Path & " not found");
      Ada.Command_Line.Set_Exit_Status (0);
      Server_Task.Launch;
      return;
   end if;

   Model_Len := Ref.Byte_Length;
   N_Chunks  := Natural ((Model_Len + LLM_Weight_Source.Chunk_Size - 1) /
                         LLM_Weight_Source.Chunk_Size);
   Ref_Bytes := new Byte_Array (0 .. Natural (Model_Len) - 1);
   Ref.Read_Seq (Ref_Bytes.all'Address, Natural (Model_Len));

   --  No ASPIDA_WEIGHT_CACHE_DIR/PASS -> on-disk cache disabled; this isolates
   --  the in-session prefetch (disk persistence is test_weight_disk's scope).

   ------------------------------------------------------------------
   --  Handshake + cold prefetch (fixed ascending fetch order).
   ------------------------------------------------------------------
   Server_Task.Launch;
   CT.In_P := S2C'Access; CT.Out_P := C2S'Access;
   Secure_Channel.Client_Handshake (Ch, CT'Access, Server_Public);
   Assert ("handshake completed", True);

   declare
      Src : LLM_Byte_Source.Byte_Source_Access :=
        LLM_Weight_Source.Open_Remote
          (Ch'Access, CT'Access, Model_Path, Model_Len);
      --  Open_Remote always returns a Remote_AEAD_Source; the fetch-log hook is
      --  concrete to that type (not on the Byte_Source interface), so reach it
      --  via a tagged view conversion. The conversion raises Constraint_Error
      --  if the source were ever not a remote one — fail-loud, by design.
      R   : LLM_Weight_Source.Remote_AEAD_Source renames
              LLM_Weight_Source.Remote_AEAD_Source (Src.all);
   begin
      --  Record the outbound fetch sequence, then prefetch every chunk in the
      --  fixed order (Prefetch_All dispatches via the interface). The log must
      --  be exactly [0, 1, ..., N_Chunks-1].
      LLM_Weight_Source.Enable_Fetch_Log (R);
      Src.Prefetch_All;

      declare
         Log_Before : constant Natural :=
           LLM_Weight_Source.Fetch_Log_Length (R);
         Ascending  : Boolean := (Log_Before = N_Chunks);
      begin
         for I in 0 .. N_Chunks - 1 loop
            if LLM_Weight_Source.Fetch_Log_At (R, I) /= Unsigned_64 (I) then
               Ascending := False;
            end if;
         end loop;
         Assert ("prefetch fetched exactly N chunks", Log_Before = N_Chunks);
         Assert ("prefetch fetch order is [0..N-1] ascending (prompt-independent)",
                 Ascending);

         --  Arm: any further Write (a weight-fetch Send) raises. The warm read
         --  must be served entirely from the in-memory cache populated above.
         CT.Armed := True;

         declare
            B         : constant Byte_Array_Access :=
              new Byte_Array (0 .. Natural (Model_Len) - 1);
            Read_OK   : Boolean;
            Log_After : Natural;
         begin
            Src.Read_Seq (B.all'Address, Natural (Model_Len));
            Read_OK   := Eq (B.all, Ref_Bytes.all);
            Log_After := LLM_Weight_Source.Fetch_Log_Length (R);
            Assert ("warm read byte-identical (in-memory cache)", Read_OK);
            Assert ("warm read made zero channel fetches (no exception)", True);
            Assert ("fetch log unchanged after warm read",
                    Log_After = Log_Before);
         end;
      end;

      LLM_Byte_Source.Free_Source (Src);
   end;

   LLM_Byte_Source.Free_Source (Ref);

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
exception
   when E : others =>
      abort Server_Task;
      Put_Line ("  (exception: " & Exception_Name (E) & " - "
                & Exception_Message (E) & ")");
      Assert ("prefetch scenario (no exception)", False);
      New_Line;
      Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
      if Failed > 0 then
         Ada.Command_Line.Set_Exit_Status (1);
      end if;
end Test_Weight_Prefetch;