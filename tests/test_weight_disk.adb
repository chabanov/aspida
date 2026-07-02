---------------------------------------------------------------------
-- Test_Weight_Disk — H19 Phase 3 integration test: the warm-cache
-- invariant (a later session reads a model with ZERO outbound fetches).
--
-- Two full sessions over real in-memory loopback channels, both with the
-- on-disk AEAD-sealed cache enabled (ASPIDA_WEIGHT_CACHE_DIR/PASS set in
-- process to a temp dir):
--
--   * Session 1 (cold): the client fetches every chunk over the channel;
--     each fetched chunk is write-through sealed to disk. Asserts the read
--     is byte-identical to a local-file read.
--
--   * Session 2 (warm): a fresh client opens a new source (empty in-memory
--     cache, same on-disk cache) and reads the whole model again. Its
--     transport is "armed" after the handshake so ANY weight-fetch Send
--     raises — proving the read was served entirely from disk (0 channel
--     fetches). Asserts byte-identical and no exception.
--
-- The two sessions share only the on-disk cache (same Model_ID => same
-- sanitized subdir => same sealed chunks); the in-memory cache does not
-- survive Close, so session 2's reads must hit disk. This is the property
-- that makes H19 weight-streaming useful: pay the network once, then run
-- locally with zero access-pattern leakage to the operator.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Directories;
with Ada.Environment_Variables;
with Ada.Exceptions;    use Ada.Exceptions;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with Crypto.X25519;
with Secure_Channel;
with LLM_Byte_Source;
use type LLM_Byte_Source.Byte_Source_Access;
with LLM_Weight_Source;

procedure Test_Weight_Disk is
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

   --  Plain loopback for session 1.
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

   --  Armed loopback for session 2: once Armed is set (after the handshake),
   --  any Write raises, so a weight-fetch Send fails loud instead of silently
   --  hitting the channel. Read is left alone (the handshake reads finish before
   --  Arming; no reads happen post-Arm if all chunks are on disk).
   type Armed_Loopback is limited new Loopback with record
      Armed : Boolean := False;
   end record;
   overriding procedure Write (T : in out Armed_Loopback; Data : Byte_Array);

   overriding procedure Write (T : in out Armed_Loopback; Data : Byte_Array) is
   begin
      if T.Armed then
         raise Secure_Channel.Handshake_Error
           with "warm session attempted a channel fetch (expected disk-only)";
      end if;
      for B of Data loop T.Out_P.Put (B); end loop;
   end Write;

   Model_Path : constant String := "svgdata/student.gguf";
   Cache_Dir  : constant String := "/tmp/aspida_wdisk_test";
   Cache_Pass : constant String := "disk-warmth-test-password";

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

   C2S_1 : aliased Pipe;
   S2C_1 : aliased Pipe;
   C2S_2 : aliased Pipe;
   S2C_2 : aliased Pipe;

   --  Session-1 server: opens the model, handshakes, serves N_Chunks (the
   --  cold read), then returns. It must NOT run with Max_Requests=0 because the
   --  client finishes reading and closes without sending more; a bounded count
   --  lets the task exit cleanly.
   task Server_Task_1 is
      entry Launch;
   end Server_Task_1;

   task body Server_Task_1 is
      ST : aliased Loopback;
      Ch : Secure_Channel.Channel;
      M  : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      M := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S_1'Access; ST.Out_P := S2C_1'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      LLM_Weight_Source.Serve_Weight_Requests
        (Ch, ST'Access, M, Model_Path, N_Chunks);
      LLM_Byte_Source.Free_Source (M);
   exception
      when others => null;
   end Server_Task_1;

   --  Session-2 server: handshakes only, then blocks on Recv (the warm client
   --  never sends). It is aborted after the client's disk-only read.
   task Server_Task_2 is
      entry Launch;
   end Server_Task_2;

   task body Server_Task_2 is
      ST : aliased Loopback;
      Ch : Secure_Channel.Channel;
      M  : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      M := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S_2'Access; ST.Out_P := S2C_2'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      --  Max_Requests = 0: serve until the channel raises (it never will,
      --  because the warm client never sends). The task is aborted by the main.
      LLM_Weight_Source.Serve_Weight_Requests (Ch, ST'Access, M, Model_Path, 0);
      LLM_Byte_Source.Free_Source (M);
   exception
      when others => null;
   end Server_Task_2;

   CT1 : aliased Loopback;
   Ch1 : aliased Secure_Channel.Channel;
   CT2 : aliased Armed_Loopback;
   Ch2 : aliased Secure_Channel.Channel;

begin
   Put_Line ("=== H19 Warm-Cache (Disk-Only Session) Test ===");
   New_Line;

   --  Ground truth.
   Ref := LLM_Byte_Source.Open_Source (Model_Path);
   if Ref = null then
      Put_Line ("  SKIP: model fixture " & Model_Path & " not found");
      Ada.Command_Line.Set_Exit_Status (0);
      Server_Task_1.Launch;
      Server_Task_2.Launch;
      return;
   end if;

   Model_Len := Ref.Byte_Length;
   N_Chunks  := Natural ((Model_Len + LLM_Weight_Source.Chunk_Size - 1) /
                         LLM_Weight_Source.Chunk_Size);
   Ref_Bytes := new Byte_Array (0 .. Natural (Model_Len) - 1);
   Ref.Read_Seq (Ref_Bytes.all'Address, Natural (Model_Len));

   --  Fresh cache dir + enable the on-disk cache for both sessions.
   if Ada.Directories.Exists (Cache_Dir) then
      Ada.Directories.Delete_Tree (Cache_Dir);
   end if;
   Ada.Directories.Create_Path (Cache_Dir);
   Ada.Environment_Variables.Set ("ASPIDA_WEIGHT_CACHE_DIR", Cache_Dir);
   Ada.Environment_Variables.Set ("ASPIDA_WEIGHT_CACHE_PASS", Cache_Pass);

   ------------------------------------------------------------------
   --  Session 1: cold read, write-through to disk.
   ------------------------------------------------------------------
   Server_Task_1.Launch;
   CT1.In_P := S2C_1'Access; CT1.Out_P := C2S_1'Access;
   Secure_Channel.Client_Handshake (Ch1, CT1'Access, Server_Public);
   Assert ("session 1 handshake completed", True);

   declare
      Src : LLM_Byte_Source.Byte_Source_Access :=
        LLM_Weight_Source.Open_Remote
          (Ch1'Access, CT1'Access, Model_Path, Model_Len);
      A   : constant Byte_Array_Access := new Byte_Array (0 .. Natural (Model_Len) - 1);
   begin
      Src.Read_Seq (A.all'Address, Natural (Model_Len));
      Assert ("session 1 cold read byte-identical (write-through to disk)",
              Eq (A.all, Ref_Bytes.all));
      LLM_Byte_Source.Free_Source (Src);   --  Close: memory freed, disk sealed
   end;

   ------------------------------------------------------------------
   --  Session 2: warm read, disk-only (armed transport proves 0 fetches).
   ------------------------------------------------------------------
   Server_Task_2.Launch;
   CT2.In_P := S2C_2'Access; CT2.Out_P := C2S_2'Access;
   Secure_Channel.Client_Handshake (Ch2, CT2'Access, Server_Public);
   Assert ("session 2 handshake completed", True);
   CT2.Armed := True;   --  any post-handshake Write (a fetch Send) now raises

   declare
      Src : LLM_Byte_Source.Byte_Source_Access :=
        LLM_Weight_Source.Open_Remote
          (Ch2'Access, CT2'Access, Model_Path, Model_Len);
      B   : constant Byte_Array_Access := new Byte_Array (0 .. Natural (Model_Len) - 1);
   begin
      --  Fresh in-memory cache; all chunks must come from the on-disk cache
      --  populated by session 1. A single miss would Send on the armed
      --  transport and raise, failing the "no exception" assertion below.
      Src.Read_Seq (B.all'Address, Natural (Model_Len));
      Assert ("session 2 warm read byte-identical (disk only)",
              Eq (B.all, Ref_Bytes.all));
      Assert ("session 2 made zero channel fetches (no exception)", True);
      LLM_Byte_Source.Free_Source (Src);
   end;

   --  Release the blocked session-2 server.
   abort Server_Task_2;

   LLM_Byte_Source.Free_Source (Ref);

   --  Cleanup env + temp dir.
   Ada.Environment_Variables.Clear ("ASPIDA_WEIGHT_CACHE_DIR");
   Ada.Environment_Variables.Clear ("ASPIDA_WEIGHT_CACHE_PASS");
   Ada.Directories.Delete_Tree (Cache_Dir);

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
exception
   when E : others =>
      abort Server_Task_1;
      abort Server_Task_2;
      Put_Line ("  (exception: " & Exception_Name (E) & " - "
                & Exception_Message (E) & ")");
      Assert ("warm-cache scenario (no exception)", False);
      Ada.Environment_Variables.Clear ("ASPIDA_WEIGHT_CACHE_DIR");
      Ada.Environment_Variables.Clear ("ASPIDA_WEIGHT_CACHE_PASS");
      begin Ada.Directories.Delete_Tree (Cache_Dir); exception when others => null; end;
      New_Line;
      Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
      if Failed > 0 then
         Ada.Command_Line.Set_Exit_Status (1);
      end if;
end Test_Weight_Disk;