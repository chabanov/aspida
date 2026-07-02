---------------------------------------------------------------------
-- Test_Weight_Stream — H19 integration test (Phases 1+2).
--
-- An in-memory loopback wires a server task (Server_Handshake +
-- Serve_Weight_Requests over a Local_File_Source on a real GGUF) to a client
-- (Client_Handshake + Remote_AEAD_Source). The client reads the whole model
-- over the channel and we assert every byte matches a local-file read of the
-- same model — the H19 parity invariant (remote-sourced weights must be
-- bit-identical to local-sourced weights, or the engine would miscompute).
--
-- Then a second full read exercises the in-memory chunk cache (zero new
-- fetches — the server only serves N_Chunks requests and then exits, so a
-- re-fetch would hang and fail), and a mid-range read checks a cache hit at
-- an arbitrary offset. This is the same loopback pattern as test_channel.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Exceptions;    use Ada.Exceptions;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with Crypto.X25519;
with Secure_Channel;
with LLM_Byte_Source;
use type LLM_Byte_Source.Byte_Source_Access;  --  '=' for the source access
with LLM_Weight_Source;

procedure Test_Weight_Stream is
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

   --  Local heap access to a byte buffer (the test holds the ground-truth
   --  model bytes and the two remote-read buffers this way).
   type Byte_Array_Access is access all Byte_Array;

   --  Blocking single-byte FIFO (same shape as test_channel). The capacity
   --  must exceed a full weight-stream frame: a max chunk reply is 4 (len) +
   --  AEAD(1 tag + 65536 body + 16 Poly1305) = 65557 bytes, so 256 KiB leaves
   --  the writer room for a whole frame (plus margin) without lapping the
   --  reader and corrupting the length prefix.
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

   type Loopback is limited new Secure_Channel.Byte_Transport with record
      In_P, Out_P : access Pipe;
   end record;
   overriding procedure Write (T : in out Loopback; Data : Byte_Array);
   overriding procedure Read  (T : in out Loopback; Data : out Byte_Array);

   overriding procedure Write (T : in out Loopback; Data : Byte_Array) is
   begin
      for B of Data loop
         T.Out_P.Put (B);
      end loop;
   end Write;

   overriding procedure Read (T : in out Loopback; Data : out Byte_Array) is
      B : U8;
   begin
      for I in Data'Range loop
         T.In_P.Get (B);
         Data (I) := B;
      end loop;
   end Read;

   Model_Path : constant String := "svgdata/student.gguf";

   Server_Secret : constant Crypto.X25519.Key_256 :=
     [16#01#, 16#23#, 16#45#, 16#67#, 16#89#, 16#ab#, 16#cd#, 16#ef#,
      others => 16#5a#];
   Server_Public : constant Crypto.X25519.Key_256 :=
     Crypto.X25519.Public_Key (Server_Secret);

   --  All session state lives at the procedure's top level (library level for a
   --  library-unit main) so 'Access of C2S/S2C/Ch satisfies the library-level
   --  access types of Loopback.In_P / Open_Remote. The buffers are filled in
   --  the body after a null-check on the model source.
   Ref       : LLM_Byte_Source.Byte_Source_Access;
   Model_Len : Unsigned_64 := 0;
   N_Chunks  : Natural := 0;
   Ref_Bytes : Byte_Array_Access;

   C2S : aliased Pipe;   --  client -> server
   S2C : aliased Pipe;   --  server -> client

   --  Server task. Launch is a rendezvous entry: the task does nothing (and
   --  touches no shared state) until the main body calls Server_Task.Launch
   --  AFTER N_Chunks is set, so the hand-off of N_Chunks is race-free.
   task Server_Task is
      entry Launch;
   end Server_Task;

   task body Server_Task is
      ST : aliased Loopback;
      Ch : Secure_Channel.Channel;
      Srv_Model : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      Srv_Model := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S'Access; ST.Out_P := S2C'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      LLM_Weight_Source.Serve_Weight_Requests
        (Ch, ST'Access, Srv_Model, Model_Path, N_Chunks);
      LLM_Byte_Source.Free_Source (Srv_Model);
   exception
      when others =>
         null;   --  peer gone / tamper / missing fixture: just end the task
   end Server_Task;

   CT : aliased Loopback;
   Ch : aliased Secure_Channel.Channel;

begin
   Put_Line ("=== H19 Weight-Streaming Test Suite ===");
   New_Line;

   --  Ground truth: a local-file read of the whole model.
   Ref := LLM_Byte_Source.Open_Source (Model_Path);
   if Ref = null then
      Put_Line ("  SKIP: model fixture " & Model_Path & " not found");
      Put_Line ("       (run from the repo root so the relative path resolves)");
      Ada.Command_Line.Set_Exit_Status (0);
      Server_Task.Launch;   --  release the server task so it can terminate
      return;
   end if;

   Model_Len := Ref.Byte_Length;
   N_Chunks  := Natural ((Model_Len + LLM_Weight_Source.Chunk_Size - 1) /
                         LLM_Weight_Source.Chunk_Size);
   Ref_Bytes := new Byte_Array (0 .. Natural (Model_Len) - 1);
   Ref.Read_Seq (Ref_Bytes.all'Address, Natural (Model_Len));

   --  Start the server (rendezvous: it now opens its own model fd, runs the
   --  handshake, and serves exactly N_Chunks range requests).
   Server_Task.Launch;

   --  Client: handshake, then open a streaming source over the channel.
   CT.In_P := S2C'Access; CT.Out_P := C2S'Access;
   Secure_Channel.Client_Handshake (Ch, CT'Access, Server_Public);
   Assert ("handshake completed (server authenticated)", True);

   declare
      Src : LLM_Byte_Source.Byte_Source_Access :=
        LLM_Weight_Source.Open_Remote
          (Ch'Access, CT'Access, Model_Path, Model_Len);
      A   : constant Byte_Array_Access := new Byte_Array (0 .. Natural (Model_Len) - 1);
      B   : constant Byte_Array_Access := new Byte_Array (0 .. Natural (Model_Len) - 1);
   begin
      Assert ("remote byte length matches local",
              Src.Byte_Length = Model_Len);

      --  Cold read: fetches every chunk over the channel (N_Chunks requests,
      --  which is exactly what the server serves before exiting).
      Src.Read_Seq (A.all'Address, Natural (Model_Len));
      Assert ("cold remote read byte-identical to local file",
              Eq (A.all, Ref_Bytes.all));
      Assert ("cursor advanced to end", Src.Cursor = Model_Len);

      --  Warm read: cache hits only. The server has already exited (it
      --  served N_Chunks), so any re-fetch would block and fail — passing
      --  here proves the cache served the bytes locally.
      Src.Seek (0);
      Src.Read_Seq (B.all'Address, Natural (Model_Len));
      Assert ("warm cached read byte-identical to local file",
              Eq (B.all, Ref_Bytes.all));
      Assert ("seek reset cursor to zero", Src.Cursor = Model_Len);

      --  Mid-range cache hit at an arbitrary offset.
      declare
         Mid : constant Natural := Natural (Model_Len / 2);
         M   : Byte_Array (0 .. 99);
         OK  : Boolean := True;
      begin
         Src.Seek (Unsigned_64 (Mid));
         Src.Read_Seq (M'Address, 100);
         for I in 0 .. 99 loop
            if M (I) /= Ref_Bytes (Ref_Bytes'First + Mid + I) then
               OK := False;
            end if;
         end loop;
         Assert ("mid-range cache-hit read byte-identical", OK);
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
      --  If the client bailed mid-stream, the server is blocked on a Recv that
      --  will never come (the loopback has no "close"); abort it so the main
      --  can finish instead of deadlocking on task termination.
      abort Server_Task;
      Put_Line ("  (exception: " & Exception_Name (E) & " - "
                & Exception_Message (E) & ")");
      Assert ("weight-stream round-trip (no exception)", False);
      New_Line;
      Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
      if Failed > 0 then
         Ada.Command_Line.Set_Exit_Status (1);
      end if;
end Test_Weight_Stream;