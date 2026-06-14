---------------------------------------------------------------------
-- Test_Channel — end-to-end Secure_Channel over an in-memory loopback:
-- a server task and the client (main) run the handshake against each
-- other through two byte pipes, then exchange authenticated records in
-- both directions. Validates handshake key agreement, server key-
-- confirmation, and framed AEAD send/recv.
--
-- (Wrong-pinned-key -> Auth_Error is guaranteed by the confirmation's
-- AEAD.Open, exercised directly in test_crypto.)
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Exceptions;    use Ada.Exceptions;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with Crypto.X25519;
with Secure_Channel;

procedure Test_Channel is
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

   function To_B (S : String) return Byte_Array is
      R : Byte_Array (0 .. S'Length - 1);
   begin
      for I in S'Range loop
         R (I - S'First) := U8 (Character'Pos (S (I)));
      end loop;
      return R;
   end To_B;

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

   --  A blocking single-byte FIFO.
   Cap : constant := 65536;
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

   --  Transport bound to an inbound and an outbound pipe.
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

   C2S : aliased Pipe;   -- client -> server
   S2C : aliased Pipe;   -- server -> client

   Server_Secret : constant Crypto.X25519.Key_256 :=
     [16#01#, 16#23#, 16#45#, 16#67#, 16#89#, 16#ab#, 16#cd#, 16#ef#,
      others => 16#5a#];
   Server_Public : constant Crypto.X25519.Key_256 :=
     Crypto.X25519.Public_Key (Server_Secret);

   --  Server: handshake, then echo "pong:" + whatever the client sent.
   task Server_Task;
   task body Server_Task is
      ST : aliased Loopback;
      Ch : Secure_Channel.Channel;
   begin
      ST.In_P := C2S'Access; ST.Out_P := S2C'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      declare
         Req : constant Byte_Array := Secure_Channel.Recv_Message (Ch, ST'Access);
      begin
         Secure_Channel.Send_Message (Ch, ST'Access, To_B ("pong:") & Req);
      end;
   exception
      when others =>
         null;   -- a failed handshake just ends the task
   end Server_Task;

begin
   Put_Line ("=== Secure_Channel Test Suite ===");
   New_Line;

   declare
      CT : aliased Loopback;
      Ch : Secure_Channel.Channel;
   begin
      CT.In_P := S2C'Access; CT.Out_P := C2S'Access;
      Secure_Channel.Client_Handshake (Ch, CT'Access, Server_Public);
      Assert ("handshake completed (server authenticated)", True);

      Secure_Channel.Send_Message (Ch, CT'Access, To_B ("ping"));
      declare
         Resp : constant Byte_Array := Secure_Channel.Recv_Message (Ch, CT'Access);
      begin
         Assert ("authenticated round-trip", Eq (Resp, To_B ("pong:ping")));
      end;
   exception
      when E : others =>
         Put_Line ("  (exception: " & Exception_Name (E) & " - "
                   & Exception_Message (E) & ")");
         Assert ("handshake / round-trip (no exception)", False);
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Channel;
