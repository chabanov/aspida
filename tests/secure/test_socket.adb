---------------------------------------------------------------------
-- Test_Socket — Secure_Channel + Socket_Transport over a REAL TCP
-- loopback connection. A server task accepts, runs the responder
-- handshake, then streams the token/done protocol; the client (main)
-- connects, runs the initiator handshake (pinned key), sends a prompt
-- and reassembles the streamed reply. No model is involved.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Interfaces;        use Interfaces;
with GNAT.Sockets;      use GNAT.Sockets;
with Crypto;            use Crypto;
with Crypto.X25519;
with Secure_Channel;
with Socket_Transport;
with Protocol;

procedure Test_Socket is
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

   Port : constant Port_Type := 39_517;
   Addr : constant Sock_Addr_Type :=
     (Family_Inet, Inet_Addr ("127.0.0.1"), Port);

   Server_Secret : constant Crypto.X25519.Key_256 :=
     [16#11#, 16#22#, 16#33#, 16#44#, others => 16#a5#];
   Server_Public : constant Crypto.X25519.Key_256 :=
     Crypto.X25519.Public_Key (Server_Secret);

   Listener : Socket_Type;

   --  Sends a token piece as a tagged record.
   procedure Send_Token
     (Ch : in out Secure_Channel.Channel;
      T  : access Secure_Channel.Byte_Transport'Class; Piece : String)
   is
      Msg : Byte_Array (0 .. Piece'Length);
   begin
      Msg (0) := Protocol.Tag_Token;
      for I in Piece'Range loop
         Msg (I - Piece'First + 1) := U8 (Character'Pos (Piece (I)));
      end loop;
      Secure_Channel.Send_Message (Ch, T, Msg);
   end Send_Token;

   task Server_Task is
      entry Start;
   end Server_Task;

   task body Server_Task is
      Conn : Socket_Type;
      From : Sock_Addr_Type;
      ST   : aliased Socket_Transport.Sock_Transport;
      Ch   : Secure_Channel.Channel;
   begin
      accept Start;                              -- proceed once main is listening
      Accept_Socket (Listener, Conn, From);
      ST.Sock := Conn;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      declare
         Prompt : constant Byte_Array :=
           Secure_Channel.Recv_Message (Ch, ST'Access);
         pragma Unreferenced (Prompt);
         Done : constant Byte_Array := [0 => Protocol.Tag_Done];
      begin
         Send_Token (Ch, ST'Access, "He");
         Send_Token (Ch, ST'Access, "llo");
         Secure_Channel.Send_Message (Ch, ST'Access, Done);
      end;
      Close_Socket (Conn);
   exception
      when others => null;
   end Server_Task;

begin
   Put_Line ("=== Secure_Channel over TCP Test Suite ===");
   New_Line;

   Create_Socket (Listener);
   Set_Socket_Option (Listener, Socket_Level, (Reuse_Address, True));
   Bind_Socket (Listener, Addr);
   Listen_Socket (Listener);
   Server_Task.Start;                            -- now safe to accept

   declare
      Client_Sock : Socket_Type;
      CT  : aliased Socket_Transport.Sock_Transport;
      Ch  : Secure_Channel.Channel;
      Acc : Unbounded_String;
   begin
      Create_Socket (Client_Sock);
      Connect_Socket (Client_Sock, Addr);
      CT.Sock := Client_Sock;

      Secure_Channel.Client_Handshake (Ch, CT'Access, Server_Public);
      Assert ("TCP handshake (server authenticated)", True);

      Secure_Channel.Send_Message
        (Ch, CT'Access, [0 => Protocol.Tag_Prompt] & To_B ("hi"));

      loop
         declare
            Rec : constant Byte_Array := Secure_Channel.Recv_Message (Ch, CT'Access);
         begin
            exit when Rec'Length >= 1 and then Rec (Rec'First) = Protocol.Tag_Done;
            if Rec'Length >= 1 and then Rec (Rec'First) = Protocol.Tag_Token then
               for I in Rec'First + 1 .. Rec'Last loop
                  Append (Acc, Character'Val (Integer (Rec (I))));
               end loop;
            end if;
         end;
      end loop;

      Assert ("streamed reply reassembles over TCP", To_String (Acc) = "Hello");
      Close_Socket (Client_Sock);
   exception
      when others =>
         Assert ("TCP round-trip (no exception)", False);
   end;

   Close_Socket (Listener);

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Socket;
