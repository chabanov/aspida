---------------------------------------------------------------------
-- Secure_Client — encrypted chat client (demo / reference).
--
-- Connects, runs the initiator handshake pinning the server's public key,
-- selects/creates a session (so the server can resume prior history), then
-- sends one prompt and prints the streamed reply in real time.
--
-- Usage:
--   ./obj/secure_client <host> <port> <server_pub_hex> <session|new> <prompt...>
--   - <session|new>: a session id to resume, or "new" for a fresh one.
--     The server prints the assigned id (reuse it to continue the chat).
---------------------------------------------------------------------

with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Command_Line;        use Ada.Command_Line;
with Ada.Strings.Unbounded;   use Ada.Strings.Unbounded;
with Interfaces;              use Interfaces;
with GNAT.Sockets;            use GNAT.Sockets;
with Crypto;                  use Crypto;
with Crypto.X25519;
with Secure_Channel;
with Socket_Transport;
with Protocol;

procedure Secure_Client is

   function Nyb (C : Character) return U8 is
     (case C is
         when '0' .. '9' => U8 (Character'Pos (C) - Character'Pos ('0')),
         when 'a' .. 'f' => U8 (Character'Pos (C) - Character'Pos ('a') + 10),
         when 'A' .. 'F' => U8 (Character'Pos (C) - Character'Pos ('A') + 10),
         when others     => 0);

   function From_Hex (S : String) return Crypto.X25519.Key_256 is
      R : Crypto.X25519.Key_256 := [others => 0];
   begin
      for I in 0 .. 31 loop
         R (I) := Nyb (S (S'First + 2 * I)) * 16 + Nyb (S (S'First + 2 * I + 1));
      end loop;
      return R;
   end From_Hex;

   --  A tagged record from a string.
   function Tagged_Msg (Tag : U8; Text : String) return Byte_Array is
      R : Byte_Array (0 .. Text'Length);
   begin
      R (0) := Tag;
      for I in Text'Range loop
         R (I - Text'First + 1) := U8 (Character'Pos (Text (I)));
      end loop;
      return R;
   end Tagged_Msg;

begin
   if Argument_Count < 5 then
      Put_Line ("usage: secure_client <host> <port> <server_pub_hex> "
                & "<session|new> <prompt...>");
      return;
   end if;

   declare
      Port    : constant Port_Type := Port_Type'Value (Argument (2));
      Srv_Pub : constant Crypto.X25519.Key_256 := From_Hex (Argument (3));
      Session : constant String :=
        (if Argument (4) = "new" then "" else Argument (4));
      Prompt  : Unbounded_String;
      Sock    : Socket_Type;
      CT      : aliased Socket_Transport.Sock_Transport;
      Ch      : Secure_Channel.Channel;
   begin
      for I in 5 .. Argument_Count loop
         if I > 5 then Append (Prompt, " "); end if;
         Append (Prompt, Argument (I));
      end loop;

      Create_Socket (Sock);
      Connect_Socket (Sock, (Family_Inet, Inet_Addr (Argument (1)), Port));
      CT.Sock := Sock;
      Secure_Channel.Client_Handshake (Ch, CT'Access, Srv_Pub);

      --  Select/create the session; print the id the server assigned.
      Secure_Channel.Send_Message
        (Ch, CT'Access, Tagged_Msg (Protocol.Tag_Session, Session));
      declare
         Rep : constant Byte_Array := Secure_Channel.Recv_Message (Ch, CT'Access);
         Id  : Unbounded_String;
      begin
         if Rep'Length >= 1 and then Rep (Rep'First) = Protocol.Tag_Session then
            for I in Rep'First + 1 .. Rep'Last loop
               Append (Id, Character'Val (Integer (Rep (I))));
            end loop;
         end if;
         Put_Line ("session: " & To_String (Id)
                   & "   (reuse this id to continue the conversation)");
      end;

      --  Send the prompt and stream the reply.
      Secure_Channel.Send_Message
        (Ch, CT'Access, Tagged_Msg (Protocol.Tag_Prompt, To_String (Prompt)));

      loop
         declare
            Rec : constant Byte_Array := Secure_Channel.Recv_Message (Ch, CT'Access);
         begin
            exit when Rec'Length = 0;
            case Rec (Rec'First) is
               when Protocol.Tag_Done =>
                  exit;
               when Protocol.Tag_Prefill =>
                  Put ("."); Flush;
               when Protocol.Tag_Token =>
                  for I in Rec'First + 1 .. Rec'Last loop
                     Put (Character'Val (Integer (Rec (I))));
                  end loop;
                  Flush;
               when others =>
                  null;
            end case;
         end;
      end loop;
      New_Line;
      Secure_Channel.Close (Ch);
      Close_Socket (Sock);
   end;
end Secure_Client;
