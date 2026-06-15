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

   --  Hex of the first N bytes (N=0 => all).
   function To_Hex (B : Byte_Array; N : Natural := 0) return String is
      Digs  : constant String := "0123456789abcdef";
      Count : constant Natural := (if N = 0 or else N > B'Length then B'Length else N);
      R : String (1 .. Count * 2);
      P : Natural := 0;
   begin
      for I in 0 .. Count - 1 loop
         R (P + 1) := Digs (Integer (Shift_Right (B (B'First + I), 4)) + 1);
         R (P + 2) := Digs (Integer (B (B'First + I) and 16#0F#) + 1);
         P := P + 2;
      end loop;
      return R;
   end To_Hex;

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
   if Argument_Count < 4 then
      Put_Line ("usage: secure_client <host> <port> <server_pub_hex> "
                & "<session|new> [prompt...]");
      Put_Line ("  omit the prompt to open an interactive chat (/quit to exit).");
      return;
   end if;

   declare
      Port    : constant Port_Type := Port_Type'Value (Argument (2));
      Srv_Pub : constant Crypto.X25519.Key_256 := From_Hex (Argument (3));
      Session : constant String :=
        (if Argument (4) = "new" then "" else Argument (4));
      Interactive : constant Boolean := Argument_Count < 5;
      Sock    : Socket_Type;
      CT      : aliased Socket_Transport.Sock_Transport;
      Ch      : Secure_Channel.Channel;

      --  Send one user turn over the open channel and stream the reply.
      --  Prefill ticks render as a transient "thinking" line that is wiped the
      --  instant the first token arrives; token bytes are buffered so only
      --  complete UTF-8 sequences reach the terminal (no split-codepoint glitch
      --  on Cyrillic / emoji while streaming).
      procedure Send_And_Stream (Text : String) is
         CR       : constant Character := Character'Val (13);
         ESC      : constant Character := Character'Val (27);
         Clear_Ln : constant String := CR & ESC & "[K";   -- ↤ + erase-to-EOL
         Pending  : Byte_Array (0 .. 4095);
         P_Len    : Natural := 0;
         Thinking : Boolean := False;
         Answered : Boolean := False;

         --  Bytes of the UTF-8 sequence that 'Lead' starts (1 for a stray
         --  continuation byte, so we never stall).
         function Seq_Len (Lead : U8) return Natural is
           (if    Lead < 16#80#  then 1
            elsif Lead >= 16#F0# then 4
            elsif Lead >= 16#E0# then 3
            elsif Lead >= 16#C0# then 2
            else  1);

         --  Emit every complete UTF-8 sequence in Pending; keep an unfinished
         --  trailing sequence buffered for the next record.
         procedure Flush_Complete is
            I : Natural := 0;
            R : Natural := 0;
         begin
            while I < P_Len loop
               declare
                  L : constant Natural := Seq_Len (Pending (I));
               begin
                  exit when I + L > P_Len;     -- tail is incomplete: hold it
                  for J in 0 .. L - 1 loop
                     Put (Character'Val (Integer (Pending (I + J))));
                  end loop;
                  I := I + L;
               end;
            end loop;
            while I + R < P_Len loop           -- shift leftover to the front
               Pending (R) := Pending (I + R); R := R + 1;
            end loop;
            P_Len := R;
         end Flush_Complete;
      begin
         Secure_Channel.Send_Message
           (Ch, CT'Access, Tagged_Msg (Protocol.Tag_Prompt, Text));
         loop
            declare
               Rec : constant Byte_Array :=
                 Secure_Channel.Recv_Message (Ch, CT'Access);
            begin
               exit when Rec'Length = 0;
               case Rec (Rec'First) is
                  when Protocol.Tag_Done => exit;
                  when Protocol.Tag_Prefill =>
                     if not Thinking then Put ("  ⏳ "); Thinking := True; end if;
                     Put ("."); Flush;
                  when Protocol.Tag_Token =>
                     if not Answered then
                        Put (Clear_Ln & "ai  ▸ "); Answered := True;
                     end if;
                     for I in Rec'First + 1 .. Rec'Last loop
                        if P_Len > Pending'Last then
                           Flush_Complete;   -- drain; keeps only a partial tail
                        end if;
                        Pending (P_Len) := Rec (I); P_Len := P_Len + 1;
                     end loop;
                     Flush_Complete; Flush;
                  when others => null;
               end case;
            end;
         end loop;
         for I in 0 .. P_Len - 1 loop          -- defensive: malformed tail
            Put (Character'Val (Integer (Pending (I))));
         end loop;
         if not Answered then Put (Clear_Ln); end if;  -- wipe a lone "thinking"
         New_Line;
      end Send_And_Stream;
   begin
      Create_Socket (Sock);
      Connect_Socket (Sock, (Family_Inet, Inet_Addr (Argument (1)), Port));
      CT.Sock := Sock;
      Secure_Channel.Client_Handshake (Ch, CT'Access, Srv_Pub);

      --  Show the user exactly what protects this session.
      New_Line;
      Put_Line ("  🔒 SECURE SESSION ESTABLISHED");
      Put_Line ("     handshake     X25519 ECDH (ephemeral) — forward secret");
      Put_Line ("     cipher        " & Secure_Channel.Cipher_Suite);
      Put_Line ("     server key    " & To_Hex (Srv_Pub));
      Put_Line ("                   (pinned ✓ — server proved it holds the secret)");
      Put_Line ("     channel bind  " & To_Hex (Secure_Channel.Channel_Binding (Ch), 16)
                & "  (both ends match → no MITM)");
      Put_Line ("     transport     every record AEAD-sealed, per-direction nonce");
      Put_Line ("     at rest       server history: ChaCha20-Poly1305 + PBKDF2-HMAC-SHA256");
      New_Line;

      --  Select/create the session; print the id the server assigned.
      Secure_Channel.Send_Message
        (Ch, CT'Access, Tagged_Msg (Protocol.Tag_Session, Session));
      declare
         Rep : constant Byte_Array := Secure_Channel.Recv_Message (Ch, CT'Access);
         Id  : Unbounded_String;
      begin
         if Rep'Length >= 1 and then Rep (Rep'First) = Protocol.Tag_Error then
            for I in Rep'First + 1 .. Rep'Last loop
               Append (Id, Character'Val (Integer (Rep (I))));
            end loop;
            Put_Line ("  ❌ server: " & To_String (Id));
            Secure_Channel.Close (Ch);
            Close_Socket (Sock);
            return;
         end if;
         if Rep'Length >= 1 and then Rep (Rep'First) = Protocol.Tag_Session then
            for I in Rep'First + 1 .. Rep'Last loop
               Append (Id, Character'Val (Integer (Rep (I))));
            end loop;
         end if;
         Put_Line ("session: " & To_String (Id)
                   & "   (reuse this id to continue the conversation)");
      end;

      --  Either stream one prompt (args) or run an interactive chat loop.
      if Interactive then
         Put_Line ("  💬 interactive chat — type a message and press Enter.");
         Put_Line ("     /quit (or Ctrl-D) to end. Every line is end-to-end encrypted.");
         New_Line;
         loop
            Put ("you ▸ "); Flush;
            exit when Ada.Text_IO.End_Of_File;
            declare
               Line : constant String := Ada.Text_IO.Get_Line;
            begin
               exit when Line = "/quit" or else Line = "/exit";
               if Line'Length > 0 then
                  Send_And_Stream (Line);
                  New_Line;
               end if;
            end;
         end loop;
      else
         declare
            Prompt : Unbounded_String;
         begin
            for I in 5 .. Argument_Count loop
               if I > 5 then Append (Prompt, " "); end if;
               Append (Prompt, Argument (I));
            end loop;
            Send_And_Stream (To_String (Prompt));
         end;
      end if;

      Put_Line ("  🔒 "
        & Secure_Channel.Records_Sent (Ch)'Image & " records sent,"
        & Secure_Channel.Records_Received (Ch)'Image
        & " received — all ChaCha20-Poly1305 encrypted + tag-authenticated.");
      Secure_Channel.Close (Ch);
      Put_Line ("  (session keys wiped from memory)");
      Close_Socket (Sock);
   end;
end Secure_Client;
