---------------------------------------------------------------------
-- Secure_Server — encrypted chat server.
--
-- Loads the Qwen model once, holds a long-term X25519 keypair (persisted
-- in server_key.bin), and prints its public key (pin this on the client).
-- For each TCP connection it runs the responder handshake, then loops:
-- decrypt a Prompt record -> LLM_Qwen.Chat streaming each token back as an
-- encrypted Token record (via Encrypting_Sink) -> a Done record.
--
-- Usage:  QWEN_MODEL_PATH=... ./obj/secure_server [port]
---------------------------------------------------------------------

with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings.Unbounded;   use Ada.Strings.Unbounded;
with Ada.Environment_Variables;
with Ada.Streams.Stream_IO;
with Ada.Directories;
with Ada.Exceptions;          use Ada.Exceptions;
with Interfaces;              use Interfaces;
with GNAT.Sockets;            use GNAT.Sockets;
with Crypto;                  use Crypto;
with Crypto.X25519;
with Crypto.Random;
with Crypto.Memory;
with Secure_Channel;
with Session_Store;
with Socket_Transport;
with Encrypting_Sink;
with Protocol;
with LLM_Qwen;

procedure Secure_Server is

   package SIO renames Ada.Streams.Stream_IO;

   Key_File : constant String := "server_key.bin";

   function Hex (B : Byte_Array) return String is
      Digs : constant String := "0123456789abcdef";
      R : String (1 .. B'Length * 2);
      P : Natural := 0;
   begin
      for X of B loop
         R (P + 1) := Digs (Integer (Shift_Right (X, 4)) + 1);
         R (P + 2) := Digs (Integer (X and 16#0F#) + 1);
         P := P + 2;
      end loop;
      return R;
   end Hex;

   procedure Load_Or_Create (Secret : out Crypto.X25519.Key_256) is
      F : SIO.File_Type;
   begin
      if Ada.Directories.Exists (Key_File) then
         SIO.Open (F, SIO.In_File, Key_File);
         Crypto.X25519.Key_256'Read (SIO.Stream (F), Secret);
         SIO.Close (F);
      else
         Crypto.Random.Fill (Secret);
         SIO.Create (F, SIO.Out_File, Key_File);
         Crypto.X25519.Key_256'Write (SIO.Stream (F), Secret);
         SIO.Close (F);
      end if;
   end Load_Or_Create;

   function Model_Path return String is
   begin
      if Ada.Environment_Variables.Exists ("QWEN_MODEL_PATH") then
         return Ada.Environment_Variables.Value ("QWEN_MODEL_PATH");
      end if;
      return "/Users/ceo/.lmstudio/models/HauhauCS/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf";
   end Model_Path;

   Port     : Port_Type := 8765;
   Secret   : Crypto.X25519.Key_256;
   Model    : LLM_Qwen.Qwen_Model;
   Listener : Socket_Type;
begin
   if Ada.Command_Line.Argument_Count >= 1 then
      Port := Port_Type'Value (Ada.Command_Line.Argument (1));
   end if;

   Load_Or_Create (Secret);
   if not Crypto.Memory.Lock (Secret'Address, Secret'Length) then
      Put_Line ("note: could not mlock the static key (swap not prevented).");
   end if;
   Put_Line ("server public key (pin this on the client):");
   Put_Line ("  " & Hex (Crypto.X25519.Public_Key (Secret)));

   Model := LLM_Qwen.Load (Model_Path);

   Create_Socket (Listener);
   Set_Socket_Option (Listener, Socket_Level, (Reuse_Address, True));
   Bind_Socket (Listener, (Family_Inet, Any_Inet_Addr, Port));
   Listen_Socket (Listener);
   Put_Line ("listening on port" & Port_Type'Image (Port));

   loop
      declare
         Conn : Socket_Type;
         From : Sock_Addr_Type;
      begin
         Accept_Socket (Listener, Conn, From);
         declare
            ST    : aliased Socket_Transport.Sock_Transport;
            Ch    : aliased Secure_Channel.Channel;
            Store : Session_Store.Store;
            Id_B  : Byte_Array (0 .. 7);
         begin
            ST.Sock := Conn;
            Secure_Channel.Server_Handshake (Ch, ST'Access, Secret);

            --  First record selects/creates the session (Tag_Session + id;
            --  empty id = new). Reply with the assigned id.
            declare
               Hello : constant Byte_Array :=
                 Secure_Channel.Recv_Message (Ch, ST'Access);
               Want  : Unbounded_String;
            begin
               if Hello'Length >= 1 and then Hello (Hello'First) = Protocol.Tag_Session then
                  for I in Hello'First + 1 .. Hello'Last loop
                     Append (Want, Character'Val (Integer (Hello (I))));
                  end loop;
               end if;
               if Length (Want) = 0 then
                  Crypto.Random.Fill (Id_B);
                  Want := To_Unbounded_String (Hex (Id_B));
               end if;
               Session_Store.Open (Store, To_String (Want));
               declare
                  Reply : Byte_Array (0 .. Length (Want));
               begin
                  Reply (0) := Protocol.Tag_Session;
                  for I in 1 .. Length (Want) loop
                     Reply (I) := U8 (Character'Pos (Element (Want, I)));
                  end loop;
                  Secure_Channel.Send_Message (Ch, ST'Access, Reply);
               end;
               Put_Line ("  session " & To_String (Want)
                 & (if Session_Store.Enabled then " (encrypted on disk)"
                    else " (not persisted)")
                 & ", resumed turns:" & Session_Store.Turn_Count (Store)'Image);
            end;

            loop
               declare
                  Req : constant Byte_Array :=
                    Secure_Channel.Recv_Message (Ch, ST'Access);
                  Sink : aliased Encrypting_Sink.Enc_Sink :=
                    (LLM_Qwen.Token_Sink with
                       Ch => Ch'Unchecked_Access, T => ST'Unchecked_Access);
                  Prompt : String (1 .. Integer'Max (0, Req'Length - 1));
               begin
                  for I in Prompt'Range loop
                     Prompt (I) := Character'Val (Integer (Req (Req'First + I)));
                  end loop;
                  declare
                     --  Multi-turn context: prior turns (user+assistant) then
                     --  the current user message last.
                     N    : constant Natural := Session_Store.Turn_Count (Store);
                     Conv : LLM_Qwen.Message_Array (1 .. 2 * N + 1);
                  begin
                     for I in 1 .. N loop
                        Conv (2 * I - 1) :=
                          (LLM_Qwen.Role_User,
                           To_Unbounded_String (Session_Store.User_Of (Store, I)));
                        Conv (2 * I) :=
                          (LLM_Qwen.Role_Assistant,
                           To_Unbounded_String (Session_Store.Assistant_Of (Store, I)));
                     end loop;
                     Conv (2 * N + 1) :=
                       (LLM_Qwen.Role_User, To_Unbounded_String (Prompt));
                     declare
                        R : constant String :=
                          LLM_Qwen.Chat (Model, Conv, 256, Sink'Access);
                     begin
                        Secure_Channel.Send_Message
                          (Ch, ST'Access, [0 => Protocol.Tag_Done]);
                        Session_Store.Append_Turn (Store, Prompt, R);
                     end;
                  end;
                  --  Scrub the decrypted prompt before its memory is reused.
                  Prompt := [others => ASCII.NUL];
               end;
            end loop;
         exception
            when others =>
               Session_Store.Close (Store);  -- drop plaintext history from RAM
               Secure_Channel.Close (Ch);    -- wipe session keys on disconnect
               raise;
         end;
      exception
         when E : others =>
            Put_Line ("  connection closed: " & Exception_Message (E));
            begin
               Close_Socket (Conn);
            exception
               when others => null;
            end;
      end;
   end loop;
end Secure_Server;
