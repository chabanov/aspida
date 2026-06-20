---------------------------------------------------------------------
-- openai_proxy — a LOCAL OpenAI-compatible HTTP endpoint that tunnels every
-- request over the encrypted Noise/ChaCha20 channel to the Aspida server.
--
-- Point any OpenAI SDK at  http://127.0.0.1:<port>/v1  with any api_key.
-- Plaintext exists only on this loopback hop and momentarily in the server's
-- RAM; on the wire it is our AEAD-sealed, pinned-key channel (no TLS middlebox,
-- no CA, no MITM). The proxy binds 127.0.0.1 ONLY.
--
-- Usage:  openai_proxy <server_host> <server_port> <server_pub_hex> [local_port]
---------------------------------------------------------------------

with Ada.Command_Line;        use Ada.Command_Line;
with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Environment_Variables;
with Ada.Streams;             use Ada.Streams;
with Ada.Strings.Fixed;
with GNAT.Sockets;            use GNAT.Sockets;
with Crypto;
with Crypto.X25519;
with Secure_Channel;
with Socket_Transport;
with Protocol;
with OpenAI;

procedure OpenAI_Proxy is

   use type Crypto.U8;

   function From_Hex (S : String) return Crypto.X25519.Key_256 is
      R : Crypto.X25519.Key_256;
      function V (C : Character) return Crypto.U8 is
        (case C is when '0' .. '9' => Crypto.U8 (Character'Pos (C) - Character'Pos ('0')),
                   when 'a' .. 'f' => Crypto.U8 (Character'Pos (C) - Character'Pos ('a') + 10),
                   when 'A' .. 'F' => Crypto.U8 (Character'Pos (C) - Character'Pos ('A') + 10),
                   when others => 0);
   begin
      for I in R'Range loop
         R (I) := V (S (S'First + 2 * I)) * 16 + V (S (S'First + 2 * I + 1));
      end loop;
      return R;
   end From_Hex;

   --  [Tag | text bytes]
   function Frame (T : Crypto.U8; S : String) return Crypto.Byte_Array is
      B : Crypto.Byte_Array (0 .. S'Length);
   begin
      B (0) := T;
      for I in 1 .. S'Length loop
         B (I) := Crypto.U8 (Character'Pos (S (S'First + I - 1)));
      end loop;
      return B;
   end Frame;

   --  Parse a Tag_Done payload "stop|length <prompt_tokens> <completion_tokens>"
   --  (whitespace-separated; absent fields default to 0/stop).
   procedure Parse_Done
     (S : String; Trunc : out Boolean; PT, CT : out Natural)
   is
      Idx : Natural := S'First;
      function Grab return String is
         Start : Natural;
      begin
         while Idx <= S'Last and then S (Idx) = ' ' loop Idx := Idx + 1; end loop;
         Start := Idx;
         while Idx <= S'Last and then S (Idx) /= ' ' loop Idx := Idx + 1; end loop;
         return S (Start .. Idx - 1);
      end Grab;
   begin
      Trunc := False; PT := 0; CT := 0;
      Trunc := Grab = "length";
      declare F : constant String := Grab; begin
         if F /= "" then PT := Natural'Value (F); end if;
      exception when others => PT := 0; end;
      declare F : constant String := Grab; begin
         if F /= "" then CT := Natural'Value (F); end if;
      exception when others => CT := 0; end;
   end Parse_Done;

   --  text of a record after its tag byte
   function Body_Of (R : Crypto.Byte_Array) return String is
      S : String (1 .. Integer'Max (0, R'Length - 1));
   begin
      for I in S'Range loop
         S (I) := Character'Val (Integer (R (R'First + I)));
      end loop;
      return S;
   end Body_Of;

   Local_Port : constant Port_Type :=
     (if Argument_Count >= 4 then Port_Type'Value (Argument (4)) else 8080);

   Sock : Socket_Type;                       -- to the server
   CT   : aliased Socket_Transport.Sock_Transport;
   Ch   : Secure_Channel.Channel;

   Listener, Client : Socket_Type;           -- local HTTP
   Model_Name : constant String := "aspida";

   ----------------------------------------------------------------
   -- Minimal HTTP over a connected client socket
   ----------------------------------------------------------------
   procedure Sock_Write (S : Socket_Type; Str : String) is
      Buf  : Stream_Element_Array (1 .. Str'Length);
      Last : Stream_Element_Offset;
      Pos  : Stream_Element_Offset := Buf'First;
   begin
      for I in Str'Range loop
         Buf (Stream_Element_Offset (I - Str'First + 1)) :=
           Stream_Element (Character'Pos (Str (I)));
      end loop;
      --  TCP may accept fewer bytes than offered; loop until the whole buffer
      --  is sent, else the final SSE chunk / [DONE] / JSON tail can be dropped
      --  (the answer "cuts off at the end").
      while Pos <= Buf'Last loop
         Send_Socket (S, Buf (Pos .. Buf'Last), Last);
         exit when Last < Pos;          -- nothing sent -> peer gone; give up
         Pos := Last + 1;
      end loop;
   end Sock_Write;

   --  Read the full HTTP request; return Method, Path and Body. Err is 0 on
   --  success, 400 for a malformed request (peer closed before the headers
   --  terminated), 413 when the headers or declared body exceed our fixed
   --  buffer (so an oversized request yields a clean error, not a dropped
   --  connection or an out-of-range index).
   procedure Read_Request (S : Socket_Type; Method, Path, Req_Body : out String;
                           M_Len, P_Len, B_Len : out Natural; Err : out Natural) is
      Buf  : Stream_Element_Array (1 .. 65536);
      Last : Stream_Element_Offset;
      Data : String (1 .. 1_048_576);
      Len  : Natural := 0;
      Hdr_End  : Natural := 0;
      CLen     : Natural := 0;
      Overflow : Boolean := False;

      function Find (Pat : String; From : Natural) return Natural is
        (Ada.Strings.Fixed.Index (Data (1 .. Len), Pat, From));
   begin
      Method := [others => ' ']; Path := [others => ' ']; Req_Body := [others => ' '];
      M_Len := 0; P_Len := 0; B_Len := 0; Err := 0;
      --  read until end of headers
      loop
         Receive_Socket (S, Buf, Last);
         exit when Last < Buf'First;            -- peer closed
         for I in 1 .. Integer (Last) loop
            Len := Len + 1;
            Data (Len) := Character'Val (Integer (Buf (Stream_Element_Offset (I))));
         end loop;
         Hdr_End := Find (Character'Val (13) & Character'Val (10)
                          & Character'Val (13) & Character'Val (10), 1);
         exit when Hdr_End > 0;
         if Len > Data'Last - 70000 then Overflow := True; exit; end if;
      end loop;
      if Hdr_End = 0 then
         Err := (if Overflow then 413 else 400);
         return;
      end if;
      --  request line: METHOD SP PATH SP HTTP/...
      declare
         SP1 : constant Natural := Find (" ", 1);
         SP2 : constant Natural := (if SP1 > 0 then Find (" ", SP1 + 1) else 0);
      begin
         if SP1 > 0 then
            M_Len := SP1 - 1;
            Method (Method'First .. Method'First + M_Len - 1) := Data (1 .. SP1 - 1);
         end if;
         if SP2 > SP1 then
            P_Len := SP2 - SP1 - 1;
            Path (Path'First .. Path'First + P_Len - 1) := Data (SP1 + 1 .. SP2 - 1);
         end if;
      end;
      --  Content-Length (case-insensitive search for the common spelling)
      declare
         CI : Natural := Find ("Content-Length:", 1);
      begin
         if CI = 0 then CI := Find ("content-length:", 1); end if;
         if CI > 0 then
            declare
               J : Natural := CI + 15;
               EOL : constant Natural := Find (Character'Val (13) & Character'Val (10), J);
            begin
               while J <= Len and then Data (J) = ' ' loop J := J + 1; end loop;
               if EOL > J then
                  --  A non-numeric / overflowed Content-Length is a malformed
                  --  request, not a 500: Natural'Value raises on garbage.
                  begin
                     CLen := Natural'Value (Ada.Strings.Fixed.Trim
                       (Data (J .. EOL - 1), Ada.Strings.Both));
                  exception
                     when others =>
                        Err := 400;
                        return;
                  end;
               end if;
            end;
         end if;
      end;
      --  read the rest of the body (Hdr_End points at the first CR of CRLFCRLF)
      declare
         Body_Start : constant Natural := Hdr_End + 4;
      begin
         --  A declared body that cannot fit the buffer is rejected up front
         --  rather than read partially (and would otherwise overrun Data).
         if CLen > Data'Last - Body_Start + 1 then
            Err := 413;
            return;
         end if;
         declare
            Have : Natural := Len - Body_Start + 1;
         begin
            while Have < CLen loop
               Receive_Socket (S, Buf, Last);
               exit when Last < Buf'First;
               for I in 1 .. Integer (Last) loop
                  exit when Len >= Data'Last;   -- never index past the buffer
                  Len := Len + 1;
                  Data (Len) := Character'Val (Integer (Buf (Stream_Element_Offset (I))));
               end loop;
               Have := Len - Body_Start + 1;
            end loop;
            B_Len := Integer'Min (CLen, Integer'Max (0, Len - Body_Start + 1));
            if B_Len > 0 then
               Req_Body (Req_Body'First .. Req_Body'First + B_Len - 1) :=
                 Data (Body_Start .. Body_Start + B_Len - 1);
            end if;
         end;
      end;
   end Read_Request;

   procedure Write_JSON (S : Socket_Type; Status, JSON_Body : String) is
      CRLF : constant String := Character'Val (13) & Character'Val (10);
   begin
      Sock_Write (S, "HTTP/1.1 " & Status & CRLF
        & "Content-Type: application/json" & CRLF
        & "Access-Control-Allow-Origin: *" & CRLF
        & "Content-Length:" & Integer'Image (JSON_Body'Length) & CRLF
        & "Connection: close" & CRLF & CRLF & JSON_Body);
   end Write_JSON;

   CRLF : constant String := Character'Val (13) & Character'Val (10);

begin
   if Argument_Count < 3 then
      Put_Line ("usage: openai_proxy <host> <port> <server_pub_hex> [local_port]");
      return;
   end if;

   --  Open the encrypted channel to the server (pin its key) once.
   Create_Socket (Sock);
   Connect_Socket (Sock, (Family_Inet, Inet_Addr (Argument (1)),
                          Port_Type'Value (Argument (2))));
   CT.Sock := Sock;
   Secure_Channel.Client_Handshake (Ch, CT'Access, From_Hex (Argument (3)));
   --  Optional client authentication (ASPIDA_CLIENT_TOKEN), sent first; a
   --  server without a token ignores it.
   if Ada.Environment_Variables.Exists ("ASPIDA_CLIENT_TOKEN") then
      Secure_Channel.Send_Message (Ch, CT'Access, Frame (Protocol.Tag_Auth,
        Ada.Environment_Variables.Value ("ASPIDA_CLIENT_TOKEN")));
   end if;
   Secure_Channel.Send_Message (Ch, CT'Access, Frame (Protocol.Tag_Session, ""));
   declare
      Rep : constant Crypto.Byte_Array := Secure_Channel.Recv_Message (Ch, CT'Access);
      pragma Unreferenced (Rep);
   begin null; end;

   Put_Line ("  🔒 encrypted tunnel up: " & Secure_Channel.Cipher_Suite);
   Put_Line ("  OpenAI-compatible endpoint: http://127.0.0.1:"
             & Ada.Strings.Fixed.Trim (Local_Port'Image, Ada.Strings.Both) & "/v1");
   Put_Line ("  (any api_key; requests are AEAD-sealed to the pinned server)");

   --  Local HTTP listener, loopback only.
   Create_Socket (Listener);
   Set_Socket_Option (Listener, Socket_Level, (Reuse_Address, True));
   Bind_Socket (Listener, (Family_Inet, Inet_Addr ("127.0.0.1"), Local_Port));
   Listen_Socket (Listener);

   loop
      declare
         Addr   : Sock_Addr_Type;
         Method : String (1 .. 16);
         Path   : String (1 .. 1024);
         RBody  : String (1 .. 1_048_576);
         ML, PL, BL, Err : Natural;
      begin
         Accept_Socket (Listener, Client, Addr);
         Read_Request (Client, Method, Path, RBody, ML, PL, BL, Err);
         declare
            Pth : constant String := Path (Path'First .. Path'First + PL - 1);
            Bdy : constant String := RBody (RBody'First .. RBody'First + BL - 1);
         begin
            if Err = 400 then
               Write_JSON (Client, "400 Bad Request",
                 OpenAI.Error_Response ("malformed HTTP request"));
            elsif Err = 413 then
               Write_JSON (Client, "413 Payload Too Large",
                 OpenAI.Error_Response ("request exceeds the proxy buffer"));
            elsif Ada.Strings.Fixed.Index (Pth, "/chat/completions") > 0 then
               Secure_Channel.Send_Message (Ch, CT'Access, Frame (Protocol.Tag_Chat, Bdy));
               --  React to the server's reply type: Tag_Resp = one JSON;
               --  Tag_Token... / Tag_Done = streaming (emit SSE).
               declare
                  Streaming : Boolean := False;
                  First     : Boolean := True;
               begin
                  loop
                     declare
                        Rec : constant Crypto.Byte_Array :=
                          Secure_Channel.Recv_Message (Ch, CT'Access);
                     begin
                        exit when Rec'Length = 0;
                        if Rec (Rec'First) = Protocol.Tag_Resp then
                           Write_JSON (Client, "200 OK", Body_Of (Rec));
                           exit;
                        elsif Rec (Rec'First) = Protocol.Tag_Token then
                           if not Streaming then
                              Sock_Write (Client, "HTTP/1.1 200 OK" & CRLF
                                & "Content-Type: text/event-stream" & CRLF
                                & "Cache-Control: no-cache" & CRLF
                                & "Access-Control-Allow-Origin: *" & CRLF
                                & "Connection: close" & CRLF & CRLF);
                              Streaming := True;
                           end if;
                           Sock_Write (Client, "data: "
                             & OpenAI.Chat_Chunk (Model_Name, Body_Of (Rec), First)
                             & CRLF & CRLF);
                           First := False;
                        elsif Rec (Rec'First) = Protocol.Tag_Done then
                           if not Streaming then
                              --  Empty reply: no Tag_Token was ever emitted, so
                              --  the SSE headers were never written. Emit them
                              --  now (plus a terminal [DONE]) so the HTTP client
                              --  sees a well-formed 200 stream instead of a bare
                              --  TCP close, which downstream parsers treat as an
                              --  error/connection-drop.
                              Sock_Write (Client, "HTTP/1.1 200 OK" & CRLF
                                & "Content-Type: text/event-stream" & CRLF
                                & "Cache-Control: no-cache" & CRLF
                                & "Access-Control-Allow-Origin: *" & CRLF
                                & "Connection: close" & CRLF & CRLF);
                              Streaming := True;
                           end if;
                           declare
                              Trunc  : Boolean;
                              PT, CT : Natural;
                           begin
                              Parse_Done (Body_Of (Rec), Trunc, PT, CT);
                              Sock_Write (Client, "data: "
                                & OpenAI.Chat_Done_Chunk
                                    (Model_Name, PT, CT,
                                     (if Trunc then "length" else "stop"))
                                & CRLF & CRLF);
                           end;
                           Sock_Write (Client, "data: [DONE]" & CRLF & CRLF);
                           exit;
                        end if;
                     end;
                  end loop;
               end;

            elsif Ada.Strings.Fixed.Index (Pth, "/models") > 0 then
               Secure_Channel.Send_Message (Ch, CT'Access, Frame (Protocol.Tag_Models, ""));
               declare
                  Rec : constant Crypto.Byte_Array := Secure_Channel.Recv_Message (Ch, CT'Access);
               begin
                  Write_JSON (Client, "200 OK", Body_Of (Rec));
               end;

            else
               Write_JSON (Client, "404 Not Found",
                 OpenAI.Error_Response ("unknown route: " & Pth));
            end if;
         end;
         Close_Socket (Client);
      exception
         when others =>
            begin Close_Socket (Client); exception when others => null; end;
      end;
   end loop;
end OpenAI_Proxy;
