---------------------------------------------------------------------
-- ws_bridge — a LOCAL web server + WebSocket<->TCP relay for the encrypted
-- chat demo. It serves the static web app (web/) and, on /ws, upgrades to a
-- WebSocket and relays the RAW bytes to the Aspida secure_server's TCP socket.
--
-- The bridge never touches the cryptography: the browser performs the whole
-- Secure_Channel handshake and seals every record itself (web/crypto.js +
-- web/channel.js). On both hops the bridge sees only ciphertext — it is a dumb
-- byte pipe whose only job is to let a browser (which cannot open raw TCP)
-- reach the server. Binds 127.0.0.1 ONLY.
--
-- Usage: ws_bridge <server_host> <server_port> <server_pub_hex> [local_port] [web_dir]
---------------------------------------------------------------------

with Ada.Command_Line;        use Ada.Command_Line;
with Ada.Environment_Variables;
with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Streams;             use Ada.Streams;
with Ada.Streams.Stream_IO;
with Ada.Directories;
with Ada.Strings;
with Ada.Strings.Fixed;       use Ada.Strings.Fixed;
with Ada.Strings.Maps;
with Ada.Strings.Maps.Constants;
with Ada.Strings.Unbounded;   use Ada.Strings.Unbounded;
with Interfaces;              use Interfaces;
with GNAT.Sockets;            use GNAT.Sockets;

procedure WS_Bridge is

   --  String -> Stream_Element_Array (declared early: used by the WS handshake).
   function To_SEA (S : String) return Stream_Element_Array is
      R : Stream_Element_Array (1 .. S'Length);
   begin
      for I in S'Range loop
         R (Stream_Element_Offset (I - S'First + 1)) := Stream_Element (Character'Pos (S (I)));
      end loop;
      return R;
   end To_SEA;

   Srv_Host : constant String := Argument (1);
   Srv_Port : constant Port_Type := Port_Type'Value (Argument (2));
   Srv_Pub  : constant String := Argument (3);
   Local_Port : constant Port_Type :=
     (if Argument_Count >= 4 then Port_Type'Value (Argument (4)) else 8888);
   Web_Dir : constant String :=
     (if Argument_Count >= 5 then Argument (5) else "web");

   WS_GUID : constant String := "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
   Dbg : constant Boolean := Ada.Environment_Variables.Exists ("ASPIDA_WSDBG");

   ------------------------------------------------------------------
   -- SHA-1 (FIPS 180-1) — needed only for the WebSocket accept hash.
   ------------------------------------------------------------------
   type Word_Array is array (Natural range <>) of Unsigned_32;
   function Rotl (X : Unsigned_32; N : Natural) return Unsigned_32 is
     (Shift_Left (X, N) or Shift_Right (X, 32 - N));

   function SHA1 (Msg : Stream_Element_Array) return Stream_Element_Array is
      H0 : Unsigned_32 := 16#67452301#;
      H1 : Unsigned_32 := 16#EFCDAB89#;
      H2 : Unsigned_32 := 16#98BADCFE#;
      H3 : Unsigned_32 := 16#10325476#;
      H4 : Unsigned_32 := 16#C3D2E1F0#;
      ML  : constant Unsigned_64 := Unsigned_64 (Msg'Length) * 8;
      Pad : constant Natural := (56 - (Msg'Length + 1) mod 64 + 64) mod 64;
      Tot : constant Natural := Msg'Length + 1 + Pad + 8;
      B   : Stream_Element_Array (0 .. Stream_Element_Offset (Tot) - 1) := [others => 0];
   begin
      for I in Msg'Range loop B (Msg'First - Msg'First + (I - Msg'First)) := Msg (I); end loop;
      B (Stream_Element_Offset (Msg'Length)) := 16#80#;
      for I in 0 .. 7 loop
         B (Stream_Element_Offset (Tot - 1 - I)) :=
           Stream_Element (Shift_Right (ML, 8 * I) and 16#FF#);
      end loop;
      declare
         Off : Stream_Element_Offset := 0;
      begin
         while Off < B'Length loop
            declare
               W : Word_Array (0 .. 79);
               A, C, D, E, F, T : Unsigned_32; KK : Unsigned_32;
               BB : Unsigned_32;
            begin
               for I in 0 .. 15 loop
                  W (I) :=
                    Shift_Left  (Unsigned_32 (B (Off + Stream_Element_Offset (4*I))),   24) or
                    Shift_Left  (Unsigned_32 (B (Off + Stream_Element_Offset (4*I+1))), 16) or
                    Shift_Left  (Unsigned_32 (B (Off + Stream_Element_Offset (4*I+2))),  8) or
                                 Unsigned_32 (B (Off + Stream_Element_Offset (4*I+3)));
               end loop;
               for I in 16 .. 79 loop
                  W (I) := Rotl (W (I-3) xor W (I-8) xor W (I-14) xor W (I-16), 1);
               end loop;
               A := H0; BB := H1; C := H2; D := H3; E := H4;
               for I in 0 .. 79 loop
                  if I < 20 then F := (BB and C) or ((not BB) and D); KK := 16#5A827999#;
                  elsif I < 40 then F := BB xor C xor D; KK := 16#6ED9EBA1#;
                  elsif I < 60 then F := (BB and C) or (BB and D) or (C and D); KK := 16#8F1BBCDC#;
                  else F := BB xor C xor D; KK := 16#CA62C1D6#;
                  end if;
                  T := Rotl (A, 5) + F + E + KK + W (I);
                  E := D; D := C; C := Rotl (BB, 30); BB := A; A := T;
               end loop;
               H0 := H0 + A; H1 := H1 + BB; H2 := H2 + C; H3 := H3 + D; H4 := H4 + E;
            end;
            Off := Off + 64;
         end loop;
      end;
      declare
         Out_B : Stream_Element_Array (0 .. 19);
         Hs : constant array (0 .. 4) of Unsigned_32 := [H0, H1, H2, H3, H4];
      begin
         for J in 0 .. 4 loop
            for I in 0 .. 3 loop
               Out_B (Stream_Element_Offset (4*J + I)) :=
                 Stream_Element (Shift_Right (Hs (J), 8 * (3 - I)) and 16#FF#);
            end loop;
         end loop;
         return Out_B;
      end;
   end SHA1;

   ------------------------------------------------------------------
   -- Base64 (standard alphabet).
   ------------------------------------------------------------------
   function Base64 (Data : Stream_Element_Array) return String is
      Alpha : constant String :=
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
      Out_Len : constant Natural := 4 * ((Data'Length + 2) / 3);
      R : String (1 .. Out_Len);
      P : Natural := 0;
      I : Stream_Element_Offset := Data'First;
   begin
      while I <= Data'Last loop
         declare
            B0 : constant Unsigned_32 := Unsigned_32 (Data (I));
            B1 : constant Unsigned_32 :=
              (if I + 1 <= Data'Last then Unsigned_32 (Data (I + 1)) else 0);
            B2 : constant Unsigned_32 :=
              (if I + 2 <= Data'Last then Unsigned_32 (Data (I + 2)) else 0);
            N  : constant Unsigned_32 :=
              Shift_Left (B0, 16) or Shift_Left (B1, 8) or B2;
         begin
            R (P + 1) := Alpha (Natural (Shift_Right (N, 18) and 63) + 1);
            R (P + 2) := Alpha (Natural (Shift_Right (N, 12) and 63) + 1);
            R (P + 3) := (if I + 1 <= Data'Last
                          then Alpha (Natural (Shift_Right (N, 6) and 63) + 1) else '=');
            R (P + 4) := (if I + 2 <= Data'Last
                          then Alpha (Natural (N and 63) + 1) else '=');
            P := P + 4;
         end;
         I := I + 3;
      end loop;
      return R;
   end Base64;

   ------------------------------------------------------------------
   -- Socket byte I/O helpers.
   ------------------------------------------------------------------
   procedure Send_All (S : Socket_Type; Data : Stream_Element_Array) is
      First : Stream_Element_Offset := Data'First;
      Last  : Stream_Element_Offset;
   begin
      while First <= Data'Last loop
         Send_Socket (S, Data (First .. Data'Last), Last);
         exit when Last < First;     -- nothing sent => peer closed
         First := Last + 1;
      end loop;
   end Send_All;

   procedure Send_Str (S : Socket_Type; Str : String) is
      Buf : Stream_Element_Array (1 .. Str'Length);
   begin
      for I in Str'Range loop
         Buf (Stream_Element_Offset (I - Str'First + 1)) :=
           Stream_Element (Character'Pos (Str (I)));
      end loop;
      Send_All (S, Buf);
   end Send_Str;

   ------------------------------------------------------------------
   -- Static file serving.
   ------------------------------------------------------------------
   function Content_Type (Path : String) return String is
   begin
      if    Tail (Path, 5) = ".html" then return "text/html; charset=utf-8";
      elsif Tail (Path, 3) = ".js"   then return "text/javascript; charset=utf-8";
      elsif Tail (Path, 4) = ".css"  then return "text/css; charset=utf-8";
      elsif Tail (Path, 4) = ".svg"  then return "image/svg+xml";
      elsif Tail (Path, 5) = ".json" then return "application/json";
      else return "application/octet-stream";
      end if;
   end Content_Type;

   procedure Serve_File (S : Socket_Type; Rel_Path : String) is
      package SIO renames Ada.Streams.Stream_IO;
      --  Map "/" -> index.html and strip the leading slash; reject "..".
      Clean : constant String :=
        (if Rel_Path = "/" or else Rel_Path = "" then "index.html"
         else Rel_Path (Rel_Path'First + 1 .. Rel_Path'Last));
      Full  : constant String := Web_Dir & "/" & Clean;
   begin
      if Index (Clean, "..") /= 0 or else not Ada.Directories.Exists (Full) then
         Send_Str (S, "HTTP/1.1 404 Not Found" & ASCII.CR & ASCII.LF
                   & "Content-Length: 0" & ASCII.CR & ASCII.LF
                   & "Connection: close" & ASCII.CR & ASCII.LF & ASCII.CR & ASCII.LF);
         return;
      end if;
      declare
         F    : SIO.File_Type;
         Size : constant Natural := Natural (Ada.Directories.Size (Full));
         Buf  : Stream_Element_Array (1 .. Stream_Element_Offset'Max (1, Stream_Element_Offset (Size)));
         Last : Stream_Element_Offset;
      begin
         SIO.Open (F, SIO.In_File, Full);
         if Size > 0 then SIO.Read (F, Buf, Last); else Last := 0; end if;
         SIO.Close (F);
         Send_Str (S, "HTTP/1.1 200 OK" & ASCII.CR & ASCII.LF
                   & "Content-Type: " & Content_Type (Clean) & ASCII.CR & ASCII.LF
                   & "Content-Length:" & Natural'Image (Size) & ASCII.CR & ASCII.LF
                   & "Cache-Control: no-store" & ASCII.CR & ASCII.LF
                   & "Connection: close" & ASCII.CR & ASCII.LF & ASCII.CR & ASCII.LF);
         if Size > 0 then Send_All (S, Buf (1 .. Last)); end if;
      end;
   end Serve_File;

   ------------------------------------------------------------------
   -- Buffered reader over a client socket (for HTTP headers + WS frames).
   ------------------------------------------------------------------
   type Reader is record
      Sock : Socket_Type;
      Buf  : Stream_Element_Array (1 .. 65536);
      Len  : Stream_Element_Offset := 0;   -- bytes valid in Buf
      Pos  : Stream_Element_Offset := 1;   -- next byte to consume
   end record;

   --  Read one byte; raises Socket_Error on close.
   function Get_Byte (R : in out Reader) return Stream_Element is
   begin
      if R.Pos > R.Len then
         Receive_Socket (R.Sock, R.Buf, R.Len);
         if R.Len = 0 then raise Socket_Error; end if;
         R.Pos := 1;
      end if;
      return B : constant Stream_Element := R.Buf (R.Pos) do
         R.Pos := R.Pos + 1;
      end return;
   end Get_Byte;

   ------------------------------------------------------------------
   -- WebSocket framing.
   ------------------------------------------------------------------
   --  Send Data as a single unmasked binary frame (server->client).
   procedure WS_Send (S : Socket_Type; Data : Stream_Element_Array) is
      L : constant Natural := Data'Length;
      Hdr : Stream_Element_Array (1 .. 10);
      HL  : Stream_Element_Offset;
   begin
      Hdr (1) := 16#82#;                         -- FIN + binary opcode
      if L < 126 then
         Hdr (2) := Stream_Element (L); HL := 2;
      elsif L < 65536 then
         Hdr (2) := 126;
         Hdr (3) := Stream_Element (Shift_Right (Unsigned_32 (L), 8) and 16#FF#);
         Hdr (4) := Stream_Element (Unsigned_32 (L) and 16#FF#);
         HL := 4;
      else
         Hdr (2) := 127;
         for I in 0 .. 7 loop
            Hdr (3 + Stream_Element_Offset (I)) :=
              Stream_Element (Shift_Right (Unsigned_64 (L), 8 * (7 - I)) and 16#FF#);
         end loop;
         HL := 10;
      end if;
      Send_All (S, Hdr (1 .. HL));
      if L > 0 then Send_All (S, Data); end if;
   end WS_Send;

   ------------------------------------------------------------------
   -- HTTP request line + header parse.
   ------------------------------------------------------------------
   procedure Read_Headers (R : in out Reader; Method, Path : out Unbounded_String;
                           WS_Key : out Unbounded_String; Is_Upgrade : out Boolean) is
      Line : Unbounded_String;
      First_Line : Boolean := True;

      procedure Read_Line is
         C : Stream_Element;
      begin
         Line := Null_Unbounded_String;
         loop
            C := Get_Byte (R);
            exit when C = Character'Pos (ASCII.LF);
            if C /= Character'Pos (ASCII.CR) then
               Append (Line, Character'Val (Integer (C)));
            end if;
         end loop;
      end Read_Line;
   begin
      Method := Null_Unbounded_String; Path := Null_Unbounded_String;
      WS_Key := Null_Unbounded_String; Is_Upgrade := False;
      loop
         Read_Line;
         exit when Length (Line) = 0;            -- blank line = end of headers
         declare
            L : constant String := To_String (Line);
         begin
            if First_Line then
               First_Line := False;
               declare
                  Sp1 : constant Natural := Index (L, " ");
                  Sp2 : constant Natural := (if Sp1 > 0 then Index (L (Sp1 + 1 .. L'Last), " ") else 0);
               begin
                  if Sp1 > 0 then Method := To_Unbounded_String (L (L'First .. Sp1 - 1)); end if;
                  if Sp1 > 0 and then Sp2 > 0 then
                     Path := To_Unbounded_String (L (Sp1 + 1 .. Sp2 - 1));
                  end if;
               end;
            else
               declare
                  LowL : constant String := Translate (L, Ada.Strings.Maps.Constants.Lower_Case_Map);
               begin
                  if Index (LowL, "upgrade:") = LowL'First
                     and then Index (LowL, "websocket") /= 0
                  then
                     Is_Upgrade := True;
                  elsif Index (LowL, "sec-websocket-key:") = LowL'First then
                     declare
                        Colon : constant Natural := Index (L, ":");
                        V : constant String := Trim (L (Colon + 1 .. L'Last), Ada.Strings.Both);
                     begin
                        WS_Key := To_Unbounded_String (V);
                     end;
                  end if;
               end;
            end if;
         end;
      end loop;
   end Read_Headers;

   ------------------------------------------------------------------
   -- One client connection (its own task so static + WS coexist).
   ------------------------------------------------------------------
   task type Conn is
      entry Start (S : Socket_Type);
   end Conn;
   type Conn_Access is access Conn;

   task body Conn is
      Client : Socket_Type;
   begin
      accept Start (S : Socket_Type) do Client := S; end Start;
      declare
         R : Reader; Method, Path, WS_Key : Unbounded_String; Upgrade : Boolean;
      begin
         R.Sock := Client;
         Read_Headers (R, Method, Path, WS_Key, Upgrade);

         if To_String (Path) = "/serverkey" then
            Send_Str (Client, "HTTP/1.1 200 OK" & ASCII.CR & ASCII.LF
              & "Content-Type: text/plain" & ASCII.CR & ASCII.LF
              & "Content-Length:" & Natural'Image (Srv_Pub'Length) & ASCII.CR & ASCII.LF
              & "Connection: close" & ASCII.CR & ASCII.LF & ASCII.CR & ASCII.LF & Srv_Pub);
            Close_Socket (Client);

         elsif Upgrade and then To_String (Path) = "/ws" then
            --  Complete the WebSocket handshake.
            declare
               Accept_Key : constant String :=
                 Base64 (SHA1 (To_SEA (To_String (WS_Key) & WS_GUID)));
            begin
               Send_Str (Client, "HTTP/1.1 101 Switching Protocols" & ASCII.CR & ASCII.LF
                 & "Upgrade: websocket" & ASCII.CR & ASCII.LF
                 & "Connection: Upgrade" & ASCII.CR & ASCII.LF
                 & "Sec-WebSocket-Accept: " & Accept_Key & ASCII.CR & ASCII.LF
                 & ASCII.CR & ASCII.LF);
            end;
            --  Open the TCP connection to the secure_server and relay both
            --  directions from a SINGLE task via a selector. Sharing one socket
            --  between two tasks (one in Send, one in Receive) is not safe under
            --  GNAT.Sockets (it surfaces as EBADF on an idle connection), so we
            --  multiplex instead of spawning a pump task.
            declare
               Srv  : Socket_Type;
               Addr : constant Sock_Addr_Type :=
                 (Family_Inet, Inet_Addr (Srv_Host), Srv_Port);
               Sel  : Selector_Type;
               RSet, ESet : Socket_Set_Type;
               Status : Selector_Status;
               Buf  : Stream_Element_Array (1 .. 65536);
               Last : Stream_Element_Offset;
               WBuf : Stream_Element_Array (0 .. 262143);   -- client WS byte accumulator
               WLen : Natural := 0;
               Done : Boolean := False;

               --  Parse every COMPLETE WebSocket frame currently in WBuf,
               --  unmask it, and forward binary/text payloads to the server.
               procedure Pump_Frames is
               begin
                  loop
                     exit when WLen < 2;
                     declare
                        Op   : constant Unsigned_32 := Unsigned_32 (WBuf (0)) and 16#0F#;
                        Mskd : constant Boolean := (Unsigned_32 (WBuf (1)) and 16#80#) /= 0;
                        L7   : constant Natural := Natural (Unsigned_32 (WBuf (1)) and 16#7F#);
                        HLen : Natural := 2;
                        PLen : Natural := L7;
                     begin
                        if L7 = 126 then
                           exit when WLen < 4;
                           PLen := Natural (WBuf (2)) * 256 + Natural (WBuf (3));
                           HLen := 4;
                        elsif L7 = 127 then
                           exit when WLen < 10;
                           PLen := 0;
                           for K in 2 .. 9 loop
                              PLen := PLen * 256 + Natural (WBuf (Stream_Element_Offset (K)));
                           end loop;
                           HLen := 10;
                        end if;
                        declare
                           MLen  : constant Natural := (if Mskd then 4 else 0);
                           Total : constant Natural := HLen + MLen + PLen;
                           MOff  : constant Natural := HLen;
                           DOff  : constant Natural := HLen + MLen;
                        begin
                           exit when PLen > WBuf'Length;       -- frame too big to ever fit
                           exit when WLen < Total;             -- wait for more bytes
                           if (Op = 1 or else Op = 2) and then PLen > 0 then
                              declare
                                 Pay : Stream_Element_Array (0 .. Stream_Element_Offset (PLen) - 1);
                              begin
                                 for J in 0 .. PLen - 1 loop
                                    Pay (Stream_Element_Offset (J)) :=
                                      (if Mskd then
                                         Stream_Element (Unsigned_32 (WBuf (Stream_Element_Offset (DOff + J)))
                                            xor Unsigned_32 (WBuf (Stream_Element_Offset (MOff + (J mod 4)))))
                                       else WBuf (Stream_Element_Offset (DOff + J)));
                                 end loop;
                                 Send_All (Srv, Pay);
                              end;
                           elsif Op = 8 then
                              Done := True;
                           end if;
                           --  Drop the consumed frame from the accumulator.
                           for J in Total .. WLen - 1 loop
                              WBuf (Stream_Element_Offset (J - Total)) := WBuf (Stream_Element_Offset (J));
                           end loop;
                           WLen := WLen - Total;
                        end;
                     end;
                     exit when Done;
                  end loop;
               end Pump_Frames;
            begin
               Create_Socket (Srv);
               Connect_Socket (Srv, Addr);
               Create_Selector (Sel);

               --  Any bytes the client already pipelined after the HTTP headers
               --  are the first WS frames: seed the accumulator with them.
               while R.Pos <= R.Len loop
                  WBuf (Stream_Element_Offset (WLen)) := R.Buf (R.Pos);
                  WLen := WLen + 1; R.Pos := R.Pos + 1;
               end loop;
               Pump_Frames;

               while not Done loop
                  Empty (RSet); Empty (ESet);
                  Set (RSet, Client); Set (RSet, Srv);
                  Check_Selector (Sel, RSet, ESet, Status);
                  if Dbg then Put_Line ("  [sel] status=" & Status'Image
                    & " srv=" & Boolean'Image (Is_Set (RSet, Srv))
                    & " cli=" & Boolean'Image (Is_Set (RSet, Client))); Flush; end if;
                  exit when Status /= Completed;

                  if Is_Set (RSet, Srv) then               -- server -> browser
                     Receive_Socket (Srv, Buf, Last);
                     if Dbg then Put_Line ("  [srv->ws]" & Last'Image); Flush; end if;
                     exit when Last = 0;
                     WS_Send (Client, Buf (1 .. Last));
                  end if;

                  if Is_Set (RSet, Client) then            -- browser -> server
                     Receive_Socket (Client, Buf, Last);
                     if Dbg then Put_Line ("  [cli->srv]" & Last'Image); Flush; end if;
                     exit when Last = 0;
                     for I in 1 .. Last loop
                        if WLen < WBuf'Length then
                           WBuf (Stream_Element_Offset (WLen)) := Buf (I);
                           WLen := WLen + 1;
                        end if;
                     end loop;
                     Pump_Frames;
                  end if;
               end loop;

               Close_Selector (Sel);
               begin Close_Socket (Srv); exception when others => null; end;
            exception
               when others => begin Close_Socket (Srv); exception when others => null; end;
            end;
            begin Close_Socket (Client); exception when others => null; end;

         elsif To_String (Method) = "GET" then
            Serve_File (Client, To_String (Path));
            Close_Socket (Client);
         else
            Send_Str (Client, "HTTP/1.1 405 Method Not Allowed" & ASCII.CR & ASCII.LF
              & "Connection: close" & ASCII.CR & ASCII.LF & ASCII.CR & ASCII.LF);
            Close_Socket (Client);
         end if;
      exception
         when others =>
            begin Close_Socket (Client); exception when others => null; end;
      end;
   end Conn;

   Listener, Client : Socket_Type;
   Addr : Sock_Addr_Type;
begin
   if Argument_Count < 3 then
      Put_Line ("usage: ws_bridge <server_host> <server_port> <server_pub_hex> "
                & "[local_port] [web_dir]");
      return;
   end if;

   --  Bind 127.0.0.1 by default (safe for local use); a public demo deploy can
   --  set ASPIDA_BIND=0.0.0.0 to listen on all interfaces.
   declare
      Bind_Addr : constant String :=
        (if Ada.Environment_Variables.Exists ("ASPIDA_BIND")
         then Ada.Environment_Variables.Value ("ASPIDA_BIND") else "127.0.0.1");
   begin
      Create_Socket (Listener);
      Set_Socket_Option (Listener, Socket_Level, (Reuse_Address, True));
      Bind_Socket (Listener, (Family_Inet, Inet_Addr (Bind_Addr), Local_Port));
   end;
   Listen_Socket (Listener);
   Put_Line ("Aspida encrypted-chat web demo");
   Put_Line ("  open  http://<host>:" & Trim (Local_Port'Image, Ada.Strings.Both) & "/");
   Put_Line ("  relaying to secure_server at " & Srv_Host & ":"
             & Trim (Srv_Port'Image, Ada.Strings.Both));
   Put_Line ("  (the bridge sees ciphertext only; the browser does the crypto)");

   loop
      Accept_Socket (Listener, Client, Addr);
      declare
         H : constant Conn_Access := new Conn;
      begin
         H.Start (Client);
      end;
   end loop;
end WS_Bridge;
