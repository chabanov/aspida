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
with Ada.Exceptions;          use Ada.Exceptions;
with Ada.Environment_Variables;
with GNAT.OS_Lib;
with Ada.Streams;             use Ada.Streams;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;
with JSON; use JSON;
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

   Listener : Socket_Type;                   -- local HTTP listener (main only)
   Model_Name : constant String := "aspida";

   --  Concurrency: a pool of worker tasks, each serving one HTTP client at a
   --  time over its OWN server channel. A long generation or a stuck client on
   --  one worker no longer blocks the others (health checks, other users) — the
   --  single serialized channel used before was the head-of-line bottleneck.
   Max_Workers : constant := 16;             --  matches the server handler pool.
                                             --  More workers than the server
                                             --  generates at once (generation is
                                             --  serialized) means a burst of
                                             --  chats always leaves a free worker
                                             --  to serve /v1/models instantly.
   Max_Queue   : constant := 64;
   type Socket_Slots is array (1 .. Max_Queue) of Socket_Type;

   --  Raised when a write to the HTTP client fails because the client went
   --  away (killed curl, browser tab closed, gateway timeout). Propagating it
   --  lets the request loop tear down THIS request's server channel promptly,
   --  which makes the server's next token-send fail and abort generation —
   --  instead of silently draining the whole response into a dead socket.
   Client_Gone : exception;

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
         begin
            Send_Socket (S, Buf (Pos .. Buf'Last), Last);
         exception
            when Socket_Error =>          -- EPIPE / ECONNRESET: peer went away
               raise Client_Gone;
         end;
         if Last < Pos then               -- nothing sent -> peer gone
            raise Client_Gone;
         end if;
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
      --  Must match the caller's RBody size (12 MiB) — this internal read
      --  buffer, not RBody, was the real ceiling: it was left at 1 MiB when
      --  RBody was bumped to 12 MiB for image edits, so any img2img request
      --  carrying a base64 source image (~2-4 MiB) 413'd here even though the
      --  output buffer could hold it. Lives on the 48 MiB Worker stack.
      Data : String (1 .. 12_582_912);
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

   --  Each worker opens its OWN encrypted channel to the server per request and
   --  closes it after, so nothing is multiplexed and a request that aborts
   --  mid-stream can never leave a half-consumed record for another request to
   --  trip over (the single reused channel was the 2026-07-13 desync deadlock).
   --  Raises Channel_Failed if the server is unreachable.
   Channel_Failed : exception;

   procedure Open_Channel
     (Sock : out Socket_Type;
      Ch   : out Secure_Channel.Channel;
      CT   : aliased in out Socket_Transport.Sock_Transport)
   is
   begin
      Create_Socket (Sock);
      Connect_Socket (Sock, (Family_Inet, Inet_Addr (Argument (1)),
                             Port_Type'Value (Argument (2))));
      --  TCP_NODELAY on the Noise hop too: the server sends one tiny AEAD
      --  frame per generated token; Nagle on this localhost socket coalesces
      --  them so the whole reply arrives at once instead of streaming.
      begin
         Set_Socket_Option (Sock, IP_Protocol_For_TCP_Level, (No_Delay, True));
      exception when others => null;
      end;
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
         Rep : constant Crypto.Byte_Array :=
           Secure_Channel.Recv_Message (Ch, CT'Access);
         pragma Unreferenced (Rep);
      begin null; end;
   exception
      when others =>
         raise Channel_Failed;
   end Open_Channel;

   --  Tear down the current channel/socket (best effort). Each HTTP request
   --  now uses its OWN fresh channel (opened per request, closed here), so a
   --  request that aborts mid-stream can NEVER leave a half-consumed record
   --  for the next request to trip over — the single shared, reused channel
   --  was the source of the 2026-07-13 desync deadlock where the proxy waited
   --  for a response frame the server had already finished and moved past.
   procedure Close_Channel
     (Ch : in out Secure_Channel.Channel; Sock : Socket_Type) is
   begin
      begin Secure_Channel.Close (Ch); exception when others => null; end;
      begin Close_Socket (Sock); exception when others => null; end;
   end Close_Channel;

   --  Queue of accepted HTTP client sockets, drained by the worker pool. Same
   --  bounded ring the server uses; Put blocks if the pool is saturated
   --  (backpressure) rather than dropping clients.
   protected Accept_Queue is
      entry Put (S : Socket_Type);
      entry Get (S : out Socket_Type);
   private
      Slots : Socket_Slots;
      Cnt   : Natural := 0;
      Hd    : Positive := 1;
      Tl    : Positive := 1;
   end Accept_Queue;

   protected body Accept_Queue is
      entry Put (S : Socket_Type) when Cnt < Max_Queue is
      begin
         Slots (Tl) := S; Tl := Tl mod Max_Queue + 1; Cnt := Cnt + 1;
      end Put;
      entry Get (S : out Socket_Type) when Cnt > 0 is
      begin
         S := Slots (Hd); Hd := Hd mod Max_Queue + 1; Cnt := Cnt - 1;
      end Get;
   end Accept_Queue;

   --  Serve one HTTP client on its OWN server channel, then close both. All the
   --  channel state (Sock/Ch/CT) is local to this call, so concurrent workers
   --  never share it. The 12 MiB request buffers live on the worker task stack
   --  (see Worker's Storage_Size).
   procedure Serve (Client : Socket_Type) is
      Sock   : Socket_Type;
      CT     : aliased Socket_Transport.Sock_Transport;
      Ch     : Secure_Channel.Channel;
      Method : String (1 .. 16);
      Path   : String (1 .. 1024);
      --  12 MiB: a /v1/images edit request carries the reference image inline
      --  as a base64 data-uri (a 1024x1024 PNG is ~1-2 MiB encoded); the old
      --  1 MiB cap truncated those. Sized to the whole request, not per-image.
      RBody  : String (1 .. 12_582_912);
      ML, PL, BL, Err : Natural;
   begin
      Read_Request (Client, Method, Path, RBody, ML, PL, BL, Err);
      --  Fresh server channel per request: no state carries between clients,
      --  so a mid-stream abort can't desync anything. Raises Channel_Failed if
      --  the server is unreachable (handled below).
      Open_Channel (Sock, Ch, CT);
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
            elsif Ada.Strings.Fixed.Index (Pth, "/chat/completions") > 0
              or else Ada.Strings.Fixed.Index (Pth, "/api/chat") > 0
            then
               --  React to the server's reply type: Tag_Resp = one JSON;
               --  Tag_Token... / Tag_Done = streaming.
               --  Ollama-native /api/chat: convert the request body and emit the
               --  reply as newline-delimited JSON (message.content/.thinking)
               --  instead of SSE. The /v1 path is unchanged (Ollama_Native=False).
               declare
                  Ollama_Native : constant Boolean :=
                    Ada.Strings.Fixed.Index (Pth, "/api/chat") > 0;
                  NL      : constant String := [1 => ASCII.LF];
                  CType   : constant String :=
                    (if Ollama_Native then "application/x-ndjson" else "text/event-stream");
                  Streaming   : Boolean := False;
                  First       : Boolean := True;
                  --  Tracks which channel the next Tag_Token belongs to.
                  --  Reason = reasoning_content (① block); false = content.
                  In_Reasoning : Boolean := False;
                  --  Cached finish_reason from Tag_Finish_Reason; used by
                  --  Chat_Done_Chunk when Tag_Done arrives (Tag_Done carries
                  --  only usage + length flag, not the finish reason itself).
                  Finish_Reason : Ada.Strings.Unbounded.Unbounded_String :=
                    Ada.Strings.Unbounded.To_Unbounded_String ("stop");
               begin
                  declare
                     Send_Body : constant String :=
                       (if Ollama_Native then OpenAI.Ollama_Body_To_OpenAI (Bdy) else Bdy);
                  begin
                     Secure_Channel.Send_Message
                       (Ch, CT'Access, Frame (Protocol.Tag_Chat, Send_Body));
                  end;
                  loop
                     declare
                        Rec : constant Crypto.Byte_Array :=
                          Secure_Channel.Recv_Message (Ch, CT'Access);
                     begin
                        exit when Rec'Length = 0;
                        if Rec (Rec'First) = Protocol.Tag_Resp then
                           if Ollama_Native then
                              Write_JSON (Client, "200 OK",
                                OpenAI.Ollama_Response_From_OpenAI
                                  (Body_Of (Rec), Model_Name));
                           else
                              Write_JSON (Client, "200 OK", Body_Of (Rec));
                           end if;
                           exit;
                        elsif Rec (Rec'First) = Protocol.Tag_Reasoning_Begin then
                           if not Streaming then
                              Sock_Write (Client, "HTTP/1.1 200 OK" & CRLF
                                & "Content-Type: " & CType & CRLF
                                & "Cache-Control: no-cache" & CRLF
                                & "Access-Control-Allow-Origin: *" & CRLF
                                & "Connection: close" & CRLF & CRLF);
                              Streaming := True;
                           end if;
                           --  Opening a reasoning channel: from now on
                           --  Tag_Token pieces populate delta.reasoning_content.
                           In_Reasoning := True;
                        elsif Rec (Rec'First) = Protocol.Tag_Tool_Call then
                           if not Streaming then
                              Sock_Write (Client, "HTTP/1.1 200 OK" & CRLF
                                & "Content-Type: " & CType & CRLF
                                & "Cache-Control: no-cache" & CRLF
                                & "Access-Control-Allow-Origin: *" & CRLF
                                & "Connection: close" & CRLF & CRLF);
                              Streaming := True;
                           end if;
                           declare
                              JSON_Body : constant String := Body_Of (Rec);
                              Idx       : constant JSON.Value_Ref :=
                                (if JSON_Body'Length > 0
                                 then JSON.Parse (JSON_Body)
                                 else null);
                           begin
                              if Idx /= null
                                and then JSON.Exists (JSON.Get (Idx, "id"))
                                and then JSON.Exists (JSON.Get (Idx, "name"))
                                and then JSON.Exists (JSON.Get (Idx, "arguments"))
                              then
                                 if Ollama_Native then
                                    Sock_Write (Client,
                                      OpenAI.Ollama_Tool_Chunk
                                        (Model_Name,
                                         JSON.As_String (JSON.Get (Idx, "id")),
                                         JSON.As_String (JSON.Get (Idx, "name")),
                                         JSON.As_String (JSON.Get (Idx, "arguments"))) & NL);
                                 else
                                    declare
                                       --  0-based stream index from the parser's
                                       --  "tc_N" id (Id_For, N >= 1). Parallel
                                       --  calls MUST carry distinct indexes or
                                       --  spec-compliant clients merge their
                                       --  argument deltas into invalid JSON.
                                       Id_S : constant String :=
                                         JSON.As_String (JSON.Get (Idx, "id"));
                                       Ord  : Natural := 0;
                                    begin
                                       if Id_S'Length > 3
                                         and then Id_S (Id_S'First .. Id_S'First + 2) = "tc_"
                                       then
                                          begin
                                             Ord := Natural'Value
                                               (Id_S (Id_S'First + 3 .. Id_S'Last)) - 1;
                                          exception
                                             when others => Ord := 0;
                                          end;
                                       end if;
                                       Sock_Write (Client, "data: "
                                         & OpenAI.Tool_Call_Chunk
                                             (Model_Name,
                                              Id_S,
                                              JSON.As_String (JSON.Get (Idx, "name")),
                                              JSON.As_String (JSON.Get (Idx, "arguments")),
                                              Ord)
                                         & CRLF & CRLF);
                                    end;
                                 end if;
                              end if;
                           end;
                        elsif Rec (Rec'First) = Protocol.Tag_Token then
                           if not Streaming then
                              Sock_Write (Client, "HTTP/1.1 200 OK" & CRLF
                                & "Content-Type: " & CType & CRLF
                                & "Cache-Control: no-cache" & CRLF
                                & "Access-Control-Allow-Origin: *" & CRLF
                                & "Connection: close" & CRLF & CRLF);
                              Streaming := True;
                           end if;
                           if Ollama_Native then
                              Sock_Write (Client,
                                OpenAI.Ollama_Chunk
                                  (Model_Name, Body_Of (Rec), In_Reasoning) & NL);
                           else
                              Sock_Write (Client, "data: "
                                & OpenAI.Chat_Chunk
                                    (Model_Name, Body_Of (Rec), First, In_Reasoning)
                                & CRLF & CRLF);
                           end if;
                           First := False;
                           --  A reasoning block typically ends before the
                           --  next event. The natural signal is the
                           --  first non-reasoning Tag_Token; we treat the
                           --  explicit Tag_Text_Begin marker as authoritative.
                        elsif Rec (Rec'First) = Protocol.Tag_Text_Begin then
                           In_Reasoning := False;
                        elsif Rec (Rec'First) = Protocol.Tag_Finish_Reason then
                           Finish_Reason :=
                             Ada.Strings.Unbounded.To_Unbounded_String
                               (Body_Of (Rec));
                        elsif Rec (Rec'First) = Protocol.Tag_Done then
                           if not Streaming then
                              Sock_Write (Client, "HTTP/1.1 200 OK" & CRLF
                                & "Content-Type: " & CType & CRLF
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
                              --  Honour the explicit finish_reason when
                              --  present; otherwise map the truncated flag
                              --  to "length".
                              declare
                                 Effective : constant String :=
                                   (if Trunc then "length"
                                    else Ada.Strings.Unbounded.To_String (Finish_Reason));
                              begin
                                 if Ollama_Native then
                                    Sock_Write (Client,
                                      OpenAI.Ollama_Done_Chunk
                                        (Model_Name, PT, CT, Effective) & NL);
                                 else
                                    Sock_Write (Client, "data: "
                                      & OpenAI.Chat_Done_Chunk
                                          (Model_Name, PT, CT, Effective)
                                      & CRLF & CRLF);
                                 end if;
                              end;
                           end;
                           if not Ollama_Native then
                              Sock_Write (Client, "data: [DONE]" & CRLF & CRLF);
                           end if;
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

            elsif Ada.Strings.Fixed.Index (Pth, "/images") > 0 then
               Secure_Channel.Send_Message (Ch, CT'Access, Frame (Protocol.Tag_Image, Bdy));
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
         Close_Channel (Ch, Sock);
         begin Close_Socket (Client); exception when others => null; end;
   exception
      when Channel_Failed =>
         --  Could not open a channel for this request: the backend is down or
         --  saturated. Tell the client and return — a single failed request
         --  must not take a worker down (the health probe restarts the service
         --  if the backend stays unreachable).
         begin
            Write_JSON (Client, "503 Service Unavailable",
              OpenAI.Error_Response ("backend unavailable", "backend_unavailable"));
         exception when others => null; end;
         begin Close_Socket (Client); exception when others => null; end;
      when others =>
         --  Client vanished mid-stream, or a per-request error. Drop this
         --  request's channel (which aborts the server-side generation) and its
         --  client socket. No shared state survives, so nothing to resync.
         Close_Channel (Ch, Sock);
         begin Close_Socket (Client); exception when others => null; end;
   end Serve;

   --  Worker pool. Each worker serves one client at a time on its own channel,
   --  so a slow generation or a stuck client on one worker never blocks the
   --  others. Big stack: Serve holds a 12 MiB request buffer (image edits).
   task type Worker with Storage_Size => 48 * 1024 * 1024;
   task body Worker is
      Client : Socket_Type;
   begin
      loop
         Accept_Queue.Get (Client);
         --  A worker must never die (its slot would be lost). Serve cleans up
         --  after itself; swallow anything that still escapes as a last resort.
         begin
            Serve (Client);
         exception
            when others =>
               begin Close_Socket (Client); exception when others => null; end;
         end;
      end loop;
   end Worker;

   Workers : array (1 .. Max_Workers) of Worker;
   pragma Unreferenced (Workers);

begin
   if Argument_Count < 3 then
      Put_Line ("usage: openai_proxy <host> <port> <server_pub_hex> [local_port]");
      GNAT.OS_Lib.OS_Exit (2);
   end if;

   --  Validate connectivity + report the cipher suite once at startup on a
   --  throwaway channel, then close it: real serving opens one per request.
   declare
      VSock : Socket_Type;
      VCT   : aliased Socket_Transport.Sock_Transport;
      VCh   : Secure_Channel.Channel;
   begin
      Open_Channel (VSock, VCh, VCT);
      Close_Channel (VCh, VSock);
   exception
      when Channel_Failed =>
         Put_Line ("error: cannot reach the secure server");
         GNAT.OS_Lib.OS_Exit (1);
   end;

   Put_Line ("  🔒 encrypted tunnel up: " & Secure_Channel.Cipher_Suite);
   Put_Line ("  OpenAI-compatible endpoint: http://127.0.0.1:"
             & Ada.Strings.Fixed.Trim (Local_Port'Image, Ada.Strings.Both) & "/v1");
   Put_Line ("  (any api_key; requests are AEAD-sealed to the pinned server;"
             & Max_Workers'Image & " workers)");

   --  Local HTTP listener, loopback only. Bind can transiently fail with
   --  EADDRINUSE right after a restart while a predecessor's listen socket is
   --  still in TIME_WAIT (or, historically, while a wedged predecessor still
   --  held the port). Retry a few times; note this is the one bind that used to
   --  escape unguarded — an EADDRINUSE here propagated to the runtime and hung
   --  the process in finalize_global_tasks (it links non-terminating library
   --  tasks), leaving the port held forever. The top-level handler below now
   --  turns any escape into an immediate OS_Exit instead.
   Create_Socket (Listener);
   Set_Socket_Option (Listener, Socket_Level, (Reuse_Address, True));
   declare
      Bound : Boolean := False;
   begin
      for Attempt in 1 .. 20 loop
         begin
            Bind_Socket
              (Listener, (Family_Inet, Inet_Addr ("127.0.0.1"), Local_Port));
            Bound := True;
            exit;
         exception
            when Socket_Error =>
               if Attempt = 20 then
                  raise;
               end if;
               delay 0.5;   --  let a predecessor's socket drain
         end;
      end loop;
      pragma Assert (Bound);
   end;
   Listen_Socket (Listener);

   --  Accept and hand each client to a worker. The acceptor itself does no I/O
   --  on the client, so it never blocks — a burst of clients queues (bounded)
   --  and drains across the pool.
   loop
      declare
         Addr : Sock_Addr_Type;
         C    : Socket_Type;
      begin
         Accept_Socket (Listener, C, Addr);
         --  Bound how long a worker may block on this client. Without a timeout
         --  a slow/stuck-but-open peer (one that stops reading our stream, or
         --  never finishes sending its request) pins a worker forever in
         --  Sock_Write / Read_Request — dead peers abort fast via RST, but a
         --  wedged reader does not. Enough of those exhaust the pool. On the
         --  loopback hop 30s is far longer than any healthy request or stream
         --  chunk; on expiry Send/Receive raise Socket_Error, which Sock_Write
         --  turns into Client_Gone and Read_Request propagates — freeing the
         --  worker either way.
         begin
            Set_Socket_Option (C, Socket_Level, (Send_Timeout, 30.0));
            Set_Socket_Option (C, Socket_Level, (Receive_Timeout, 30.0));
            --  TCP_NODELAY: SSE streams write one tiny "data: {token}" line per
            --  generated token. Without disabling Nagle the OS coalesces these
            --  small writes and flushes them in bursts, so the client sees the
            --  whole reply arrive at once instead of token-by-token. Disable
            --  Nagle so each streamed chunk goes out immediately.
            Set_Socket_Option
              (C, IP_Protocol_For_TCP_Level, (No_Delay, True));
         exception
            when others => null;
         end;
         Accept_Queue.Put (C);
      exception
         when others =>
            --  A bad accept must not kill the acceptor loop.
            null;
      end;
   end loop;
exception
   --  Last line of defence. ANY exception that reaches here (the listener
   --  Bind's EADDRINUSE, etc.) must NOT unwind into finalize_global_tasks —
   --  that blocks forever on this binary's non-terminating library tasks and
   --  leaves :LOCAL_PORT held, which is the shape of the 2026-07-13 proxy hang.
   --  Log and force-exit so systemd restarts a clean instance.
   when E : others =>
      Put_Line ("proxy fatal: " & Exception_Name (E) & ": " & Exception_Message (E));
      Flush;   --  OS_Exit skips finalization; flush so the reason hits the log
      GNAT.OS_Lib.OS_Exit (1);
end OpenAI_Proxy;
