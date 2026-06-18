---------------------------------------------------------------------
-- Secure_Server — concurrent encrypted chat server.
--
-- Loads the model once (shared, read-only — backend chosen from the GGUF
-- architecture: Llama / Qwen / Gemma) and serves many clients at the same
-- time: a pool of handler tasks each take a connection from a bounded
-- queue, run the handshake + session, and chat. I/O, handshakes and
-- persistence run in parallel across connections; the engine interleaves
-- concurrent generations per step (and the Llama backend batches them
-- through a shared forward pass). Infer_Lock is now a no-op kept for
-- structure — serialization is handled inside the engine.
--
-- Usage:  QWEN_MODEL_PATH=<any-supported-gguf> ./obj/secure_server [port]
--         (the env var name is historical; it accepts any supported GGUF)
---------------------------------------------------------------------

with Ada.Text_IO;             use Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings.Unbounded;   use Ada.Strings.Unbounded;
with Ada.Environment_Variables;
with Ada.Streams.Stream_IO;
with Ada.Directories;
with Ada.Real_Time;
with Ada.Exceptions;          use Ada.Exceptions;
with Interfaces;              use Interfaces;
with Interfaces.C;
with GNAT.Sockets;            use GNAT.Sockets;
with GNAT.OS_Lib;
with Crypto;                  use Crypto;
with Crypto.X25519;
with Crypto.Random;
with Crypto.Memory;
with Secure_Channel;
with Socket_Transport;
with Session_Store;
with Encrypting_Sink;
with Protocol;
with LLM_Qwen;
with LLM_Engine;
with LLM_Catalog;
with LLM_Sampler;
with OpenAI;

procedure Secure_Server is

   package SIO renames Ada.Streams.Stream_IO;

   Key_File     : constant String := "server_key.bin";
   Pub_File     : constant String := "server_pub.hex";
   Sel_File     : constant String := "active_model";  -- persisted model choice
   Reload_Code  : constant := 75;   -- exit code: "switch model, supervisor reloads"
   Default_Port : constant := 8765;
   Max_Clients  : constant := 8;     -- concurrent handler tasks
   Max_Queue    : constant := 64;    -- pending-connection backlog

   --  Generation cap per turn. Default 256; override with ASPIDA_MAX_TOKENS
   --  (e.g. a small value keeps a CPU demo snappy by bounding worst-case
   --  latency — short answers still stop early on EOS).
   function Max_Reply_Tokens return Integer is
   begin
      if Ada.Environment_Variables.Exists ("ASPIDA_MAX_TOKENS") then
         begin
            return Integer'Max (1, Integer'Value
              (Ada.Environment_Variables.Value ("ASPIDA_MAX_TOKENS")));
         exception when others => null;
         end;
      end if;
      return 2048;   -- generous default; a length-bound deploy sets ASPIDA_MAX_TOKENS
   end Max_Reply_Tokens;

   --  How long a connection may stay silent before its handler reclaims the
   --  slot (anti-DoS: a client that handshakes then never sends must not pin a
   --  handler task forever). Generous so a user thinking between turns is fine;
   --  override with ASPIDA_IDLE_TIMEOUT (seconds).
   function Idle_Timeout return Duration is
   begin
      if Ada.Environment_Variables.Exists ("ASPIDA_IDLE_TIMEOUT") then
         begin
            return Duration'Value
              (Ada.Environment_Variables.Value ("ASPIDA_IDLE_TIMEOUT"));
         exception
            when others => null;   -- malformed -> fall through to default
         end;
      end if;
      return 600.0;
   end Idle_Timeout;

   function Env_Int (Name : String; D : Integer) return Integer is
   begin
      if Ada.Environment_Variables.Exists (Name) then
         return Integer'Value (Ada.Environment_Variables.Value (Name));
      end if;
      return D;
   exception when others => return D; end Env_Int;

   --  Anti-DoS: cap NEW connections to Rate_Max per Rate_Window seconds
   --  (global, across all sources). Default 0 = disabled, so the bridge-fronted
   --  demo (every peer is 127.0.0.1) is unaffected unless an operator opts in
   --  on a directly-exposed deployment. Tunable via ASPIDA_RATE_MAX /
   --  ASPIDA_RATE_WINDOW.
   Rate_Max    : constant Integer  := Env_Int ("ASPIDA_RATE_MAX", 0);
   Rate_Window : constant Duration :=
     Duration (Integer'Max (1, Env_Int ("ASPIDA_RATE_WINDOW", 1)));

   --  Best-effort: restrict a sensitive file to owner read/write (0600). Used
   --  for the server's static private key, whose leak is full impersonation.
   procedure Set_Owner_Only (Path : String) is
      use Interfaces.C;
      function C_Chmod (P : char_array; Mode : int) return int
        with Import, Convention => C, External_Name => "chmod";
      Discard : constant int := C_Chmod (To_C (Path), 8#600#);
      pragma Unreferenced (Discard);   -- best-effort; failure is non-fatal
   begin
      null;
   end Set_Owner_Only;

   --  Optional shared-secret client authentication. Empty (unset) => disabled,
   --  preserving today's open behaviour (and the bridge-fronted demo). When
   --  set, every client must present this token before a session begins.
   function Client_Token return String is
   begin
      if Ada.Environment_Variables.Exists ("ASPIDA_CLIENT_TOKEN") then
         return Ada.Environment_Variables.Value ("ASPIDA_CLIENT_TOKEN");
      end if;
      return "";
   end Client_Token;

   Required_Token : constant String  := Client_Token;
   Auth_Required  : constant Boolean := Required_Token'Length > 0;

   function To_Bytes (S : String) return Byte_Array is
      B : Byte_Array (1 .. S'Length);
   begin
      for I in 1 .. S'Length loop
         B (I) := U8 (Character'Pos (S (S'First + I - 1)));
      end loop;
      return B;
   end To_Bytes;

   Token_Bytes : constant Byte_Array := To_Bytes (Required_Token);

   --  Generation sampling, configured via environment (default = greedy, so
   --  behaviour is unchanged unless explicitly opted in):
   --    ASPIDA_TEMP, ASPIDA_TOP_P, ASPIDA_TOP_K,
   --    ASPIDA_REPEAT_PENALTY, ASPIDA_REPEAT_LAST_N, ASPIDA_SEED.
   function Sampling_Cfg return LLM_Sampler.Params is
      P : LLM_Sampler.Params := LLM_Sampler.Greedy;
      function FEnv (Name : String; D : Float) return Float is
      begin
         if Ada.Environment_Variables.Exists (Name) then
            return Float'Value (Ada.Environment_Variables.Value (Name));
         end if;
         return D;
      exception when others => return D; end FEnv;
      function IEnv (Name : String; D : Integer) return Integer is
      begin
         if Ada.Environment_Variables.Exists (Name) then
            return Integer'Value (Ada.Environment_Variables.Value (Name));
         end if;
         return D;
      exception when others => return D; end IEnv;
   begin
      P.Temperature    := FEnv ("ASPIDA_TEMP", 0.0);
      P.Top_P          := FEnv ("ASPIDA_TOP_P", 1.0);
      P.Top_K          := IEnv ("ASPIDA_TOP_K", 0);
      P.Repeat_Penalty := FEnv ("ASPIDA_REPEAT_PENALTY", 1.0);
      P.Repeat_Last_N  := IEnv ("ASPIDA_REPEAT_LAST_N", 64);
      P.Seed           := Long_Long_Integer (IEnv ("ASPIDA_SEED", 0));
      return P;
   end Sampling_Cfg;

   Sampler_Cfg : constant LLM_Sampler.Params := Sampling_Cfg;

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

   function Load_Or_Create_Key return Crypto.X25519.Key_256 is
      F : SIO.File_Type;
      K : Crypto.X25519.Key_256;
   begin
      if Ada.Directories.Exists (Key_File) then
         SIO.Open (F, SIO.In_File, Key_File);
         Crypto.X25519.Key_256'Read (SIO.Stream (F), K);
         SIO.Close (F);
      else
         Crypto.Random.Fill (K);
         SIO.Create (F, SIO.Out_File, Key_File);
         Crypto.X25519.Key_256'Write (SIO.Stream (F), K);
         SIO.Close (F);
      end if;
      --  Lock down perms whether freshly created or pre-existing (older keys
      --  may have been written world-readable).
      Set_Owner_Only (Key_File);
      return K;
   end Load_Or_Create_Key;

   --  A model switch is only honored when a supervisor is present to restart
   --  the process (e.g. `make serve`, which sets ASPIDA_AUTORELOAD). Without
   --  it, a selection is persisted but applies on the next manual start.
   function Autoreload return Boolean is
     (Ada.Environment_Variables.Exists ("ASPIDA_AUTORELOAD"));

   --  The runtime model selection persisted by a previous Tag_Select ("" if
   --  none / unreadable). First non-empty line of Sel_File.
   function Read_Selected return String is
      F : Ada.Text_IO.File_Type;
   begin
      if not Ada.Directories.Exists (Sel_File) then
         return "";
      end if;
      Ada.Text_IO.Open (F, Ada.Text_IO.In_File, Sel_File);
      return L : constant String :=
        (if Ada.Text_IO.End_Of_File (F) then "" else Ada.Text_IO.Get_Line (F))
      do
         Ada.Text_IO.Close (F);
      end return;
   exception
      when others => return "";
   end Read_Selected;

   procedure Write_Selected (Path : String) is
      F : Ada.Text_IO.File_Type;
   begin
      Ada.Text_IO.Create (F, Ada.Text_IO.Out_File, Sel_File);
      Ada.Text_IO.Put_Line (F, Path);
      Ada.Text_IO.Close (F);
   exception
      when others => null;   -- non-fatal: selection just won't persist
   end Write_Selected;

   --  Resolve the active model: explicit env wins (deployments pin it), then a
   --  persisted runtime selection, then a built-in default.
   function Model_Path return String is
   begin
      if Ada.Environment_Variables.Exists ("QWEN_MODEL_PATH") then
         return Ada.Environment_Variables.Value ("QWEN_MODEL_PATH");
      end if;
      declare
         Sel : constant String := Read_Selected;
      begin
         if Sel'Length > 0 and then Ada.Directories.Exists (Sel) then
            return Sel;
         end if;
      end;
      return "/Users/ceo/.lmstudio/models/HauhauCS/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf";
   end Model_Path;

   --  Assigned in the main body (guarded), NOT at elaboration: a bad/corrupt/
   --  unsupported model must yield a clean operator error and a non-zero exit,
   --  not an unhandled exception before main. Handler tasks block on the empty
   --  connection queue until the accept loop runs (post-load), so they never
   --  observe Model before it is set.
   Model    : LLM_Engine.Engine;
   Secret   : constant Crypto.X25519.Key_256 := Load_Or_Create_Key;
   Listener : Socket_Type;

   --  Generations no longer serialize as a whole. Correctness (shared GPU
   --  buffers + the all-core worker pool) is now enforced per forward step by
   --  LLM_Step_Lock, so concurrent sessions INTERLEAVE token-by-token instead
   --  of one waiting behind the other. These stay as no-ops so the handler's
   --  existing acquire/release structure is unchanged. (Concurrency is still
   --  bounded by Max_Clients connection slots.)
   protected Infer_Lock is
      procedure Acquire;
      procedure Release;
   end Infer_Lock;

   protected body Infer_Lock is
      procedure Acquire is
      begin
         null;
      end Acquire;
      procedure Release is
      begin
         null;
      end Release;
   end Infer_Lock;

   --  Global new-connection rate limiter (see Rate_Max above). Disabled when
   --  Rate_Max <= 0.
   protected Rate_Limiter is
      procedure Check (Ok : out Boolean);
   private
      Window_Start : Ada.Real_Time.Time := Ada.Real_Time.Time_First;
      Count        : Natural := 0;
      Started      : Boolean := False;
   end Rate_Limiter;

   protected body Rate_Limiter is
      procedure Check (Ok : out Boolean) is
         use Ada.Real_Time;
         Now : constant Time := Clock;
      begin
         if Rate_Max <= 0 then
            Ok := True;
            return;
         end if;
         if not Started
           or else To_Duration (Now - Window_Start) >= Rate_Window
         then
            Window_Start := Now;
            Count        := 0;
            Started      := True;
         end if;
         Count := Count + 1;
         Ok := Count <= Rate_Max;
      end Check;
   end Rate_Limiter;

   --  Bounded hand-off of accepted sockets to the handler pool.
   type Socket_Slots is array (1 .. Max_Queue) of Socket_Type;
   protected Conn_Queue is
      entry Put (S : Socket_Type);
      entry Get (S : out Socket_Type);
   private
      Slots : Socket_Slots;
      Cnt   : Natural := 0;
      Hd    : Positive := 1;
      Tl    : Positive := 1;
   end Conn_Queue;

   protected body Conn_Queue is
      entry Put (S : Socket_Type) when Cnt < Max_Queue is
      begin
         Slots (Tl) := S; Tl := Tl mod Max_Queue + 1; Cnt := Cnt + 1;
      end Put;
      entry Get (S : out Socket_Type) when Cnt > 0 is
      begin
         S := Slots (Hd); Hd := Hd mod Max_Queue + 1; Cnt := Cnt - 1;
      end Get;
   end Conn_Queue;

   --  At most one live connection per session id (the encrypted history file
   --  and the conversation context must not be raced by two connections).
   type Id_Array is array (1 .. Max_Clients) of Unbounded_String;
   protected Active_Sessions is
      procedure Acquire (Id : String; Ok : out Boolean);
      procedure Release (Id : String);
   private
      Ids : Id_Array;
      Cnt : Natural := 0;
   end Active_Sessions;

   protected body Active_Sessions is
      procedure Acquire (Id : String; Ok : out Boolean) is
      begin
         for I in 1 .. Cnt loop
            if To_String (Ids (I)) = Id then
               Ok := False; return;            -- already in use
            end if;
         end loop;
         if Cnt < Max_Clients then
            Cnt := Cnt + 1; Ids (Cnt) := To_Unbounded_String (Id); Ok := True;
         else
            Ok := False;                        -- registry full
         end if;
      end Acquire;
      procedure Release (Id : String) is
      begin
         for I in 1 .. Cnt loop
            if To_String (Ids (I)) = Id then
               Ids (I) := Ids (Cnt); Cnt := Cnt - 1; return;
            end if;
         end loop;
      end Release;
   end Active_Sessions;

   ------------------------------------------------------------------
   -- One client connection: handshake -> session -> chat turns.
   ------------------------------------------------------------------
   procedure Handle_Connection (Conn : Socket_Type) is
      ST    : aliased Socket_Transport.Sock_Transport;
      Ch    : aliased Secure_Channel.Channel;
      Store : Session_Store.Store;
      Id_B  : Byte_Array (0 .. 7);
      Sid   : Unbounded_String;          -- acquired session id (for Release)

      Auth_Failed : exception;

      --  Consume an optional leading Tag_Auth record (validating it against the
      --  configured token, in constant time), then return the real session
      --  hello. Works in every on/off combination: a server without a token
      --  just ignores a leading auth record; one with a token rejects any
      --  client that omits it or sends the wrong one.
      function Get_Session_Hello return Byte_Array is
         procedure Reject (Msg : String) is
            Err : Byte_Array (0 .. Msg'Length);
         begin
            Err (0) := Protocol.Tag_Error;
            for I in Msg'Range loop
               Err (I - Msg'First + 1) := U8 (Character'Pos (Msg (I)));
            end loop;
            Secure_Channel.Send_Message (Ch, ST'Access, Err);
         end Reject;
         First : constant Byte_Array := Secure_Channel.Recv_Message (Ch, ST'Access);
      begin
         if First'Length >= 1 and then First (First'First) = Protocol.Tag_Auth then
            if Auth_Required then
               declare
                  Tok : Byte_Array (1 .. First'Length - 1);
               begin
                  for I in Tok'Range loop
                     Tok (I) := First (First'First + I);
                  end loop;
                  if not Crypto.Const_Time_Equal (Tok, Token_Bytes) then
                     Reject ("authentication failed");
                     raise Auth_Failed;
                  end if;
               end;
            end if;
            --  token record consumed; the real session hello is the next one
            return Secure_Channel.Recv_Message (Ch, ST'Access);
         elsif Auth_Required then
            Reject ("authentication required");
            raise Auth_Failed;
         else
            return First;
         end if;
      end Get_Session_Hello;
   begin
      ST.Sock := Conn;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Secret);

      --  First record selects/creates the session (Tag_Session + id;
      --  empty id = new). Reply with the assigned id.
      declare
         Hello : constant Byte_Array := Get_Session_Hello;
         Want  : Unbounded_String;
         Ok    : Boolean;
      begin
         if Hello'Length >= 1 and then Hello (Hello'First) = Protocol.Tag_Session then
            for I in Hello'First + 1 .. Hello'Last loop
               Append (Want, Character'Val (Integer (Hello (I))));
            end loop;
         end if;
         if Length (Want) = 0 then
            Crypto.Random.Fill (Id_B);
            Want := To_Unbounded_String (Hex (Id_B));
         elsif not Session_Store.Valid_Id (To_String (Want)) then
            --  Client-supplied id reaches the filesystem path; reject anything
            --  that isn't a safe [A-Za-z0-9_-]{1,64} token (path-traversal).
            declare
               Msg : constant String := "invalid session id";
               Err : Byte_Array (0 .. Msg'Length);
            begin
               Err (0) := Protocol.Tag_Error;
               for I in Msg'Range loop
                  Err (I - Msg'First + 1) := U8 (Character'Pos (Msg (I)));
               end loop;
               Secure_Channel.Send_Message (Ch, ST'Access, Err);
            end;
            Put_Line ("  [rejected] invalid session id");
            Secure_Channel.Close (Ch);
            Close_Socket (Conn);
            return;
         end if;

         --  Refuse a second live connection to the same session.
         Active_Sessions.Acquire (To_String (Want), Ok);
         if not Ok then
            declare
               Msg : constant String := "session busy or server full";
               Err : Byte_Array (0 .. Msg'Length);
            begin
               Err (0) := Protocol.Tag_Error;
               for I in Msg'Range loop
                  Err (I - Msg'First + 1) := U8 (Character'Pos (Msg (I)));
               end loop;
               Secure_Channel.Send_Message (Ch, ST'Access, Err);
            end;
            Put_Line ("  [" & To_String (Want) & "] rejected (busy/full)");
            Secure_Channel.Close (Ch);
            Close_Socket (Conn);
            return;
         end if;
         Sid := Want;

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
         Put_Line ("  [" & To_String (Want) & "] connected, resumed turns:"
           & Session_Store.Turn_Count (Store)'Image);
      end;

      loop
         declare
            Req : constant Byte_Array := Secure_Channel.Recv_Message (Ch, ST'Access);
            Tag : constant Crypto.U8 :=
              (if Req'Length >= 1 then Req (Req'First) else 0);
            Sink : aliased Encrypting_Sink.Enc_Sink :=
              (LLM_Qwen.Token_Sink with
                 Ch => Ch'Unchecked_Access, T => ST'Unchecked_Access);
            Prompt : String (1 .. Integer'Max (0, Req'Length - 1));

            --  Frame Tag + text into one AEAD record and send it.
            procedure Send_Tagged (T : Crypto.U8; S : String) is
               Out_B : Byte_Array (0 .. S'Length);
            begin
               Out_B (0) := T;
               for I in 1 .. S'Length loop
                  Out_B (I) := Crypto.U8 (Character'Pos (S (S'First + I - 1)));
               end loop;
               Secure_Channel.Send_Message (Ch, ST'Access, Out_B);
            end Send_Tagged;
         begin
            for I in Prompt'Range loop
               Prompt (I) := Character'Val (Integer (Req (Req'First + I)));
            end loop;

            --  OpenAI-compatible routes (tunneled JSON). The proxy maps HTTP.
            if Tag = Protocol.Tag_Models then
               --  Full catalog of every model on this system + which is
               --  active, so a client can show a picker.
               Send_Tagged (Protocol.Tag_Resp,
                 OpenAI.Catalog_Response (Model_Path, Autoreload));

            elsif Tag = Protocol.Tag_Select then
               --  Switch the active model. The body is the chosen model id
               --  (its path, exactly as returned in the catalog). Validate it
               --  against the catalog, persist it, and reload if supervised.
               declare
                  use type LLM_Catalog.Model_Status;
                  Cat   : constant LLM_Catalog.Entry_Vectors.Vector :=
                    LLM_Catalog.Discover;
                  Found : Boolean := False;
               begin
                  for E of Cat loop
                     if To_String (E.Path) = Prompt
                       and then E.Status = LLM_Catalog.Supported
                     then
                        Found := True;
                        exit;
                     end if;
                  end loop;

                  if not Found then
                     Send_Tagged (Protocol.Tag_Resp, OpenAI.Select_Result
                       (False, False, "unknown or unsupported model"));
                  elsif Prompt = Model_Path then
                     Send_Tagged (Protocol.Tag_Resp, OpenAI.Select_Result
                       (True, False, "already active"));
                  else
                     Write_Selected (Prompt);
                     if Autoreload then
                        Send_Tagged (Protocol.Tag_Resp, OpenAI.Select_Result
                          (True, True, "switching model; reloading"));
                        Put_Line ("model switch -> " & Prompt & "; reloading");
                        delay 0.3;   -- let the reply flush to the client
                        GNAT.OS_Lib.OS_Exit (Reload_Code);
                     else
                        Send_Tagged (Protocol.Tag_Resp, OpenAI.Select_Result
                          (True, False, "selected; restart the server to apply"));
                     end if;
                  end if;
               end;

            elsif Tag = Protocol.Tag_Chat then
               declare
                  Locked : Boolean := False;
               begin
                  declare
                     Rq      : constant OpenAI.Request := OpenAI.Parse_Chat (Prompt);
                     Eff_Max : constant Integer :=
                       Integer'Min (Integer'Max (1, Rq.Max_Tokens), Max_Reply_Tokens);
                     P       : LLM_Sampler.Params := Rq.Params;
                     St      : aliased LLM_Engine.Gen_Stats;
                     function Finish return String is
                       (if St.Truncated then "length" else "stop");
                  begin
                     if Rq.N = 0 then
                        Send_Tagged (Protocol.Tag_Resp,
                          OpenAI.Error_Response ("messages required"));
                     else
                        --  Never trust client params: clamp.
                        if P.Temperature < 0.0 then P.Temperature := 0.0;
                        elsif P.Temperature > 2.0 then P.Temperature := 2.0; end if;
                        if P.Top_P <= 0.0 or else P.Top_P > 1.0 then P.Top_P := 1.0; end if;
                        if P.Top_K < 0 then P.Top_K := 0; end if;
                        Infer_Lock.Acquire; Locked := True;
                        if Rq.Stream then
                           declare
                              R : constant String := LLM_Engine.Chat
                                (Model, Rq.Messages, Eff_Max, Sink'Access, P,
                                 St'Access);
                              pragma Unreferenced (R);
                           begin null; end;
                           Infer_Lock.Release; Locked := False;
                           if St.Overflow then
                              Send_Tagged (Protocol.Tag_Resp, OpenAI.Error_Response
                                ("prompt exceeds the model context window",
                                 "context_length_exceeded"));
                           else
                              --  Carry finish_reason + usage on the done record so
                              --  the proxy emits a standards-correct final chunk.
                              Send_Tagged (Protocol.Tag_Done,
                                Finish & St.Prompt_Tokens'Image
                                       & St.Completion_Tokens'Image);
                           end if;
                        else
                           declare
                              R : constant String := LLM_Engine.Chat
                                (Model, Rq.Messages, Eff_Max, null, P, St'Access);
                           begin
                              Infer_Lock.Release; Locked := False;
                              if St.Overflow then
                                 Send_Tagged (Protocol.Tag_Resp, OpenAI.Error_Response
                                   ("prompt exceeds the model context window",
                                    "context_length_exceeded"));
                              else
                                 Send_Tagged (Protocol.Tag_Resp,
                                   OpenAI.Chat_Response
                                     (To_String (Rq.Model), R,
                                      Prompt_Tokens     => St.Prompt_Tokens,
                                      Completion_Tokens => St.Completion_Tokens,
                                      Finish            => Finish));
                              end if;
                           end;
                        end if;
                     end if;
                  end;
               exception
                  when others =>
                     if Locked then Infer_Lock.Release; end if;
                     Send_Tagged (Protocol.Tag_Resp,
                       OpenAI.Error_Response ("bad request"));
               end;

            else
            declare
               N    : constant Natural := Session_Store.Turn_Count (Store);
               Conv : LLM_Qwen.Message_Array (1 .. 2 * N + 1);
               Locked : Boolean := False;
               R_Val  : Unbounded_String;
            begin
               for I in 1 .. N loop
                  Conv (2 * I - 1) := (LLM_Qwen.Role_User,
                    To_Unbounded_String (Session_Store.User_Of (Store, I)));
                  Conv (2 * I) := (LLM_Qwen.Role_Assistant,
                    To_Unbounded_String (Session_Store.Assistant_Of (Store, I)));
               end loop;
               Conv (2 * N + 1) :=
                 (LLM_Qwen.Role_User, To_Unbounded_String (Prompt));

               --  Only one generation runs at a time across all clients.
               Infer_Lock.Acquire; Locked := True;
               R_Val := To_Unbounded_String
                 (LLM_Engine.Chat
                    (Model, Conv, Max_Reply_Tokens, Sink'Access, Sampler_Cfg));
               Infer_Lock.Release; Locked := False;

               Secure_Channel.Send_Message (Ch, ST'Access, [0 => Protocol.Tag_Done]);
               Session_Store.Append_Turn (Store, Prompt, To_String (R_Val));
            exception
               when others =>
                  if Locked then Infer_Lock.Release; end if;
                  raise;
            end;
            end if;
         end;
      end loop;
   exception
      when E : others =>
         Put_Line ("  connection closed: " & Exception_Message (E));
         --  Each cleanup step is isolated: a failure in one must not skip the
         --  others (a leaked socket or unreleased session would compound).
         if Length (Sid) > 0 then
            begin Active_Sessions.Release (To_String (Sid));
            exception when others => null; end;
         end if;
         begin Session_Store.Close (Store);
         exception when others => null; end;
         begin Secure_Channel.Close (Ch);
         exception when others => null; end;
         begin Close_Socket (Conn);
         exception when others => null; end;
   end Handle_Connection;

   task type Handler;
   task body Handler is
      Conn : Socket_Type;
   begin
      loop
         Conn_Queue.Get (Conn);
         --  A handler task must never die: even though Handle_Connection has its
         --  own handler, an exception escaping it (e.g. from cleanup) would kill
         --  this task and permanently shrink the pool. Swallow as a last resort.
         begin
            Handle_Connection (Conn);
         exception
            when others => null;
         end;
      end loop;
   end Handler;

   Handlers : array (1 .. Max_Clients) of Handler;
   pragma Unreferenced (Handlers);

   Port      : Port_Type := Default_Port;
   Bind_Addr : Inet_Addr_Type := Any_Inet_Addr;
begin
   --  Load the model first, with a guard: fail fast and clearly on a bad,
   --  corrupt or unsupported-quantization model rather than crashing before
   --  the server is even up.
   begin
      Model := LLM_Engine.Load (Model_Path);
   exception
      when E : others =>
         Put_Line ("fatal: cannot load model """ & Model_Path & """: "
                   & Exception_Message (E));
         GNAT.OS_Lib.OS_Exit (1);
   end;

   if Ada.Command_Line.Argument_Count >= 1 then
      Port := Port_Type'Value (Ada.Command_Line.Argument (1));
   end if;

   --  Optionally restrict the listener to one address (e.g. 127.0.0.1 when
   --  fronted by a reverse proxy / bridge). Default Any preserves direct
   --  native-client access; an empty or "0.0.0.0" value also means Any.
   if Ada.Environment_Variables.Exists ("ASPIDA_BIND") then
      declare
         B : constant String := Ada.Environment_Variables.Value ("ASPIDA_BIND");
      begin
         if B /= "" and then B /= "0.0.0.0" then
            Bind_Addr := Inet_Addr (B);
         end if;
      exception
         when others =>
            Put_Line ("note: ASPIDA_BIND='" & B & "' invalid; binding Any.");
      end;
   end if;

   if not Crypto.Memory.Lock (Secret'Address, Secret'Length) then
      Put_Line ("note: could not mlock the static key (swap not prevented).");
   end if;
   Put_Line ("server public key (pin this on the client):");
   Put_Line ("  " & Hex (Crypto.X25519.Public_Key (Secret)));
   --  Also drop it next to the binary so `make chat` can pin it automatically.
   declare
      F : Ada.Text_IO.File_Type;
   begin
      Ada.Text_IO.Create (F, Ada.Text_IO.Out_File, Pub_File);
      Ada.Text_IO.Put_Line (F, Hex (Crypto.X25519.Public_Key (Secret)));
      Ada.Text_IO.Close (F);
   exception
      when others => null;   -- non-fatal: the key is already on screen
   end;

   Create_Socket (Listener);
   Set_Socket_Option (Listener, Socket_Level, (Reuse_Address, True));
   Bind_Socket (Listener, (Family_Inet, Bind_Addr, Port));
   Listen_Socket (Listener);
   Put_Line ("listening on port" & Port_Type'Image (Port)
     & " (up to" & Integer'Image (Max_Clients) & " concurrent clients)");

   --  Discovery: enumerate every model present on this system (metadata only,
   --  no weights loaded), so it is clear what is available and which is active.
   --  The active model is still the one given by QWEN_MODEL_PATH; selecting a
   --  different one at runtime is a separate step.
   declare
      use type LLM_Catalog.Model_Status;
      Catalog : constant LLM_Catalog.Entry_Vectors.Vector := LLM_Catalog.Discover;
      Runnable : Natural := 0;
   begin
      Put_Line ("models available on this system (roots: "
        & LLM_Catalog.Roots_Description & "):");
      for E of Catalog loop
         if E.Status = LLM_Catalog.Supported then
            Runnable := Runnable + 1;
         end if;
         Put_Line ("  " & LLM_Catalog.Describe (E));
      end loop;
      Put_Line ("  -->" & Runnable'Image & " runnable; active:" & Model_Path);
   exception
      when others =>
         Put_Line ("model discovery skipped (non-fatal).");
   end;

   loop
      declare
         Conn : Socket_Type;
         From : Sock_Addr_Type;
      begin
         Accept_Socket (Listener, Conn, From);
         declare
            Permit : Boolean;
         begin
            Rate_Limiter.Check (Permit);
            if not Permit then
               --  Over the new-connection rate: drop immediately so a flood
               --  cannot exhaust the queue/handler pool.
               begin Close_Socket (Conn); exception when others => null; end;
            else
               --  Reclaim the slot from a silent peer: a read that stalls past
               --  the idle timeout raises in the handler, which then cleans up.
               Set_Socket_Option
                 (Conn, Socket_Level, (Receive_Timeout, Idle_Timeout));
               Conn_Queue.Put (Conn);     -- a free handler will pick it up
            end if;
         end;
      end;
   end loop;
end Secure_Server;
