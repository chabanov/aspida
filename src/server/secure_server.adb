---------------------------------------------------------------------
-- Secure_Server — concurrent encrypted chat server.
--
-- Loads the Qwen model once (shared, read-only) and serves many clients
-- at the same time: a pool of handler tasks each take a connection from a
-- bounded queue, run the handshake + session, and chat. Inference itself
-- is serialized by Infer_Lock (one model + a thread pool that already uses
-- every core, so concurrent generations would only contend) — but I/O,
-- handshakes and persistence run in parallel across connections.
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
with Socket_Transport;
with Session_Store;
with Encrypting_Sink;
with Protocol;
with LLM_Qwen;
with LLM_Engine;
with LLM_Sampler;

procedure Secure_Server is

   package SIO renames Ada.Streams.Stream_IO;

   Key_File     : constant String := "server_key.bin";
   Pub_File     : constant String := "server_pub.hex";
   Default_Port : constant := 8765;
   Max_Clients  : constant := 8;     -- concurrent handler tasks
   Max_Queue    : constant := 64;    -- pending-connection backlog
   Max_Reply_Tokens : constant := 256;   -- generation cap per turn

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
      return K;
   end Load_Or_Create_Key;

   function Model_Path return String is
   begin
      if Ada.Environment_Variables.Exists ("QWEN_MODEL_PATH") then
         return Ada.Environment_Variables.Value ("QWEN_MODEL_PATH");
      end if;
      return "/Users/ceo/.lmstudio/models/HauhauCS/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/"
        & "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf";
   end Model_Path;

   --  Loaded during elaboration, before the handler tasks activate.
   Model    : constant LLM_Engine.Engine := LLM_Engine.Load (Model_Path);
   Secret   : constant Crypto.X25519.Key_256 := Load_Or_Create_Key;
   Listener : Socket_Type;

   --  Serializes the actual generation (single model + all-core pool).
   protected Infer_Lock is
      entry Acquire;
      procedure Release;
   private
      Held : Boolean := False;
   end Infer_Lock;

   protected body Infer_Lock is
      entry Acquire when not Held is
      begin
         Held := True;
      end Acquire;
      procedure Release is
      begin
         Held := False;
      end Release;
   end Infer_Lock;

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
   begin
      ST.Sock := Conn;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Secret);

      --  First record selects/creates the session (Tag_Session + id;
      --  empty id = new). Reply with the assigned id.
      declare
         Hello : constant Byte_Array := Secure_Channel.Recv_Message (Ch, ST'Access);
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
            Sink : aliased Encrypting_Sink.Enc_Sink :=
              (LLM_Qwen.Token_Sink with
                 Ch => Ch'Unchecked_Access, T => ST'Unchecked_Access);
            Prompt : String (1 .. Integer'Max (0, Req'Length - 1));
         begin
            for I in Prompt'Range loop
               Prompt (I) := Character'Val (Integer (Req (Req'First + I)));
            end loop;
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

   Port : Port_Type := Default_Port;
begin
   if Ada.Command_Line.Argument_Count >= 1 then
      Port := Port_Type'Value (Ada.Command_Line.Argument (1));
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
   Bind_Socket (Listener, (Family_Inet, Any_Inet_Addr, Port));
   Listen_Socket (Listener);
   Put_Line ("listening on port" & Port_Type'Image (Port)
     & " (up to" & Integer'Image (Max_Clients) & " concurrent clients)");

   loop
      declare
         Conn : Socket_Type;
         From : Sock_Addr_Type;
      begin
         Accept_Socket (Listener, Conn, From);
         --  Reclaim the slot from a silent peer: a read that stalls past the
         --  idle timeout raises in the handler, which then cleans up.
         Set_Socket_Option
           (Conn, Socket_Level, (Receive_Timeout, Idle_Timeout));
         Conn_Queue.Put (Conn);        -- a free handler will pick it up
      end;
   end loop;
end Secure_Server;
