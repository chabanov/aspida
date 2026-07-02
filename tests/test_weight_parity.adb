---------------------------------------------------------------------
-- Test_Weight_Parity — H19 Phase 8 validation.
--
-- Three lines of evidence that H19 weight-streaming is a true drop-in for
-- the local-file inference path:
--
--   (A) PARITY — the same model (svgdata/student.gguf), loaded once from a
--       local file and once from a Remote_AEAD_Source over an in-memory
--       loopback channel, produces BIT-IDENTICAL Chat output under greedy
--       sampling (temp <= 0 => argmax, seed-irrelevant). Same weights (the
--       remote read is byte-identical, proven by test_weight_stream) => same
--       FP values => same argmax => same tokens. This test confirms it
--       end-to-end through the real engine: LLM_Engine.Load (Path) vs
--       LLM_Engine.Load_From_Source (Src) + Chat.
--
--   (B) COLD BANDWIDTH + LATENCY — the remote Load_From_Source IS the cold
--       fetch: it pulls every chunk over the channel once (the whole model).
--       We measure the wall-clock time and report the model size, chunk
--       count, and throughput. This is the one-time cost H19 pays per cold
--       cache.
--
--   (C) ZERO-LEAK ACCESS PATTERN — two INDEPENDENT cold reads (two fresh
--       sources, two handshakes) record their outbound fetch sequences via
--       the opt-in fetch log. We assert the two sequences are IDENTICAL and
--       cover every chunk. The fetch pattern is determined by the byte
--       layout alone — no prompt is involved in fetching (the engine loads
--       the whole model into RAM at load time; inference reads zero source
--       bytes) — so any session, for any prompt, sees the same pattern.
--       (The reads here are sequential full-file reads, the streaming-client
--       access pattern; the engine's per-tensor seek order is a different but
--       equally fixed, prompt-independent order. Phase 5's test_weight_prefetch
--       proves the stronger ascending-[0..N-1] pattern when Prefetch_All is
--       used.)
--
-- WHY SUBPROCESSES: the Llama continuous-batch scheduler is a process-wide
-- singleton (Sched.Init is gated by Init_Guard and accepted exactly once), so
-- it binds to the FIRST model a Chat runs on and never rebinds. Chatting on a
-- second Llama model in the same process would reuse the first (freed) model.
-- The production server serves one llama model per process, so the singleton
-- is correct there; for this test we run each Chat leg in its OWN child
-- process (this binary invoked as `local` / `remote`) and compare their
-- hex-encoded outputs from the orchestrator. The zero-leak section uses bare
-- source reads (no engine, no scheduler) so it runs in-process.
--
-- No on-disk cache is enabled (ASPIDA_WEIGHT_CACHE_DIR unset) so every
-- remote read goes over the channel — isolating the cold path. Warm-cache
-- zero-fetch is test_weight_disk's scope.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Exceptions;    use Ada.Exceptions;
with Ada.Calendar;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with Crypto.X25519;
with Secure_Channel;
with LLM_Byte_Source;
use type LLM_Byte_Source.Byte_Source_Access;
with LLM_Weight_Source;
with LLM_Engine;
with LLM_Qwen;
with Ada.Strings.Unbounded;
use type Ada.Strings.Unbounded.Unbounded_String;
with Ada.Directories;
with GNAT.OS_Lib;       use GNAT.OS_Lib;

procedure Test_Weight_Parity is
   use Ada.Text_IO;
   use Ada.Calendar;

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

   type Byte_Array_Access is access all Byte_Array;

   --  Blocking single-byte FIFO. Cap exceeds a full weight-stream frame
   --  (4 + AEAD(1 + 65536 + 16) = 65557) so the writer cannot lap the reader.
   Cap : constant := 262_144;
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

   type Loopback is limited new Secure_Channel.Byte_Transport with record
      In_P, Out_P : access Pipe;
   end record;
   overriding procedure Write (T : in out Loopback; Data : Byte_Array);
   overriding procedure Read  (T : in out Loopback; Data : out Byte_Array);

   overriding procedure Write (T : in out Loopback; Data : Byte_Array) is
   begin
      for B of Data loop T.Out_P.Put (B); end loop;
   end Write;
   overriding procedure Read (T : in out Loopback; Data : out Byte_Array) is
      B : U8;
   begin
      for I in Data'Range loop T.In_P.Get (B); Data (I) := B; end loop;
   end Read;

   Model_Path : constant String := "svgdata/student.gguf";

   Server_Secret : constant Crypto.X25519.Key_256 :=
     [16#01#, 16#23#, 16#45#, 16#67#, 16#89#, 16#ab#, 16#cd#, 16#ef#,
      others => 16#5a#];
   Server_Public : constant Crypto.X25519.Key_256 :=
     Crypto.X25519.Public_Key (Server_Secret);

   --  The conversation used for the parity check. Greedy (temp <= 0) so the
   --  output is a deterministic function of the weights alone — same weights
   --  => same tokens, which is the parity invariant.
   Conv : constant LLM_Qwen.Message_Array :=
     [1 => (LLM_Qwen.Role_User,
            Ada.Strings.Unbounded.To_Unbounded_String
              ("Compute: 2 + 3 ="))];

   Max_New : constant := 16;

   --  Hex-encode a String so a child can emit its answer as a single safe
   --  line and the orchestrator can compare by exact string equality.
   Hex_Digits : constant String := "0123456789abcdef";
   function To_Hex (S : String) return String is
      R : String (1 .. S'Length * 2);
   begin
      for K in 0 .. S'Length - 1 loop
         declare
            C : constant Natural := Character'Pos (S (S'First + K));
         begin
            R (1 + 2 * K) := Hex_Digits (1 + C / 16);
            R (2 + 2 * K) := Hex_Digits (1 + C mod 16);
         end;
      end loop;
      return R;
   end To_Hex;

   --  Run a greedy Chat on an engine and return the Answer text. The engine
   --  must outlive the call (caller unloads it).
   function Chat_Answer (E : LLM_Engine.Engine) return String is
      R : constant LLM_Qwen.Chat_Result :=
        LLM_Engine.Chat (E, Conv, Max_New, Params => LLM_Engine.Greedy);
   begin
      return Ada.Strings.Unbounded.To_String (R.Answer);
   end Chat_Answer;

   --  Library-level state (see test_weight_stream for why 'Access needs this).
   Model_Len  : Unsigned_64 := 0;
   N_Chunks   : Natural := 0;
   Have_Model : Boolean := False;

   procedure Probe_Model is
      Probe : LLM_Byte_Source.Byte_Source_Access;
   begin
      Probe := LLM_Byte_Source.Open_Source (Model_Path);
      Have_Model := (Probe /= null);
      if Have_Model then
         Model_Len := Probe.Byte_Length;
         N_Chunks  := Natural ((Model_Len + LLM_Weight_Source.Chunk_Size - 1) /
                               LLM_Weight_Source.Chunk_Size);
         LLM_Byte_Source.Free_Source (Probe);
      end if;
   end Probe_Model;

   --  Shared pipes for the parity remote leg (server A).
   C2S_A : aliased Pipe;
   S2C_A : aliased Pipe;

   task Server_A is
      entry Launch;
   end Server_A;
   task body Server_A is
      ST : aliased Loopback;
      Ch : Secure_Channel.Channel;
      M  : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      M := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S_A'Access; ST.Out_P := S2C_A'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      --  Serve the cold Load_From_Source (the whole model). The client
      --  finishes loading and never sends again, so serve until aborted.
      LLM_Weight_Source.Serve_Weight_Requests (Ch, ST'Access, M, Model_Path, 0);
      LLM_Byte_Source.Free_Source (M);
   exception
      when others => null;
   end Server_A;

   --  Pipes + servers for the two zero-leak cold reads. Two separate sessions
   --  => two independent fetch logs to compare.
   C2S_C1 : aliased Pipe;  S2C_C1 : aliased Pipe;
   C2S_C2 : aliased Pipe;  S2C_C2 : aliased Pipe;

   task Server_C1 is entry Launch; end Server_C1;
   task body Server_C1 is
      ST : aliased Loopback; Ch : Secure_Channel.Channel;
      M  : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      M := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S_C1'Access; ST.Out_P := S2C_C1'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      LLM_Weight_Source.Serve_Weight_Requests (Ch, ST'Access, M, Model_Path, 0);
      LLM_Byte_Source.Free_Source (M);
   exception
      when others => null;
   end Server_C1;

   task Server_C2 is entry Launch; end Server_C2;
   task body Server_C2 is
      ST : aliased Loopback; Ch : Secure_Channel.Channel;
      M  : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      M := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S_C2'Access; ST.Out_P := S2C_C2'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      LLM_Weight_Source.Serve_Weight_Requests (Ch, ST'Access, M, Model_Path, 0);
      LLM_Byte_Source.Free_Source (M);
   exception
      when others => null;
   end Server_C2;

   --  Run one cold full-model read over a fresh channel and capture the
   --  outbound fetch log (chunk index of every channel fetch, in order,
   --  reduced mod 256 — exact for any model with < 256 chunks). Returns the
   --  log as a heap array (sized to N_Chunks) and its length.
   procedure Cold_Read_Log
     (C2S, S2C : access Pipe;
      Log      : out Byte_Array_Access;
      Log_Len  : out Natural)
   is
      CT  : aliased Loopback;
      Ch  : aliased Secure_Channel.Channel;
      Src : LLM_Byte_Source.Byte_Source_Access;
   begin
      CT.In_P := S2C; CT.Out_P := C2S;
      Secure_Channel.Client_Handshake (Ch, CT'Access, Server_Public);
      Src := LLM_Weight_Source.Open_Remote
        (Ch'Access, CT'Access, Model_Path, Model_Len);
      declare
         R : LLM_Weight_Source.Remote_AEAD_Source renames
               LLM_Weight_Source.Remote_AEAD_Source (Src.all);
         Buf : constant Byte_Array_Access :=
           new Byte_Array (0 .. Natural (Model_Len) - 1);
      begin
         LLM_Weight_Source.Enable_Fetch_Log (R);
         Src.Read_Seq (Buf.all'Address, Natural (Model_Len));
         Log_Len := LLM_Weight_Source.Fetch_Log_Length (R);
         Log := new Byte_Array (0 .. N_Chunks - 1);
         for I in 0 .. Natural'Min (Log_Len, N_Chunks) - 1 loop
            Log (I) := Crypto.U8
              (LLM_Weight_Source.Fetch_Log_At (R, I) mod 256);
         end loop;
      end;
      LLM_Byte_Source.Free_Source (Src);
   end Cold_Read_Log;

   ------------------------------------------------------------------
   --  CHILD MODE: load the model from a LOCAL file, greedy Chat, emit the
   --  hex-encoded answer on stdout (prefix PARITY_). Runs in its own process
   --  so its scheduler binds to this model alone.
   ------------------------------------------------------------------
   procedure Run_Local_Leg is
      E : LLM_Engine.Engine;
   begin
      E := LLM_Engine.Load (Model_Path);
      declare
         Ans : constant String := Chat_Answer (E);
      begin
         LLM_Engine.Unload (E);
         Put_Line ("PARITY_OK");
         Put_Line ("PARITY_ANS " & To_Hex (Ans));
      end;
   exception
      when E : others =>
         Put_Line ("PARITY_ERR " & Exception_Name (E) & ": "
                   & Exception_Message (E));
   end Run_Local_Leg;

   ------------------------------------------------------------------
   --  CHILD MODE: load the model from a Remote_AEAD_Source over the loopback
   --  (timed), greedy Chat, emit hex answer + cold-load timing on stdout.
   ------------------------------------------------------------------
   procedure Run_Remote_Leg is
      CT : aliased Loopback;
      Ch : aliased Secure_Channel.Channel;
      T0 : Ada.Calendar.Time;
      T1 : Ada.Calendar.Time;
   begin
      Server_A.Launch;
      CT.In_P := S2C_A'Access; CT.Out_P := C2S_A'Access;
      Secure_Channel.Client_Handshake (Ch, CT'Access, Server_Public);
      declare
         Src : constant LLM_Byte_Source.Byte_Source_Access :=
           LLM_Weight_Source.Open_Remote
             (Ch'Access, CT'Access, Model_Path, Model_Len);
         E_Rem : LLM_Engine.Engine;
      begin
         T0 := Ada.Calendar.Clock;
         E_Rem := LLM_Engine.Load_From_Source (Src);  --  consumes Src
         T1 := Ada.Calendar.Clock;
         declare
            Ans : constant String := Chat_Answer (E_Rem);
            Dur : constant Duration := T1 - T0;
         begin
            LLM_Engine.Unload (E_Rem);
            Put_Line ("PARITY_OK");
            Put_Line ("PARITY_ANS " & To_Hex (Ans));
            --  dur in milliseconds (integer) for easy parsing.
            Put_Line ("PARITY_COLD " & Natural (Model_Len)'Image & " "
                      & N_Chunks'Image & " "
                      & Integer'Image (Integer (Dur * 1000.0)));
         end;
      end;
   exception
      when E : others =>
         Put_Line ("PARITY_ERR " & Exception_Name (E) & ": "
                   & Exception_Message (E));
         begin abort Server_A; exception when others => null; end;
   end Run_Remote_Leg;

   ------------------------------------------------------------------
   --  ORCHESTRATOR: spawn the local + remote children, capture their stdout,
   --  compare the hex answers, then run the in-process zero-leak section.
   ------------------------------------------------------------------
   Self : constant String := "obj/test_weight_parity";

   function Slurp (Path : String) return String is
      F : File_Type;
      R : Ada.Strings.Unbounded.Unbounded_String;
   begin
      Open (F, In_File, Path);
      while not End_Of_File (F) loop
         Ada.Strings.Unbounded.Append (R, Get_Line (F));
         Ada.Strings.Unbounded.Append (R, ASCII.LF);
      end loop;
      Close (F);
      return Ada.Strings.Unbounded.To_String (R);
   end Slurp;

   --  Extract the first line beginning with Marker from a blob ("" if none).
   function First_Line_With (Blob, Marker : String) return String is
      First : Natural := Blob'First;
   begin
      while First <= Blob'Last loop
         declare
            Last : Natural := First;
         begin
            while Last <= Blob'Last and then Blob (Last) /= ASCII.LF loop
               Last := Last + 1;
            end loop;
            --  line is Blob (First .. Last - 1) (Last is the LF or past end)
            declare
               Lo : constant Natural := First;
               Hi : constant Natural := (if Last > First then Last - 1 else First - 1);
            begin
               if Hi - Lo + 1 >= Marker'Length
                 and then Blob (Lo .. Lo + Marker'Length - 1) = Marker
               then
                  return Blob (Lo .. Hi);
               end if;
            end;
            First := Last + 1;
         end;
      end loop;
      return "";
   end First_Line_With;

   --  Return the substring after a leading prefix, or "" if the line is too
   --  short to contain it.
   function Strip_Prefix (Line, Prefix : String) return String is
   begin
      if Line'Length > Prefix'Length
        and then Line (Line'First .. Line'First + Prefix'Length - 1) = Prefix
      then
         return Line (Line'First + Prefix'Length .. Line'Last);
      else
         return "";
      end if;
   end Strip_Prefix;

   procedure Run_Orchestrator is
      package U renames Ada.Strings.Unbounded;
      Cap_Local  : constant String := "/tmp/aspida_parity_local.txt";
      Cap_Remote : constant String := "/tmp/aspida_parity_remote.txt";
      Args_Local  : Argument_List :=
        [new String'("local")];
      Args_Remote : Argument_List :=
        [new String'("remote")];
      Ok_L, Ok_R : Boolean;
      RC_L, RC_R : Integer;
      Blob_L, Blob_R : U.Unbounded_String;
      Ans_L, Ans_R   : U.Unbounded_String;
      Has_L, Has_R   : Boolean;
      Cold_Line      : U.Unbounded_String;
   begin
      Put_Line ("--- (A) Parity + (B) cold bandwidth/latency ---");

      --  Spawn each Chat leg in its own process; stdout -> capture file.
      Spawn (Self, Args_Local,  Cap_Local,  Ok_L, RC_L, Err_To_Out => True);
      Spawn (Self, Args_Remote, Cap_Remote, Ok_R, RC_R, Err_To_Out => True);
      for A of Args_Local  loop Free (A); end loop;
      for A of Args_Remote loop Free (A); end loop;

      Blob_L := U.To_Unbounded_String (Slurp (Cap_Local));
      Blob_R := U.To_Unbounded_String (Slurp (Cap_Remote));
      begin Ada.Directories.Delete_File (Cap_Local);  exception when others => null; end;
      begin Ada.Directories.Delete_File (Cap_Remote); exception when others => null; end;

      --  A child prints "PARITY_OK" then "PARITY_ANS <hex>". Missing marker =>
      --  the child faulted before emitting (or spawn failed).
      Has_L := First_Line_With (U.To_String (Blob_L), "PARITY_OK") /= "";
      Has_R := First_Line_With (U.To_String (Blob_R), "PARITY_OK") /= "";
      Ans_L := U.To_Unbounded_String
        (First_Line_With (U.To_String (Blob_L), "PARITY_ANS "));
      Ans_R := U.To_Unbounded_String
        (First_Line_With (U.To_String (Blob_R), "PARITY_ANS "));

      Assert ("local child completed without fault", Has_L);
      Assert ("remote child completed without fault", Has_R);

      --  Strip the marker prefix to compare the hex payloads.
      Ans_L := U.To_Unbounded_String
        (Strip_Prefix (U.To_String (Ans_L), "PARITY_ANS "));
      Ans_R := U.To_Unbounded_String
        (Strip_Prefix (U.To_String (Ans_R), "PARITY_ANS "));

      Assert ("local produced non-empty output", U.Length (Ans_L) > 0);
      Assert ("remote produced non-empty output", U.Length (Ans_R) > 0);
      Assert ("remote output == local output (bit-identical, greedy)",
              Has_L and then Has_R and then Ans_L = Ans_R
              and then U.Length (Ans_L) > 0);

      Cold_Line := U.To_Unbounded_String
        (First_Line_With (U.To_String (Blob_R), "PARITY_COLD "));
      if U.Length (Cold_Line) > 0 then
         Put_Line ("  cold load:"
                   & Strip_Prefix (U.To_String (Cold_Line), "PARITY_COLD ")
                   & " (bytes, chunks, ms)");
      end if;
   end Run_Orchestrator;

   procedure Run_Zero_Leak is
      L1, L2     : Byte_Array_Access;
      N1, N2     : Natural;
      Identical  : Boolean := False;
      Covers_All : Boolean := False;
   begin
      Server_C1.Launch;
      Server_C2.Launch;
      Cold_Read_Log (C2S_C1'Access, S2C_C1'Access, L1, N1);
      Cold_Read_Log (C2S_C2'Access, S2C_C2'Access, L2, N2);

      Identical  := (N1 = N2);
      Covers_All := (N1 = N_Chunks);
      for I in 0 .. N1 - 1 loop
         if I > L1'Last or else I > L2'Last or else L1 (I) /= L2 (I) then
            Identical := False;
         end if;
      end loop;

      Assert ("both cold reads fetched exactly N chunks",
              N1 = N_Chunks and then N2 = N_Chunks);
      Assert ("two cold reads fetched every chunk (full model)", Covers_All);
      Assert ("two cold reads have identical fetch sequence (prompt-independent)",
              Identical);
      Put_Line ("  fetch log length:" & N1'Image & " chunks (== N_Chunks)");
   exception
      when E : others =>
         Put_Line ("  (zero-leak leg exception: " & Exception_Name (E) & " - "
                   & Exception_Message (E) & ")");
         Assert ("zero-leak scenario (no exception)", False);
   end Run_Zero_Leak;

   Mode : constant String :=
     (if Ada.Command_Line.Argument_Count >= 1
      then Ada.Command_Line.Argument (1) else "");

begin
   Probe_Model;

   if Mode = "local" then
      --  OS_Exit (not a normal return): this child called Chat, which starts
      --  the Llama scheduler task (no terminate alt inside its loop). A normal
      --  return would wait for that task forever at process exit. OS_Exit
      --  terminates the process immediately, matching secure_server's reload
      --  path. (GNAT.OS_Lib already withed + used above.)
      if not Have_Model then
         Put_Line ("PARITY_ERR model fixture not found");
         GNAT.OS_Lib.OS_Exit (1);
      else
         Run_Local_Leg;
         GNAT.OS_Lib.OS_Exit (0);
      end if;
   elsif Mode = "remote" then
      if not Have_Model then
         Put_Line ("PARITY_ERR model fixture not found");
         GNAT.OS_Lib.OS_Exit (1);
      end if;
      Run_Remote_Leg;
      GNAT.OS_Lib.OS_Exit (0);
   end if;

   --  Orchestrator (no mode arg).
   Put_Line ("=== H19 Phase 8 Validation: Parity + Bandwidth + Zero-Leak ===");
   New_Line;
   if not Have_Model then
      Put_Line ("  SKIP: model fixture " & Model_Path & " not found");
      Ada.Command_Line.Set_Exit_Status (0);
      abort Server_A; abort Server_C1; abort Server_C2;
      return;
   end if;

   Run_Orchestrator;

   New_Line;
   Put_Line ("--- (C) Zero-leak access pattern (two cold reads) ---");
   Run_Zero_Leak;
   abort Server_A; abort Server_C1; abort Server_C2;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
exception
   when E : others =>
      abort Server_A; abort Server_C1; abort Server_C2;
      Put_Line ("  (top-level exception: " & Exception_Name (E) & " - "
                & Exception_Message (E) & ")");
      Assert ("test harness (no top-level exception)", False);
      New_Line;
      Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
      if Failed > 0 then
         Ada.Command_Line.Set_Exit_Status (1);
      end if;
end Test_Weight_Parity;