---------------------------------------------------------------------
-- Test_Weight_Partial — H19 Phase 7 validation (partial-model warm).
--
-- Three lines of evidence that the partial-warm path is a correct drop-in
-- for the eager full load:
--
--   (A) PARITY — the same model (svgdata/student.gguf, 2 transformer blocks)
--       is loaded three ways and Chat is run under greedy sampling
--       (temp <= 0 => argmax, deterministic in the weights):
--         (i)  LLM_Engine.Load (Path)                          — local eager
--         (ii) LLM_Engine.Load_From_Source (Src)              — remote eager
--         (iii)LLM_Engine.Load_From_Source_Partial (Src, K=1) — remote partial
--       The partial path loads the head + block 1 EAGERLY and streams block
--       2 in the BACKGROUND while Chat runs; the forward pass blocks on
--       M.Warm.Wait (2) only if it out-runs the fetcher. Because the weights
--       are the same bytes (proven byte-identical by test_weight_stream) and
--       the forward pass is deterministic, all three MUST produce bit-identical
--       token sequences. This proves the per-block Wait is race-free: no torn
--       reads, no use-before-loaded, no fetcher/forward ordering hazard.
--
--   (B) PARTIAL PATH EXERCISED — the partial child's stdout must contain the
--       "layers hot" log line, confirming the background fetcher actually ran
--       (K < block_count, not the degenerate eager fallback).
--
--   (C) TIMING — load wall-clock is reported for the eager-remote vs partial
--       legs. The partial load RETURNS before all blocks are in RAM (it only
--       waits for the first K), so on a model large enough that fetch dominates
--       compute, partial load is strictly faster. The 2-block fixture is too
--       small for a guaranteed speedup, so we REPORT timings (not assert a
--       strict inequality) — the correctness parity above is the real gate.
--
-- WHY SUBPROCESSES: the Llama continuous-batch scheduler is a process-wide
-- singleton (see test_weight_parity for the full rationale). Each Chat leg runs
-- in its own child process (this binary invoked as `local` / `full` /
-- `partial`); the orchestrator compares their hex-encoded answers.
--
-- No on-disk cache (ASPIDA_WEIGHT_CACHE_DIR unset) so every remote read goes
-- over the channel — isolating the cold path.
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

procedure Test_Weight_Partial is
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

   --  K: layers loaded eagerly by the partial path. The student fixture has
   --  2 transformer blocks, so K=1 warms block 1 and streams block 2 in the
   --  background — the real partial path (K < block_count).
   K_Warm : constant Positive := 1;

   Server_Secret : constant Crypto.X25519.Key_256 :=
     [16#01#, 16#23#, 16#45#, 16#67#, 16#89#, 16#ab#, 16#cd#, 16#ef#,
      others => 16#5a#];
   Server_Public : constant Crypto.X25519.Key_256 :=
     Crypto.X25519.Public_Key (Server_Secret);

   Conv : constant LLM_Qwen.Message_Array :=
     [1 => (LLM_Qwen.Role_User,
            Ada.Strings.Unbounded.To_Unbounded_String
              ("Compute: 2 + 3 ="))];

   Max_New : constant := 16;

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

   function Chat_Answer (E : LLM_Engine.Engine) return String is
      R : constant LLM_Qwen.Chat_Result :=
        LLM_Engine.Chat (E, Conv, Max_New, Params => LLM_Engine.Greedy);
   begin
      return Ada.Strings.Unbounded.To_String (R.Answer);
   end Chat_Answer;

   --  Library-level state (see test_weight_stream for why 'Access needs this).
   Model_Len  : Unsigned_64 := 0;
   Have_Model : Boolean := False;

   procedure Probe_Model is
      Probe : LLM_Byte_Source.Byte_Source_Access;
   begin
      Probe := LLM_Byte_Source.Open_Source (Model_Path);
      Have_Model := (Probe /= null);
      if Have_Model then
         Model_Len := Probe.Byte_Length;
         LLM_Byte_Source.Free_Source (Probe);
      end if;
   end Probe_Model;

   --  Pipes for the eager-remote leg (server FULL).
   C2S_F : aliased Pipe;  S2C_F : aliased Pipe;
   task Server_F is entry Launch; end Server_F;
   task body Server_F is
      ST : aliased Loopback; Ch : Secure_Channel.Channel;
      M  : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      M := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S_F'Access; ST.Out_P := S2C_F'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      LLM_Weight_Source.Serve_Weight_Requests (Ch, ST'Access, M, Model_Path, 0);
      LLM_Byte_Source.Free_Source (M);
   exception
      when others => null;
   end Server_F;

   --  Pipes for the partial-remote leg (server P).
   C2S_P : aliased Pipe;  S2C_P : aliased Pipe;
   task Server_P is entry Launch; end Server_P;
   task body Server_P is
      ST : aliased Loopback; Ch : Secure_Channel.Channel;
      M  : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      M := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S_P'Access; ST.Out_P := S2C_P'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      LLM_Weight_Source.Serve_Weight_Requests (Ch, ST'Access, M, Model_Path, 0);
      LLM_Byte_Source.Free_Source (M);
   exception
      when others => null;
   end Server_P;

   ------------------------------------------------------------------
   --  CHILD: local eager load -> greedy Chat -> emit hex answer.
   ------------------------------------------------------------------
   procedure Run_Local_Leg is
      E : LLM_Engine.Engine;
   begin
      E := LLM_Engine.Load (Model_Path);
      declare
         Ans : constant String := Chat_Answer (E);
      begin
         LLM_Engine.Unload (E);
         Put_Line ("PARTIAL_OK");
         Put_Line ("PARTIAL_ANS " & To_Hex (Ans));
      end;
   exception
      when E : others =>
         Put_Line ("PARTIAL_ERR " & Exception_Name (E) & ": "
                   & Exception_Message (E));
   end Run_Local_Leg;

   ------------------------------------------------------------------
   --  CHILD: remote EAGER load (Load_From_Source) -> Chat -> emit answer
   --  + load timing. The baseline against which partial is compared.
   ------------------------------------------------------------------
   procedure Run_Full_Leg is
      CT : aliased Loopback;  Ch : aliased Secure_Channel.Channel;
      T0 : Ada.Calendar.Time; T1 : Ada.Calendar.Time;
   begin
      Server_F.Launch;
      CT.In_P := S2C_F'Access; CT.Out_P := C2S_F'Access;
      Secure_Channel.Client_Handshake (Ch, CT'Access, Server_Public);
      declare
         Src : constant LLM_Byte_Source.Byte_Source_Access :=
           LLM_Weight_Source.Open_Remote
             (Ch'Access, CT'Access, Model_Path, Model_Len);
         E : LLM_Engine.Engine;
      begin
         T0 := Ada.Calendar.Clock;
         E := LLM_Engine.Load_From_Source (Src);   --  consumes Src (full load)
         T1 := Ada.Calendar.Clock;
         declare
            Ans : constant String := Chat_Answer (E);
         begin
            LLM_Engine.Unload (E);
            Put_Line ("PARTIAL_OK");
            Put_Line ("PARTIAL_ANS " & To_Hex (Ans));
            Put_Line ("PARTIAL_LOAD_MS " &
                      Integer'Image (Integer ((T1 - T0) * 1000.0)));
         end;
      end;
   exception
      when E : others =>
         Put_Line ("PARTIAL_ERR " & Exception_Name (E) & ": "
                   & Exception_Message (E));
         begin abort Server_F; exception when others => null; end;
   end Run_Full_Leg;

   ------------------------------------------------------------------
   --  CHILD: remote PARTIAL load (Load_From_Source_Partial, K=K_Warm) ->
   --  Chat -> emit answer + load timing. The forward pass blocks per-layer
   --  on M.Warm.Wait if it out-runs the background fetcher.
   ------------------------------------------------------------------
   procedure Run_Partial_Leg is
      CT : aliased Loopback;  Ch : aliased Secure_Channel.Channel;
      T0 : Ada.Calendar.Time; T1 : Ada.Calendar.Time;
   begin
      Server_P.Launch;
      CT.In_P := S2C_P'Access; CT.Out_P := C2S_P'Access;
      Secure_Channel.Client_Handshake (Ch, CT'Access, Server_Public);
      declare
         Src : constant LLM_Byte_Source.Byte_Source_Access :=
           LLM_Weight_Source.Open_Remote
             (Ch'Access, CT'Access, Model_Path, Model_Len);
         E : LLM_Engine.Engine;
      begin
         T0 := Ada.Calendar.Clock;
         E := LLM_Engine.Load_From_Source_Partial (Src, K_Warm);  --  consumes Src
         T1 := Ada.Calendar.Clock;
         declare
            Ans : constant String := Chat_Answer (E);
         begin
            LLM_Engine.Unload (E);
            Put_Line ("PARTIAL_OK");
            Put_Line ("PARTIAL_ANS " & To_Hex (Ans));
            Put_Line ("PARTIAL_LOAD_MS " &
                      Integer'Image (Integer ((T1 - T0) * 1000.0)));
         end;
      end;
   exception
      when E : others =>
         Put_Line ("PARTIAL_ERR " & Exception_Name (E) & ": "
                   & Exception_Message (E));
         begin abort Server_P; exception when others => null; end;
   end Run_Partial_Leg;

   ------------------------------------------------------------------
   --  ORCHESTRATOR: spawn local + full + partial children, capture stdout,
   --  compare all three hex answers, confirm the partial path ran.
   ------------------------------------------------------------------
   Self : constant String := "obj/test_weight_partial";

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

   function Contains (Blob, Needle : String) return Boolean is
   begin
      for I in Blob'First .. Blob'Last - Needle'Length + 1 loop
         if Blob (I .. I + Needle'Length - 1) = Needle then
            return True;
         end if;
      end loop;
      return False;
   end Contains;

   procedure Run_Orchestrator is
      package U renames Ada.Strings.Unbounded;
      Cap_Local   : constant String := "/tmp/aspida_partial_local.txt";
      Cap_Full   : constant String := "/tmp/aspida_partial_full.txt";
      Cap_Partial : constant String := "/tmp/aspida_partial_part.txt";
      Args_Local   : Argument_List := [new String'("local")];
      Args_Full    : Argument_List := [new String'("full")];
      Args_Partial : Argument_List := [new String'("partial")];
      Ok_L, Ok_F, Ok_P : Boolean;
      RC_L, RC_F, RC_P : Integer;
      Blob_L, Blob_F, Blob_P : U.Unbounded_String;
      Ans_L, Ans_F, Ans_P : U.Unbounded_String;
      Has_L, Has_F, Has_P : Boolean;
      Load_F, Load_P      : U.Unbounded_String;
   begin
      Put_Line ("--- (A) Parity: local vs remote-eager vs remote-partial ---");

      Spawn (Self, Args_Local,   Cap_Local,   Ok_L, RC_L, Err_To_Out => True);
      Spawn (Self, Args_Full,    Cap_Full,    Ok_F, RC_F, Err_To_Out => True);
      Spawn (Self, Args_Partial, Cap_Partial, Ok_P, RC_P, Err_To_Out => True);
      for A of Args_Local   loop Free (A); end loop;
      for A of Args_Full    loop Free (A); end loop;
      for A of Args_Partial loop Free (A); end loop;

      Blob_L := U.To_Unbounded_String (Slurp (Cap_Local));
      Blob_F := U.To_Unbounded_String (Slurp (Cap_Full));
      Blob_P := U.To_Unbounded_String (Slurp (Cap_Partial));
      begin Ada.Directories.Delete_File (Cap_Local);   exception when others => null; end;
      begin Ada.Directories.Delete_File (Cap_Full);    exception when others => null; end;
      begin Ada.Directories.Delete_File (Cap_Partial); exception when others => null; end;

      Has_L := First_Line_With (U.To_String (Blob_L), "PARTIAL_OK") /= "";
      Has_F := First_Line_With (U.To_String (Blob_F), "PARTIAL_OK") /= "";
      Has_P := First_Line_With (U.To_String (Blob_P), "PARTIAL_OK") /= "";

      Assert ("local child completed without fault",   Has_L);
      Assert ("full child completed without fault",   Has_F);
      Assert ("partial child completed without fault", Has_P);

      Ans_L := U.To_Unbounded_String
        (Strip_Prefix (First_Line_With (U.To_String (Blob_L), "PARTIAL_ANS "), "PARTIAL_ANS "));
      Ans_F := U.To_Unbounded_String
        (Strip_Prefix (First_Line_With (U.To_String (Blob_F), "PARTIAL_ANS "), "PARTIAL_ANS "));
      Ans_P := U.To_Unbounded_String
        (Strip_Prefix (First_Line_With (U.To_String (Blob_P), "PARTIAL_ANS "), "PARTIAL_ANS "));

      Assert ("local produced non-empty output",   U.Length (Ans_L) > 0);
      Assert ("full produced non-empty output",    U.Length (Ans_F) > 0);
      Assert ("partial produced non-empty output", U.Length (Ans_P) > 0);

      --  The core Phase 7 invariant: the partial-warm path produces the SAME
      --  tokens as both eager paths — race-free per-block streaming.
      Assert ("partial == full (bit-identical, greedy)",
              Has_F and then Has_P and then Ans_P = Ans_F
              and then U.Length (Ans_P) > 0);
      Assert ("partial == local (bit-identical, greedy)",
              Has_L and then Has_P and then Ans_P = Ans_L
              and then U.Length (Ans_P) > 0);
      Assert ("full == local (sanity)",
              Has_L and then Has_F and then Ans_F = Ans_L
              and then U.Length (Ans_F) > 0);

      --  (B) The partial path must actually have run the background fetcher
      --  (not the K >= block_count degenerate eager fallback). The log line
      --  "layers hot" is printed only on the K < block_count path.
      Assert ("partial child ran the background fetcher (K < block_count)",
              Has_P and then Contains (U.To_String (Blob_P), "layers hot"));

      --  (C) Report load timings (eager vs partial). Reported, not asserted:
      --  the 2-block fixture is too small for a guaranteed partial speedup.
      Load_F := U.To_Unbounded_String
        (Strip_Prefix (First_Line_With (U.To_String (Blob_F), "PARTIAL_LOAD_MS "), "PARTIAL_LOAD_MS "));
      Load_P := U.To_Unbounded_String
        (Strip_Prefix (First_Line_With (U.To_String (Blob_P), "PARTIAL_LOAD_MS "), "PARTIAL_LOAD_MS "));
      Put_Line ("  load (eager) :" & U.To_String (Load_F) & " ms");
      Put_Line ("  load (partial):" & U.To_String (Load_P) & " ms  (K =" & K_Warm'Image & " warm)");
   end Run_Orchestrator;

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
         Put_Line ("PARTIAL_ERR model fixture not found");
         GNAT.OS_Lib.OS_Exit (1);
      else
         Run_Local_Leg;
         GNAT.OS_Lib.OS_Exit (0);
      end if;
   elsif Mode = "full" then
      if not Have_Model then
         Put_Line ("PARTIAL_ERR model fixture not found");
         GNAT.OS_Lib.OS_Exit (1);
      end if;
      Run_Full_Leg;
      GNAT.OS_Lib.OS_Exit (0);
   elsif Mode = "partial" then
      if not Have_Model then
         Put_Line ("PARTIAL_ERR model fixture not found");
         GNAT.OS_Lib.OS_Exit (1);
      end if;
      Run_Partial_Leg;
      GNAT.OS_Lib.OS_Exit (0);
   end if;

   --  Orchestrator (no mode arg).
   Put_Line ("=== H19 Phase 7 Validation: Partial-Warm Parity ===");
   New_Line;
   if not Have_Model then
      Put_Line ("  SKIP: model fixture " & Model_Path & " not found");
      Ada.Command_Line.Set_Exit_Status (0);
      return;
   end if;

   Run_Orchestrator;
   abort Server_F; abort Server_P;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
exception
   when E : others =>
      abort Server_F; abort Server_P;
      Put_Line ("  (top-level exception: " & Exception_Name (E) & " - "
                & Exception_Message (E) & ")");
      Assert ("test harness (no top-level exception)", False);
      New_Line;
      Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
      if Failed > 0 then
         Ada.Command_Line.Set_Exit_Status (1);
      end if;
end Test_Weight_Partial;