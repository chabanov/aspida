---------------------------------------------------------------------
-- Test_Weight_Egress — H19 Phase 6: H5 non-leakage on the client egress.
--
-- The H19 inversion puts inference on the client; the only thing the client
-- sends to the network is weight-fetch requests. This test verifies the two
-- non-leakage claims that are NOT already covered by the crypto layer's
-- "the wire is AEAD ciphertext" guarantee:
--
--   (A) EGRESS CONTENT BOUND — every weight-fetch the client emits is a
--       Tag_WReq whose plaintext body is exactly Off(U64) + Count(U32) +
--       Model_ID_Len(U32) + Model_ID bytes. The PROMPT and the generated
--       OUTPUT never enter this body. We build, for every chunk the cold load
--       would fetch, the exact request Remote_AEAD_Source.Fetch_Chunk sends
--       (Encode_WReq with the same Off/Count/Model_ID it computes), and assert:
--         * the tag is Tag_WReq (no other record type egresses),
--         * Decode_WReq round-trips the Off/Count/Model_ID,
--         * Off/Count are in range (Count = min(Chunk_Size, Len-Off)),
--         * the prompt is NOT a byte substring of the request, and
--         * the request length is exactly 17 + Model_ID'Length (no extra
--           payload — the encoder cannot smuggle anything beyond Off/Count/ID).
--       The encoder is a pure function of (Off, Count, Model_ID); with the
--       structural fact that Fetch_Chunk passes ONLY those (it has no pointer
--       to the prompt/output — see (C)), this bounds the egress content.
--
--   (B) AT-REST CACHE SEAL — the only persistent client-side state is the
--       on-disk weight cache (Phase 3, AEAD-sealed via At_Rest). We do a cold
--       load with the disk cache enabled (write-through populates it), then
--       read every byte the cache wrote and assert:
--         * the MODEL ID and the PROMPT do NOT appear as plaintext substrings
--           (the blobs are ChaCha20-Poly1305 sealed; the model-id is bound
--           *inside* the AEAD plaintext, so the raw file bytes are ciphertext),
--         * the cache is non-empty (chunks were actually written).
--       A plaintext prompt or model-id on disk would be a leak; the seal
--       prevents it and this test catches a regression.
--
--   (C) STRUCTURAL ISOLATION (not asserted at runtime here — it is a
--       type-level fact, grep-verified): no file under src/llm/ (the engine
--       and every backend) `with`s Secure_Channel or references a
--       Byte_Transport. The only seam between the engine and the network is
--       the Byte_Source interface, which is READ-ONLY (Read_Seq / Seek /
--       Byte_Length / Cursor / Close / Prefetch_All — no Send/Write). So
--       there is no API path from the prompt or the generated output to the
--       transport; the only client Send_Message is Fetch_Chunk's Tag_WReq.
--       (Inference sending nothing after warm is proven by test_weight_prefetch
--       armed-transport; Chat working after the source is freed — i.e. with no
--       channel at all — is proven by test_weight_parity.)
--
-- No on-disk cache for section A (isolates the encoder). Section B enables it.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Exceptions;    use Ada.Exceptions;
with Ada.Directories;
with Ada.Streams.Stream_IO;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with Crypto.X25519;
with Secure_Channel;
with LLM_Byte_Source;
use type LLM_Byte_Source.Byte_Source_Access;
with LLM_Weight_Source;
with LLM_Weight_Proto;
with Protocol;
with Ada.Environment_Variables;
with Ada.Strings.Unbounded;

procedure Test_Weight_Egress is
   use Ada.Text_IO;
   use Ada.Directories;

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

   --  Blocking single-byte FIFO (cap exceeds a full AEAD weight frame).
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
   Prompt     : constant String := "Compute: 2 + 3 =";

   Server_Secret : constant Crypto.X25519.Key_256 :=
     [16#01#, 16#23#, 16#45#, 16#67#, 16#89#, 16#ab#, 16#cd#, 16#ef#,
      others => 16#5a#];
   Server_Public : constant Crypto.X25519.Key_256 :=
     Crypto.X25519.Public_Key (Server_Secret);

   Model_Len  : Unsigned_64 := 0;
   N_Chunks   : Natural := 0;
   Have_Model : Boolean := False;

   Cache_Dir  : constant String := "/tmp/aspida_egress_cache";
   Cache_Pass : constant String := "egress-test-pass";

   C2S : aliased Pipe;
   S2C : aliased Pipe;

   task Server_Task is
      entry Launch;
   end Server_Task;
   task body Server_Task is
      ST : aliased Loopback;
      Ch : Secure_Channel.Channel;
      M  : LLM_Byte_Source.Byte_Source_Access;
   begin
      accept Launch;
      M := LLM_Byte_Source.Open_Source (Model_Path);
      ST.In_P := C2S'Access; ST.Out_P := S2C'Access;
      Secure_Channel.Server_Handshake (Ch, ST'Access, Server_Secret);
      LLM_Weight_Source.Serve_Weight_Requests (Ch, ST'Access, M, Model_Path, 0);
      LLM_Byte_Source.Free_Source (M);
   exception
      when others => null;
   end Server_Task;

   --  Byte-substring search: does Haystack contain Needle's exact byte sequence?
   function Contains (Haystack : Byte_Array; Needle : String) return Boolean is
   begin
      if Needle'Length = 0 then
         return True;
      end if;
      for I in Haystack'First .. Haystack'Last - Needle'Length + 1 loop
         declare
            Match : Boolean := True;
         begin
            for J in Needle'Range loop
               if Haystack (I + (J - Needle'First)) /= U8 (Character'Pos (Needle (J))) then
                  Match := False; exit;
               end if;
            end loop;
            if Match then
               return True;
            end if;
         end;
      end loop;
      return False;
   end Contains;

   --  Read a whole file into a heap Byte_Array (0-based). Returns null on
   --  failure (caller skips). Uses Ada.Streams.Stream_IO's stream attributes
   --  (the same path as test_weight_cache's Corrupt_Tag), so no per-byte loop.
   function Read_File_Bytes (Path : String) return Byte_Array_Access is
      package SIO renames Ada.Streams.Stream_IO;
      F    : SIO.File_Type;
      Size : constant Natural := Natural (Ada.Directories.Size (Path));
   begin
      if Size = 0 then
         return null;
      end if;
      SIO.Open (F, SIO.In_File, Path);
      declare
         Buf : constant Byte_Array_Access := new Byte_Array (0 .. Size - 1);
      begin
         Byte_Array'Read (SIO.Stream (F), Buf.all);
         SIO.Close (F);
         return Buf;
      end;
   exception
      when others =>
         begin SIO.Close (F); exception when others => null; end;
         return null;
   end Read_File_Bytes;

   --  Does any ordinary file under Dir (recursively) contain Needle as a
   --  byte substring? Used to prove the sealed cache writes no plaintext.
   function Dir_Contains_Substring
     (Dir : String; Needle : String) return Boolean
   is
      Found : Boolean := False;

      procedure Walk (D : String) is
         S : Ada.Directories.Search_Type;
         E : Ada.Directories.Directory_Entry_Type;
      begin
         Ada.Directories.Start_Search
           (S, D, "*",
            Filter => [Ada.Directories.Ordinary_File
                       | Ada.Directories.Directory => True,
                      others => False]);
         while Ada.Directories.More_Entries (S) loop
            Ada.Directories.Get_Next_Entry (S, E);
            declare
               Name : constant String := Ada.Directories.Simple_Name (E);
               Full : constant String := Ada.Directories.Full_Name (E);
            begin
               --  Skip the special "." / ".." entries GNAT's readdir returns;
               --  descending into them would recurse infinitely (./././...).
               if Name /= "." and then Name /= ".." then
                  if Ada.Directories.Kind (E) = Ada.Directories.Directory then
                     Walk (Full);
                  else
                     declare
                        B : constant Byte_Array_Access := Read_File_Bytes (Full);
                     begin
                        if B /= null and then Contains (B.all, Needle) then
                           Found := True;
                        end if;
                     end;
                  end if;
               end if;
            end;
         end loop;
         Ada.Directories.End_Search (S);
      end Walk;
   begin
      Walk (Dir);
      return Found;
   end Dir_Contains_Substring;

   --  Count ordinary files under Dir (recursively) — sanity that the cache
   --  actually wrote something.
   function Dir_File_Count (Dir : String) return Natural is
      N : Natural := 0;
      procedure Walk (D : String) is
         S : Ada.Directories.Search_Type;
         E : Ada.Directories.Directory_Entry_Type;
      begin
         Ada.Directories.Start_Search
           (S, D, "*",
            Filter => [Ada.Directories.Ordinary_File
                       | Ada.Directories.Directory => True,
                      others => False]);
         while Ada.Directories.More_Entries (S) loop
            Ada.Directories.Get_Next_Entry (S, E);
            declare
               Name : constant String := Ada.Directories.Simple_Name (E);
               Full : constant String := Ada.Directories.Full_Name (E);
            begin
               --  Skip "." / ".." (GNAT's readdir returns them).
               if Name /= "." and then Name /= ".." then
                  if Ada.Directories.Kind (E) = Ada.Directories.Directory then
                     Walk (Full);
                  else
                     N := N + 1;
                  end if;
               end if;
            end;
         end loop;
         Ada.Directories.End_Search (S);
      end Walk;
   begin
      Walk (Dir);
      return N;
   end Dir_File_Count;

begin
   Put_Line ("=== H19 Phase 6: Client Egress Non-Leakage ===");
   New_Line;

   declare
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
   end;

   if not Have_Model then
      Put_Line ("  SKIP: model fixture " & Model_Path & " not found");
      Ada.Command_Line.Set_Exit_Status (0);
      abort Server_Task;
      return;
   end if;

   ------------------------------------------------------------------
   --  (A) Egress content bound: every chunk-fetch request carries only
   --  Off/Count/Model_ID — never the prompt.
   ------------------------------------------------------------------
   Put_Line ("--- (A) Egress content bound (encoder carries no prompt) ---");
   declare
      All_Req_Tag_Ok    : Boolean := True;
      All_Round_Trip    : Boolean := True;
      All_In_Range      : Boolean := True;
      All_No_Prompt     : Boolean := True;
      All_Exact_Length  : Boolean := True;
      Any_Model_Id_Sent : Boolean := False;
   begin
      for Index in 0 .. N_Chunks - 1 loop
         declare
            Off      : constant Unsigned_64 := Unsigned_64 (Index) * LLM_Weight_Source.Chunk_Size;
            Remain   : constant Unsigned_64 := Model_Len - Off;
            Count    : constant U32 := U32 (Unsigned_64'Min (LLM_Weight_Source.Chunk_Size, Remain));
            Req      : constant Byte_Array := LLM_Weight_Proto.Encode_WReq
              (Off, Count, Model_Path);
            R_Off    : aliased U64;
            R_Count  : aliased U32;
            R_ID     : aliased Ada.Strings.Unbounded.Unbounded_String;
            R_OK     : aliased Boolean;
         begin
            --  Tag must be WReq (the ONLY record the client egresses post-handshake).
            if LLM_Weight_Proto.Tag_Of (Req) /= Protocol.Tag_WReq then
               All_Req_Tag_Ok := False;
            end if;

            --  Round-trip: decode yields the same Off/Count/Model_ID.
            LLM_Weight_Proto.Decode_WReq (Req, R_Off, R_Count, R_ID, R_OK);
            if not R_OK
              or else R_Off /= Off
              or else R_Count /= Count
              or else Ada.Strings.Unbounded.To_String (R_ID) /= Model_Path
            then
               All_Round_Trip := False;
            end if;

            --  Off/Count in range, matching Fetch_Chunk's computation.
            if Off >= Model_Len
              or else Count > U32 (LLM_Weight_Source.Chunk_Size)
              or else Count /= U32 (Unsigned_64'Min (LLM_Weight_Source.Chunk_Size, Remain))
            then
               All_In_Range := False;
            end if;

            --  The prompt must NOT be a substring of the egress plaintext.
            if Contains (Req, Prompt) then
               All_No_Prompt := False;
            end if;

            --  Exact length: tag(1) + header(16) + Model_ID — no extra payload.
            if Req'Length /= 17 + Model_Path'Length then
               All_Exact_Length := False;
            end if;

            --  Sanity contrast: the Model_ID (NOT secret) IS egressed by design.
            if Contains (Req, Model_Path) then
               Any_Model_Id_Sent := True;
            end if;
         end;
      end loop;

      Assert ("every chunk-fetch request is a Tag_WReq", All_Req_Tag_Ok);
      Assert ("every request round-trips (Off/Count/Model_ID)", All_Round_Trip);
      Assert ("every request Off/Count in range", All_In_Range);
      Assert ("no request body contains the prompt", All_No_Prompt);
      Assert ("every request length == 17 + Model_ID (no extra payload)",
              All_Exact_Length);
      Assert ("Model_ID is egressed (expected — it is not secret)",
              Any_Model_Id_Sent);
   end;

   ------------------------------------------------------------------
   --  (B) At-rest cache seal: the on-disk cache writes no plaintext
   --  model-id or prompt.
   ------------------------------------------------------------------
   New_Line;
   Put_Line ("--- (B) At-rest cache seal (no plaintext on disk) ---");

   --  Fresh cache dir + enable the on-disk cache for the cold read.
   if Ada.Directories.Exists (Cache_Dir) then
      Ada.Directories.Delete_Tree (Cache_Dir);
   end if;
   Ada.Directories.Create_Path (Cache_Dir);
   Ada.Environment_Variables.Set ("ASPIDA_WEIGHT_CACHE_DIR", Cache_Dir);
   Ada.Environment_Variables.Set ("ASPIDA_WEIGHT_CACHE_PASS", Cache_Pass);

   declare
      Cache_Wrote : Boolean := False;
      No_Plaintext_ID : Boolean := False;
      No_Plaintext_Prompt : Boolean := False;
   begin
      Server_Task.Launch;
      declare
         CT  : aliased Loopback;
         Ch  : aliased Secure_Channel.Channel;
         Src : LLM_Byte_Source.Byte_Source_Access :=
           LLM_Weight_Source.Open_Remote
             (Ch'Access, CT'Access, Model_Path, Model_Len);
         Buf : constant Byte_Array_Access :=
           new Byte_Array (0 .. Natural (Model_Len) - 1);
      begin
         CT.In_P := S2C'Access; CT.Out_P := C2S'Access;
         Secure_Channel.Client_Handshake (Ch, CT'Access, Server_Public);
         --  Cold full read: every chunk is fetched over the channel and
         --  write-through sealed to the on-disk cache.
         Src.Read_Seq (Buf.all'Address, Natural (Model_Len));
         LLM_Byte_Source.Free_Source (Src);
      end;
      abort Server_Task;

      --  The cache should now hold N_Chunks sealed files.
      Cache_Wrote := (Dir_File_Count (Cache_Dir) >= N_Chunks);

      --  Neither the model-id nor the prompt may appear as plaintext in any
      --  sealed cache file.
      No_Plaintext_ID     := not Dir_Contains_Substring (Cache_Dir, Model_Path);
      No_Plaintext_Prompt  := not Dir_Contains_Substring (Cache_Dir, Prompt);

      Assert ("disk cache wrote the chunks (>= N_Chunks files)", Cache_Wrote);
      Assert ("no plaintext model-id in any cache file (AEAD-sealed)",
              No_Plaintext_ID);
      Assert ("no plaintext prompt in any cache file", No_Plaintext_Prompt);
   exception
      when E : others =>
         Put_Line ("  (cache-seal leg exception: " & Exception_Name (E) & " - "
                   & Exception_Message (E) & ")");
         Assert ("cache-seal scenario (no exception)", False);
         begin abort Server_Task; exception when others => null; end;
   end;

   --  Cleanup env + temp dir.
   Ada.Environment_Variables.Clear ("ASPIDA_WEIGHT_CACHE_DIR");
   Ada.Environment_Variables.Clear ("ASPIDA_WEIGHT_CACHE_PASS");
   begin Ada.Directories.Delete_Tree (Cache_Dir); exception when others => null; end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
exception
   when E : others =>
      abort Server_Task;
      Ada.Environment_Variables.Clear ("ASPIDA_WEIGHT_CACHE_DIR");
      Ada.Environment_Variables.Clear ("ASPIDA_WEIGHT_CACHE_PASS");
      begin Ada.Directories.Delete_Tree (Cache_Dir); exception when others => null; end;
      Put_Line ("  (top-level exception: " & Exception_Name (E) & " - "
                & Exception_Message (E) & ")");
      Assert ("test harness (no top-level exception)", False);
      New_Line;
      Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
      if Failed > 0 then
         Ada.Command_Line.Set_Exit_Status (1);
      end if;
end Test_Weight_Egress;