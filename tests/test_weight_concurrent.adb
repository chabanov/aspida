---------------------------------------------------------------------
-- Test_Weight_Concurrent — H19 regression: concurrent positional reads of
-- ONE shared Local_File_Source are race-free.
--
-- The weight server (Serve_Weight_Requests) reads each requested range from a
-- single shared Model source. An earlier version used Seek (Off) + Read_Seq,
-- two operations over one mutable fd cursor: two clients served in parallel
-- could interleave A.Seek / B.Seek / A.Read and A would read from B's offset.
-- The fix routes every range read through Read_At_Pos, backed by pread(2),
-- which carries the offset in the syscall and never touches the fd cursor.
--
-- This test drives that primitive directly under real contention: N worker
-- tasks share ONE Local_File_Source (one fd) and each performs many
-- Read_At_Pos calls at randomized offsets, comparing every byte against a
-- reference image of the whole file read once up front. Under the old
-- Seek+Read_Seq path this races and mismatches; under pread it is exact.
--
-- Model-free: uses svgdata/student.gguf purely as a convenient real file of
-- bytes (any file works); no engine, no channel, no scheduler.
---------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Command_Line;
with Ada.Directories;
with Interfaces;            use Interfaces;
with Crypto;
with LLM_Byte_Source;
use type LLM_Byte_Source.Byte_Source_Access;

procedure Test_Weight_Concurrent is

   Path : constant String := "svgdata/student.gguf";

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Check (Cond : Boolean; Label : String) is
   begin
      if Cond then
         Passed := Passed + 1;
         Put_Line ("  PASS: " & Label);
      else
         Failed := Failed + 1;
         Put_Line ("  FAIL: " & Label);
      end if;
   end Check;

begin
   Put_Line ("=== H19 Concurrent Positional-Read (pread) Race Test ===");

   if not Ada.Directories.Exists (Path) then
      Put_Line ("  SKIP: " & Path & " not present");
      return;
   end if;

   if Natural (Ada.Directories.Size (Path)) = 0 then
      Put_Line ("  SKIP: " & Path & " is empty (need bytes to range-read)");
      return;
   end if;

   declare
      Len : constant Interfaces.Unsigned_64 :=
        Interfaces.Unsigned_64 (Ada.Directories.Size (Path));
      --  Reference image: the whole file, read once, sequentially.
      Ref : Crypto.Byte_Array (0 .. Natural (Len) - 1);
      Src : LLM_Byte_Source.Byte_Source_Access :=
        LLM_Byte_Source.Open_Source (Path);

      N_Workers : constant := 8;
      Iters     : constant := 400;
      type Result_Arr is array (1 .. N_Workers) of Boolean;
      pragma Atomic_Components (Result_Arr);
      All_OK : Result_Arr := [others => True];
   begin
      Check (Src /= null, "opened the shared source");
      if Src = null then
         Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
         return;
      end if;

      --  Build the reference with a positional read of the entire file.
      Src.Read_At_Pos (0, Ref'Address, Ref'Length);
      Check (True, "reference image read (" & Ref'Length'Image & " bytes)");

      --  N worker tasks share the SAME Src (same fd). Each reads many random
      --  sub-ranges via Read_At_Pos and compares against Ref. A cursor race
      --  would corrupt at least one worker's bytes; pread cannot.
      declare
         --  Tiny deterministic per-worker PRNG (xorshift), so each worker
         --  hits a different, repeatable offset sequence with no shared RNG.
         task type Worker is
            entry Start (Id : Positive);
         end Worker;

         task body Worker is
            My_Id : Positive;
            State : Interfaces.Unsigned_64;
         begin
            accept Start (Id : Positive) do
               My_Id := Id;
            end Start;
            State := Interfaces.Unsigned_64 (My_Id) * 2_654_435_761 + 1;

            for Step in 1 .. Iters loop
               --  xorshift64* -> a pseudo-random offset/length in range.
               State := State xor Interfaces.Shift_Right (State, 12);
               State := State xor Interfaces.Shift_Left  (State, 25);
               State := State xor Interfaces.Shift_Right (State, 27);
               declare
                  R      : constant Interfaces.Unsigned_64 :=
                    State * 2_685_821_657_736_338_717;
                  Max_L  : constant Interfaces.Unsigned_64 :=
                    Interfaces.Unsigned_64'Min (4096, Len);
                  L      : constant Natural :=
                    Natural (R mod Max_L) + 1;            --  1 .. Max_L
                  Off_Sp : constant Interfaces.Unsigned_64 :=
                    (if Len > Interfaces.Unsigned_64 (L)
                     then Len - Interfaces.Unsigned_64 (L) else 0);
                  Off    : constant Interfaces.Unsigned_64 :=
                    (if Off_Sp = 0 then 0
                     else Interfaces.Shift_Right (R, 20) mod (Off_Sp + 1));
                  Buf    : Crypto.Byte_Array (0 .. L - 1);
               begin
                  Src.Read_At_Pos (Off, Buf'Address, L);
                  for J in 0 .. L - 1 loop
                     if Buf (J) /= Ref (Natural (Off) + J) then
                        All_OK (My_Id) := False;
                        exit;
                     end if;
                  end loop;
               end;
               exit when not All_OK (My_Id);
            end loop;
         end Worker;

         Pool : array (1 .. N_Workers) of Worker;
      begin
         for I in Pool'Range loop
            Pool (I).Start (I);
         end loop;
         --  Tasks complete at end of this declare block (implicit wait).
      end;

      declare
         Every_OK : Boolean := True;
      begin
         for I in All_OK'Range loop
            if not All_OK (I) then
               Every_OK := False;
            end if;
         end loop;
         Check (Every_OK,
                "all 8 workers x 400 concurrent positional reads matched the "
                & "reference (no cursor race)");
      end;

      --  Cursor untouched by Read_At_Pos: after all the positional reads the
      --  shared source's cursor is still 0 (we never called Seek/Read_Seq).
      Check (Src.Cursor = 0, "Read_At_Pos left the fd cursor unmoved");

      --  Out-of-range positional read fails loud, never an OOB read.
      declare
         Dummy : Crypto.Byte_Array (0 .. 15);
         Raised : Boolean := False;
      begin
         Src.Read_At_Pos (Len, Dummy'Address, 16);   --  Off == Len, Count > 0
         Check (False, "past-end Read_At_Pos should have raised");
      exception
         when LLM_Byte_Source.Malformed_Source =>
            Raised := True;
            Check (Raised, "past-end Read_At_Pos raises Malformed_Source");
      end;

      LLM_Byte_Source.Free_Source (Src);
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   end if;
end Test_Weight_Concurrent;
