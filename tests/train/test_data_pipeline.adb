------------------------------------------------------------------------
-- test_data_pipeline — corpus ingest, next-token windowing, DP sharding. $0.
------------------------------------------------------------------------

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Streams.Stream_IO;
with Data_Pipeline;         use Data_Pipeline;

procedure Test_Data_Pipeline is
   Pass : Boolean := True;
   procedure Chk (Name : String; Cond : Boolean) is
   begin
      Put_Line ("  " & (if Cond then "PASS" else "FAIL") & ": " & Name);
      if not Cond then Pass := False; end if;
   end Chk;
begin
   Put_Line ("=== Data_Pipeline (ingest / windows / shard) ===");

   --  write a known 10-byte file, load it back
   declare
      package SIO renames Ada.Streams.Stream_IO;
      F : SIO.File_Type; Path : constant String := "/tmp/aspida_dp_test.bin";
   begin
      SIO.Create (F, SIO.Out_File, Path);                 -- raw bytes 10..19
      for B in 10 .. 19 loop
         Ada.Streams.Stream_Element'Write (SIO.Stream (F), Ada.Streams.Stream_Element (B));
      end loop;
      SIO.Close (F);
      declare T : constant Token_Vec := Load_Bytes (Path);
      begin
         Chk ("load: 10 bytes", T'Length = 10);
         Chk ("load: first=10, last=19", T (T'First) = 10 and then T (T'Last) = 19);
      end;
   end;

   --  windowing on an in-memory stream 0..9 (Len=10)
   declare
      T : constant Token_Vec := [for I in 0 .. 9 => I];
      Seq : constant := 4; Stride : constant := 2;
      Ids, Tgts : Token_Vec (1 .. Seq);
   begin
      --  Len=10, Seq=4, Stride=2: highest start = 10-1-4 = 5 -> windows at start
      --  0,2,4 -> count = (10-1-4)/2 + 1 = 3
      Chk ("window count = 3", Window_Count (10, Seq, Stride) = 3);
      Window (T, 1, Seq, Stride, Ids, Tgts);   -- start 0
      Chk ("w1 ids = 0,1,2,3", Ids = [0,1,2,3]);
      Chk ("w1 tgts = 1,2,3,4 (shift)", Tgts = [1,2,3,4]);
      Window (T, 3, Seq, Stride, Ids, Tgts);   -- start 4
      Chk ("w3 ids = 4,5,6,7", Ids = [4,5,6,7]);
      Chk ("w3 tgts = 5,6,7,8", Tgts = [5,6,7,8]);
   end;

   --  data-parallel shard: 10 examples over 3 ranks -> 4,3,3 ; disjoint + cover all
   declare
      Count : constant := 10; World : constant := 3;
      F0, L0, F1, L1, F2, L2 : Integer;
   begin
      Shard (Count, World, 0, F0, L0);
      Shard (Count, World, 1, F1, L1);
      Shard (Count, World, 2, F2, L2);
      Chk ("rank0 = 0..3", F0 = 0 and then L0 = 3);
      Chk ("rank1 = 4..6", F1 = 4 and then L1 = 6);
      Chk ("rank2 = 7..9", F2 = 7 and then L2 = 9);
      Chk ("contiguous + covers all", L0 + 1 = F1 and then L1 + 1 = F2 and then L2 = Count - 1);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_Data_Pipeline;
