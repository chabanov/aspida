------------------------------------------------------------------------
-- test_kv_pool — paged KV-cache allocator (PagedAttention phase 1):
-- on-demand block allocation, position->block mapping, refcounted prefix
-- sharing, and free-on-close.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with KV_Pool;     use KV_Pool;

procedure Test_KV_Pool is
   Pass : Boolean := True;
   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;
begin
   Put_Line ("=== KV_Pool (paged KV allocator) ===");

   --  Allocation only crosses a block boundary every Block_Size positions.
   declare
      M : Manager (Total_Blocks => 10, Block_Size => 4, Max_Seqs => 4);
   begin
      Check ("starts with all blocks free", Free_Blocks (M) = 10);
      Open_Seq (M, 1);
      for I in 1 .. 4 loop Append (M, 1); end loop;     -- exactly one block
      Check ("4 positions = 1 block", Free_Blocks (M) = 9
             and then Blocks_Used (M, 1) = 1);
      Append (M, 1);                                     -- 5th -> 2nd block
      Check ("5 positions = 2 blocks", Free_Blocks (M) = 8
             and then Blocks_Used (M, 1) = 2);
      for I in 1 .. 3 loop Append (M, 1); end loop;      -- total 8 (pos 0..7)

      --  position -> (block, offset)
      declare
         B0, B4, B5 : Positive; O0, O4, O5 : Natural;
      begin
         Locate (M, 1, 0, B0, O0);
         Locate (M, 1, 4, B4, O4);
         Locate (M, 1, 5, B5, O5);
         Check ("locate offsets", O0 = 0 and then O4 = 0 and then O5 = 1);
         Check ("pos 0 and 4 are different blocks", B0 /= B4);
         Check ("pos 4 and 5 share a block", B4 = B5);
      end;

      Close_Seq (M, 1);
      Check ("close returns all blocks", Free_Blocks (M) = 10);
   end;

   --  Prefix sharing: a shared prefix costs blocks once.
   declare
      M : Manager (Total_Blocks => 10, Block_Size => 4, Max_Seqs => 4);
   begin
      Open_Seq (M, 1);
      for I in 1 .. 8 loop Append (M, 1); end loop;      -- 2 blocks, free 10->8
      Check ("seq1 used 2 blocks", Free_Blocks (M) = 8);

      Open_Seq (M, 2);
      Share_Prefix (M, Dst => 2, Src => 1, N => 8);      -- share, no new alloc
      Check ("prefix share allocates nothing", Free_Blocks (M) = 8);
      Check ("shared seq sees the prefix", Length (M, 2) = 8
             and then Blocks_Used (M, 2) = 2);

      Close_Seq (M, 2);                                  -- prefix still held by 1
      Check ("closing sharer keeps shared blocks", Free_Blocks (M) = 8);
      Close_Seq (M, 1);                                  -- last holder -> freed
      Check ("closing owner frees the blocks", Free_Blocks (M) = 10);
   end;

   --  Exhaustion raises rather than corrupts.
   declare
      M   : Manager (Total_Blocks => 2, Block_Size => 4, Max_Seqs => 2);
      Hit : Boolean := False;
   begin
      Open_Seq (M, 1);
      begin
         for I in 1 .. 100 loop Append (M, 1); end loop;  -- 2 blocks then boom
      exception
         when KV_Pool.Out_Of_Blocks => Hit := True;
      end;
      Check ("pool exhaustion raises Out_Of_Blocks", Hit);
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_KV_Pool;
