---------------------------------------------------------------------
-- KV_Pool — paged KV-cache bookkeeping (the data structure behind
-- PagedAttention / vLLM). Pure allocation logic, no tensor payload: a fixed
-- pool of equal-size blocks (pages) handed out on demand, a per-sequence block
-- table mapping logical position -> (physical block, offset), and reference
-- counts so a shared prefix (e.g. a common system prompt) costs memory once.
--
-- This is phase 1 of PagedAttention: it captures the memory model (no
-- per-sequence over-allocation + prefix sharing) and is validated in
-- isolation. Phase 2 stores actual K/V tensors in the blocks and routes the
-- attention gather through the block table.
---------------------------------------------------------------------

package KV_Pool is

   type Manager (Total_Blocks, Block_Size, Max_Seqs : Positive) is limited private;

   subtype Seq_Id is Positive;

   Out_Of_Blocks : exception;   -- pool exhausted
   Bad_Sequence  : exception;   -- unopened seq / out-of-range position

   --  Physical blocks not currently allocated to any sequence.
   function Free_Blocks (M : Manager) return Natural;

   --  Total physical blocks a sequence references (shared blocks count once
   --  for the pool but appear in each sequence's table).
   function Blocks_Used (M : Manager; S : Seq_Id) return Natural;

   function Length (M : Manager; S : Seq_Id) return Natural;

   procedure Open_Seq  (M : in out Manager; S : Seq_Id);
   procedure Close_Seq (M : in out Manager; S : Seq_Id);

   --  Extend sequence S by one logical position; pulls a fresh block from the
   --  pool only when crossing a block boundary. Raises Out_Of_Blocks if empty.
   procedure Append (M : in out Manager; S : Seq_Id);

   --  Physical location of logical position Pos (0-based).
   procedure Locate
     (M : Manager; S : Seq_Id; Pos : Natural;
      Block : out Positive; Offset : out Natural);

   --  Share Src's first N logical positions with Dst (prefix reuse): Dst's
   --  table points at Src's prefix blocks, their refcounts rise, so the prefix
   --  is stored once. Dst must be freshly opened (empty).
   procedure Share_Prefix (M : in out Manager; Dst, Src : Seq_Id; N : Natural);

private

   subtype Block_Index is Positive;

   type Refcounts  is array (Positive range <>) of Natural;
   --  Block table: Table (Seq, Slot) = physical block (0 = unset). A sequence
   --  can reference at most Total_Blocks blocks.
   type Table_2D   is array (Positive range <>, Positive range <>) of Natural;
   type Lengths    is array (Positive range <>) of Natural;
   type Open_Flags is array (Positive range <>) of Boolean;

   type Manager (Total_Blocks, Block_Size, Max_Seqs : Positive) is limited record
      Ref     : Refcounts (1 .. Total_Blocks)               := [others => 0];
      Free    : Natural                                     := Total_Blocks;
      Table   : Table_2D (1 .. Max_Seqs, 1 .. Total_Blocks);  -- zeroed in Open
      Len     : Lengths (1 .. Max_Seqs)                     := [others => 0];
      Is_Open : Open_Flags (1 .. Max_Seqs)                  := [others => False];
   end record;

end KV_Pool;
