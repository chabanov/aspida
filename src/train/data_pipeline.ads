------------------------------------------------------------------------
-- Data_Pipeline — turns an engineer's domain corpus into next-token training
-- windows for the GPU Student, with data-parallel sharding. Byte-level
-- tokenisation here keeps it deterministic and vocab-free; the platform swaps in
-- a trained BPE (the engine already has a BPE trainer) without changing the
-- windowing/sharding contract below.
--
-- Next-token LM: window w gives Ids(1..Seq) = corpus[start..start+Seq-1] and
-- Tgts(i) = corpus[start+i] (each position predicts the following token).
------------------------------------------------------------------------

package Data_Pipeline is

   type Token_Vec is array (Natural range <>) of Integer;

   --  Byte-level tokens (0..255) of a file. Empty if the file is empty.
   function Load_Bytes (Path : String) return Token_Vec;

   --  Number of next-token windows of length Seq at the given Stride.
   function Window_Count (Len, Seq, Stride : Positive) return Natural;

   --  Fill Ids/Tgts (each length Seq) for window W (1-based, 1 .. Window_Count).
   --  Tgts is Ids shifted by one (the LM target).
   procedure Window
     (T : Token_Vec; W, Seq, Stride : Positive; Ids, Tgts : out Token_Vec)
     with Pre => Ids'Length = Seq and then Tgts'Length = Seq;

   --  Data-parallel shard: rank R (0-based, of World) gets the contiguous slice
   --  First .. Last of [0 .. Count-1] (balanced; remainder to the low ranks).
   --  Last < First means an empty shard.
   procedure Shard
     (Count, World : Positive; R : Natural; First, Last : out Integer)
     with Pre => R < World;

end Data_Pipeline;
