------------------------------------------------------------------------
-- Data_Pipeline body — byte ingest, next-token windowing, data-parallel shard.
------------------------------------------------------------------------

with Ada.Streams.Stream_IO; use Ada.Streams.Stream_IO;
with Ada.Streams;

package body Data_Pipeline is

   function Load_Bytes (Path : String) return Token_Vec is
      F : File_Type;
   begin
      Open (F, In_File, Path);
      declare
         Len : constant Natural := Natural (Size (F));
         T   : Token_Vec (0 .. (if Len = 0 then -1 else Len - 1));
         S   : constant Stream_Access := Stream (F);
         B   : Ada.Streams.Stream_Element;
      begin
         for I in T'Range loop
            Ada.Streams.Stream_Element'Read (S, B);
            T (I) := Integer (B);
         end loop;
         Close (F);
         return T;
      end;
   exception
      when others =>
         if Is_Open (F) then Close (F); end if;
         raise;
   end Load_Bytes;

   function Window_Count (Len, Seq, Stride : Positive) return Natural is
   begin
      --  window w valid if w*Stride + Seq <= Len-1 (need one extra token for the
      --  last target). Highest start = Len-1-Seq.
      if Len < Seq + 1 then
         return 0;
      else
         return (Len - 1 - Seq) / Stride + 1;
      end if;
   end Window_Count;

   procedure Window
     (T : Token_Vec; W, Seq, Stride : Positive; Ids, Tgts : out Token_Vec)
   is
      Start : constant Natural := (W - 1) * Stride;   -- W is 1-based
   begin
      for I in 0 .. Seq - 1 loop
         Ids  (Ids'First  + I) := T (T'First + Start + I);
         Tgts (Tgts'First + I) := T (T'First + Start + I + 1);   -- next token
      end loop;
   end Window;

   procedure Shard
     (Count, World : Positive; R : Natural; First, Last : out Integer)
   is
      Base : constant Natural := Count / World;
      Remd : constant Natural := Count mod World;
      Size : constant Natural := Base + (if R < Remd then 1 else 0);
   begin
      First := R * Base + Integer'Min (R, Remd);
      Last  := First + Size - 1;   -- Last < First => empty shard
   end Shard;

end Data_Pipeline;
