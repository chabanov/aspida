---------------------------------------------------------------------
-- KV_Pool body.
---------------------------------------------------------------------

package body KV_Pool is

   function Free_Blocks (M : Manager) return Natural is (M.Free);
   function Length      (M : Manager; S : Seq_Id) return Natural is (M.Len (S));

   --  Number of block-table slots a sequence occupies = ceil(len/block_size).
   function Slots_Of (M : Manager; S : Seq_Id) return Natural is
     ((M.Len (S) + M.Block_Size - 1) / M.Block_Size);

   function Blocks_Used (M : Manager; S : Seq_Id) return Natural is
     (Slots_Of (M, S));

   procedure Zero_Row (M : in out Manager; S : Seq_Id) is
   begin
      for K in 1 .. M.Total_Blocks loop M.Table (S, K) := 0; end loop;
   end Zero_Row;

   procedure Open_Seq (M : in out Manager; S : Seq_Id) is
   begin
      M.Is_Open (S) := True;
      M.Len (S)     := 0;
      Zero_Row (M, S);
   end Open_Seq;

   --  Find a physical block with refcount 0.
   function First_Free (M : Manager) return Block_Index is
   begin
      for B in M.Ref'Range loop
         if M.Ref (B) = 0 then return B; end if;
      end loop;
      raise Out_Of_Blocks;
   end First_Free;

   procedure Append (M : in out Manager; S : Seq_Id) is
   begin
      if not M.Is_Open (S) then raise Bad_Sequence; end if;
      if M.Len (S) mod M.Block_Size = 0 then       -- crossing a block boundary
         if M.Free = 0 then raise Out_Of_Blocks; end if;
         declare
            B : constant Block_Index := First_Free (M);
         begin
            M.Ref (B) := 1;
            M.Free := M.Free - 1;
            M.Table (S, Slots_Of (M, S) + 1) := B;   -- next slot
         end;
      end if;
      M.Len (S) := M.Len (S) + 1;
   end Append;

   procedure Locate
     (M : Manager; S : Seq_Id; Pos : Natural;
      Block : out Positive; Offset : out Natural)
   is
      Slot : constant Positive := Pos / M.Block_Size + 1;
   begin
      if not M.Is_Open (S) or else Pos >= M.Len (S)
        or else M.Table (S, Slot) = 0
      then
         raise Bad_Sequence;
      end if;
      Block  := M.Table (S, Slot);
      Offset := Pos mod M.Block_Size;
   end Locate;

   procedure Share_Prefix (M : in out Manager; Dst, Src : Seq_Id; N : Natural) is
      N_Slots : constant Natural := (N + M.Block_Size - 1) / M.Block_Size;
   begin
      if not M.Is_Open (Dst) or else not M.Is_Open (Src)
        or else M.Len (Dst) /= 0 or else N > M.Len (Src)
      then
         raise Bad_Sequence;
      end if;
      for K in 1 .. N_Slots loop
         declare
            B : constant Natural := M.Table (Src, K);
         begin
            M.Table (Dst, K) := B;         -- point at the same physical block
            M.Ref (B) := M.Ref (B) + 1;    -- shared -> bump refcount, no alloc
         end;
      end loop;
      M.Len (Dst) := N;
   end Share_Prefix;

   procedure Close_Seq (M : in out Manager; S : Seq_Id) is
   begin
      if not M.Is_Open (S) then return; end if;
      for K in 1 .. Slots_Of (M, S) loop
         declare
            B : constant Natural := M.Table (S, K);
         begin
            if B /= 0 and then M.Ref (B) > 0 then
               M.Ref (B) := M.Ref (B) - 1;
               if M.Ref (B) = 0 then       -- last holder -> back to the pool
                  M.Free := M.Free + 1;
               end if;
            end if;
         end;
      end loop;
      M.Is_Open (S) := False;
      M.Len (S)     := 0;
      Zero_Row (M, S);
   end Close_Seq;

end KV_Pool;
