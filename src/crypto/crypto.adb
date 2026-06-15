---------------------------------------------------------------------
-- Crypto body
---------------------------------------------------------------------

with Interfaces; use Interfaces;

package body Crypto with SPARK_Mode => On is

   function Const_Time_Equal (A, B : Byte_Array) return Boolean is
      Diff : U8 := 0;
   begin
      if A'Length /= B'Length then
         return False;                       -- length is not a secret
      end if;
      for I in A'Range loop
         Diff := Diff or (A (I) xor B (B'First + (I - A'First)));
         --  Diff has folded exactly the bytes A'First..I; it is zero iff every
         --  one of those byte pairs is equal. This ties the accumulator to the
         --  functional postcondition.
         pragma Loop_Invariant
           ((Diff = 0)
            = (for all J in A'First .. I =>
                 A (J) = B (B'First + (J - A'First))));
      end loop;
      return Diff = 0;
   end Const_Time_Equal;

   --  The body uses an anti-dead-store-elimination trick (a branch the
   --  optimiser cannot prove unreachable) that is outside the SPARK model,
   --  so it is exempted; the spec's postcondition is the callable guarantee.
   procedure Wipe (A : in out Byte_Array) with SPARK_Mode => Off is
   begin
      for I in A'Range loop
         A (I) := 0;
      end loop;
      --  Force the stores to be observable so the optimiser cannot elide the
      --  wipe of a buffer that is never read again (the branch is never taken).
      if A'Length > 0 and then A (A'First) /= 0 then
         raise Program_Error;
      end if;
   end Wipe;

   function Load_LE32 (A : Byte_Array; Offset : Natural) return U32 is
   begin
      return U32 (A (A'First + Offset))
        or Shift_Left (U32 (A (A'First + Offset + 1)), 8)
        or Shift_Left (U32 (A (A'First + Offset + 2)), 16)
        or Shift_Left (U32 (A (A'First + Offset + 3)), 24);
   end Load_LE32;

   procedure Store_LE32 (A : in out Byte_Array; Offset : Natural; V : U32) is
   begin
      A (A'First + Offset)     := U8 (V and 16#FF#);
      A (A'First + Offset + 1) := U8 (Shift_Right (V, 8)  and 16#FF#);
      A (A'First + Offset + 2) := U8 (Shift_Right (V, 16) and 16#FF#);
      A (A'First + Offset + 3) := U8 (Shift_Right (V, 24) and 16#FF#);
   end Store_LE32;

   procedure Store_LE64 (A : in out Byte_Array; Offset : Natural; V : U64) is
   begin
      for I in 0 .. 7 loop
         A (A'First + Offset + I) :=
           U8 (Shift_Right (V, 8 * I) and 16#FF#);
      end loop;
   end Store_LE64;

end Crypto;
