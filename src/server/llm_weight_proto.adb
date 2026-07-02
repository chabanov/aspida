---------------------------------------------------------------------
-- LLM_Weight_Proto body — little-endian packing of the H19 wire records.
--
-- Encoding uses the SPARK-proved Crypto.Store_LE32 / Store_LE64 (proved
-- little-endian store with bounds preconditions). Decode reads back with a
-- local Load_LE64 (Crypto exposes Load_LE32 but not Load_LE64); it is a
-- plain 8-byte little-endian fold, guarded by the up-front length check so
-- it never reads past the record end.
---------------------------------------------------------------------

with Protocol;

package body LLM_Weight_Proto with SPARK_Mode => On is

   --  The fixed header that follows the tag in a WReq: U64 + U32 + U32.
   WReq_Header : constant Natural := 8 + 4 + 4;  --  16 bytes (after the tag)

   --  Little-endian 64-bit load. Caller guarantees A holds 8 bytes at Off
   --  (the decode path checks the record length first).
   function Load_LE64 (A : Crypto.Byte_Array; Off : Natural) return Crypto.U64 is
      R : Crypto.U64 := 0;
   begin
      for I in 0 .. 7 loop
         R := R or Interfaces.Shift_Left
           (Crypto.U64 (A (A'First + Off + I)), 8 * I);
      end loop;
      return R;
   end Load_LE64;

   function Tag_Of (Msg : Crypto.Byte_Array) return Crypto.U8 is
   begin
      if Msg'Length = 0 then
         return 0;
      end if;
      return Msg (Msg'First);
   end Tag_Of;

   function Encode_WReq
     (Off      : Crypto.U64;
      Count    : Crypto.U32;
      Model_ID : String) return Crypto.Byte_Array
   is
      Size : constant Natural := 1 + WReq_Header + Model_ID'Length;
      R    : Crypto.Byte_Array (0 .. Size - 1);
      pragma Assert (R'Last - R'First = Size - 1);
   begin
      --  Zero first so SPARK flow sees R as initialized; every byte is then
      --  explicitly set by the tag, the LE stores, and the Model_ID copy, so
      --  the result is identical to a direct build.
      R := [others => 0];
      R (0) := Protocol.Tag_WReq;
      Crypto.Store_LE64 (R, 1, Off);
      Crypto.Store_LE32 (R, 9, Count);
      Crypto.Store_LE32 (R, 13, Crypto.U32 (Model_ID'Length));
      for I in Model_ID'Range loop
         R (17 + (I - Model_ID'First)) := Crypto.U8 (Character'Pos (Model_ID (I)));
      end loop;
      return R;
   end Encode_WReq;

   function Encode_WData (Data : Crypto.Byte_Array) return Crypto.Byte_Array is
      R : Crypto.Byte_Array (0 .. Data'Length);
   begin
      R := [others => 0];
      R (0) := Protocol.Tag_WData;
      for I in Data'Range loop
         R (1 + (I - Data'First)) := Data (I);
      end loop;
      return R;
   end Encode_WData;

   function Encode_WErr (Reason : String) return Crypto.Byte_Array is
      R : Crypto.Byte_Array (0 .. Reason'Length);
   begin
      R := [others => 0];
      R (0) := Protocol.Tag_WErr;
      for I in Reason'Range loop
         R (1 + (I - Reason'First)) := Crypto.U8 (Character'Pos (Reason (I)));
      end loop;
      return R;
   end Encode_WErr;

   procedure Decode_WReq
     (Msg      : Crypto.Byte_Array;
      Off      : out Crypto.U64;
      Count    : out Crypto.U32;
      Model_ID : out Ada.Strings.Unbounded.Unbounded_String;
      OK       : out Boolean)
     with SPARK_Mode => Off
   is
      use Ada.Strings.Unbounded;
   begin
      Off      := 0;
      Count    := 0;
      Model_ID := Null_Unbounded_String;
      OK       := False;

      --  Tag (1) + header (16) must be present before any field is read.
      if Msg'Length < 1 + WReq_Header then
         return;
      end if;

      --  The LE helpers index relative to A'First, so pass the field offsets
      --  (1, 9, 13) — not Msg'First-relative — to avoid a double shift.
      Off   := Load_LE64 (Msg, 1);
      Count := Crypto.Load_LE32 (Msg, 9);

      declare
         ID_Len : constant Crypto.U32 := Crypto.Load_LE32 (Msg, 13);
         First  : constant Natural     := Msg'First + 1 + WReq_Header;  --  first model-id byte (absolute)
      begin
         --  A length that would overflow Natural or run past the record end is
         --  a malformed request, not a raise. The frame cap (1 MiB) bounds
         --  Msg'Length, so 2_000_000 is a safe upper guard before the Natural
         --  conversion.
         if ID_Len > 2_000_000 then
            return;
         end if;
         if First + Natural (ID_Len) > Msg'Last + 1 then
            return;
         end if;

         declare
            S : String (1 .. Natural (ID_Len));
         begin
            for I in 0 .. Natural (ID_Len) - 1 loop
               S (1 + I) := Character'Val (Msg (First + I));
            end loop;
            Model_ID := To_Unbounded_String (S);
         end;

         OK := True;
      end;
   end Decode_WReq;

end LLM_Weight_Proto;