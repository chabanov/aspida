---------------------------------------------------------------------
-- Crypto.HKDF body — RFC 5869 (HMAC-SHA256)
---------------------------------------------------------------------

package body Crypto.HKDF is

   use Crypto.SHA256;

   procedure Extract
     (Salt : Byte_Array; IKM : Byte_Array; PRK : out Digest)
   is
   begin
      --  HMAC with an empty key pads to all-zero block, which equals the
      --  RFC's "HashLen zeros" salt, so no special case is needed.
      HMAC (Salt, IKM, PRK);
   end Extract;

   procedure Expand
     (PRK : Digest; Info : Byte_Array; Output : out Byte_Array)
   is
      N     : constant Natural := (Output'Length + 31) / 32;   -- ceil(L/HashLen)
      Prev  : Byte_Array (0 .. 31) := [others => 0];
      Prev_Len : Natural := 0;                                 -- T(0) is empty
      Pos   : Natural := 0;
   begin
      for I in 1 .. N loop
         declare
            --  Input = T(i-1) | Info | byte(i)
            In_Buf : Byte_Array (0 .. Prev_Len + Info'Length + 1 - 1);
            P      : Natural := 0;
            T      : Digest;
         begin
            for J in 0 .. Prev_Len - 1 loop
               In_Buf (P) := Prev (J); P := P + 1;
            end loop;
            for J in Info'Range loop
               In_Buf (P) := Info (J); P := P + 1;
            end loop;
            In_Buf (P) := U8 (I);

            HMAC (PRK, In_Buf, T);

            declare
               Take : constant Natural := Natural'Min (32, Output'Length - Pos);
            begin
               for J in 0 .. Take - 1 loop
                  Output (Output'First + Pos + J) := T (J);
               end loop;
               Pos := Pos + Take;
            end;

            Prev (0 .. 31) := T;
            Prev_Len := 32;
         end;
      end loop;
   end Expand;

end Crypto.HKDF;
