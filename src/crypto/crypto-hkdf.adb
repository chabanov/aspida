---------------------------------------------------------------------
-- Crypto.HKDF body — RFC 5869 (HMAC-SHA256)
---------------------------------------------------------------------

package body Crypto.HKDF with SPARK_Mode => On is

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
      Output := [others => 0];   -- fully initialised for flow; filled below
      for I in 1 .. N loop
         declare
            --  Input = T(i-1) | Info | byte(i). Sized for the maximum
            --  (Prev is at most 32 bytes) so the bound is constant, not a
            --  variable subtype constraint (which SPARK forbids); only the
            --  filled prefix In_Buf (0 .. P - 1) is fed to HMAC.
            In_Buf : Byte_Array (0 .. 32 + Info'Length) := [others => 0];
            P      : Natural := 0;
            T      : Digest := [others => 0];
         begin
            for J in 0 .. Prev_Len - 1 loop
               In_Buf (P) := Prev (J); P := P + 1;
            end loop;
            for J in Info'Range loop
               In_Buf (P) := Info (J); P := P + 1;
            end loop;
            In_Buf (P) := U8 (I); P := P + 1;

            HMAC (PRK, In_Buf (0 .. P - 1), T);

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
            Wipe (In_Buf);   -- holds T(i-1), a secret
            Wipe (T);
         end;
      end loop;
      Wipe (Prev);           -- last secret block
   end Expand;

end Crypto.HKDF;
