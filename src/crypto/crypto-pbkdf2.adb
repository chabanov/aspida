---------------------------------------------------------------------
-- Crypto.PBKDF2 body — RFC 8018 §5.2 (PBKDF2-HMAC-SHA256)
---------------------------------------------------------------------

with Interfaces; use Interfaces;
with Crypto.SHA256;

package body Crypto.PBKDF2 is

   procedure Derive
     (Password   : Byte_Array;
      Salt       : Byte_Array;
      Iterations : Positive;
      DK         : out Byte_Array)
   is
      use Crypto.SHA256;
      H_Len    : constant := 32;
      N_Blocks : constant Natural := (DK'Length + H_Len - 1) / H_Len;
      Pos      : Natural := 0;
   begin
      for I in 1 .. N_Blocks loop
         declare
            --  U_1 = HMAC(P, Salt || INT_BE32(i))
            Msg  : Byte_Array (0 .. Salt'Length + 4 - 1);
            U    : Digest;
            U_Nx : Digest;
            T    : Digest;
         begin
            for J in Salt'Range loop
               Msg (J - Salt'First) := Salt (J);
            end loop;
            Msg (Salt'Length)     := U8 (Shift_Right (U32 (I), 24) and 16#FF#);
            Msg (Salt'Length + 1) := U8 (Shift_Right (U32 (I), 16) and 16#FF#);
            Msg (Salt'Length + 2) := U8 (Shift_Right (U32 (I), 8)  and 16#FF#);
            Msg (Salt'Length + 3) := U8 (U32 (I) and 16#FF#);

            HMAC (Password, Msg, U);
            T := U;
            for C in 2 .. Iterations loop
               HMAC (Password, Byte_Array (U), U_Nx);
               U := U_Nx;
               for K in T'Range loop
                  T (K) := T (K) xor U (K);
               end loop;
            end loop;

            declare
               Take : constant Natural := Natural'Min (H_Len, DK'Length - Pos);
            begin
               for K in 0 .. Take - 1 loop
                  DK (DK'First + Pos + K) := T (K);
               end loop;
               Pos := Pos + Take;
            end;
         end;
      end loop;
   end Derive;

end Crypto.PBKDF2;
