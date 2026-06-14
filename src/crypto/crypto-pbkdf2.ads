---------------------------------------------------------------------
-- Crypto.PBKDF2 — password-based key derivation (PBKDF2-HMAC-SHA256,
-- RFC 8018 §5.2). Stretches a password + salt into key material over a
-- configurable iteration count (work factor) to slow brute force.
---------------------------------------------------------------------

package Crypto.PBKDF2 is

   procedure Derive
     (Password   : Byte_Array;
      Salt       : Byte_Array;
      Iterations : Positive;
      DK         : out Byte_Array);

end Crypto.PBKDF2;
