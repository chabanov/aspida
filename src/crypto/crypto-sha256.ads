---------------------------------------------------------------------
-- Crypto.SHA256 — SHA-256 hash (FIPS 180-4) and HMAC-SHA256 (RFC 2104 /
-- FIPS 198-1). Used by HKDF for key derivation in the handshake.
---------------------------------------------------------------------

package Crypto.SHA256 is

   subtype Digest is Byte_Array (0 .. 31);   -- 256-bit output
   Block_Size : constant := 64;              -- 512-bit input block

   function Hash (M : Byte_Array) return Digest;

   procedure HMAC (Key, Msg : Byte_Array; Mac : out Digest);

end Crypto.SHA256;
