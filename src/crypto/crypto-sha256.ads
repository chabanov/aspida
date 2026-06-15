---------------------------------------------------------------------
-- Crypto.SHA256 — SHA-256 hash (FIPS 180-4) and HMAC-SHA256 (RFC 2104 /
-- FIPS 198-1). Used by HKDF for key derivation in the handshake.
---------------------------------------------------------------------

--  SPARK_Mode: formally analysed for absence of run-time errors and flow.
--  The round arithmetic is all 32-bit modular (no overflow checks); the only
--  justified checks concern message-length arithmetic on a Byte_Array that
--  would have to span >2 GiB from index 0 (not constructible in practice).
package Crypto.SHA256 with SPARK_Mode => On is

   subtype Digest is Byte_Array (0 .. 31);   -- 256-bit output
   Block_Size : constant := 64;              -- 512-bit input block

   function Hash (M : Byte_Array) return Digest
     with Global => null;

   procedure HMAC (Key, Msg : Byte_Array; Mac : out Digest)
     with Global => null;

end Crypto.SHA256;
