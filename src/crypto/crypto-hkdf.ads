---------------------------------------------------------------------
-- Crypto.HKDF — HMAC-based Extract-and-Expand KDF (RFC 5869, SHA-256).
--
-- Extract concentrates input keying material (e.g. an X25519 shared
-- secret) plus an optional salt into a pseudorandom key; Expand stretches
-- that PRK into as many context-bound output bytes as needed (e.g. the
-- per-direction session keys + nonces for the secure channel).
---------------------------------------------------------------------

with Crypto.SHA256;

--  SPARK_Mode: flow-analysed (initialisation, data dependencies, non-aliasing).
package Crypto.HKDF with SPARK_Mode => On is

   --  PRK = HMAC-SHA256(salt, IKM). Empty salt is treated as 32 zero bytes
   --  (equivalent under HMAC), per RFC 5869 §2.2.
   procedure Extract
     (Salt : Byte_Array; IKM : Byte_Array; PRK : out Crypto.SHA256.Digest)
     with Global => null;

   --  Fill Output with OKM = T(1) | T(2) | ... (RFC 5869 §2.3). Output'Length
   --  must be <= 255 * 32, and Info must be small (a context label; callers
   --  pass a few bytes) so the per-block input buffer 32 + Info'Length + 1
   --  stays well within a Natural index. Max_Info_Len is generous for any
   --  sane context string and keeps the buffer bound trivially provable.
   Max_Info_Len : constant := 1024;

   procedure Expand
     (PRK : Crypto.SHA256.Digest; Info : Byte_Array; Output : out Byte_Array)
     with Global => null,
          Pre    => Output'Length <= 255 * 32
                    and then Info'Length <= Max_Info_Len;

end Crypto.HKDF;
