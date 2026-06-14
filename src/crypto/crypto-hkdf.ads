---------------------------------------------------------------------
-- Crypto.HKDF — HMAC-based Extract-and-Expand KDF (RFC 5869, SHA-256).
--
-- Extract concentrates input keying material (e.g. an X25519 shared
-- secret) plus an optional salt into a pseudorandom key; Expand stretches
-- that PRK into as many context-bound output bytes as needed (e.g. the
-- per-direction session keys + nonces for the secure channel).
---------------------------------------------------------------------

with Crypto.SHA256;

package Crypto.HKDF is

   --  PRK = HMAC-SHA256(salt, IKM). Empty salt is treated as 32 zero bytes
   --  (equivalent under HMAC), per RFC 5869 §2.2.
   procedure Extract
     (Salt : Byte_Array; IKM : Byte_Array; PRK : out Crypto.SHA256.Digest);

   --  Fill Output with OKM = T(1) | T(2) | ... (RFC 5869 §2.3). Output'Length
   --  must be <= 255 * 32.
   procedure Expand
     (PRK : Crypto.SHA256.Digest; Info : Byte_Array; Output : out Byte_Array)
     with Pre => Output'Length <= 255 * 32;

end Crypto.HKDF;
