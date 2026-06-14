---------------------------------------------------------------------
-- Crypto.Random — cryptographically secure random bytes.
--
-- Sourced from the OS CSPRNG via getentropy(2) (macOS/BSD; also glibc
-- 2.25+), which draws from the kernel pool. Used for ephemeral keys and
-- nonces. Raises if the OS cannot provide entropy (never silently weak).
---------------------------------------------------------------------

package Crypto.Random is

   procedure Fill (Buf : out Byte_Array);

end Crypto.Random;
