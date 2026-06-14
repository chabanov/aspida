---------------------------------------------------------------------
-- Crypto — root of the self-contained pure-Ada crypto library.
--
-- Common byte/word types and the security-sensitive helpers shared by
-- every primitive: constant-time comparison (no secret-dependent branch)
-- and best-effort secret zeroization. Little-endian load/store match the
-- conventions of ChaCha20/Poly1305/X25519 (RFC 8439 / RFC 7748).
---------------------------------------------------------------------

with Interfaces;

package Crypto is

   subtype U8  is Interfaces.Unsigned_8;
   subtype U32 is Interfaces.Unsigned_32;
   subtype U64 is Interfaces.Unsigned_64;

   type Byte_Array is array (Natural range <>) of U8;

   --  Equality that runs in time independent of WHERE the inputs differ
   --  (folds all byte diffs before testing). Length is not secret, so a
   --  length mismatch may short-circuit. Use this for MAC/tag comparison.
   function Const_Time_Equal (A, B : Byte_Array) return Boolean;

   --  Overwrite secret material with zeros. Written to resist dead-store
   --  elimination (the post-loop read forces the writes to be observable),
   --  so a key/plaintext buffer is actually cleared, not optimised away.
   procedure Wipe (A : in out Byte_Array);

   --  Little-endian load/store of 32- and 64-bit words from/to a byte slice.
   function  Load_LE32 (A : Byte_Array; Offset : Natural) return U32;
   procedure Store_LE32 (A : in out Byte_Array; Offset : Natural; V : U32);
   procedure Store_LE64 (A : in out Byte_Array; Offset : Natural; V : U64);

end Crypto;
