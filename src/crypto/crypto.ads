---------------------------------------------------------------------
-- Crypto — root of the self-contained pure-Ada crypto library.
--
-- Common byte/word types and the security-sensitive helpers shared by
-- every primitive: constant-time comparison (no secret-dependent branch)
-- and best-effort secret zeroization. Little-endian load/store match the
-- conventions of ChaCha20/Poly1305/X25519 (RFC 8439 / RFC 7748).
---------------------------------------------------------------------

with Interfaces;

--  SPARK_Mode: this root unit is formally analysed (flow + absence of
--  run-time errors) by gnatprove. The contracts below make the index
--  preconditions explicit so callers' accesses are proved in range.
package Crypto with SPARK_Mode => On is

   use type Interfaces.Unsigned_8;

   subtype U8  is Interfaces.Unsigned_8;
   subtype U32 is Interfaces.Unsigned_32;
   subtype U64 is Interfaces.Unsigned_64;

   type Byte_Array is array (Natural range <>) of U8;

   --  Equality that runs in time independent of WHERE the inputs differ
   --  (folds all byte diffs before testing). Length is not secret, so a
   --  length mismatch may short-circuit. Use this for MAC/tag comparison.
   --  The postcondition pins the functional meaning (proved bit-exact).
   function Const_Time_Equal (A, B : Byte_Array) return Boolean
     with Global => null,
          Post   => Const_Time_Equal'Result =
                      (A'Length = B'Length
                       and then (for all I in A'Range =>
                                   A (I) = B (B'First + (I - A'First))));

   --  Overwrite secret material with zeros. Written to resist dead-store
   --  elimination (the post-loop read forces the writes to be observable),
   --  so a key/plaintext buffer is actually cleared, not optimised away.
   --  The body steps outside SPARK (the anti-DSE trick is an optimisation
   --  hack), but the postcondition is the guarantee callers rely on.
   procedure Wipe (A : in out Byte_Array)
     with Global            => null,
          Always_Terminates => True,
          Post              => (for all I in A'Range => A (I) = 0);

   --  Little-endian load/store of 32- and 64-bit words from/to a byte slice.
   --  Preconditions guarantee the four/eight accessed bytes are in range.
   --  Bounds are stated via A'Last/A'First (not A'Length, whose computation
   --  could overflow Natural for a full-range array) so gnatprove proves them.
   function  Load_LE32 (A : Byte_Array; Offset : Natural) return U32
     with Global => null,
          Pre    => A'Last >= A'First
                    and then A'Last - A'First >= 3
                    and then Offset <= A'Last - A'First - 3;
   procedure Store_LE32 (A : in out Byte_Array; Offset : Natural; V : U32)
     with Global => null,
          Pre    => A'Last >= A'First
                    and then A'Last - A'First >= 3
                    and then Offset <= A'Last - A'First - 3;
   procedure Store_LE64 (A : in out Byte_Array; Offset : Natural; V : U64)
     with Global => null,
          Pre    => A'Last >= A'First
                    and then A'Last - A'First >= 7
                    and then Offset <= A'Last - A'First - 7;

end Crypto;
