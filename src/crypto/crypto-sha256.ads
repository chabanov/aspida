---------------------------------------------------------------------
-- Crypto.SHA256 — SHA-256 hash (FIPS 180-4) and HMAC-SHA256 (RFC 2104 /
-- FIPS 198-1). Used by HKDF for key derivation in the handshake.
--
-- Two interfaces:
--   * Hash (M)        — one-shot, whole-message. SPARK-proved (make prove).
--   * Context         — incremental streaming (Init -> Update* -> Final) for
--     inputs larger than memory (a streamed GGUF). Final is bit-identical to
--     Hash of the concatenation of all Update chunks. Context reuses the
--     same Compress step as Hash (one implementation); it is SPARK-analysed
--     alongside the rest of this unit and cross-checked against Hash on
--     boundary inputs (0/1/63/64/65/127/128 bytes) in test_weight_pin.
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

   --  Incremental streaming hash. Init starts a fresh hash; Update feeds bytes
   --  in any number of chunks of any length; Final returns the digest and
   --  wipes the buffer. Equivalent to Hash of the concatenation of all chunks.
   --
   --  Proof scope: Hash and HMAC are fully SPARK-proved (make prove). The
   --  streaming Context (Update/Final) is SPARK_Mode => Off — its correctness
   --  rests on reusing the same proved Compress step and on a bit-identical
   --  cross-check against Hash over boundary inputs (0/1/63/64/65/127/128
   --  bytes and a full streamed model) in test_weight_pin. SPARK cannot bound
   --  a streamed chunk's length away from Natural'Last, so the loop-invariant
   --  length arithmetic is not dischargeable; this matches the existing
   --  Hash annotation that message lengths here are far below 2 GiB.
   type Context is private;

   procedure Init (Ctx : out Context)
     with Global => null;

   procedure Update (Ctx : in out Context; Data : Byte_Array)
     with Global => null;

   --  Final is a procedure (not a function) because SPARK forbids functions
   --  with in-out parameters; Ctx is wiped as a side effect.
   procedure Final (Ctx : in out Context; D : out Digest)
     with Global => null;

private

   --  The 8 working-state words. Exposed (package-private) so Context can hold
   --  the state and Compress can be shared between Hash and Context.
   type Words8 is array (0 .. 7) of U32;

   type Context is record
      H       : Words8 := [others => 0];
      Buf     : Byte_Array (0 .. Block_Size - 1) := [others => 0];
      Buf_Len : Natural := 0;            --  partial bytes pending in Buf (0..63)
      Bit_Len : U64 := 0;               --  total message length in bits
   end record;

end Crypto.SHA256;