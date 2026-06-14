---------------------------------------------------------------------
-- Crypto.ChaCha20 — the ChaCha20 stream cipher (RFC 8439 §2.1–2.4)
--
-- 256-bit key, 96-bit nonce, 32-bit block counter (the IETF variant).
-- All arithmetic is on 32-bit words; the algorithm is inherently
-- constant-time (no secret-dependent branches or memory indexing).
---------------------------------------------------------------------

package Crypto.ChaCha20 is

   subtype Key_256  is Byte_Array (0 .. 31);   -- 32-byte key
   subtype Nonce_96 is Byte_Array (0 .. 11);   -- 12-byte nonce
   subtype Block_64 is Byte_Array (0 .. 63);   -- one keystream block

   --  One 64-byte keystream block for the given counter (RFC 8439 §2.3).
   procedure Keystream_Block
     (Key : Key_256; Nonce : Nonce_96; Counter : U32; B : out Block_64);

   --  XOR the keystream into Input, producing Output (same length), starting
   --  at the given block counter (RFC 8439 §2.4). Encrypt and decrypt are the
   --  same operation. Output'Length must equal Input'Length.
   procedure XOR_Stream
     (Key     : Key_256;
      Nonce   : Nonce_96;
      Counter : U32;
      Input   : Byte_Array;
      Output  : out Byte_Array)
     with Pre => Output'Length = Input'Length;

end Crypto.ChaCha20;
