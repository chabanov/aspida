---------------------------------------------------------------------
-- Crypto.AEAD — ChaCha20-Poly1305 AEAD (RFC 8439 §2.8)
--
-- Authenticated encryption with associated data. Encrypt-then-MAC:
-- the ciphertext (and the AAD) are authenticated by a Poly1305 tag whose
-- one-time key is derived from a ChaCha20 keystream block (§2.6). Open
-- verifies the tag in constant time and refuses to release plaintext on
-- any mismatch. A (key, nonce) pair MUST never be reused.
---------------------------------------------------------------------

package Crypto.AEAD is

   subtype Key_256  is Byte_Array (0 .. 31);
   subtype Nonce_96 is Byte_Array (0 .. 11);
   subtype Tag_128  is Byte_Array (0 .. 15);

   procedure Seal
     (Key        : Key_256;
      Nonce      : Nonce_96;
      AAD        : Byte_Array;
      Plaintext  : Byte_Array;
      Ciphertext : out Byte_Array;
      Tag        : out Tag_128)
     with Pre => Ciphertext'Length = Plaintext'Length;

   --  Returns True and writes Plaintext only if the tag authenticates;
   --  otherwise returns False and leaves Plaintext zeroed.
   function Open
     (Key        : Key_256;
      Nonce      : Nonce_96;
      AAD        : Byte_Array;
      Ciphertext : Byte_Array;
      Tag        : Tag_128;
      Plaintext  : out Byte_Array) return Boolean
     with Pre => Plaintext'Length = Ciphertext'Length;

end Crypto.AEAD;
