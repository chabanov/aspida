---------------------------------------------------------------------
-- Crypto.AEAD body — ChaCha20-Poly1305 (RFC 8439 §2.6–2.8)
---------------------------------------------------------------------

with Crypto.ChaCha20;
with Crypto.Poly1305;

package body Crypto.AEAD with SPARK_Mode => On is

   --  Poly1305 key generation (RFC 8439 §2.6): first 32 bytes of the
   --  ChaCha20 keystream block at counter 0.
   procedure Poly_Key_Gen
     (Key : Key_256; Nonce : Nonce_96; OTK : out Poly1305.Key_256)
   is
      Block : ChaCha20.Block_64;
   begin
      ChaCha20.Keystream_Block (Key, Nonce, 0, Block);
      OTK := Block (0 .. 31);
      Wipe (Block);
   end Poly_Key_Gen;

   --  Tag over: AAD | pad16 | Ciphertext | pad16 | le64(|AAD|) | le64(|CT|).
   procedure Compute_Tag
     (Key        : Key_256;
      Nonce      : Nonce_96;
      AAD        : Byte_Array;
      Ciphertext : Byte_Array;
      Tag        : out Tag_128)
   is
      AAD_Pad : constant Natural := (16 - AAD'Length mod 16) mod 16;
      CT_Pad  : constant Natural := (16 - Ciphertext'Length mod 16) mod 16;
      Mac_Len : constant Natural :=
        AAD'Length + AAD_Pad + Ciphertext'Length + CT_Pad + 16;
      Mac_Data : Byte_Array (0 .. Mac_Len - 1) := [others => 0];
      OTK : Poly1305.Key_256;
      P   : Natural := 0;
   begin
      Poly_Key_Gen (Key, Nonce, OTK);

      for I in AAD'Range loop
         Mac_Data (P) := AAD (I); P := P + 1;
      end loop;
      P := P + AAD_Pad;                              -- zero padding (already 0)
      for I in Ciphertext'Range loop
         Mac_Data (P) := Ciphertext (I); P := P + 1;
      end loop;
      P := P + CT_Pad;
      Store_LE64 (Mac_Data, P, U64 (AAD'Length));        P := P + 8;
      Store_LE64 (Mac_Data, P, U64 (Ciphertext'Length));

      Poly1305.MAC (OTK, Mac_Data, Tag);
      Wipe (OTK);
      Wipe (Mac_Data);   -- holds AAD || ciphertext; scrub before returning
   end Compute_Tag;

   procedure Seal
     (Key        : Key_256;
      Nonce      : Nonce_96;
      AAD        : Byte_Array;
      Plaintext  : Byte_Array;
      Ciphertext : out Byte_Array;
      Tag        : out Tag_128)
   is
   begin
      ChaCha20.XOR_Stream (Key, Nonce, 1, Plaintext, Ciphertext);
      Compute_Tag (Key, Nonce, AAD, Ciphertext, Tag);
   end Seal;

   function Open
     (Key        : Key_256;
      Nonce      : Nonce_96;
      AAD        : Byte_Array;
      Ciphertext : Byte_Array;
      Tag        : Tag_128;
      Plaintext  : out Byte_Array) return Boolean
     with SPARK_Mode => Off
   is
      Expected : Tag_128;
   begin
      Compute_Tag (Key, Nonce, AAD, Ciphertext, Expected);
      if not Const_Time_Equal (Expected, Tag) then
         Plaintext := [others => 0];            -- never release on auth failure
         Wipe (Expected);                       -- scrub the computed tag
         return False;
      end if;
      ChaCha20.XOR_Stream (Key, Nonce, 1, Ciphertext, Plaintext);
      Wipe (Expected);                          -- scrub the computed tag
      return True;
   end Open;

end Crypto.AEAD;
