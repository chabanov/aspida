---------------------------------------------------------------------
-- At_Rest body
--
-- File layout (all authenticated; the 36-byte header is the AEAD AAD):
--   [0..3]   magic "ASR1"
--   [4..7]   iterations (big-endian u32)
--   [8..23]  PBKDF2 salt (16 bytes)
--   [24..35] AEAD nonce (12 bytes)
--   [36..]   ciphertext
--   [end-16] Poly1305 tag (16 bytes)
---------------------------------------------------------------------

with Interfaces;            use Interfaces;
with Ada.Streams.Stream_IO;
with Crypto.Random;
with Crypto.PBKDF2;
with Crypto.AEAD;

package body At_Rest is

   use Crypto;
   package SIO renames Ada.Streams.Stream_IO;
   use type SIO.Count;

   Magic    : constant Byte_Array (0 .. 3) :=
     [Character'Pos ('A'), Character'Pos ('S'),
      Character'Pos ('R'), Character'Pos ('1')];
   Hdr_Len  : constant := 36;
   Salt_Off : constant := 8;
   Non_Off  : constant := 24;
   Max_File : constant := 64 * 1024 * 1024;   -- 64 MiB sane cap on a store file

   procedure Put_BE32 (A : in out Byte_Array; Off : Natural; V : U32) is
   begin
      A (A'First + Off)     := U8 (Shift_Right (V, 24) and 16#FF#);
      A (A'First + Off + 1) := U8 (Shift_Right (V, 16) and 16#FF#);
      A (A'First + Off + 2) := U8 (Shift_Right (V, 8)  and 16#FF#);
      A (A'First + Off + 3) := U8 (V and 16#FF#);
   end Put_BE32;

   function Get_BE32 (A : Byte_Array; Off : Natural) return U32 is
     (Shift_Left (U32 (A (A'First + Off)), 24)
      or Shift_Left (U32 (A (A'First + Off + 1)), 16)
      or Shift_Left (U32 (A (A'First + Off + 2)), 8)
      or U32 (A (A'First + Off + 3)));

   procedure Save
     (Path       : String;
      Password   : Crypto.Byte_Array;
      Plaintext  : Crypto.Byte_Array;
      Iterations : Positive := 200_000)
   is
      Header : Byte_Array (0 .. Hdr_Len - 1);
      Salt   : Byte_Array (0 .. 15);
      Nonce  : Byte_Array (0 .. 11);
      Key    : Byte_Array (0 .. 31);
      CT     : Byte_Array (0 .. Plaintext'Length - 1);
      Tag    : Crypto.AEAD.Tag_128;
      F      : SIO.File_Type;
   begin
      Crypto.Random.Fill (Salt);
      Crypto.Random.Fill (Nonce);
      Crypto.PBKDF2.Derive (Password, Salt, Iterations, Key);

      for I in 0 .. 3 loop Header (I) := Magic (I); end loop;
      Put_BE32 (Header, 4, U32 (Iterations));
      for I in 0 .. 15 loop Header (Salt_Off + I) := Salt (I); end loop;
      for I in 0 .. 11 loop Header (Non_Off + I)  := Nonce (I); end loop;

      Crypto.AEAD.Seal (Key, Nonce, Header, Plaintext, CT, Tag);
      Wipe (Key);

      SIO.Create (F, SIO.Out_File, Path);
      Byte_Array'Write (SIO.Stream (F), Header);
      if CT'Length > 0 then
         Byte_Array'Write (SIO.Stream (F), CT);
      end if;
      Byte_Array'Write (SIO.Stream (F), Byte_Array (Tag));
      SIO.Close (F);
   end Save;

   function Load
     (Path : String; Password : Crypto.Byte_Array) return Crypto.Byte_Array
   is
      F : SIO.File_Type;
   begin
      SIO.Open (F, SIO.In_File, Path);
      --  Bound the allocation by the on-disk size before reading: a corrupt or
      --  hostile file must not be able to force a multi-gigabyte read (OOM DoS).
      if SIO.Size (F) > SIO.Count (Max_File) then
         SIO.Close (F);
         raise Format_Error with "encrypted store too large";
      end if;
      declare
         Size : constant Natural := Natural (SIO.Size (F));
         Buf  : Byte_Array (0 .. Size - 1);
      begin
         if Size >= 1 then
            Byte_Array'Read (SIO.Stream (F), Buf);
         end if;
         SIO.Close (F);

         if Size < Hdr_Len + 16 then
            raise Format_Error with "file too short";
         end if;
         for I in 0 .. 3 loop
            if Buf (I) /= Magic (I) then
               raise Format_Error with "bad magic";
            end if;
         end loop;

         declare
            Iters  : constant U32 := Get_BE32 (Buf, 4);
            Salt   : Byte_Array (0 .. 15);
            Nonce  : Byte_Array (0 .. 11);
            Header : Byte_Array (0 .. Hdr_Len - 1);
            CT_Len : constant Natural := Size - Hdr_Len - 16;
            CT     : Byte_Array (0 .. CT_Len - 1);
            Tag    : Crypto.AEAD.Tag_128;
            Key    : Byte_Array (0 .. 31);
            PT     : Byte_Array (0 .. CT_Len - 1);
            OK     : Boolean;
         begin
            for I in 0 .. Hdr_Len - 1 loop Header (I) := Buf (I); end loop;
            for I in 0 .. 15 loop Salt (I)  := Buf (Salt_Off + I); end loop;
            for I in 0 .. 11 loop Nonce (I) := Buf (Non_Off + I);  end loop;
            for I in 0 .. CT_Len - 1 loop CT (I) := Buf (Hdr_Len + I); end loop;
            for I in 0 .. 15 loop Tag (I) := Buf (Hdr_Len + CT_Len + I); end loop;

            Crypto.PBKDF2.Derive (Password, Salt, Positive (Iters), Key);
            OK := Crypto.AEAD.Open (Key, Nonce, Header, CT, Tag, PT);
            Wipe (Key);
            if not OK then
               raise Decrypt_Error with "wrong password or corrupted file";
            end if;
            return PT;
         end;
      end;
   end Load;

end At_Rest;
