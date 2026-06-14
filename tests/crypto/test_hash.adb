---------------------------------------------------------------------
-- Test_Hash — SHA-256 / HMAC-SHA256 / HKDF validated against the
-- official vectors: FIPS 180-4 (SHA-256), RFC 4231 (HMAC-SHA256),
-- RFC 5869 (HKDF).
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with Crypto.SHA256;
with Crypto.HKDF;

procedure Test_Hash is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Cond : Boolean) is
   begin
      if Cond then
         Put_Line ("  PASS: " & Name); Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name); Failed := Failed + 1;
      end if;
   end Assert;

   function Nyb (C : Character) return U8 is
   begin
      case C is
         when '0' .. '9' => return U8 (Character'Pos (C) - Character'Pos ('0'));
         when 'a' .. 'f' => return U8 (Character'Pos (C) - Character'Pos ('a') + 10);
         when 'A' .. 'F' => return U8 (Character'Pos (C) - Character'Pos ('A') + 10);
         when others     => return 0;
      end case;
   end Nyb;

   function H (S : String) return Byte_Array is
      Tmp : Byte_Array (0 .. S'Length);
      N   : Natural := 0;
      I   : Integer := S'First;
   begin
      while I <= S'Last loop
         if S (I) = ' ' then
            I := I + 1;
         else
            Tmp (N) := Nyb (S (I)) * 16 + Nyb (S (I + 1));
            N := N + 1; I := I + 2;
         end if;
      end loop;
      return Tmp (0 .. N - 1);
   end H;

   function To_B (S : String) return Byte_Array is
      R : Byte_Array (0 .. S'Length - 1);
   begin
      for I in S'Range loop
         R (I - S'First) := U8 (Character'Pos (S (I)));
      end loop;
      return R;
   end To_B;

   function Eq (A, B : Byte_Array) return Boolean is
   begin
      if A'Length /= B'Length then
         return False;
      end if;
      for I in 0 .. A'Length - 1 loop
         if A (A'First + I) /= B (B'First + I) then
            return False;
         end if;
      end loop;
      return True;
   end Eq;

   function Rep (B : U8; N : Natural) return Byte_Array is
      R : constant Byte_Array (0 .. N - 1) := [others => B];
   begin
      return R;
   end Rep;

begin
   Put_Line ("=== SHA-256 / HMAC / HKDF Test Suite ===");
   New_Line;

   --  FIPS 180-4 SHA-256.
   Assert ("SHA-256 empty",
     Eq (Byte_Array (Crypto.SHA256.Hash (H (""))),
         H ("e3b0c44298fc1c149afbf4c8996fb924"
          & "27ae41e4649b934ca495991b7852b855")));
   Assert ("SHA-256 abc",
     Eq (Byte_Array (Crypto.SHA256.Hash (To_B ("abc"))),
         H ("ba7816bf8f01cfea414140de5dae2223"
          & "b00361a396177a9cb410ff61f20015ad")));
   Assert ("SHA-256 two-block",
     Eq (Byte_Array (Crypto.SHA256.Hash (To_B
           ("abcdbcdecdefdefgefghfghighijhijk"
          & "ijkljklmklmnlmnomnopnopq"))),
         H ("248d6a61d20638b8e5c026930c3e6039"
          & "a33ce45964ff2167f6ecedd419db06c1")));

   --  RFC 4231 HMAC-SHA256.
   declare
      Mac : Crypto.SHA256.Digest;
   begin
      Crypto.SHA256.HMAC (Rep (16#0b#, 20), To_B ("Hi There"), Mac);
      Assert ("HMAC-SHA256 RFC 4231 TC1",
        Eq (Byte_Array (Mac),
            H ("b0344c61d8db38535ca8afceaf0bf12b"
             & "881dc200c9833da726e9376c2e32cff7")));
      Crypto.SHA256.HMAC (To_B ("Jefe"),
        To_B ("what do ya want for nothing?"), Mac);
      Assert ("HMAC-SHA256 RFC 4231 TC2",
        Eq (Byte_Array (Mac),
            H ("5bdcc146bf60754e6a042426089575c7"
             & "5a003f089d2739839dec58b964ec3843")));
   end;

   --  RFC 5869 HKDF-SHA256.
   declare
      PRK : Crypto.SHA256.Digest;
      OKM : Byte_Array (0 .. 41);   -- L = 42
   begin
      --  Test Case 1 (with salt and info).
      Crypto.HKDF.Extract (H ("000102030405060708090a0b0c"),
                           Rep (16#0b#, 22), PRK);
      Assert ("HKDF TC1 PRK",
        Eq (Byte_Array (PRK),
            H ("077709362c2e32df0ddc3f0dc47bba63"
             & "90b6c73bb50f9c3122ec844ad7c2b3e5")));
      Crypto.HKDF.Expand (PRK, H ("f0f1f2f3f4f5f6f7f8f9"), OKM);
      Assert ("HKDF TC1 OKM",
        Eq (OKM,
            H ("3cb25f25faacd57a90434f64d0362f2a"
             & "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
             & "34007208d5b887185865")));

      --  Test Case 3 (empty salt and info).
      Crypto.HKDF.Extract (H (""), Rep (16#0b#, 22), PRK);
      Assert ("HKDF TC3 PRK",
        Eq (Byte_Array (PRK),
            H ("19ef24a32c717b167f33a91d6f648bdf"
             & "96596776afdb6377ac434c1c293ccb04")));
      Crypto.HKDF.Expand (PRK, H (""), OKM);
      Assert ("HKDF TC3 OKM",
        Eq (OKM,
            H ("8da4e775a563c18f715f802a063c5a31"
             & "b8a11f5c5ee1879ec3454e5f3c738d2d"
             & "9d201395faa4b61a96c8")));
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Hash;
