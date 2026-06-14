---------------------------------------------------------------------
-- Test_Crypto — ChaCha20-Poly1305 AEAD validated bit-exact against the
-- official RFC 8439 test vectors (§2.4.2 ChaCha20, §2.5.2 Poly1305,
-- §2.8.2 AEAD), plus round-trip and tamper-detection checks.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with Crypto.ChaCha20;
with Crypto.Poly1305;
with Crypto.AEAD;

procedure Test_Crypto is
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

   --  Parse a hex string (spaces ignored) into bytes.
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
            N := N + 1;
            I := I + 2;
         end if;
      end loop;
      return Tmp (0 .. N - 1);
   end H;

   --  Text to bytes.
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

   Sunscreen : constant String :=
     "Ladies and Gentlemen of the class of '99: If I could offer you only "
     & "one tip for the future, sunscreen would be it.";

begin
   Put_Line ("=== RFC 8439 ChaCha20-Poly1305 Test Suite ===");
   New_Line;

   --  RFC 8439 §2.4.2 — ChaCha20 encryption.
   declare
      Key   : constant Byte_Array := H ("000102030405060708090a0b0c0d0e0f"
                                       & "101112131415161718191a1b1c1d1e1f");
      Nonce : constant Byte_Array := H ("000000000000004a00000000");
      PT    : constant Byte_Array := To_B (Sunscreen);
      Want  : constant Byte_Array := H (
         "6e2e359a2568f98041ba0728dd0d6981"
       & "e97e7aec1d4360c20a27afccfd9fae0b"
       & "f91b65c5524733ab8f593dabcd62b357"
       & "1639d624e65152ab8f530c359f0861d8"
       & "07ca0dbf500d6a6156a38e088a22b65e"
       & "52bc514d16ccf806818ce91ab7793736"
       & "5af90bbf74a35be6b40b8eedf2785e42"
       & "874d");
      CT : Byte_Array (PT'Range);
   begin
      Crypto.ChaCha20.XOR_Stream (Key, Nonce, 1, PT, CT);
      Assert ("ChaCha20 §2.4.2 encryption", Eq (CT, Want));
   end;

   --  RFC 8439 §2.5.2 — Poly1305.
   declare
      Key : constant Byte_Array := H ("85d6be7857556d337f4452fe42d506a8"
                                     & "0103808afb0db2fd4abff6af4149f51b");
      Msg : constant Byte_Array := To_B ("Cryptographic Forum Research Group");
      Want : constant Byte_Array := H ("a8061dc1305136c6c22b8baf0c0127a9");
      Tag  : Crypto.Poly1305.Tag_128;
   begin
      Crypto.Poly1305.MAC (Key, Msg, Tag);
      Assert ("Poly1305 §2.5.2 MAC", Eq (Tag, Want));
   end;

   --  RFC 8439 §2.8.2 — AEAD seal (ciphertext + tag), open, and tamper.
   declare
      Key   : constant Byte_Array := H ("808182838485868788898a8b8c8d8e8f"
                                       & "909192939495969798999a9b9c9d9e9f");
      Nonce : constant Byte_Array := H ("070000004041424344454647");
      AAD   : constant Byte_Array := H ("50515253c0c1c2c3c4c5c6c7");
      PT    : constant Byte_Array := To_B (Sunscreen);
      Want_CT : constant Byte_Array := H (
         "d31a8d34648e60db7b86afbc53ef7ec2"
       & "a4aded51296e08fea9e2b5a736ee62d6"
       & "3dbea45e8ca9671282fafb69da92728b"
       & "1a71de0a9e060b2905d6a5b67ecd3b36"
       & "92ddbd7f2d778b8c9803aee328091b58"
       & "fab324e4fad675945585808b4831d7bc"
       & "3ff4def08e4b7a9de576d26586cec64b"
       & "6116");
      Want_Tag : constant Byte_Array := H ("1ae10b594f09e26a7e902ecbd0600691");
      CT  : Byte_Array (PT'Range);
      Tag : Crypto.AEAD.Tag_128;
   begin
      Crypto.AEAD.Seal (Key, Nonce, AAD, PT, CT, Tag);
      Assert ("AEAD §2.8.2 ciphertext", Eq (CT, Want_CT));
      Assert ("AEAD §2.8.2 tag", Eq (Tag, Want_Tag));

      --  Round-trip open.
      declare
         Got : Byte_Array (PT'Range);
         OK  : constant Boolean :=
           Crypto.AEAD.Open (Key, Nonce, AAD, CT, Tag, Got);
      begin
         Assert ("AEAD open authenticates", OK);
         Assert ("AEAD open recovers plaintext", Eq (Got, PT));
      end;

      --  Tamper: flip one ciphertext byte -> must reject.
      declare
         Bad : Byte_Array := CT;
         Got : Byte_Array (PT'Range);
         OK  : Boolean;
      begin
         Bad (Bad'First) := Bad (Bad'First) xor 1;
         OK := Crypto.AEAD.Open (Key, Nonce, AAD, Bad, Tag, Got);
         Assert ("AEAD rejects tampered ciphertext", not OK);
      end;

      --  Tamper: flip one tag byte -> must reject.
      declare
         Bad_Tag : Crypto.AEAD.Tag_128 := Tag;
         Got : Byte_Array (PT'Range);
         OK  : Boolean;
      begin
         Bad_Tag (Bad_Tag'Last) := Bad_Tag (Bad_Tag'Last) xor 16#80#;
         OK := Crypto.AEAD.Open (Key, Nonce, AAD, CT, Bad_Tag, Got);
         Assert ("AEAD rejects tampered tag", not OK);
      end;
   end;

   --  Utility checks.
   declare
      A : constant Byte_Array := H ("00112233");
      B : constant Byte_Array := H ("00112233");
      C : constant Byte_Array := H ("00112234");
      W : Byte_Array := H ("deadbeef");
   begin
      Assert ("Const_Time_Equal equal", Const_Time_Equal (A, B));
      Assert ("Const_Time_Equal differ", not Const_Time_Equal (A, C));
      Wipe (W);
      Assert ("Wipe zeroes", W (W'First) = 0 and W (W'Last) = 0);
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Crypto;
