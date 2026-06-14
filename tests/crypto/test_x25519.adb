---------------------------------------------------------------------
-- Test_X25519 — X25519 validated against RFC 7748: the §5.2 scalar-mult
-- vectors and the §6.1 Diffie-Hellman key agreement (Alice/Bob).
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with Crypto.X25519;

procedure Test_X25519 is
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

   function H (S : String) return Crypto.X25519.Key_256 is
      R : Crypto.X25519.Key_256;
      N : Natural := 0;
      I : Integer := S'First;
   begin
      while I <= S'Last and then N <= 31 loop
         if S (I) = ' ' then
            I := I + 1;
         else
            R (N) := Nyb (S (I)) * 16 + Nyb (S (I + 1));
            N := N + 1; I := I + 2;
         end if;
      end loop;
      return R;
   end H;

   function Eq (A, B : Crypto.X25519.Key_256) return Boolean is
   begin
      for I in A'Range loop
         if A (I) /= B (I) then
            return False;
         end if;
      end loop;
      return True;
   end Eq;

begin
   Put_Line ("=== RFC 7748 X25519 Test Suite ===");
   New_Line;

   --  §5.2 scalar-multiplication vectors.
   Assert ("X25519 §5.2 vector 1",
     Eq (Crypto.X25519.Scalar_Mult
           (H ("a546e36bf0527c9d3b16154b82465edd"
             & "62144c0ac1fc5a18506a2244ba449ac4"),
            H ("e6db6867583030db3594c1a424b15f7c"
             & "726624ec26b3353b10a903a6d0ab1c4c")),
         H ("c3da55379de9c6908e94ea4df28d084f"
          & "32eccf03491c71f754b4075577a28552")));

   Assert ("X25519 §5.2 vector 2",
     Eq (Crypto.X25519.Scalar_Mult
           (H ("4b66e9d4d1b4673c5ad22691957d6af5"
             & "c11b6421e0ea01d42ca4169e7918ba0d"),
            H ("e5210f12786811d3f4b7959d0538ae2c"
             & "31dbe7106fc03c3efc4cd549c715a493")),
         H ("95cbde9476e8907d7aade45cb4b873f8"
          & "8b595a68799fa152e6f8f7647aac7957")));

   --  §6.1 Diffie-Hellman key agreement.
   declare
      A_Priv : constant Crypto.X25519.Key_256 :=
        H ("77076d0a7318a57d3c16c17251b26645"
         & "df4c2f87ebc0992ab177fba51db92c2a");
      A_Pub  : constant Crypto.X25519.Key_256 :=
        H ("8520f0098930a754748b7ddcb43ef75a"
         & "0dbf3a0d26381af4eba4a98eaa9b4e6a");
      B_Priv : constant Crypto.X25519.Key_256 :=
        H ("5dab087e624a8a4b79e17f8b83800ee6"
         & "6f3bb1292618b6fd1c2f8b27ff88e0eb");
      B_Pub  : constant Crypto.X25519.Key_256 :=
        H ("de9edb7d7b7dc1b4d35b61c2ece43537"
         & "3f8343c85b78674dadfc7e146f882b4f");
      Shared : constant Crypto.X25519.Key_256 :=
        H ("4a5d9d5ba4ce2de1728e3bf480350f25"
         & "e07e21c947d19e3376f09b3c1e161742");
   begin
      Assert ("public key (Alice)", Eq (Crypto.X25519.Public_Key (A_Priv), A_Pub));
      Assert ("public key (Bob)",   Eq (Crypto.X25519.Public_Key (B_Priv), B_Pub));
      Assert ("ECDH Alice*Bob_pub",
        Eq (Crypto.X25519.Scalar_Mult (A_Priv, B_Pub), Shared));
      Assert ("ECDH Bob*Alice_pub",
        Eq (Crypto.X25519.Scalar_Mult (B_Priv, A_Pub), Shared));
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_X25519;
