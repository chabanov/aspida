---------------------------------------------------------------------
-- Test_PBKDF2 — PBKDF2-HMAC-SHA256 against the RFC 7914 §11 vectors.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Interfaces;     use Interfaces;
with Crypto;         use Crypto;
with Crypto.PBKDF2;

procedure Test_PBKDF2 is
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
     (case C is
         when '0' .. '9' => U8 (Character'Pos (C) - Character'Pos ('0')),
         when 'a' .. 'f' => U8 (Character'Pos (C) - Character'Pos ('a') + 10),
         when others     => 0);

   function H (S : String) return Byte_Array is
      R : Byte_Array (0 .. S'Length / 2 - 1);
   begin
      for I in R'Range loop
         R (I) := Nyb (S (S'First + 2 * I)) * 16 + Nyb (S (S'First + 2 * I + 1));
      end loop;
      return R;
   end H;

   function To_B (S : String) return Byte_Array is
      R : Byte_Array (0 .. S'Length - 1);
   begin
      for I in S'Range loop R (I - S'First) := U8 (Character'Pos (S (I))); end loop;
      return R;
   end To_B;

   function Eq (A, B : Byte_Array) return Boolean is
   begin
      if A'Length /= B'Length then return False; end if;
      for I in 0 .. A'Length - 1 loop
         if A (A'First + I) /= B (B'First + I) then return False; end if;
      end loop;
      return True;
   end Eq;

   DK : Byte_Array (0 .. 63);
begin
   Put_Line ("=== PBKDF2-HMAC-SHA256 (RFC 7914 §11) Test Suite ===");
   New_Line;

   Crypto.PBKDF2.Derive (To_B ("passwd"), To_B ("salt"), 1, DK);
   Assert ("PBKDF2 passwd/salt/c=1",
     Eq (DK,
         H ("55ac046e56e3089fec1691c22544b605"
          & "f94185216dde0465e68b9d57c20dacbc"
          & "49ca9cccf179b645991664b39d77ef31"
          & "7c71b845b1e30bd509112041d3a19783")));

   Crypto.PBKDF2.Derive (To_B ("Password"), To_B ("NaCl"), 80_000, DK);
   Assert ("PBKDF2 Password/NaCl/c=80000",
     Eq (DK,
         H ("4ddcd8f60b98be21830cee5ef22701f9"
          & "641a4418d04c0414aeff08876b34ab56"
          & "a1d425a1225833549adb841b51c9b317"
          & "6a272bdebba1d078478f62b397f33c8d")));

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_PBKDF2;
