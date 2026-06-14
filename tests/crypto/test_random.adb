---------------------------------------------------------------------
-- Test_Random — sanity checks for the OS CSPRNG (no fixed vectors are
-- possible). Verifies non-zero output, that two draws differ, and that
-- the multi-chunk path (> 256 bytes per getentropy call) works.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Interfaces;     use Interfaces;
with Crypto;         use Crypto;
with Crypto.Random;

procedure Test_Random is
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

   function All_Zero (A : Byte_Array) return Boolean is
   begin
      for X of A loop
         if X /= 0 then
            return False;
         end if;
      end loop;
      return True;
   end All_Zero;

   function Equal (A, B : Byte_Array) return Boolean is
   begin
      for I in A'Range loop
         if A (I) /= B (I) then
            return False;
         end if;
      end loop;
      return True;
   end Equal;

   A, B : Byte_Array (0 .. 31);
   Big  : Byte_Array (0 .. 299) := [others => 0];   -- > 256: exercises chunking
begin
   Put_Line ("=== Crypto.Random Test Suite ===");
   New_Line;

   Crypto.Random.Fill (A);
   Crypto.Random.Fill (B);
   Assert ("draw is not all zero", not All_Zero (A));
   Assert ("two draws differ", not Equal (A, B));

   Crypto.Random.Fill (Big);
   Assert (">256-byte draw filled (not all zero)", not All_Zero (Big));

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Random;
