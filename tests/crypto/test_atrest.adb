---------------------------------------------------------------------
-- Test_Atrest — password-encrypted session store: round-trip, wrong
-- password rejection, and tamper rejection.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Streams.Stream_IO;
with Interfaces;     use Interfaces;
with Crypto;         use Crypto;
with At_Rest;

procedure Test_Atrest is
   use Ada.Text_IO;
   package SIO renames Ada.Streams.Stream_IO;

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

   Path : constant String := "/tmp/aspida_atrest_test.bin";
   Pw   : constant Byte_Array := To_B ("correct horse battery staple");
   Data : constant Byte_Array := To_B ("encrypted session history: turn 1, turn 2...");
begin
   Put_Line ("=== At_Rest (encrypted store) Test Suite ===");
   New_Line;

   At_Rest.Save (Path, Pw, Data, Iterations => 1000);

   --  Correct password recovers the data.
   declare
      Got : constant Byte_Array := At_Rest.Load (Path, Pw);
   begin
      Assert ("round-trip with correct password", Eq (Got, Data));
   end;

   --  Wrong password is rejected.
   declare
      Got : Byte_Array := To_B ("");
      pragma Unreferenced (Got);
      Raised : Boolean := False;
   begin
      Got := At_Rest.Load (Path, To_B ("wrong password"));
      Assert ("wrong password rejected", False);
   exception
      when At_Rest.Decrypt_Error =>
         Raised := True;
         Assert ("wrong password rejected", Raised);
   end;

   --  Tampering with the ciphertext is rejected (read, flip a byte, rewrite).
   declare
      F : SIO.File_Type;
   begin
      SIO.Open (F, SIO.In_File, Path);
      declare
         Sz  : constant Natural := Natural (SIO.Size (F));
         Buf : Byte_Array (0 .. Sz - 1);
      begin
         Byte_Array'Read (SIO.Stream (F), Buf);
         SIO.Close (F);
         Buf (Sz - 20) := Buf (Sz - 20) xor 16#FF#;   -- a ciphertext byte
         SIO.Create (F, SIO.Out_File, Path);
         Byte_Array'Write (SIO.Stream (F), Buf);
         SIO.Close (F);
      end;
   end;
   declare
      Got : Byte_Array := To_B ("");
      pragma Unreferenced (Got);
      Raised : Boolean := False;
   begin
      Got := At_Rest.Load (Path, Pw);
      Assert ("tampered file rejected", False);
   exception
      when At_Rest.Decrypt_Error =>
         Raised := True;
         Assert ("tampered file rejected", Raised);
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Atrest;
