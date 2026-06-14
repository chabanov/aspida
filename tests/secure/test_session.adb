---------------------------------------------------------------------
-- Test_Session — encrypted persistent history: a session is written
-- encrypted, reloaded+decrypted in a fresh store, and rejected under a
-- wrong master password.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Environment_Variables;
with Ada.Directories;
with Ada.Strings.Fixed;
with Ada.Streams.Stream_IO;
with Session_Store;
with At_Rest;

procedure Test_Session is
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

   function Has (Hay, Needle : String) return Boolean is
     (Ada.Strings.Fixed.Index (Hay, Needle) > 0);

   Id   : constant String := "unit-test-session";
   File : constant String := "sessions/" & Id & ".session";
begin
   Put_Line ("=== Session_Store (encrypted history) Test Suite ===");
   New_Line;

   if Ada.Directories.Exists (File) then
      Ada.Directories.Delete_File (File);
   end if;

   --  Persistence enabled by setting the master password.
   Ada.Environment_Variables.Set ("ASPIDA_STORE_PASSWORD", "master-secret-pw");
   Assert ("persistence enabled with password set", Session_Store.Enabled);

   --  Write two turns; the file is created encrypted.
   declare
      S : Session_Store.Store;
   begin
      Session_Store.Open (S, Id);
      Session_Store.Append_Turn (S, "Привіт", "Вітаю!");
      Session_Store.Append_Turn (S, "Як справи?", "Чудово, дякую.");
      Assert ("two turns recorded", Session_Store.Turn_Count (S) = 2);
      Session_Store.Close (S);
   end;
   Assert ("encrypted history file created", Ada.Directories.Exists (File));

   --  The raw bytes on disk must NOT contain the plaintext.
   declare
      package SIO renames Ada.Streams.Stream_IO;
      F : SIO.File_Type;
   begin
      SIO.Open (F, SIO.In_File, File);
      declare
         Sz  : constant Natural := Natural (SIO.Size (F));
         Raw : String (1 .. Sz);
      begin
         String'Read (SIO.Stream (F), Raw);
         SIO.Close (F);
         Assert ("plaintext absent from encrypted file",
           not Has (Raw, "Вітаю!") and then not Has (Raw, "Чудово"));
      end;
   end;

   --  Reload in a fresh store -> decrypts the transcript.
   declare
      S : Session_Store.Store;
   begin
      Session_Store.Open (S, Id);
      declare
         T : constant String := Session_Store.Transcript (S);
      begin
         Assert ("reloaded transcript has turn 1", Has (T, "Вітаю!"));
         Assert ("reloaded transcript has turn 2", Has (T, "Чудово, дякую."));
      end;
      Session_Store.Close (S);
   end;

   --  Wrong master password -> reload is rejected.
   Ada.Environment_Variables.Set ("ASPIDA_STORE_PASSWORD", "WRONG-pw");
   declare
      S : Session_Store.Store;
      Raised : Boolean := False;
   begin
      Session_Store.Open (S, Id);
      Session_Store.Close (S);
      Assert ("wrong password rejected on reload", False);
   exception
      when At_Rest.Decrypt_Error =>
         Raised := True;
         Assert ("wrong password rejected on reload", Raised);
   end;

   --  Cleanup.
   if Ada.Directories.Exists (File) then
      Ada.Directories.Delete_File (File);
   end if;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Session;
