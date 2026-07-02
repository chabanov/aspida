---------------------------------------------------------------------
-- Test_Weight_Cache — unit tests for the H19 on-disk AEAD-sealed chunk
-- cache (Phase 3 of H19_WEIGHT_STREAM_ROADMAP.md).
--
-- Validates, against a real temp directory:
--   * Store -> Load round-trips byte-identically
--   * Has reports presence/absence correctly
--   * a blob written under key A and read under key B -> Cache_Miss
--     (never silent cross-model corruption)
--   * a tampered blob -> Cache_Error (AEAD tag failure, loud)
--   * disabled cache (empty dir) -> Enabled False, Store no-op, Has False
--   * multiple distinct chunks coexist and load independently
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Directories;
with Ada.Streams.Stream_IO;
with Interfaces;        use Interfaces;
with Crypto;            use Crypto;
with LLM_Weight_Cache;

procedure Test_Weight_Cache is
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

   Temp_Dir : constant String := "/tmp/aspida_wcache_test";
   Pass     : constant Byte_Array := To_B ("test-password-0123456789");

   package SIO renames Ada.Streams.Stream_IO;

   --  Flip the last byte of a sealed blob (the Poly1305 tag) so the AEAD Open
   --  must fail -> At_Rest.Decrypt_Error -> Cache_Error.
   procedure Corrupt_Tag (Path : String) is
      F    : SIO.File_Type;
      Size : constant Natural := Natural (Ada.Directories.Size (Path));
   begin
      SIO.Open (F, SIO.In_File, Path);
      declare
         Buf : Byte_Array (0 .. Size - 1);
      begin
         Byte_Array'Read (SIO.Stream (F), Buf);
         SIO.Close (F);
         Buf (Size - 1) := Buf (Size - 1) xor 16#FF#;
         SIO.Create (F, SIO.Out_File, Path);
         Byte_Array'Write (SIO.Stream (F), Buf);
         SIO.Close (F);
      end;
   end Corrupt_Tag;

begin
   Put_Line ("=== H19 Weight-Cache Test Suite ===");
   New_Line;

   --  Clean slate.
   if Ada.Directories.Exists (Temp_Dir) then
      Ada.Directories.Delete_Tree (Temp_Dir);
   end if;
   Ada.Directories.Create_Path (Temp_Dir);

   ------------------------------------------------------------------
   --  Round-trip + Has + multi-chunk
   ------------------------------------------------------------------
   declare
      C : LLM_Weight_Cache.Weight_Cache :=
        LLM_Weight_Cache.Open (Temp_Dir, Pass, "model-abc");
   begin
      Assert ("cache enabled with dir + pass", LLM_Weight_Cache.Enabled (C));

      Assert ("fresh cache has no chunk 0", not LLM_Weight_Cache.Has (C, 0));

      LLM_Weight_Cache.Store (C, 0, To_B ("hello-weight-chunk-0"));
      Assert ("chunk 0 present after store", LLM_Weight_Cache.Has (C, 0));
      Assert ("chunk 1 still absent", not LLM_Weight_Cache.Has (C, 1));

      declare
         D : constant LLM_Weight_Cache.Byte_Array_Access :=
           LLM_Weight_Cache.Load (C, 0);
      begin
         Assert ("round-trip chunk 0 byte-identical",
                 Eq (D.all, To_B ("hello-weight-chunk-0")));
      end;

      --  A second, non-adjacent chunk coexists.
      LLM_Weight_Cache.Store (C, 5, To_B ("chunk-five-has-different-bytes"));
      declare
         D0 : constant LLM_Weight_Cache.Byte_Array_Access :=
           LLM_Weight_Cache.Load (C, 0);
         D5 : constant LLM_Weight_Cache.Byte_Array_Access :=
           LLM_Weight_Cache.Load (C, 5);
      begin
         Assert ("chunk 0 still intact after storing chunk 5",
                 Eq (D0.all, To_B ("hello-weight-chunk-0")));
         Assert ("chunk 5 round-trips byte-identical",
                 Eq (D5.all, To_B ("chunk-five-has-different-bytes")));
      end;

      LLM_Weight_Cache.Close (C);
   end;

   ------------------------------------------------------------------
   --  Wrong key -> Cache_Miss (the blob from key A must not surface as B)
   --
   --  To exercise the in-blob key check (rather than a mere "file absent"
   --  miss), store under a key whose Sanitize() collides with a second key,
   --  so the wrong-key open finds the same on-disk file but the bound key
   --  prefix inside the AEAD plaintext does not match. "model abc" (space)
   --  and "model_abc" both sanitize to the subdir "model_abc".
   ------------------------------------------------------------------
   declare
      C0 : LLM_Weight_Cache.Weight_Cache :=
        LLM_Weight_Cache.Open (Temp_Dir, Pass, "model abc");
   begin
      --  Seed the colliding subdir with a blob bound to "model abc".
      LLM_Weight_Cache.Store (C0, 0, To_B ("belongs-to-model-abc"));
      LLM_Weight_Cache.Close (C0);
   end;

   declare
      CB : LLM_Weight_Cache.Weight_Cache :=
        LLM_Weight_Cache.Open (Temp_Dir, Pass, "model_abc");
   begin
      --  Same dir + pass, DIFFERENT key that sanitizes to the same subdir.
      Assert ("second cache enabled", LLM_Weight_Cache.Enabled (CB));
      Assert ("blob present on disk under same sanitized subdir",
              LLM_Weight_Cache.Has (CB, 0));
      begin
         declare
            D : constant LLM_Weight_Cache.Byte_Array_Access :=
              LLM_Weight_Cache.Load (CB, 0);
         begin
            --  Unreachable on a correct miss; reference D (always-false
            --  length test) so -gnatwk does not flag it unused.
            Assert ("wrong-key load raised Cache_Miss", D.all'Length < 0);
         end;
      exception
         when LLM_Weight_Cache.Cache_Miss =>
            Assert ("wrong-key load raised Cache_Miss", True);
         when others =>
            Assert ("wrong-key load raised Cache_Miss", False);
      end;
      LLM_Weight_Cache.Close (CB);
   end;

   ------------------------------------------------------------------
   --  Tamper -> Cache_Error (AEAD tag failure, never silent)
   ------------------------------------------------------------------
   declare
      C : LLM_Weight_Cache.Weight_Cache :=
        LLM_Weight_Cache.Open (Temp_Dir, Pass, "model-tamper");
   begin
      LLM_Weight_Cache.Store (C, 0, To_B ("seal-me-please"));
      --  Reach into the file and flip the tag byte.
      declare
         Sub : constant String := Temp_Dir & "/" & "model-tamper";
      begin
         Corrupt_Tag (Sub & "/chunk_0.enc");
      end;
      begin
         declare
            D : constant LLM_Weight_Cache.Byte_Array_Access :=
              LLM_Weight_Cache.Load (C, 0);
         begin
            --  Unreachable on a correct tamper detection; reference D.
            Assert ("tampered load raised Cache_Error", D.all'Length < 0);
         end;
      exception
         when LLM_Weight_Cache.Cache_Error =>
            Assert ("tampered load raised Cache_Error", True);
         when others =>
            Assert ("tampered load raised Cache_Error", False);
      end;
      LLM_Weight_Cache.Close (C);
   end;

   ------------------------------------------------------------------
   --  Disabled (empty dir) -> Enabled False, Store no-op, Has False
   ------------------------------------------------------------------
   declare
      C : LLM_Weight_Cache.Weight_Cache :=
        LLM_Weight_Cache.Open ("", Pass, "model-disabled");
   begin
      Assert ("empty dir disables cache", not LLM_Weight_Cache.Enabled (C));
      LLM_Weight_Cache.Store (C, 0, To_B ("wont-be-written"));
      Assert ("disabled Store left no chunk", not LLM_Weight_Cache.Has (C, 0));
      LLM_Weight_Cache.Close (C);
   end;

   ------------------------------------------------------------------
   --  Disabled (empty pass) -> also disabled
   ------------------------------------------------------------------
   declare
      C : LLM_Weight_Cache.Weight_Cache :=
        LLM_Weight_Cache.Open (Temp_Dir, To_B (""), "model-nopass");
   begin
      Assert ("empty pass disables cache", not LLM_Weight_Cache.Enabled (C));
      LLM_Weight_Cache.Close (C);
   end;

   --  Cleanup.
   Ada.Directories.Delete_Tree (Temp_Dir);

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Weight_Cache;