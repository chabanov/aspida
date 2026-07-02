---------------------------------------------------------------------
-- Test_Weight_Pin — H19 Phase 4 (attestation-pinned model hash).
--
-- Three groups:
--   (A) SHA-256 streaming Context vs one-shot Hash on the block boundary
--       inputs 0/1/63/64/65/127/128 bytes (single-chunk AND a 2-chunk split).
--       This is the cross-check that the SPARK_Mode=>Off Context is
--       bit-identical to the proved Hash.
--   (B) Hash_Source (streaming a whole Byte_Source) vs Hash of the same bytes
--       read into memory, on svgdata/student.gguf — the H19 parity invariant
--       for the pin hash itself.
--   (C) Pin parse + Verify: correct pin passes; wrong digest -> Pin_Error;
--       model-id mismatch -> Pin_Error; bare hash passes; no pin (Empty_Pin)
--       is a no-op; malformed specs -> Pin_Error.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;               use Ada.Strings;
with Ada.Strings.Fixed;         use Ada.Strings.Fixed;
with Ada.Strings.Unbounded;     use Ada.Strings.Unbounded;
with Interfaces;                use Interfaces;
with Crypto;                    use Crypto;
with Crypto.SHA256;
with LLM_Byte_Source;
with LLM_Weight_Pin;
use type LLM_Byte_Source.Byte_Source_Access;  --  '=' for the source access

procedure Test_Weight_Pin is
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

   function Eq (A, B : Crypto.SHA256.Digest) return Boolean is
     (A = B);

   function Hex_Char (V : Natural) return Character is
     (if V < 10 then Character'Val (Character'Pos ('0') + V)
      else Character'Val (Character'Pos ('a') + V - 10));

   function Hex_Image (D : Crypto.SHA256.Digest) return String is
      R : String (1 .. 64);
   begin
      for I in 0 .. 31 loop
         R (2 * I + 1) := Hex_Char (Natural (D (I)) / 16);
         R (2 * I + 2) := Hex_Char (Natural (D (I)) mod 16);
      end loop;
      return R;
   end Hex_Image;

   --  Deterministic non-zero buffer of length N.
   function Make_Buf (N : Natural) return Byte_Array is
      B : Byte_Array (0 .. N - 1);
   begin
      for I in 0 .. N - 1 loop
         B (I) := U8 ((I * 31 + 7) mod 256);
      end loop;
      return B;
   end Make_Buf;

   Model_Path : constant String := "svgdata/student.gguf";

   type Byte_Array_Access is access all Byte_Array;

begin
   Put_Line ("=== H19 Weight-Pin Test Suite ===");
   New_Line;

   ------------------------------------------------------------------
   -- (A) Context vs Hash on boundary inputs
   ------------------------------------------------------------------
   declare
      Sizes : constant array (1 .. 7) of Natural :=
        [0, 1, 63, 64, 65, 127, 128];
   begin
      for K in Sizes'Range loop
         declare
            N  : constant Natural     := Sizes (K);
            M  : constant Byte_Array   := Make_Buf (N);
            D1 : Crypto.SHA256.Digest;
            D2 : Crypto.SHA256.Digest;
            D3 : Crypto.SHA256.Digest;
            C  : Crypto.SHA256.Context;
         begin
            D1 := Crypto.SHA256.Hash (M);

            Crypto.SHA256.Init (C);
            Crypto.SHA256.Update (C, M);
            Crypto.SHA256.Final (C, D2);

            Crypto.SHA256.Init (C);
            Crypto.SHA256.Update (C, M (0 .. N / 2 - 1));
            Crypto.SHA256.Update (C, M (N / 2 .. N - 1));
            Crypto.SHA256.Final (C, D3);

            Assert ("boundary N=" & Trim (N'Img, Both)
                    & " single-chunk = Hash", Eq (D1, D2));
            Assert ("boundary N=" & Trim (N'Img, Both)
                    & " 2-chunk split = Hash", Eq (D1, D3));
         end;
      end loop;
   end;

   ------------------------------------------------------------------
   -- (B) Hash_Source vs in-memory Hash on the real model fixture
   --     + (C) pin parse / verify
   ------------------------------------------------------------------
   declare
      Ref : LLM_Byte_Source.Byte_Source_Access :=
              LLM_Byte_Source.Open_Source (Model_Path);
   begin
      if Ref = null then
         Put_Line ("  SKIP: model fixture " & Model_Path & " not found");
         Put_Line ("       (run from the repo root so the relative path resolves)");
         Ada.Command_Line.Set_Exit_Status (0);
         Put_Line ("Results: " & Trim (Passed'Img, Both) & " passed, "
                   & Trim (Failed'Img, Both) & " failed.");
         return;
      end if;

      declare
         Len   : constant Unsigned_64 := Ref.Byte_Length;
         Bytes : constant Byte_Array_Access :=
                   new Byte_Array (0 .. Natural (Len) - 1);
         D_Mem, D_Src : Crypto.SHA256.Digest;
      begin
         Ref.Read_Seq (Bytes.all'Address, Natural (Len));
         D_Mem := Crypto.SHA256.Hash (Bytes.all);
         D_Src := LLM_Weight_Pin.Hash_Source (Ref.all);
         Assert ("Hash_Source(student.gguf) = Hash of bytes", Eq (D_Mem, D_Src));

         --  (C) Pin parse + Verify against the same source.
         declare
            Hex : constant String := Hex_Image (D_Mem);

            --  Wrong-but-valid pin: last hex digit changed to a different one.
            Wrong_Last : constant Character :=
              (if Hex (Hex'Last) = '0' then '1' else '0');
            Wrong_Hex  : constant String :=
              Hex (Hex'First .. Hex'Last - 1) & Wrong_Last;

            P_Correct : constant LLM_Weight_Pin.Pin :=
              LLM_Weight_Pin.Parse ("student@" & Hex);
            P_Wrong   : constant LLM_Weight_Pin.Pin :=
              LLM_Weight_Pin.Parse ("student@" & Wrong_Hex);
            P_Other   : constant LLM_Weight_Pin.Pin :=
              LLM_Weight_Pin.Parse ("other@" & Hex);
            P_Bare    : constant LLM_Weight_Pin.Pin :=
              LLM_Weight_Pin.Parse (Hex);

            --  Result flags, declared here so the Asserts (outside the inner
            --  handled blocks) can see them. Each inner block runs one Verify
            --  / Parse and sets its flag; the Assert reads it.
            OK_Correct, OK_Bare, OK_Empty : Boolean := True;
            Raised_Wrong, Raised_Other    : Boolean := False;
            All_Bad                       : Boolean := True;
         begin
            --  Correct pin: Verify passes.
            begin
               LLM_Weight_Pin.Verify (Ref.all, "student", P_Correct);
            exception
               when others => OK_Correct := False;
            end;
            Assert ("correct pin verifies", OK_Correct);

            --  Wrong digest: Verify raises Pin_Error.
            begin
               LLM_Weight_Pin.Verify (Ref.all, "student", P_Wrong);
            exception
               when LLM_Weight_Pin.Pin_Error => Raised_Wrong := True;
               when others                   => Raised_Wrong := False;
            end;
            Assert ("wrong digest -> Pin_Error", Raised_Wrong);

            --  Model-id mismatch: Verify raises Pin_Error (no hash).
            begin
               LLM_Weight_Pin.Verify (Ref.all, "student", P_Other);
            exception
               when LLM_Weight_Pin.Pin_Error => Raised_Other := True;
               when others                   => Raised_Other := False;
            end;
            Assert ("model-id mismatch -> Pin_Error", Raised_Other);

            --  Bare hash: Verify passes regardless of Model_ID.
            begin
               LLM_Weight_Pin.Verify (Ref.all, "anything", P_Bare);
            exception
               when others => OK_Bare := False;
            end;
            Assert ("bare-hash pin verifies", OK_Bare);

            --  No pin: Empty_Pin is a no-op.
            begin
               LLM_Weight_Pin.Verify (Ref.all, "student",
                                      LLM_Weight_Pin.Empty_Pin);
            exception
               when others => OK_Empty := False;
            end;
            Assert ("empty pin is a no-op", OK_Empty);

            --  Malformed specs -> Pin_Error at Parse. Parse is called in the
            --  statement part (not an initializer) so a raise is caught by the
            --  block's handler (declarative-part exceptions are NOT covered).
            declare
               Bad : constant array (1 .. 4) of Unbounded_String :=
                 [To_Unbounded_String ("xyz"),
                  To_Unbounded_String ("id@zz"),
                  To_Unbounded_String ("@" & Hex),
                  To_Unbounded_String ("id@")];
            begin
               for I in Bad'Range loop
                  declare
                     Junk : LLM_Weight_Pin.Pin;
                  begin
                     --  Parse is in the statement part so its raise is caught
                     --  by this block's handler (declarative-part raises are
                     --  NOT covered). Junk is read via Is_Empty so it is not
                     --  an unused assignment on the no-raise path.
                     Junk := LLM_Weight_Pin.Parse (To_String (Bad (I)));
                     if not LLM_Weight_Pin.Is_Empty (Junk) then
                        All_Bad := False;   --  parsed but should have raised
                     end if;
                  exception
                     when LLM_Weight_Pin.Pin_Error => null;  --  expected
                     when others                   => All_Bad := False;
                  end;
               end loop;
            end;
            Assert ("malformed specs -> Pin_Error", All_Bad);
         end;
      end;

      LLM_Byte_Source.Free_Source (Ref);
   end;

   New_Line;
   Put_Line ("Results: " & Trim (Passed'Img, Both) & " passed, "
             & Trim (Failed'Img, Both) & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Weight_Pin;