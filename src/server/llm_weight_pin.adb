---------------------------------------------------------------------
-- LLM_Weight_Pin body — see spec.
---------------------------------------------------------------------

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Interfaces;            use Interfaces;
with Crypto;                use Crypto;
--  Crypto.SHA256 and LLM_Byte_Source are withed by the spec, so they are
--  visible in the body without a redundant with-clause here.

package body LLM_Weight_Pin is

   Hex_Len : constant := 64;   --  SHA-256 digest = 32 bytes = 64 hex chars

   --  Hex digit -> 0..15. Returns 16 for a non-hex char (the caller rejects
   --  via Pin_Error before using a 16, so the bad value never indexes).
   function Nyb (C : Character) return Natural is
   begin
      case C is
         when '0' .. '9' => return Character'Pos (C) - Character'Pos ('0');
         when 'a' .. 'f' => return Character'Pos (C) - Character'Pos ('a') + 10;
         when 'A' .. 'F' => return Character'Pos (C) - Character'Pos ('A') + 10;
         when others     => return 16;
      end case;
   end Nyb;

   --  Parse 64 hex chars (any origin) into a Digest. Raises Pin_Error on a
   --  wrong length or a non-hex char.
   function Parse_Hex (Hex : String) return Crypto.SHA256.Digest is
      D : Crypto.SHA256.Digest := [others => 0];
   begin
      if Hex'Length /= Hex_Len then
         raise Pin_Error with "pin digest must be 64 hex chars";
      end if;
      for I in 0 .. 31 loop
         declare
            Hi : constant Natural := Nyb (Hex (Hex'First + 2 * I));
            Lo : constant Natural := Nyb (Hex (Hex'First + 2 * I + 1));
         begin
            if Hi > 15 or else Lo > 15 then
               raise Pin_Error with "pin digest has a non-hex character";
            end if;
            D (I) := U8 (Hi * 16 + Lo);
         end;
      end loop;
      return D;
   end Parse_Hex;

   function Parse (Spec : String) return Pin is
      At_Pos : Integer := -1;
   begin
      if Spec'Length = 0 then
         return Empty_Pin;
      end if;

      --  Find the first '@' separating model id from digest.
      for I in Spec'Range loop
         if Spec (I) = '@' then
            At_Pos := I;
            exit;
         end if;
      end loop;

      if At_Pos = -1 then
         --  Bare digest, no model id.
         return (Has_ID => False,
                 ID     => Null_Unbounded_String,
                 Dig    => Parse_Hex (Spec));
      else
         --  <model_id>@<hex>. Reject an empty id or an empty digest.
         if At_Pos = Spec'First then
            raise Pin_Error with "pin has empty model id before '@'";
         end if;
         if At_Pos = Spec'Last then
            raise Pin_Error with "pin has empty digest after '@'";
         end if;
         return (Has_ID => True,
                 ID     => To_Unbounded_String
                             (Spec (Spec'First .. At_Pos - 1)),
                 Dig    => Parse_Hex (Spec (At_Pos + 1 .. Spec'Last)));
      end if;
   end Parse;

   function Is_Empty (P : Pin) return Boolean is
     (not P.Has_ID and then Length (P.ID) = 0
      and then (for all I in P.Dig'Range => P.Dig (I) = 0));

   function Pin_Model_ID (P : Pin) return String is
     (To_String (P.ID));

   function Pin_Digest (P : Pin) return Crypto.SHA256.Digest is
     (P.Dig);

   ------------------------------------------------------------------
   -- Hashing + verification
   ------------------------------------------------------------------

   --  Stream buffer: 4 KiB keeps Hash_Source off the secondary stack while
   --  still amortizing the per-Update call. Any size is bit-identical (the
   --  Context is chunk-agnostic).
   Buf_Len : constant := 4096;

   function Hash_Source
     (S : in out LLM_Byte_Source.Byte_Source'Class)
      return Crypto.SHA256.Digest
   is
      Total : constant Unsigned_64 := LLM_Byte_Source.Byte_Length (S);
      Ctx   : Crypto.SHA256.Context;
      Buf   : aliased Byte_Array (0 .. Buf_Len - 1);
      Pos   : Unsigned_64 := 0;
   begin
      Crypto.SHA256.Init (Ctx);
      LLM_Byte_Source.Seek (S, 0);

      while Pos < Total loop
         declare
            Remaining : constant Unsigned_64 := Total - Pos;
            N         : constant Natural :=
              (if Remaining >= Unsigned_64 (Buf_Len)
               then Buf_Len
               else Natural (Remaining));
         begin
            LLM_Byte_Source.Read_Seq (S, Buf'Address, N);
            Crypto.SHA256.Update (Ctx, Buf (0 .. N - 1));
            Pos := Pos + Unsigned_64 (N);
         end;
      end loop;

      declare
         D : Crypto.SHA256.Digest;
      begin
         Crypto.SHA256.Final (Ctx, D);
         return D;
      end;
   end Hash_Source;

   procedure Verify
     (S        : in out LLM_Byte_Source.Byte_Source'Class;
      Model_ID : String;
      P        : Pin)
   is
   begin
      if Is_Empty (P) then
         return;  --  no pin: integrity is the AEAD channel alone (Phase 1-3)
      end if;

      if P.Has_ID and then Pin_Model_ID (P) /= Model_ID then
         raise Pin_Error with "pin model-id mismatch";
      end if;

      declare
         D : constant Crypto.SHA256.Digest := Hash_Source (S);
      begin
         if D /= P.Dig then
            raise Pin_Error with "weight hash mismatch";
         end if;
      end;
   end Verify;

end LLM_Weight_Pin;