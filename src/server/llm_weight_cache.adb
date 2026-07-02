---------------------------------------------------------------------
-- LLM_Weight_Cache body — on-disk AEAD-sealed chunk storage via At_Rest.
--
-- Each chunk is sealed with the per-cache password and the bound Cache_Key
-- is prepended to the plaintext (inside the AEAD), so a blob read under the
-- wrong key is detected and reported as a miss rather than served as the
-- wrong model's weights. At_Rest's atomic .tmp + fsync + rename makes a
-- Store crash-safe (no truncated/zero-length blob).
---------------------------------------------------------------------

with Ada.Directories;
with Ada.Strings;           use Ada.Strings;
with Ada.Strings.Fixed;     use Ada.Strings.Fixed;
with Ada.Unchecked_Deallocation;
with At_Rest;

package body LLM_Weight_Cache is

   use Ada.Strings.Unbounded;       --  spec withs it
   use Crypto;                       --  spec withs it
   use type Interfaces.Unsigned_8;   --  '=' on bytes
   use type Interfaces.Unsigned_32;  --  arithmetic on Key_Len

   procedure Free is new Ada.Unchecked_Deallocation (Crypto.Byte_Array, Byte_Array_Access);

   --  Map an arbitrary Cache_Key to a single path-safe directory segment:
   --  keep [A-Za-z0-9._-], replace everything else (slashes, spaces, colons,
   --  NUL, ...) with '_'. A non-empty key never yields an empty result, but
   --  guard anyway so the path is always well-formed.
   function Sanitize (S : String) return String is
      R : String (1 .. S'Length);
   begin
      for I in S'Range loop
         declare
            C : constant Character := S (I);
         begin
            if (C in 'A' .. 'Z' or else C in 'a' .. 'z'
                 or else C in '0' .. '9' or else C = '.' or else C = '_'
                 or else C = '-') then
               R (1 + (I - S'First)) := C;
            else
               R (1 + (I - S'First)) := '_';
            end if;
         end;
      end loop;
      if R'Length = 0 then
         return "_";
      end if;
      return R;
   end Sanitize;

   function Chunk_Path
     (C : Weight_Cache; Index : Interfaces.Unsigned_64) return String is
   begin
      return To_String (C.Dir) & "/" & Sanitize (To_String (C.Key))
        & "/chunk_" & Trim (Interfaces.Unsigned_64'Image (Index), Both) & ".enc";
   end Chunk_Path;

   procedure Put_BE32 (A : in out Crypto.Byte_Array; Off : Natural; V : Crypto.U32) is
   begin
      A (A'First + Off)     := U8 (Interfaces.Shift_Right (V, 24) and 16#FF#);
      A (A'First + Off + 1) := U8 (Interfaces.Shift_Right (V, 16) and 16#FF#);
      A (A'First + Off + 2) := U8 (Interfaces.Shift_Right (V, 8)  and 16#FF#);
      A (A'First + Off + 3) := U8 (V and 16#FF#);
   end Put_BE32;

   function Get_BE32 (A : Crypto.Byte_Array; Off : Natural) return Crypto.U32 is
     (Interfaces.Shift_Left (U32 (A (A'First + Off)), 24)
      or Interfaces.Shift_Left (U32 (A (A'First + Off + 1)), 16)
      or Interfaces.Shift_Left (U32 (A (A'First + Off + 2)), 8)
      or U32 (A (A'First + Off + 3)));

   function Open
     (Dir       : String;
      Pass      : Crypto.Byte_Array;
      Cache_Key : String) return Weight_Cache
   is
      --  Default-initialize to the disabled state; the enabled path overwrites
      --  the relevant fields. Weight_Cache is non-limited, so a default-
      --  initialized local can be mutated and then returned by value (the
      --  owned Pass access transfers to the copy; the local is not
      --  finalized away because Weight_Cache is not controlled).
      R : Weight_Cache;
   begin
      --  Persistence is opt-in: an empty dir or password leaves the cache
      --  disabled (in-memory only), so default behavior is unchanged.
      if Dir'Length = 0 or else Pass'Length = 0 then
         return R;
      end if;

      R.On  := True;
      R.Dir := To_Unbounded_String (Dir);
      R.Key := To_Unbounded_String (Cache_Key);
      R.Pass := new Crypto.Byte_Array'(Pass);   --  owned copy; wiped on Close

      --  Create the per-model subdir (and parents) once so later Stores can
      --  write straight into it. Create_Path is idempotent on an existing dir.
      Ada.Directories.Create_Path (Dir & "/" & Sanitize (Cache_Key));
      return R;
   end Open;

   function Enabled (C : Weight_Cache) return Boolean is
     (C.On);

   function Has
     (C     : Weight_Cache;
      Index : Interfaces.Unsigned_64) return Boolean is
   begin
      if not C.On then
         return False;
      end if;
      return Ada.Directories.Exists (Chunk_Path (C, Index));
   end Has;

   function Load
     (C     : Weight_Cache;
      Index : Interfaces.Unsigned_64) return Byte_Array_Access
   is
      Path : constant String := Chunk_Path (C, Index);

      --  Unseal + parse in a nested function. The AEAD open (At_Rest.Load) is
      --  evaluated in the nested function's declarative part; if it raises, the
      --  exception propagates out of the nested function and is then raised at
      --  the call site below -- i.e. inside THIS block's handled sequence of
      --  statements, where the handler can map it to Cache_Error. (A block's
      --  own exception handlers do not cover its declarative part, so the
      --  unseal cannot live directly in Load's declarative part.)
      function Unseal_And_Parse return Byte_Array_Access is
         PT : constant Crypto.Byte_Array := At_Rest.Load (Path, C.Pass.all);
         K  : constant String := To_String (C.Key);
      begin
         --  Parse [Key_Len BE32][Key bytes][Chunk bytes].
         if PT'Length < 4 then
            raise Cache_Error with "chunk blob too short";
         end if;

         declare
            K_Len : constant Crypto.U32 := Get_BE32 (PT, 0);
         begin
            if Natural (K_Len) > PT'Length - 4 then
               raise Cache_Error with "chunk blob key length out of range";
            end if;

            --  A key mismatch means this blob belongs to a different model
            --  (e.g. two models sanitized to the same subdir). Report a miss so
            --  the caller refetches and overwrites -- never serve the wrong
            --  bytes.
            if Natural (K_Len) /= K'Length then
               raise Cache_Miss with "chunk cached under a different key";
            end if;
            for I in 0 .. K'Length - 1 loop
               if PT (4 + I) /= Crypto.U8 (Character'Pos (K (K'First + I))) then
                  raise Cache_Miss with "chunk cached under a different key";
               end if;
            end loop;

            declare
               Chunk_Len : constant Natural := PT'Length - 4 - K'Length;
               D         : constant Byte_Array_Access :=
                 new Crypto.Byte_Array (0 .. Chunk_Len - 1);
            begin
               for I in 0 .. Chunk_Len - 1 loop
                  D (I) := PT (4 + K'Length + I);
               end loop;
               return D;
            end;
         end;
      end Unseal_And_Parse;
   begin
      if not C.On then
         raise Cache_Miss with "cache disabled";
      end if;

      if not Ada.Directories.Exists (Path) then
         raise Cache_Miss with "chunk not cached";
      end if;

      return Unseal_And_Parse;
   exception
      when At_Rest.Decrypt_Error | At_Rest.Format_Error =>
         raise Cache_Error with "sealed chunk corrupt or wrong password";
   end Load;

   procedure Store
     (C     : in out Weight_Cache;
      Index : Interfaces.Unsigned_64;
      Data  : Crypto.Byte_Array)
   is
      Path : constant String := Chunk_Path (C, Index);
      K    : constant String := To_String (C.Key);
      PT   : Crypto.Byte_Array (0 .. 3 + K'Length + Data'Length);
   begin
      if not C.On then
         return;   --  disabled: no-op (in-memory cache still holds the chunk)
      end if;

      Put_BE32 (PT, 0, Crypto.U32 (K'Length));
      for I in K'Range loop
         PT (4 + (I - K'First)) := Crypto.U8 (Character'Pos (K (I)));
      end loop;
      for I in Data'Range loop
         PT (4 + K'Length + (I - Data'First)) := Data (I);
      end loop;

      At_Rest.Save (Path, C.Pass.all, PT);
   end Store;

   procedure Close (C : in out Weight_Cache) is
      P : Byte_Array_Access := C.Pass;
   begin
      if C.On then
         if P /= null then
            Crypto.Wipe (P.all);   --  zero the password before release
            Free (P);              --  P becomes null
         end if;
         C.On := False;
         C.Pass := null;
      end if;
   end Close;

end LLM_Weight_Cache;