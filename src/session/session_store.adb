---------------------------------------------------------------------
-- Session_Store body
--
-- Turns are kept as (user, assistant) pairs and serialized with a simple
-- length-prefixed format ([u_len BE32][u][a_len BE32][a] per turn) before
-- being encrypted at rest by At_Rest.
---------------------------------------------------------------------

with Interfaces;             use Interfaces;
with Interfaces.C;           use type Interfaces.C.int;
with Ada.Strings.Unbounded;  use Ada.Strings.Unbounded;
with Ada.Containers.Vectors;
with Ada.Environment_Variables;
with Ada.Directories;
with Ada.Unchecked_Deallocation;
with Crypto;                 use Crypto;
with At_Rest;

package body Session_Store is

   Var      : constant String := "ASPIDA_STORE_PASSWORD";
   Dir      : constant String := "sessions";
   PBKDF_It : constant := 600_000;
   LF       : constant Character := Character'Val (10);

   --  int unsetenv(const char *name);  POSIX unsetenv(2). Removes the variable
   --  from the process environment so it no longer appears in /proc/<pid>/environ
   --  and is not inherited by child processes.
   function C_Unsetenv
     (Name : Interfaces.C.char_array) return Interfaces.C.int
     with Import, Convention => C, External_Name => "unsetenv";

   --  int chmod(const char *path, mode_t mode);  POSIX chmod(2). Used to lock
   --  the sessions directory to owner-only so other local users cannot read the
   --  (encrypted-at-rest) session files.
   function C_Chmod
     (Pathp : Interfaces.C.char_array; Mode : Interfaces.C.int)
      return Interfaces.C.int
     with Import, Convention => C, External_Name => "chmod";

   --  The master password is captured from ASPIDA_STORE_PASSWORD into this
   --  cached buffer and the variable is then scrubbed from the environment so a
   --  child process can never observe it. Whenever the variable is found set
   --  again (e.g. an operator rotates the password), the cache is refreshed
   --  from the new value and the variable scrubbed again — so the live value
   --  always wins, but never lingers in the environment. Pw_Loaded records that
   --  at least one capture attempt has happened; Pw_Set records whether a
   --  non-empty password is currently held (an unset/empty password disables
   --  persistence — see Enabled).
   type Byte_Array_Access is access Byte_Array;
   procedure Free_Pw is
     new Ada.Unchecked_Deallocation (Byte_Array, Byte_Array_Access);
   Pw_Loaded : Boolean := False;
   Pw_Set    : Boolean := False;
   Pw_Cache  : Byte_Array_Access := null;

   procedure Load_Password is
      V : constant String := Ada.Environment_Variables.Value (Var, "");
   begin
      if V'Length > 0 then
         --  Variable is currently set: (re)load the cache from it — this also
         --  honors a password rotation between the cached value and now.
         if Pw_Cache /= null then
            Wipe (Pw_Cache.all);
            Free_Pw (Pw_Cache);
         end if;
         Pw_Cache := new Byte_Array (0 .. V'Length - 1);
         for I in V'Range loop
            Pw_Cache (I - V'First) := U8 (Character'Pos (V (I)));
         end loop;
         Pw_Set := True;
         --  Scrub the variable so a child process can never observe it. The
         --  return value is intentionally ignored.
         declare
            Dummy : Interfaces.C.int;
            pragma Unreferenced (Dummy);
         begin
            Dummy := C_Unsetenv (Interfaces.C.To_C (Var));
         end;
      elsif not Pw_Loaded then
         --  First call and the variable was never set: persistence is disabled.
         Pw_Set := False;
      end if;
      --  Otherwise the variable is empty but a password was captured earlier;
      --  keep the cache (the variable was already scrubbed on that call).
      Pw_Loaded := True;
   end Load_Password;

   type Turn_Rec is record
      U : Unbounded_String;
      A : Unbounded_String;
   end record;

   package Turn_Vectors is new Ada.Containers.Vectors (Positive, Turn_Rec);

   type Store_Rec is record
      Id    : Unbounded_String;
      Turns : Turn_Vectors.Vector;
   end record;

   procedure Free is new Ada.Unchecked_Deallocation (Store_Rec, Store);

   function Enabled return Boolean is
   begin
      Load_Password;
      return Pw_Set;
   end Enabled;

   function Valid_Id (Id : String) return Boolean is
   begin
      if Id'Length = 0 or else Id'Length > 64 then
         return False;
      end if;
      for C of Id loop
         case C is
            when '0' .. '9' | 'a' .. 'z' | 'A' .. 'Z' | '_' | '-' => null;
            when others => return False;
         end case;
      end loop;
      return True;
   end Valid_Id;

   --  Return a copy of the cached master password (empty if unset). The caller
   --  owns the copy and must Wipe it; the cache itself stays resident for the
   --  process lifetime. Reads from the one-time cache, never the environment.
   function Password_Bytes return Byte_Array is
   begin
      Load_Password;
      if Pw_Set and then Pw_Cache /= null then
         return R : Byte_Array (Pw_Cache'Range) do
            R := Pw_Cache.all;
         end return;
      else
         return R : Byte_Array (1 .. 0) do
            null;
         end return;
      end if;
   end Password_Bytes;

   function Path (S : Store) return String is
     (Dir & "/" & To_String (S.Id) & ".session");

   ------------------------------------------------------------------
   -- (De)serialisation of the turn list.
   ------------------------------------------------------------------

   function Serialize (V : Turn_Vectors.Vector) return Byte_Array is
      Total : Natural := 0;
   begin
      for T of V loop
         Total := Total + 8 + Length (T.U) + Length (T.A);
      end loop;
      return R : Byte_Array (0 .. Integer'Max (0, Total) - 1) do
         declare
            P : Natural := 0;
            procedure Put_Str (Str : String) is
            begin
               R (P)     := U8 (Shift_Right (U32 (Str'Length), 24) and 16#FF#);
               R (P + 1) := U8 (Shift_Right (U32 (Str'Length), 16) and 16#FF#);
               R (P + 2) := U8 (Shift_Right (U32 (Str'Length), 8)  and 16#FF#);
               R (P + 3) := U8 (U32 (Str'Length) and 16#FF#);
               P := P + 4;
               for I in Str'Range loop
                  R (P) := U8 (Character'Pos (Str (I))); P := P + 1;
               end loop;
            end Put_Str;
         begin
            for T of V loop
               Put_Str (To_String (T.U));
               Put_Str (To_String (T.A));
            end loop;
         end;
      end return;
   end Serialize;

   Corrupt_Store : exception;   -- length prefixes inconsistent with the blob

   procedure Deserialize (Data : Byte_Array; V : in out Turn_Vectors.Vector) is
      P : Natural := 0;   -- offset from Data'First
      function Rd_Len return Natural is
      begin
         if P + 4 > Data'Length then
            raise Corrupt_Store with "truncated length prefix";
         end if;
         declare
            L : constant U32 :=
              Shift_Left (U32 (Data (Data'First + P)), 24)
              or Shift_Left (U32 (Data (Data'First + P + 1)), 16)
              or Shift_Left (U32 (Data (Data'First + P + 2)), 8)
              or U32 (Data (Data'First + P + 3));
         begin
            P := P + 4;
            return Natural (L);
         end;
      end Rd_Len;
      function Rd_Str (N : Natural) return String is
      begin
         if P + N > Data'Length then
            raise Corrupt_Store with "length prefix exceeds remaining data";
         end if;
         return R : String (1 .. N) do
            for I in 1 .. N loop
               R (I) := Character'Val (Integer (Data (Data'First + P + I - 1)));
            end loop;
            P := P + N;
         end return;
      end Rd_Str;
   begin
      V.Clear;
      --  Every read is bounds-checked against Data'Length, so a corrupt (but
      --  authenticated) blob raises Corrupt_Store rather than reading past the
      --  buffer or allocating a multi-gigabyte String from a bogus prefix.
      while P < Data'Length loop
         declare
            U_Len : constant Natural := Rd_Len;
            U     : constant String  := Rd_Str (U_Len);
            A_Len : constant Natural := Rd_Len;
            A     : constant String  := Rd_Str (A_Len);
         begin
            V.Append (Turn_Rec'(To_Unbounded_String (U), To_Unbounded_String (A)));
         end;
      end loop;
   end Deserialize;

   procedure Save (S : Store) is
      Pw    : Byte_Array := Password_Bytes;
      Bytes : Byte_Array := Serialize (S.Turns);
   begin
      if not Ada.Directories.Exists (Dir) then
         Ada.Directories.Create_Path (Dir);
         --  Lock the sessions directory to owner-only (rwx) right after
         --  creation so other local users cannot enumerate or read the
         --  encrypted-at-rest session files.
         if C_Chmod (Interfaces.C.To_C (Dir), 8#700#) /= 0 then
            raise Ada.Directories.Use_Error with "chmod of sessions dir failed";
         end if;
      end if;
      At_Rest.Save (Path (S), Pw, Bytes, Iterations => PBKDF_It);
      Wipe (Bytes);
      Wipe (Pw);
   end Save;

   ------------------------------------------------------------------
   -- Public API.
   ------------------------------------------------------------------

   procedure Open (S : out Store; Id : String) is
   begin
      --  Defence in depth: never let an unvalidated id reach the filesystem
      --  path (callers should pre-validate, but this is the last line).
      if not Valid_Id (Id) then
         raise Program_Error with "invalid session id";
      end if;
      S := new Store_Rec;
      S.Id := To_Unbounded_String (Id);
      S.Turns.Clear;

      if Enabled and then Ada.Directories.Exists (Path (S)) then
         declare
            Pw   : Byte_Array := Password_Bytes;
            Data : Byte_Array := At_Rest.Load (Path (S), Pw);
         begin
            Deserialize (Data, S.Turns);
            Wipe (Data);
            Wipe (Pw);
         end;
      end if;
   end Open;

   procedure Append_Turn (S : in out Store; User, Assistant : String) is
   begin
      S.Turns.Append
        (Turn_Rec'(To_Unbounded_String (User), To_Unbounded_String (Assistant)));
      if Enabled then
         Save (S);
      end if;
   end Append_Turn;

   function Turn_Count (S : Store) return Natural is
     (if S = null then 0 else Natural (S.Turns.Length));

   function User_Of (S : Store; I : Positive) return String is
     (To_String (S.Turns (I).U));

   function Assistant_Of (S : Store; I : Positive) return String is
     (To_String (S.Turns (I).A));

   function Transcript (S : Store) return String is
      R : Unbounded_String;
   begin
      if S = null then
         return "";
      end if;
      for T of S.Turns loop
         Append (R, "### User" & LF & To_String (T.U) & LF & LF
                    & "### Assistant" & LF & To_String (T.A) & LF & LF);
      end loop;
      return To_String (R);
   end Transcript;

   procedure Close (S : in out Store) is
   begin
      if S /= null then
         S.Turns.Clear;           -- drop plaintext history from RAM
         Free (S);
      end if;
   end Close;

end Session_Store;
