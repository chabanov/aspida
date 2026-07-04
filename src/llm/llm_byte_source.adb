---------------------------------------------------------------------
-- LLM_Byte_Source body — Local_File_Source via POSIX fd + lseek + read.
--
-- The POSIX thin bindings and the short-read loop lived in llm_gguf.adb
-- historically; they move here unchanged so the GGUF parser is source-agnostic
-- and a future Remote_AEAD_Source can swap in without touching the parser.
---------------------------------------------------------------------

with Ada.Unchecked_Deallocation;
with System.Storage_Elements;

package body LLM_Byte_Source is

   use Interfaces;  --  makes Unsigned_64 arithmetic/comparison operators visible

   --------------------------------------------------------------------
   -- POSIX I/O imports (thin bindings to libc / libSystem)
   --------------------------------------------------------------------
   type C_Int is new Integer;
   type C_Size is mod 2**64;
   type C_Off is new Long_Long_Integer;

   function C_Open  (Path : String; Flags : C_Int; Mode : C_Int) return C_Int
     with Import, Convention => C, External_Name => "open";
   function C_Read  (FD : C_Int; Buf : System.Address; Count : C_Size) return C_Int
     with Import, Convention => C, External_Name => "read";
   --  pread: positional read at an absolute offset that does NOT use or move
   --  the fd's file-offset. POSIX guarantees it is atomic w.r.t. the offset,
   --  so concurrent tasks sharing one fd never race on the cursor.
   function C_PRead (FD : C_Int; Buf : System.Address; Count : C_Size;
                     Offset : C_Off) return C_Int
     with Import, Convention => C, External_Name => "pread";
   function C_LSeek (FD : C_Int; Offset : C_Off; Whence : C_Int) return C_Off
     with Import, Convention => C, External_Name => "lseek";
   function C_Close (FD : C_Int) return C_Int
     with Import, Convention => C, External_Name => "close";

   O_RDONLY : constant C_Int := 0;
   SEEK_SET : constant C_Int := 0;
   SEEK_END : constant C_Int := 2;

   --  Explicitly discard the return code of a side-effecting POSIX call
   --  (close / lseek) whose result we intentionally ignore.
   procedure Ignore (Unused : C_Int) is null;
   procedure Ignore (Unused : C_Off) is null;

   --------------------------------------------------------------------
   -- Concrete class-wide helpers
   --------------------------------------------------------------------

   procedure Read_At
     (S     : in out Byte_Source'Class;
      Off   : Interfaces.Unsigned_64;
      Addr  : System.Address;
      Count : Natural) is
   begin
      S.Seek (Off);
      S.Read_Seq (Addr, Count);
   end Read_At;

   procedure Free_Source (S : in out Byte_Source_Access) is
      procedure Dealloc is new Ada.Unchecked_Deallocation
        (Byte_Source'Class, Byte_Source_Access);
   begin
      if S /= null then
         S.Close;        --  idempotent dispatch: release the fd / connection
         Dealloc (S);    --  S becomes null
      end if;
   end Free_Source;

   --------------------------------------------------------------------
   -- Local_File_Source
   --------------------------------------------------------------------

   function Open_Source (Path : String) return Byte_Source_Access is
      FD      : C_Int;
      End_Pos : C_Off;
   begin
      --  C's open() needs a NUL-terminated string; an Ada String is not, so
      --  append the terminator explicitly. Passing a bare Ada String read past
      --  the boundary until a random 0 — it happened to work for a string
      --  literal but not for an env-var value (same fix as the old LLM_GGUF).
      FD := C_Open (Path & Character'Val (0), O_RDONLY, 0);
      if FD < 0 then
         return null;
      end if;

      --  Probe the file length once (cached in Len; this is the bound the GGUF
      --  parser validates every tensor offset against), then rewind to the
      --  start so the header parse reads from byte 0.
      End_Pos := C_LSeek (FD, 0, SEEK_END);
      if End_Pos < 0 then
         Ignore (C_Close (FD));
         return null;
      end if;
      Ignore (C_LSeek (FD, 0, SEEK_SET));

      --  Allocate on the heap and widen to the class-wide access the engine
      --  holds. Returning the allocator directly (rather than via an
      --  intermediate anonymous access variable) keeps the accessibility level
      --  at the library level, so Free_Source can later deallocate it via the
      --  Byte_Source_Access pool.
      return new Local_File_Source'
        (FD  => Integer (FD),
         Len => Interfaces.Unsigned_64 (End_Pos),
         Pos => 0);
   end Open_Source;

   --  Read exactly Count bytes, looping over short reads (read() may legally
   --  return fewer than requested — at EOF, on a signal, or on a pipe). The
   --  cursor advances by exactly Count on success.
   overriding procedure Read_Seq
     (S     : in out Local_File_Source;
      Addr  : System.Address;
      Count : Natural)
   is
      use System.Storage_Elements;
      Remaining : Natural          := Count;
      Cur       : System.Address   := Addr;
      N         : C_Int;
   begin
      while Remaining > 0 loop
         N := C_Read (C_Int (S.FD), Cur, C_Size (Remaining));
         if N <= 0 then
            raise Malformed_Source with "short read from byte source";
         end if;
         Remaining := Remaining - Natural (N);
         Cur := Cur + Storage_Offset (N);
      end loop;
      S.Pos := S.Pos + Interfaces.Unsigned_64 (Count);
   end Read_Seq;

   overriding function Byte_Length
     (S : Local_File_Source) return Interfaces.Unsigned_64 is
   begin
      return S.Len;
   end Byte_Length;

   overriding function Cursor
     (S : Local_File_Source) return Interfaces.Unsigned_64 is
   begin
      return S.Pos;
   end Cursor;

   overriding procedure Seek
     (S   : in out Local_File_Source;
      Off : Interfaces.Unsigned_64) is
   begin
      --  A seek past the end is rejected here (rather than letting the next
      --  read short-read) so a malformed tensor offset fails loud at the seek,
      --  matching the GGUF parser's fail-loud-on-OOB discipline.
      if Off > S.Len then
         raise Malformed_Source with "seek past end of byte source";
      end if;
      Ignore (C_LSeek (C_Int (S.FD), C_Off (Off), SEEK_SET));
      S.Pos := Off;
   end Seek;

   --  Positional read via pread: reads Count bytes at absolute Off without
   --  using or moving the fd cursor, so several tasks can read one shared
   --  read-only fd (e.g. the weight server serving concurrent clients from one
   --  GGUF) with no cursor race. Loops over short reads exactly like Read_Seq.
   overriding procedure Read_At_Pos
     (S     : in out Local_File_Source;
      Off   : Interfaces.Unsigned_64;
      Addr  : System.Address;
      Count : Natural)
   is
      use System.Storage_Elements;
      Remaining : Natural        := Count;
      Cur       : System.Address := Addr;
      Cur_Off   : Interfaces.Unsigned_64 := Off;
      N         : C_Int;
   begin
      --  Fail loud on an out-of-range request (matching Seek's discipline) so
      --  a hostile offset/count never reads past the file into adjacent bytes.
      if Off > S.Len
        or else Interfaces.Unsigned_64 (Count) > S.Len - Off
      then
         raise Malformed_Source with "positional read past end of byte source";
      end if;
      while Remaining > 0 loop
         N := C_PRead (C_Int (S.FD), Cur, C_Size (Remaining), C_Off (Cur_Off));
         if N <= 0 then
            raise Malformed_Source with "short pread from byte source";
         end if;
         Remaining := Remaining - Natural (N);
         Cur     := Cur + Storage_Offset (N);
         Cur_Off := Cur_Off + Interfaces.Unsigned_64 (N);
      end loop;
   end Read_At_Pos;

   overriding procedure Close (S : in out Local_File_Source) is
   begin
      if S.FD >= 0 then
         Ignore (C_Close (C_Int (S.FD)));
         S.FD := -1;
      end if;
   end Close;

end LLM_Byte_Source;