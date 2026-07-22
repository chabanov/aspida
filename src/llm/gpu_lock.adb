with Interfaces.C; use Interfaces.C;

package body GPU_Lock is

   function C_Open (Path : char_array; Flags, Mode : int) return int
     with Import, Convention => C, External_Name => "open";
   function C_Flock (Fd, Op : int) return int
     with Import, Convention => C, External_Name => "flock";
   function C_Close (Fd : int) return int
     with Import, Convention => C, External_Name => "close";

   O_RDWR  : constant int := 2;
   O_CREAT : constant int := 8#100#;   -- 0x40 on Linux
   LOCK_SH : constant int := 1;
   LOCK_UN : constant int := 8;
   Mode    : constant int := 8#644#;

   Lock_Path : constant char_array := To_C ("/tmp/aspida_gpu.lock");

   procedure Acquire_Shared (H : out Handle) is
      Fd : constant int := C_Open (Lock_Path, O_CREAT + O_RDWR, Mode);
   begin
      H.Fd := Integer (Fd);
      if Fd >= 0 then
         --  Result checked in a condition so -gnatwe doesn't flag a discard;
         --  a flock failure just means we proceed without serialisation.
         if C_Flock (Fd, LOCK_SH) = 0 then null; end if;
      end if;
   exception
      when others => H.Fd := -1;
   end Acquire_Shared;

   procedure Release (H : in out Handle) is
   begin
      if H.Fd >= 0 then
         if C_Flock (int (H.Fd), LOCK_UN) = 0 then null; end if;
         if C_Close (int (H.Fd)) = 0 then null; end if;
         H.Fd := -1;
      end if;
   exception
      when others => H.Fd := -1;
   end Release;

end GPU_Lock;
