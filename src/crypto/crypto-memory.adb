---------------------------------------------------------------------
-- Crypto.Memory body
---------------------------------------------------------------------

with Interfaces.C;

package body Crypto.Memory is

   --  int mlock(const void *addr, size_t len);
   function C_Mlock
     (Addr : System.Address; Len : Interfaces.C.size_t) return Interfaces.C.int
     with Import, Convention => C, External_Name => "mlock";

   function Lock (First : System.Address; Length : Natural) return Boolean is
      use type Interfaces.C.int;
   begin
      if Length = 0 then
         return True;
      end if;
      return C_Mlock (First, Interfaces.C.size_t (Length)) = 0;
   end Lock;

end Crypto.Memory;
