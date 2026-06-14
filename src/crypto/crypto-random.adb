---------------------------------------------------------------------
-- Crypto.Random body — OS CSPRNG via getentropy(2)
---------------------------------------------------------------------

with Interfaces.C;
with System;

package body Crypto.Random is

   --  int getentropy(void *buf, size_t buflen);  -- max 256 bytes per call.
   function Getentropy
     (Buf : System.Address; Len : Interfaces.C.size_t) return Interfaces.C.int
     with Import, Convention => C, External_Name => "getentropy";

   procedure Fill (Buf : out Byte_Array) is
      use type Interfaces.C.int;
      N   : constant Natural := Buf'Length;
      Pos : Natural := 0;
   begin
      while Pos < N loop
         declare
            Chunk : constant Natural := Natural'Min (256, N - Pos);
            RC    : Interfaces.C.int;
         begin
            RC := Getentropy (Buf (Buf'First + Pos)'Address,
                              Interfaces.C.size_t (Chunk));
            if RC /= 0 then
               raise Program_Error with "Crypto.Random: getentropy failed";
            end if;
            Pos := Pos + Chunk;
         end;
      end loop;
   end Fill;

end Crypto.Random;
