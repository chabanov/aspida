--  Phase-0 FFI proof (Ada side): call the CUDA vector-add kernel through the
--  C ABI and verify the GPU result. This proves the Ada -> C -> CUDA path that
--  Stream B's GPU backend will use for real kernels.
with Ada.Text_IO;   use Ada.Text_IO;
with Interfaces.C;  use Interfaces.C;
with System;

procedure Vadd_Main is
   N : constant := 1024;

   --  Contiguous (C-convention) arrays of C float, passed by address.
   type Real_Array is array (0 .. N - 1) of aliased C_float;
   pragma Convention (C, Real_Array);

   A, B, C : Real_Array;

   procedure Vadd
     (Pa, Pb, Pc : System.Address; Len : int)
     with Import => True, Convention => C, External_Name => "vadd";

   Ok : Boolean := True;
begin
   for I in A'Range loop
      A (I) := C_float (I);
      B (I) := C_float (2 * I);
   end loop;

   Vadd (A'Address, B'Address, C'Address, int (N));

   for I in C'Range loop
      if C (I) /= C_float (3 * I) then
         Ok := False;
      end if;
   end loop;

   Put_Line ("c[0]    =" & C_float'Image (C (0)));
   Put_Line ("c[1023] =" & C_float'Image (C (N - 1)) & "  (expected 3069.0)");
   Put_Line (if Ok then "FFI OK: Ada -> C -> CUDA vector add correct on GPU"
             else "FFI FAIL: mismatch");
end Vadd_Main;
