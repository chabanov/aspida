---------------------------------------------------------------------
-- LLM_GPU body — dlopen the CUDA shim and dispatch matvec to it.
---------------------------------------------------------------------

with Ada.Environment_Variables;
with Ada.Unchecked_Conversion;
with Interfaces.C;          use Interfaces.C;
with Interfaces.C.Strings;  use Interfaces.C.Strings;

package body LLM_GPU is

   use type System.Address;

   function C_dlopen (Name : chars_ptr; Flag : int) return System.Address
     with Import => True, Convention => C, External_Name => "dlopen";
   function C_dlsym (Handle : System.Address; Name : chars_ptr) return System.Address
     with Import => True, Convention => C, External_Name => "dlsym";

   RTLD_NOW : constant int := 2;

   --  Matches  void aspida_gpu_matvec(const void* w, long wbytes, int kind,
   --                                  int in, int out, const float* x, float* y)
   type MatVec_Fn is access procedure
     (W : System.Address; WB : Interfaces.C.long; Kind : int;
      In_D : int; Out_D : int; X : System.Address; Y : System.Address)
     with Convention => C;

   function To_Fn is new Ada.Unchecked_Conversion (System.Address, MatVec_Fn);

   --  Matches  void aspida_gpu_matmul(const void* w, long wbytes, int kind,
   --                  int in, int out, int batch, const float* x, float* y)
   type MatMul_Fn is access procedure
     (W : System.Address; WB : Interfaces.C.long; Kind : int;
      In_D : int; Out_D : int; Batch : int; X : System.Address; Y : System.Address)
     with Convention => C;

   function To_MM is new Ada.Unchecked_Conversion (System.Address, MatMul_Fn);

   Fn        : MatVec_Fn := null;
   MM_Fn     : MatMul_Fn := null;

   --  The lazy dlopen must run exactly once even when several handler tasks
   --  call Available()/Init concurrently at startup. A bare Boolean flag had a
   --  check-then-act race (two tasks could both see False and both dlopen).
   --  The protected procedure serialises the first call; later callers take
   --  the early-out under the lock. Fn/MM_Fn stay as package-level access
   --  values so the hot MatVec/MatMul paths read them without locking.
   protected Init_Guard is
      procedure Run;
   private
      Done : Boolean := False;
   end Init_Guard;

   protected body Init_Guard is
      procedure Run is
         H : System.Address;
      begin
         if Done then
            return;
         end if;
         Done := True;
         if not Ada.Environment_Variables.Exists ("ASPIDA_GPU") then
            return;
         end if;
         declare
            Lib : constant String :=
              (if Ada.Environment_Variables.Exists ("ASPIDA_GPU_LIB")
               then Ada.Environment_Variables.Value ("ASPIDA_GPU_LIB")
               else "./libaspidagpu.so");
            CL  : chars_ptr := New_String (Lib);
         begin
            H := C_dlopen (CL, RTLD_NOW);
            Free (CL);
            if H = System.Null_Address then
               return;
            end if;
            declare
               CS : chars_ptr := New_String ("aspida_gpu_matvec");
               A  : System.Address;
            begin
               A := C_dlsym (H, CS);
               Free (CS);
               if A /= System.Null_Address then
                  Fn := To_Fn (A);
               end if;
            end;
            declare
               CS : chars_ptr := New_String ("aspida_gpu_matmul");
               A  : System.Address;
            begin
               A := C_dlsym (H, CS);
               Free (CS);
               if A /= System.Null_Address then
                  MM_Fn := To_MM (A);
               end if;
            end;
         end;
      end Run;
   end Init_Guard;

   procedure Init is
   begin
      Init_Guard.Run;
   end Init;

   function Available return Boolean is
   begin
      Init;
      return Fn /= null;
   end Available;

   procedure MatVec
     (W_Addr  : System.Address;
      W_Bytes : Long_Long_Integer;
      Kind    : Integer;
      In_Dim  : Integer;
      Out_Dim : Integer;
      X       : System.Address;
      Y       : System.Address) is
   begin
      Fn (W_Addr, Interfaces.C.long (W_Bytes), int (Kind),
          int (In_Dim), int (Out_Dim), X, Y);
   end MatVec;

   function Has_MatMul return Boolean is
   begin
      Init;
      return MM_Fn /= null;
   end Has_MatMul;

   procedure MatMul
     (W_Addr  : System.Address;
      W_Bytes : Long_Long_Integer;
      Kind    : Integer;
      In_Dim  : Integer;
      Out_Dim : Integer;
      Batch   : Integer;
      X       : System.Address;
      Y       : System.Address) is
   begin
      MM_Fn (W_Addr, Interfaces.C.long (W_Bytes), int (Kind),
             int (In_Dim), int (Out_Dim), int (Batch), X, Y);
   end MatMul;

end LLM_GPU;
