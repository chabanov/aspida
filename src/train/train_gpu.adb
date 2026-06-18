---------------------------------------------------------------------
-- Train_GPU body — dlopen the training shim and dispatch FP32 ops to it.
---------------------------------------------------------------------

with Ada.Environment_Variables;
with Ada.Unchecked_Conversion;
with Interfaces.C;          use Interfaces.C;
with Interfaces.C.Strings;  use Interfaces.C.Strings;
with System;

package body Train_GPU is

   use type System.Address;

   function C_dlopen (Name : chars_ptr; Flag : int) return System.Address
     with Import => True, Convention => C, External_Name => "dlopen";
   function C_dlsym (Handle : System.Address; Name : chars_ptr) return System.Address
     with Import => True, Convention => C, External_Name => "dlsym";
   RTLD_NOW : constant int := 2;

   --  void f(const float*,const float*,float*,int,int,int)
   type MM_Fn is access procedure
     (A, B, C : System.Address; M, K, N : int) with Convention => C;
   --  void f(const float*,const float*,float*,int,int)
   type SM_Fn is access procedure
     (P, DP, DS : System.Address; R, N : int) with Convention => C;
   --  void f(const float*,const float*,const float*,float*,float*,int,int,float)
   type RN_Fn is access procedure
     (X, G, DY, DX, DG : System.Address; R, D : int; Eps : C_Float)
     with Convention => C;

   function To_MM is new Ada.Unchecked_Conversion (System.Address, MM_Fn);
   function To_SM is new Ada.Unchecked_Conversion (System.Address, SM_Fn);
   function To_RN is new Ada.Unchecked_Conversion (System.Address, RN_Fn);

   Fwd_Fn, DA_Fn, DB_Fn : MM_Fn := null;
   SBwd_Fn : SM_Fn := null;
   RBwd_Fn : RN_Fn := null;
   Init_Done : Boolean := False;

   function Sym (H : System.Address; Name : String) return System.Address is
      CS : chars_ptr := New_String (Name);
      A  : constant System.Address := C_dlsym (H, CS);
   begin
      Free (CS);
      return A;
   end Sym;

   procedure Init is
      H : System.Address;
   begin
      if Init_Done then
         return;
      end if;
      Init_Done := True;
      if not Ada.Environment_Variables.Exists ("ASPIDA_TRAIN_GPU") then
         return;
      end if;
      declare
         Lib : constant String :=
           (if Ada.Environment_Variables.Exists ("ASPIDA_TRAIN_LIB")
            then Ada.Environment_Variables.Value ("ASPIDA_TRAIN_LIB")
            else "./libaspidatrain.so");
         CL  : chars_ptr := New_String (Lib);
      begin
         H := C_dlopen (CL, RTLD_NOW);
         Free (CL);
         if H = System.Null_Address then
            return;
         end if;
         declare
            A1 : constant System.Address := Sym (H, "aspida_mm_fwd");
            A2 : constant System.Address := Sym (H, "aspida_mm_dA");
            A3 : constant System.Address := Sym (H, "aspida_mm_dB");
            A4 : constant System.Address := Sym (H, "aspida_softmax_bwd");
            A5 : constant System.Address := Sym (H, "aspida_rmsnorm_bwd");
         begin
            if A1 /= System.Null_Address then Fwd_Fn := To_MM (A1); end if;
            if A2 /= System.Null_Address then DA_Fn  := To_MM (A2); end if;
            if A3 /= System.Null_Address then DB_Fn  := To_MM (A3); end if;
            if A4 /= System.Null_Address then SBwd_Fn := To_SM (A4); end if;
            if A5 /= System.Null_Address then RBwd_Fn := To_RN (A5); end if;
         end;
      end;
   end Init;

   function Available return Boolean is
   begin
      Init;
      return Fwd_Fn /= null and then DA_Fn /= null and then DB_Fn /= null
        and then SBwd_Fn /= null and then RBwd_Fn /= null;
   end Available;

   procedure Need is
   begin
      if not Available then raise Not_Available; end if;
   end Need;

   procedure MM_Fwd (A, B : F32_Array; C : out F32_Array; M, K, N : Positive) is
   begin
      Need;
      Fwd_Fn (A'Address, B'Address, C'Address, int (M), int (K), int (N));
   end MM_Fwd;

   procedure MM_DA (DC, B : F32_Array; DA : out F32_Array; M, K, N : Positive) is
   begin
      Need;
      DA_Fn (DC'Address, B'Address, DA'Address, int (M), int (K), int (N));
   end MM_DA;

   procedure MM_DB (A, DC : F32_Array; DB : out F32_Array; M, K, N : Positive) is
   begin
      Need;
      DB_Fn (A'Address, DC'Address, DB'Address, int (M), int (K), int (N));
   end MM_DB;

   procedure Softmax_Bwd
     (P, DP : F32_Array; DS : out F32_Array; R, N : Positive) is
   begin
      Need;
      SBwd_Fn (P'Address, DP'Address, DS'Address, int (R), int (N));
   end Softmax_Bwd;

   procedure RMSNorm_Bwd
     (X, G, DY : F32_Array; DX, DG : out F32_Array;
      R, D : Positive; Eps : Float := 1.0E-6) is
   begin
      Need;
      RBwd_Fn (X'Address, G'Address, DY'Address, DX'Address, DG'Address,
               int (R), int (D), C_Float (Eps));
   end RMSNorm_Bwd;

end Train_GPU;
