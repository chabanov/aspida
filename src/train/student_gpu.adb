------------------------------------------------------------------------
-- Student_GPU body — dlopen the resident Student shim and dispatch.
-- Mirrors Train_GPU_Resident: lazy one-time dlopen, dlsym the C ABI,
-- function-pointer conversions, graceful fallback when the shim is absent.
------------------------------------------------------------------------

with Interfaces.C;          use Interfaces.C;
with Interfaces.C.Strings;  use Interfaces.C.Strings;
with Ada.Environment_Variables;
with Ada.Unchecked_Conversion;

package body Student_GPU is

   use type System.Address;

   function C_dlopen (Name : chars_ptr; Flag : int) return System.Address
     with Import, Convention => C, External_Name => "dlopen";
   function C_dlsym (Handle : System.Address; Name : chars_ptr) return System.Address
     with Import, Convention => C, External_Name => "dlsym";
   RTLD_NOW : constant int := 2;

   type Create_Fn is access function
     (V, D, F, S, L, H : int) return System.Address with Convention => C;
   type Set_Fn    is access procedure
     (H : System.Address; Ids, Tgts : System.Address) with Convention => C;
   type Step_Fn   is access function
     (H : System.Address; LR : C_Float) return C_Float with Convention => C;
   type Free_Fn   is access procedure (H : System.Address) with Convention => C;
   type Micro_Fn  is access function (H : System.Address) return C_Float with Convention => C;
   type Apply_Fn  is access procedure
     (H : System.Address; LR : C_Float; G : int) with Convention => C;
   type Distill_Fn is access procedure
     (H : System.Address; Ids, Q : System.Address) with Convention => C;
   type NParams_Fn is access function (H : System.Address) return int
     with Convention => C;
   type GetW_Fn    is access procedure
     (H : System.Address; Out_W : System.Address) with Convention => C;

   function To_Create is new Ada.Unchecked_Conversion (System.Address, Create_Fn);
   function To_Set    is new Ada.Unchecked_Conversion (System.Address, Set_Fn);
   function To_Step   is new Ada.Unchecked_Conversion (System.Address, Step_Fn);
   function To_Free   is new Ada.Unchecked_Conversion (System.Address, Free_Fn);
   function To_Micro  is new Ada.Unchecked_Conversion (System.Address, Micro_Fn);
   function To_Apply  is new Ada.Unchecked_Conversion (System.Address, Apply_Fn);
   function To_Distill is new Ada.Unchecked_Conversion (System.Address, Distill_Fn);
   function To_NParams is new Ada.Unchecked_Conversion (System.Address, NParams_Fn);
   function To_GetW    is new Ada.Unchecked_Conversion (System.Address, GetW_Fn);

   Loaded : Boolean := False;
   Ok     : Boolean := False;
   P_Create : Create_Fn := null;
   P_Set    : Set_Fn    := null;
   P_Step   : Step_Fn   := null;
   P_Free   : Free_Fn   := null;
   P_Micro  : Micro_Fn  := null;
   P_Apply  : Apply_Fn  := null;
   P_Distill : Distill_Fn := null;
   P_NParams : NParams_Fn := null;
   P_GetW    : GetW_Fn    := null;

   procedure Init is
      H : System.Address;
      function Sym (Name : String) return System.Address is
         CS : chars_ptr := New_String (Name);
         A  : System.Address;
      begin
         A := C_dlsym (H, CS); Free (CS); return A;
      end Sym;
   begin
      if Loaded then return; end if;
      Loaded := True;
      declare
         Path : constant String :=
           (if Ada.Environment_Variables.Exists ("ASPIDA_STUDENT_LIB")
            then Ada.Environment_Variables.Value ("ASPIDA_STUDENT_LIB")
            else "./libaspidastudent.so");
         CL : chars_ptr := New_String (Path);
         AC, ASd, AStp, AF : System.Address;
      begin
         H := C_dlopen (CL, RTLD_NOW); Free (CL);
         if H = System.Null_Address then return; end if;
         AC := Sym ("stu_create"); ASd := Sym ("stu_set_data");
         AStp := Sym ("stu_step");  AF := Sym ("stu_free");
         declare
            AMi : constant System.Address := Sym ("stu_micro");
            AAp : constant System.Address := Sym ("stu_apply");
            ADi : constant System.Address := Sym ("stu_set_distill");
            ANp : constant System.Address := Sym ("stu_nparams");
            AGw : constant System.Address := Sym ("stu_get_weights");
         begin
            if AC /= System.Null_Address and then ASd /= System.Null_Address
              and then AStp /= System.Null_Address and then AF /= System.Null_Address
              and then AMi /= System.Null_Address and then AAp /= System.Null_Address
              and then ADi /= System.Null_Address and then ANp /= System.Null_Address
              and then AGw /= System.Null_Address
            then
               P_Create := To_Create (AC); P_Set := To_Set (ASd);
               P_Step := To_Step (AStp);   P_Free := To_Free (AF);
               P_Micro := To_Micro (AMi);  P_Apply := To_Apply (AAp);
               P_Distill := To_Distill (ADi);
               P_NParams := To_NParams (ANp);
               P_GetW    := To_GetW (AGw);
               Ok := True;
            end if;
         end;
      end;
   end Init;

   function Available return Boolean is
   begin Init; return Ok; end Available;

   function Create (Voc, Dim, Ff, Seq, Layers, Heads : Positive) return Session is
   begin
      if not Available then raise Not_Available; end if;
      return P_Create (int (Voc), int (Dim), int (Ff), int (Seq),
                       int (Layers), int (Heads));
   end Create;

   procedure Set_Data (S : Session; Ids, Targets : Int_Array) is
   begin
      if not Available then raise Not_Available; end if;
      P_Set (S, Ids (Ids'First)'Address, Targets (Targets'First)'Address);
   end Set_Data;

   procedure Set_Distill (S : Session; Ids : Int_Array; Q : F32_Array) is
   begin
      if not Available then raise Not_Available; end if;
      P_Distill (S, Ids (Ids'First)'Address, Q (Q'First)'Address);
   end Set_Distill;

   function Step (S : Session; LR : Float) return Float is
   begin
      if not Available then raise Not_Available; end if;
      return Float (P_Step (S, C_Float (LR)));
   end Step;

   function N_Params (S : Session) return Natural is
   begin
      if not Available then raise Not_Available; end if;
      return Natural (P_NParams (S));
   end N_Params;

   procedure Get_Weights (S : Session; Out_W : out F32_Array) is
   begin
      if not Available then raise Not_Available; end if;
      P_GetW (S, Out_W (Out_W'First)'Address);
   end Get_Weights;

   function Micro (S : Session) return Float is
   begin
      if not Available then raise Not_Available; end if;
      return Float (P_Micro (S));
   end Micro;

   procedure Apply (S : Session; LR : Float; G : Positive) is
   begin
      if not Available then raise Not_Available; end if;
      P_Apply (S, C_Float (LR), int (G));
   end Apply;

   procedure Free (S : Session) is
   begin
      if Ok and then S /= System.Null_Address then P_Free (S); end if;
   end Free;

end Student_GPU;
