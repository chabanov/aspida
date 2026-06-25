------------------------------------------------------------------------
-- Train_GPU_Resident body — dlopen the resident-session shim and dispatch.
-- Mirrors LLM_GPU: lazy one-time dlopen, dlsym the C ABI, function-pointer
-- conversions, graceful fallback (Available = False) when the shim is absent.
------------------------------------------------------------------------

with Interfaces.C;          use Interfaces.C;
with Interfaces.C.Strings;  use Interfaces.C.Strings;
with Ada.Environment_Variables;
with Ada.Unchecked_Conversion;

package body Train_GPU_Resident is

   use type System.Address;

   function C_dlopen (Name : chars_ptr; Flag : int) return System.Address
     with Import, Convention => C, External_Name => "dlopen";
   function C_dlsym (Handle : System.Address; Name : chars_ptr) return System.Address
     with Import, Convention => C, External_Name => "dlsym";
   RTLD_NOW : constant int := 2;

   --  C ABI of the shim.
   type Create_Fn is access function
     (L, B, D : int; LR : C_Float) return System.Address with Convention => C;
   type Set_Fn is access procedure
     (H : System.Address; X, T : System.Address) with Convention => C;
   type Step_Fn is access function
     (H : System.Address) return C_Float with Convention => C;
   type Free_Fn is access procedure (H : System.Address) with Convention => C;
   type Loss_Fn is access function (H : System.Address) return C_Float with Convention => C;
   type Grad_Fn is access function
     (H : System.Address; Layer, Idx : int) return C_Float with Convention => C;
   type WGet_Fn is access function
     (H : System.Address; Layer, Idx : int) return C_Float with Convention => C;
   type WSet_Fn is access procedure
     (H : System.Address; Layer, Idx : int; V : C_Float) with Convention => C;

   function To_Create is new Ada.Unchecked_Conversion (System.Address, Create_Fn);
   function To_Set    is new Ada.Unchecked_Conversion (System.Address, Set_Fn);
   function To_Step   is new Ada.Unchecked_Conversion (System.Address, Step_Fn);
   function To_Free   is new Ada.Unchecked_Conversion (System.Address, Free_Fn);
   function To_Loss   is new Ada.Unchecked_Conversion (System.Address, Loss_Fn);
   function To_Grad   is new Ada.Unchecked_Conversion (System.Address, Grad_Fn);
   function To_WGet   is new Ada.Unchecked_Conversion (System.Address, WGet_Fn);
   function To_WSet   is new Ada.Unchecked_Conversion (System.Address, WSet_Fn);

   Loaded  : Boolean := False;
   Ok      : Boolean := False;
   P_Create : Create_Fn := null;
   P_Set    : Set_Fn    := null;
   P_Step   : Step_Fn   := null;
   P_Free   : Free_Fn   := null;
   P_Loss   : Loss_Fn   := null;
   P_Grad   : Grad_Fn   := null;
   P_WGet   : WGet_Fn   := null;
   P_WSet   : WSet_Fn   := null;

   procedure Init is
      H : System.Address;
      function Sym (Name : String) return System.Address is
         CS : chars_ptr := New_String (Name);
         A  : System.Address;
      begin
         A := C_dlsym (H, CS);
         Free (CS);
         return A;
      end Sym;
   begin
      if Loaded then
         return;
      end if;
      Loaded := True;
      declare
         Path : constant String :=
           (if Ada.Environment_Variables.Exists ("ASPIDA_TRAIN_LIB")
            then Ada.Environment_Variables.Value ("ASPIDA_TRAIN_LIB")
            else "./libaspidatrain.so");
         CL : chars_ptr := New_String (Path);
         AC, AS, AStp, AF, AL, AG, AWg, AWs : System.Address;
      begin
         H := C_dlopen (CL, RTLD_NOW);
         Free (CL);
         if H = System.Null_Address then
            return;
         end if;
         AC := Sym ("art_create");
         AS := Sym ("art_set_data");
         AStp := Sym ("art_step");
         AF := Sym ("art_free");
         AL := Sym ("art_loss_only");
         AG := Sym ("art_grad_at");
         AWg := Sym ("art_w_get");
         AWs := Sym ("art_w_set");
         if AC /= System.Null_Address and then AS /= System.Null_Address
           and then AStp /= System.Null_Address and then AF /= System.Null_Address
           and then AL /= System.Null_Address and then AG /= System.Null_Address
           and then AWg /= System.Null_Address and then AWs /= System.Null_Address
         then
            P_Create := To_Create (AC);
            P_Set    := To_Set (AS);
            P_Step   := To_Step (AStp);
            P_Free   := To_Free (AF);
            P_Loss   := To_Loss (AL);
            P_Grad   := To_Grad (AG);
            P_WGet   := To_WGet (AWg);
            P_WSet   := To_WSet (AWs);
            Ok := True;
         end if;
      end;
   end Init;

   function Available return Boolean is
   begin
      Init;
      return Ok;
   end Available;

   function Create (L, B, D : Integer; LR : Float) return Session is
   begin
      if not Available then
         raise Not_Available;
      end if;
      return P_Create (int (L), int (B), int (D), C_Float (LR));
   end Create;

   procedure Set_Data (S : Session; X, T : F32_Array) is
   begin
      if not Available then
         raise Not_Available;
      end if;
      P_Set (S, X (X'First)'Address, T (T'First)'Address);
   end Set_Data;

   function Step (S : Session) return Float is
   begin
      if not Available then
         raise Not_Available;
      end if;
      return Float (P_Step (S));
   end Step;

   procedure Free (S : Session) is
   begin
      if Ok and then S /= System.Null_Address then
         P_Free (S);
      end if;
   end Free;

   function Loss_Only (S : Session) return Float is
   begin
      if not Available then raise Not_Available; end if;
      return Float (P_Loss (S));
   end Loss_Only;

   function Grad_At (S : Session; Layer, Idx : Integer) return Float is
   begin
      if not Available then raise Not_Available; end if;
      return Float (P_Grad (S, int (Layer), int (Idx)));
   end Grad_At;

   function W_Get (S : Session; Layer, Idx : Integer) return Float is
   begin
      if not Available then raise Not_Available; end if;
      return Float (P_WGet (S, int (Layer), int (Idx)));
   end W_Get;

   procedure W_Set (S : Session; Layer, Idx : Integer; V : Float) is
   begin
      if not Available then raise Not_Available; end if;
      P_WSet (S, int (Layer), int (Idx), C_Float (V));
   end W_Set;

end Train_GPU_Resident;
