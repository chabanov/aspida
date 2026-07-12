---------------------------------------------------------------------
-- LLM_Qwen_GPU body — dlopen the resident CUDA entry points.
--
-- Same dlopen/dlsym + Unchecked_Conversion pattern as LLM_GPU, gated behind
-- ASPIDA_GPU_RESIDENT so the per-matvec path stays the default until the
-- resident kernels have cleared their bit-exactness gates. Shares the shim
-- (and therefore the resident weight cache) with LLM_GPU.
---------------------------------------------------------------------

with Ada.Environment_Variables;
with Ada.Unchecked_Conversion;
with Interfaces.C;          use Interfaces.C;
with Interfaces.C.Strings;  use Interfaces.C.Strings;

package body LLM_Qwen_GPU is

   use type System.Address;

   function C_dlopen (Name : chars_ptr; Flag : int) return System.Address
     with Import => True, Convention => C, External_Name => "dlopen";
   function C_dlsym (Handle : System.Address; Name : chars_ptr) return System.Address
     with Import => True, Convention => C, External_Name => "dlsym";

   RTLD_NOW : constant int := 2;

   --  void aspida_gpu_moe_decode(const float* x, int dim, int n_exp, int top_k,
   --      int intermed,
   --      const void* rw, long rb, int rk,
   --      const void* gw, long gb, int gk,   -- gate expert (3D)
   --      const void* uw, long ub, int uk,   -- up   expert (3D)
   --      const void* dw, long db, int dk,   -- down expert (3D)
   --      const void* sgw,long sgb,int sgk,  -- shared gate (2D)
   --      const void* suw,long sub,int suk,  -- shared up   (2D)
   --      const void* sdw,long sdb,int sdk,  -- shared down (2D)
   --      const float* sgi, int sgi_len, float* y);
   type MoE_Fn is access procedure
     (X : System.Address; Dim, N_Exp, Top_K, Intermed : int;
      RW : System.Address; RB : Interfaces.C.long; RK : int;
      GW : System.Address; GB : Interfaces.C.long; GK : int;
      UW : System.Address; UB : Interfaces.C.long; UK : int;
      DW : System.Address; DB : Interfaces.C.long; DK : int;
      SGW : System.Address; SGB : Interfaces.C.long; SGK : int;
      SUW : System.Address; SUB : Interfaces.C.long; SUK : int;
      SDW : System.Address; SDB : Interfaces.C.long; SDK : int;
      SGI : System.Address; SGI_Len : int; Y : System.Address)
     with Convention => C;

   function To_Fn is new Ada.Unchecked_Conversion (System.Address, MoE_Fn);

   Fn : MoE_Fn := null;

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
         --  Opt-in: resident path stays off until explicitly enabled.
         if not Ada.Environment_Variables.Exists ("ASPIDA_GPU_RESIDENT") then
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
               CS : chars_ptr := New_String ("aspida_gpu_moe_decode");
               A  : System.Address;
            begin
               A := C_dlsym (H, CS);
               Free (CS);
               if A /= System.Null_Address then
                  Fn := To_Fn (A);
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

   procedure MoE_Decode
     (X               : System.Address;
      Dim             : Integer;
      N_Experts       : Integer;
      Top_K           : Integer;
      Intermed        : Integer;
      Router          : GPU_Weight;
      Gate_Exp        : GPU_Weight;
      Up_Exp          : GPU_Weight;
      Down_Exp        : GPU_Weight;
      Shared_Gate     : GPU_Weight;
      Shared_Up       : GPU_Weight;
      Shared_Down     : GPU_Weight;
      Shared_Gate_Inp : System.Address;
      Gate_Inp_Len    : Integer;
      Y               : System.Address) is
   begin
      Fn (X, int (Dim), int (N_Experts), int (Top_K), int (Intermed),
          Router.Addr,      Interfaces.C.long (Router.Bytes),      int (Router.Kind),
          Gate_Exp.Addr,    Interfaces.C.long (Gate_Exp.Bytes),    int (Gate_Exp.Kind),
          Up_Exp.Addr,      Interfaces.C.long (Up_Exp.Bytes),      int (Up_Exp.Kind),
          Down_Exp.Addr,    Interfaces.C.long (Down_Exp.Bytes),    int (Down_Exp.Kind),
          Shared_Gate.Addr, Interfaces.C.long (Shared_Gate.Bytes), int (Shared_Gate.Kind),
          Shared_Up.Addr,   Interfaces.C.long (Shared_Up.Bytes),   int (Shared_Up.Kind),
          Shared_Down.Addr, Interfaces.C.long (Shared_Down.Bytes), int (Shared_Down.Kind),
          Shared_Gate_Inp,  int (Gate_Inp_Len), Y);
   end MoE_Decode;

end LLM_Qwen_GPU;
