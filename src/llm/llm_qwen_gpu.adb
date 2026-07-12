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

   --  void aspida_gpu_moe_experts(const float* x, int dim, int top_k,
   --      int intermed, int n_exp,
   --      const int* top_idx, const float* top_w,
   --      const void* gw, long gb, int gk,   -- gate expert (3D)
   --      const void* uw, long ub, int uk,   -- up   expert (3D)
   --      const void* dw, long db, int dk,   -- down expert (3D)
   --      const void* sgw,long sgb,int sgk,  -- shared gate (2D)
   --      const void* suw,long sub,int suk,  -- shared up   (2D)
   --      const void* sdw,long sdb,int sdk,  -- shared down (2D)
   --      const float* sgi, int sgi_len, float* y);
   type MoE_Fn is access procedure
     (X : System.Address; Dim, Top_K, Intermed, N_Exp : int;
      Top_Idx : System.Address; Top_W : System.Address;
      GW : System.Address; GB : Interfaces.C.long; GK : int;
      UW : System.Address; UB : Interfaces.C.long; UK : int;
      DW : System.Address; DB : Interfaces.C.long; DK : int;
      SGW : System.Address; SGB : Interfaces.C.long; SGK : int;
      SUW : System.Address; SUB : Interfaces.C.long; SUK : int;
      SDW : System.Address; SDB : Interfaces.C.long; SDK : int;
      SGI : System.Address; SGI_Len : int; Y : System.Address)
     with Convention => C;

   function To_Fn is new Ada.Unchecked_Conversion (System.Address, MoE_Fn);

   --  int aspida_gpu_dnet_new(int nv, int khd, int vhd, int qo, int kernel)
   type Dnet_New_Fn is access function
     (NV, KHD, VHD, QO, Kernel : int) return int
     with Convention => C;
   function To_DNew is new Ada.Unchecked_Conversion (System.Address, Dnet_New_Fn);

   --  void aspida_gpu_dnet_step(int handle, const float* x, int dim,
   --      qkv_w,b,k, al_w,b,k, be_w,b,k, ga_w,b,k, out_w,b,k,
   --      conv_w,b, a_w,b, dt_w,b, norm_w,b,
   --      int nv,khd,vhd,qo,q_dim,n_k_heads,v_dim,kernel, float* out)
   type Dnet_Step_Fn is access procedure
     (Handle : int; X : System.Address; Dim : int;
      QW : System.Address; QB : Interfaces.C.long; QK : int;
      AW : System.Address; AB : Interfaces.C.long; AK : int;
      BW : System.Address; BB : Interfaces.C.long; BK : int;
      GW : System.Address; GB : Interfaces.C.long; GK : int;
      OW : System.Address; OB : Interfaces.C.long; OK : int;
      CW : System.Address; CB : Interfaces.C.long;
      AAW : System.Address; AAB : Interfaces.C.long;
      DW : System.Address; DB : Interfaces.C.long;
      NW : System.Address; NB : Interfaces.C.long;
      NV, KHD, VHD, QO, Q_Dim, N_K_Heads, V_Dim, Kernel : int;
      Y : System.Address)
     with Convention => C;
   function To_DStep is new Ada.Unchecked_Conversion (System.Address, Dnet_Step_Fn);

   Fn       : MoE_Fn := null;
   DNew_Fn  : Dnet_New_Fn := null;
   DStep_Fn : Dnet_Step_Fn := null;

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
               CS : chars_ptr := New_String ("aspida_gpu_moe_experts");
               A  : System.Address;
            begin
               A := C_dlsym (H, CS);
               Free (CS);
               if A /= System.Null_Address then
                  Fn := To_Fn (A);
               end if;
            end;
            declare
               CS : chars_ptr := New_String ("aspida_gpu_dnet_new");
               A  : System.Address;
            begin
               A := C_dlsym (H, CS);
               Free (CS);
               if A /= System.Null_Address then
                  DNew_Fn := To_DNew (A);
               end if;
            end;
            declare
               CS : chars_ptr := New_String ("aspida_gpu_dnet_step");
               A  : System.Address;
            begin
               A := C_dlsym (H, CS);
               Free (CS);
               if A /= System.Null_Address then
                  DStep_Fn := To_DStep (A);
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

   procedure MoE_Experts
     (X               : System.Address;
      Dim             : Integer;
      Top_K           : Integer;
      Intermed        : Integer;
      N_Experts       : Integer;
      Top_Idx         : System.Address;
      Top_W           : System.Address;
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
      Fn (X, int (Dim), int (Top_K), int (Intermed), int (N_Experts),
          Top_Idx, Top_W,
          Gate_Exp.Addr,    Interfaces.C.long (Gate_Exp.Bytes),    int (Gate_Exp.Kind),
          Up_Exp.Addr,      Interfaces.C.long (Up_Exp.Bytes),      int (Up_Exp.Kind),
          Down_Exp.Addr,    Interfaces.C.long (Down_Exp.Bytes),    int (Down_Exp.Kind),
          Shared_Gate.Addr, Interfaces.C.long (Shared_Gate.Bytes), int (Shared_Gate.Kind),
          Shared_Up.Addr,   Interfaces.C.long (Shared_Up.Bytes),   int (Shared_Up.Kind),
          Shared_Down.Addr, Interfaces.C.long (Shared_Down.Bytes), int (Shared_Down.Kind),
          Shared_Gate_Inp,  int (Gate_Inp_Len), Y);
   end MoE_Experts;

   function Dnet_Available return Boolean is
   begin
      Init;
      return DNew_Fn /= null and then DStep_Fn /= null;
   end Dnet_Available;

   function Dnet_New (NV, KHD, VHD, QO, Kernel : Integer) return Integer is
   begin
      Init;
      if DNew_Fn = null then
         return -1;
      end if;
      return Integer (DNew_Fn (int (NV), int (KHD), int (VHD), int (QO), int (Kernel)));
   end Dnet_New;

   procedure Dnet_Step
     (Handle  : Integer;
      X       : System.Address;
      Dim     : Integer;
      QKV_W   : GPU_Weight;
      Alpha_W : GPU_Weight;
      Beta_W  : GPU_Weight;
      Gate_W  : GPU_Weight;
      Out_W   : GPU_Weight;
      Conv_W  : System.Address;
      Conv_B  : Long_Long_Integer;
      A_W     : System.Address;
      A_B     : Long_Long_Integer;
      Dt_W    : System.Address;
      Dt_B    : Long_Long_Integer;
      Norm_W  : System.Address;
      Norm_B  : Long_Long_Integer;
      NV, KHD, VHD, QO, Q_Dim, N_K_Heads, V_Dim, Kernel : Integer;
      Y       : System.Address) is
   begin
      DStep_Fn (int (Handle), X, int (Dim),
                QKV_W.Addr,   Interfaces.C.long (QKV_W.Bytes),   int (QKV_W.Kind),
                Alpha_W.Addr, Interfaces.C.long (Alpha_W.Bytes), int (Alpha_W.Kind),
                Beta_W.Addr,  Interfaces.C.long (Beta_W.Bytes),  int (Beta_W.Kind),
                Gate_W.Addr,  Interfaces.C.long (Gate_W.Bytes),  int (Gate_W.Kind),
                Out_W.Addr,   Interfaces.C.long (Out_W.Bytes),   int (Out_W.Kind),
                Conv_W, Interfaces.C.long (Conv_B),
                A_W,    Interfaces.C.long (A_B),
                Dt_W,   Interfaces.C.long (Dt_B),
                Norm_W, Interfaces.C.long (Norm_B),
                int (NV), int (KHD), int (VHD), int (QO), int (Q_Dim),
                int (N_K_Heads), int (V_Dim), int (Kernel), Y);
   end Dnet_Step;

end LLM_Qwen_GPU;
