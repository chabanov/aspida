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

   --  int aspida_gpu_fattn_new(int max_len, int kvd, int nq)
   type Fattn_New_Fn is access function
     (Max_Len, KVD, NQ : int) return int
     with Convention => C;
   function To_FNew is new Ada.Unchecked_Conversion (System.Address, Fattn_New_Fn);

   --  void aspida_gpu_fattn_step(...) — see llm_qwen_gpu.ads / gpu_matvec.cu.
   type Fattn_Step_Fn is access procedure
     (Handle : int; X : System.Address; Dim : int;
      QW : System.Address; QB : Interfaces.C.long; QK : int;
      KW : System.Address; KB : Interfaces.C.long; KK : int;
      VW : System.Address; VB : Interfaces.C.long; VK : int;
      OW : System.Address; OB : Interfaces.C.long; OK : int;
      QN : System.Address; QNB : Interfaces.C.long;
      KN : System.Address; KNB : Interfaces.C.long;
      NQ, NKV, HD, Pos : int;
      RD : int; Base : C_float; Freq_Scale, M_Scale : C_float;
      Yarn_On : int; Corr_Lo, Corr_Hi : C_float;
      FF : System.Address; FFB : Interfaces.C.long;
      Use_FF, Interleaved, Sec_Total : int;
      Y : System.Address)
     with Convention => C;
   function To_FStep is new Ada.Unchecked_Conversion (System.Address, Fattn_Step_Fn);

   type Free_H_Fn is access procedure (Handle : int) with Convention => C;
   function To_FreeH is new Ada.Unchecked_Conversion (System.Address, Free_H_Fn);

   type Void_Fn is access procedure with Convention => C;
   function To_Void is new Ada.Unchecked_Conversion (System.Address, Void_Fn);
   type Int_Fn is access function return int with Convention => C;
   function To_Int is new Ada.Unchecked_Conversion (System.Address, Int_Fn);

   type Chain_Dnet_Fn is access procedure
     (AN : System.Address; ANB : Interfaces.C.long;
      PN : System.Address; PNB : Interfaces.C.long;
      QW : System.Address; QB : Interfaces.C.long; QK : int;
      AW : System.Address; AB : Interfaces.C.long; AK : int;
      BW : System.Address; BB : Interfaces.C.long; BK : int;
      GW : System.Address; GB : Interfaces.C.long; GK : int;
      OW : System.Address; OB : Interfaces.C.long; OK : int;
      CW : System.Address; CB : Interfaces.C.long;
      AAW : System.Address; AAB : Interfaces.C.long;
      DW : System.Address; DB : Interfaces.C.long;
      NW : System.Address; NB : Interfaces.C.long;
      NV, KHD, VHD, QO, Q_Dim, N_K_Heads, V_Dim, Kernel : int)
     with Convention => C;
   function To_CDnet is new Ada.Unchecked_Conversion (System.Address, Chain_Dnet_Fn);

   type Chain_Fattn_Fn is access procedure
     (AN : System.Address; ANB : Interfaces.C.long;
      PN : System.Address; PNB : Interfaces.C.long;
      QW : System.Address; QB : Interfaces.C.long; QK : int;
      KW : System.Address; KB : Interfaces.C.long; KK : int;
      VW : System.Address; VB : Interfaces.C.long; VK : int;
      OW : System.Address; OB : Interfaces.C.long; OK : int;
      QN : System.Address; QNB : Interfaces.C.long;
      KN : System.Address; KNB : Interfaces.C.long;
      NQ, NKV, HD : int;
      RD : int; Base, Freq_Scale, M_Scale : C_float;
      Yarn_On : int; Corr_Lo, Corr_Hi : C_float;
      FF : System.Address; FFB : Interfaces.C.long;
      Use_FF, Interleaved, Sec_Total : int)
     with Convention => C;
   function To_CFattn is new Ada.Unchecked_Conversion (System.Address, Chain_Fattn_Fn);

   type Chain_MoE_Fn is access procedure
     (RW : System.Address; RB : Interfaces.C.long; RK : int;
      GW : System.Address; GB : Interfaces.C.long; GK : int;
      UW : System.Address; UB : Interfaces.C.long; UK : int;
      DW : System.Address; DB : Interfaces.C.long; DK : int;
      SGW : System.Address; SGB : Interfaces.C.long; SGK : int;
      SUW : System.Address; SUB : Interfaces.C.long; SUK : int;
      SDW : System.Address; SDB : Interfaces.C.long; SDK : int;
      SGI : System.Address; SGIB : Interfaces.C.long; SGIL : int;
      N_Exp, Top_K, Intermed : int)
     with Convention => C;
   function To_CMoE is new Ada.Unchecked_Conversion (System.Address, Chain_MoE_Fn);

   type Chain_Model_Fn is access procedure
     (E : System.Address; EB : Interfaces.C.long;
      F : System.Address; FB : Interfaces.C.long;
      L : System.Address; LB : Interfaces.C.long; LK : int;
      Dim, Vocab : int)
     with Convention => C;
   function To_CModel is new Ada.Unchecked_Conversion (System.Address, Chain_Model_Fn);

   type Chain_Fwd_Fn is access procedure
     (Row, Pos : int; Handles : System.Address; Logits : System.Address)
     with Convention => C;
   function To_CFwd is new Ada.Unchecked_Conversion (System.Address, Chain_Fwd_Fn);

   type Chain_Begin_Fn is access procedure (Handles : System.Address)
     with Convention => C;
   function To_CBegin is new Ada.Unchecked_Conversion (System.Address, Chain_Begin_Fn);

   type Chain_Batch_Fn is access procedure
     (B : int; Rows, Pos, Handles, Logits : System.Address) with Convention => C;
   function To_CBatch is new Ada.Unchecked_Conversion (System.Address, Chain_Batch_Fn);

   type Chain_Prefill_Fn is access procedure
     (P : int; Rows : System.Address; Pos_Start : int;
      Handles, Last_Logits : System.Address) with Convention => C;
   function To_CPre is new Ada.Unchecked_Conversion (System.Address, Chain_Prefill_Fn);

   Fn       : MoE_Fn := null;
   DNew_Fn  : Dnet_New_Fn := null;
   DStep_Fn : Dnet_Step_Fn := null;
   FNew_Fn  : Fattn_New_Fn := null;
   FStep_Fn : Fattn_Step_Fn := null;
   DFree_Fn : Free_H_Fn := null;
   FFree_Fn : Free_H_Fn := null;
   CReset_Fn : Void_Fn := null;
   CReady_Fn : Int_Fn := null;
   CDnet_Fn  : Chain_Dnet_Fn := null;
   CFattn_Fn : Chain_Fattn_Fn := null;
   CMoE_Fn   : Chain_MoE_Fn := null;
   CModel_Fn : Chain_Model_Fn := null;
   CFwd_Fn   : Chain_Fwd_Fn := null;
   CBegin_Fn : Chain_Begin_Fn := null;
   CEnd_Fn   : Void_Fn := null;
   CBatch_Fn : Chain_Batch_Fn := null;
   CErr_Fn   : Int_Fn := null;   --  aspida_gpu_last_error
   CPre_Fn   : Chain_Prefill_Fn := null;

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
            declare
               CS : chars_ptr := New_String ("aspida_gpu_fattn_new");
               A  : System.Address;
            begin
               A := C_dlsym (H, CS);
               Free (CS);
               if A /= System.Null_Address then
                  FNew_Fn := To_FNew (A);
               end if;
            end;
            declare
               CS : chars_ptr := New_String ("aspida_gpu_fattn_step");
               A  : System.Address;
            begin
               A := C_dlsym (H, CS);
               Free (CS);
               if A /= System.Null_Address then
                  FStep_Fn := To_FStep (A);
               end if;
            end;
            declare
               procedure Look (Name : String; Dst : out System.Address) is
                  CS : chars_ptr := New_String (Name);
               begin
                  Dst := C_dlsym (H, CS);
                  Free (CS);
               end Look;
               A : System.Address;
            begin
               Look ("aspida_gpu_dnet_free", A);
               if A /= System.Null_Address then DFree_Fn := To_FreeH (A); end if;
               Look ("aspida_gpu_fattn_free", A);
               if A /= System.Null_Address then FFree_Fn := To_FreeH (A); end if;
               Look ("aspida_gpu_chain_reset", A);
               if A /= System.Null_Address then CReset_Fn := To_Void (A); end if;
               Look ("aspida_gpu_chain_ready", A);
               if A /= System.Null_Address then CReady_Fn := To_Int (A); end if;
               Look ("aspida_gpu_chain_dnet", A);
               if A /= System.Null_Address then CDnet_Fn := To_CDnet (A); end if;
               Look ("aspida_gpu_chain_fattn", A);
               if A /= System.Null_Address then CFattn_Fn := To_CFattn (A); end if;
               Look ("aspida_gpu_chain_moe", A);
               if A /= System.Null_Address then CMoE_Fn := To_CMoE (A); end if;
               Look ("aspida_gpu_chain_model", A);
               if A /= System.Null_Address then CModel_Fn := To_CModel (A); end if;
               Look ("aspida_gpu_chain_forward", A);
               if A /= System.Null_Address then CFwd_Fn := To_CFwd (A); end if;
               Look ("aspida_gpu_chain_begin", A);
               if A /= System.Null_Address then CBegin_Fn := To_CBegin (A); end if;
               Look ("aspida_gpu_chain_end", A);
               if A /= System.Null_Address then CEnd_Fn := To_Void (A); end if;
               Look ("aspida_gpu_chain_forward_batch", A);
               if A /= System.Null_Address then CBatch_Fn := To_CBatch (A); end if;
               Look ("aspida_gpu_last_error", A);
               if A /= System.Null_Address then CErr_Fn := To_Int (A); end if;
               Look ("aspida_gpu_chain_prefill", A);
               if A /= System.Null_Address then CPre_Fn := To_CPre (A); end if;
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

   function Fattn_Available return Boolean is
   begin
      Init;
      return FNew_Fn /= null and then FStep_Fn /= null;
   end Fattn_Available;

   function Fattn_New (Max_Len, KVD, NQ : Integer) return Integer is
   begin
      Init;
      if FNew_Fn = null then
         return -1;
      end if;
      return Integer (FNew_Fn (int (Max_Len), int (KVD), int (NQ)));
   end Fattn_New;

   procedure Fattn_Step
     (Handle   : Integer;
      X        : System.Address;
      Dim      : Integer;
      Q_W      : GPU_Weight;
      K_W      : GPU_Weight;
      V_W      : GPU_Weight;
      O_W      : GPU_Weight;
      Q_Norm   : System.Address;
      QN_B     : Long_Long_Integer;
      K_Norm   : System.Address;
      KN_B     : Long_Long_Integer;
      NQ, NKV, HD, Pos : Integer;
      RD       : Integer;
      Base     : Float;
      Freq_Scale, M_Scale : Float;
      Yarn_On  : Integer;
      Corr_Lo, Corr_Hi : Float;
      FF       : System.Address;
      FF_B     : Long_Long_Integer;
      Use_FF, Interleaved, Sec_Total : Integer;
      Y        : System.Address) is
   begin
      FStep_Fn (int (Handle), X, int (Dim),
                Q_W.Addr, Interfaces.C.long (Q_W.Bytes), int (Q_W.Kind),
                K_W.Addr, Interfaces.C.long (K_W.Bytes), int (K_W.Kind),
                V_W.Addr, Interfaces.C.long (V_W.Bytes), int (V_W.Kind),
                O_W.Addr, Interfaces.C.long (O_W.Bytes), int (O_W.Kind),
                Q_Norm, Interfaces.C.long (QN_B),
                K_Norm, Interfaces.C.long (KN_B),
                int (NQ), int (NKV), int (HD), int (Pos),
                int (RD), C_float (Base), C_float (Freq_Scale), C_float (M_Scale),
                int (Yarn_On), C_float (Corr_Lo), C_float (Corr_Hi),
                FF, Interfaces.C.long (FF_B),
                int (Use_FF), int (Interleaved), int (Sec_Total), Y);
   end Fattn_Step;

   procedure Dnet_Free (Handle : Integer) is
   begin
      if DFree_Fn /= null and then Handle >= 0 then
         DFree_Fn (int (Handle));
      end if;
   end Dnet_Free;

   procedure Fattn_Free (Handle : Integer) is
   begin
      if FFree_Fn /= null and then Handle >= 0 then
         FFree_Fn (int (Handle));
      end if;
   end Fattn_Free;

   function Chain_Available return Boolean is
   begin
      Init;
      return CDnet_Fn /= null and then CFattn_Fn /= null
        and then CMoE_Fn /= null and then CModel_Fn /= null
        and then CFwd_Fn /= null and then CReady_Fn /= null;
   end Chain_Available;

   procedure Chain_Reset is
   begin
      if CReset_Fn /= null then
         CReset_Fn.all;
      end if;
   end Chain_Reset;

   procedure Chain_Dnet
     (Attn_Norm : System.Address; AN_B : Long_Long_Integer;
      Post_Norm : System.Address; PN_B : Long_Long_Integer;
      QKV_W, Alpha_W, Beta_W, Gate_W, Out_W : GPU_Weight;
      Conv_W : System.Address; Conv_B : Long_Long_Integer;
      A_W    : System.Address; A_B    : Long_Long_Integer;
      Dt_W   : System.Address; Dt_B   : Long_Long_Integer;
      Norm_W : System.Address; Norm_B : Long_Long_Integer;
      NV, KHD, VHD, QO, Q_Dim, N_K_Heads, V_Dim, Kernel : Integer) is
   begin
      CDnet_Fn (Attn_Norm, Interfaces.C.long (AN_B),
                Post_Norm, Interfaces.C.long (PN_B),
                QKV_W.Addr, Interfaces.C.long (QKV_W.Bytes), int (QKV_W.Kind),
                Alpha_W.Addr, Interfaces.C.long (Alpha_W.Bytes), int (Alpha_W.Kind),
                Beta_W.Addr, Interfaces.C.long (Beta_W.Bytes), int (Beta_W.Kind),
                Gate_W.Addr, Interfaces.C.long (Gate_W.Bytes), int (Gate_W.Kind),
                Out_W.Addr, Interfaces.C.long (Out_W.Bytes), int (Out_W.Kind),
                Conv_W, Interfaces.C.long (Conv_B),
                A_W, Interfaces.C.long (A_B),
                Dt_W, Interfaces.C.long (Dt_B),
                Norm_W, Interfaces.C.long (Norm_B),
                int (NV), int (KHD), int (VHD), int (QO), int (Q_Dim),
                int (N_K_Heads), int (V_Dim), int (Kernel));
   end Chain_Dnet;

   procedure Chain_Fattn
     (Attn_Norm : System.Address; AN_B : Long_Long_Integer;
      Post_Norm : System.Address; PN_B : Long_Long_Integer;
      Q_W, K_W, V_W, O_W : GPU_Weight;
      Q_Norm : System.Address; QN_B : Long_Long_Integer;
      K_Norm : System.Address; KN_B : Long_Long_Integer;
      NQ, NKV, HD : Integer;
      RD : Integer; Base, Freq_Scale, M_Scale : Float;
      Yarn_On : Integer; Corr_Lo, Corr_Hi : Float;
      FF : System.Address; FF_B : Long_Long_Integer;
      Use_FF, Interleaved, Sec_Total : Integer) is
   begin
      CFattn_Fn (Attn_Norm, Interfaces.C.long (AN_B),
                 Post_Norm, Interfaces.C.long (PN_B),
                 Q_W.Addr, Interfaces.C.long (Q_W.Bytes), int (Q_W.Kind),
                 K_W.Addr, Interfaces.C.long (K_W.Bytes), int (K_W.Kind),
                 V_W.Addr, Interfaces.C.long (V_W.Bytes), int (V_W.Kind),
                 O_W.Addr, Interfaces.C.long (O_W.Bytes), int (O_W.Kind),
                 Q_Norm, Interfaces.C.long (QN_B),
                 K_Norm, Interfaces.C.long (KN_B),
                 int (NQ), int (NKV), int (HD),
                 int (RD), C_float (Base), C_float (Freq_Scale), C_float (M_Scale),
                 int (Yarn_On), C_float (Corr_Lo), C_float (Corr_Hi),
                 FF, Interfaces.C.long (FF_B),
                 int (Use_FF), int (Interleaved), int (Sec_Total));
   end Chain_Fattn;

   procedure Chain_MoE
     (Router, Gate_Exp, Up_Exp, Down_Exp,
      Shared_Gate, Shared_Up, Shared_Down : GPU_Weight;
      SGI : System.Address; SGI_B : Long_Long_Integer; SGI_Len : Integer;
      N_Experts, Top_K, Intermed : Integer) is
   begin
      CMoE_Fn (Router.Addr, Interfaces.C.long (Router.Bytes), int (Router.Kind),
               Gate_Exp.Addr, Interfaces.C.long (Gate_Exp.Bytes), int (Gate_Exp.Kind),
               Up_Exp.Addr, Interfaces.C.long (Up_Exp.Bytes), int (Up_Exp.Kind),
               Down_Exp.Addr, Interfaces.C.long (Down_Exp.Bytes), int (Down_Exp.Kind),
               Shared_Gate.Addr, Interfaces.C.long (Shared_Gate.Bytes), int (Shared_Gate.Kind),
               Shared_Up.Addr, Interfaces.C.long (Shared_Up.Bytes), int (Shared_Up.Kind),
               Shared_Down.Addr, Interfaces.C.long (Shared_Down.Bytes), int (Shared_Down.Kind),
               SGI, Interfaces.C.long (SGI_B), int (SGI_Len),
               int (N_Experts), int (Top_K), int (Intermed));
   end Chain_MoE;

   procedure Chain_Model
     (Embed : System.Address; Embed_B : Long_Long_Integer;
      FNorm : System.Address; FNorm_B : Long_Long_Integer;
      LM    : System.Address; LM_B    : Long_Long_Integer; LM_K : Integer;
      Dim, Vocab : Integer) is
   begin
      CModel_Fn (Embed, Interfaces.C.long (Embed_B),
                 FNorm, Interfaces.C.long (FNorm_B),
                 LM, Interfaces.C.long (LM_B), int (LM_K), int (Dim), int (Vocab));
   end Chain_Model;

   function Chain_Ready return Boolean is
   begin
      Init;
      return CReady_Fn /= null and then CReady_Fn.all /= 0;
   end Chain_Ready;

   procedure Chain_Begin (Handles : System.Address) is
   begin
      if CBegin_Fn /= null then
         CBegin_Fn (Handles);
      end if;
   end Chain_Begin;

   procedure Chain_End is
   begin
      if CEnd_Fn /= null then
         CEnd_Fn.all;
      end if;
   end Chain_End;

   --  A failed GPU op must abort this generation (so Decode_Tokens frees its
   --  state and the handler releases the inference lock) rather than leave the
   --  token loop running on poisoned device state or a wedged handler holding
   --  the lock while the GPU sits idle — the 2026-07-13 prod GPU-0% wedge.
   procedure Check_GPU is
   begin
      if CErr_Fn /= null then
         declare
            E : constant int := CErr_Fn.all;
         begin
            if E /= 0 then
               raise GPU_Error with "CUDA error" & int'Image (E);
            end if;
         end;
      end if;
   end Check_GPU;

   procedure Chain_Forward
     (Embed_Row : Integer;
      Pos       : Integer;
      Handles   : System.Address;
      Logits    : System.Address) is
   begin
      CFwd_Fn (int (Embed_Row), int (Pos), Handles, Logits);
      Check_GPU;
   end Chain_Forward;

   function Chain_Batch_Available return Boolean is
   begin
      Init;
      return CBatch_Fn /= null;
   end Chain_Batch_Available;

   function Chain_Prefill_Available return Boolean is
   begin
      Init;
      --  ASPIDA_NO_PREFILL forces the per-token path (A/B bit-exactness check).
      if Ada.Environment_Variables.Exists ("ASPIDA_NO_PREFILL") then
         return False;
      end if;
      return CPre_Fn /= null;
   end Chain_Prefill_Available;

   procedure Chain_Prefill
     (P : Integer; Rows : System.Address; Pos_Start : Integer;
      Handles : System.Address; Last_Logits : System.Address) is
   begin
      CPre_Fn (int (P), Rows, int (Pos_Start), Handles, Last_Logits);
      Check_GPU;
   end Chain_Prefill;

   procedure Chain_Forward_Batch
     (B : Integer; Rows, Pos, Handles, Logits : System.Address) is
   begin
      CBatch_Fn (int (B), Rows, Pos, Handles, Logits);
      Check_GPU;
   end Chain_Forward_Batch;

end LLM_Qwen_GPU;
