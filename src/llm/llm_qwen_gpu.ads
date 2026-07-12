---------------------------------------------------------------------
-- LLM_Qwen_GPU — resident-GPU decode blocks for the Qwen/MoE backend.
--
-- This is the "resident forward" path (see GPU_RESIDENT_FORWARD.md):
-- instead of offloading one matvec at a time (LLM_GPU.MatVec, which round-trips
-- the activation host<->device on every weight), a whole decode block runs on
-- the device with the hidden state kept resident, so per token only the input
-- goes in and the result comes out once.
--
-- Increment 1 covers the MoE FFN block. Every entry is validated bit-exact
-- against its CPU oracle (LLM_MoE.Forward) before it is allowed on the hot path.
-- Weights are addressed by the SAME host pointers LLM_GPU already caches in
-- VRAM (Raw_Address), so no extra upload — the device weight cache is shared.
--
-- Available is False unless ASPIDA_GPU_RESIDENT is set and libaspidagpu.so
-- exports the resident entry points; callers fall back to the CPU/per-matvec
-- path, so this package is always safe to link.
---------------------------------------------------------------------

with System;

package LLM_Qwen_GPU is

   --  True iff the resident shim is loaded and aspida_gpu_moe_decode resolved.
   function Available return Boolean;

   --  A quantized weight the device already has cached, described the way the
   --  CUDA side needs it: host pointer (the VRAM cache key), byte length, and
   --  the K-quant kind code (0=Q4_K, 1=Q6_K, 2=Q5_K, 3=Q3_K, 4=Q2_K; -1 other).
   type GPU_Weight is record
      Addr  : System.Address    := System.Null_Address;
      Bytes : Long_Long_Integer := 0;
      Kind  : Integer           := -1;
   end record;

   --  Fused MoE experts: y[dim] := combine(selected experts) + shared expert.
   --
   --  The router GEMV, stable softmax and greedy top-K stay on the CPU (they are
   --  tiny, and the router is frequently non-K-quant in mixed quants) — the
   --  caller (LLM_MoE.Forward) computes Top_Idx (0-based expert indices) and the
   --  renormalised Top_W, then this runs ONLY the expensive part on the device
   --  with the activation resident: the K selected SwiGLU experts (gate/up/down
   --  3D slices) -> weighted combine -> shared expert (optional sigmoid gate).
   --  Gate/Up expert blocks are 3D [n_experts, intermed, dim]; Down is
   --  [n_experts, dim, intermed]; per-expert byte stride = Bytes / N_Experts.
   --
   --  Top_Idx points at a C int[Top_K] (0-based), Top_W at a C float[Top_K].
   --  Shared_Gate_Inp is the dense gate-input vector (length Dim) or a null
   --  address with Gate_Inp_Len <= 1 when the layer has no shared gate.
   --
   --  Precondition (caller): every expert/shared GPU_Weight.Kind >= 0 and
   --  Available. If any is non-K-quant the caller must use the CPU path.
   procedure MoE_Experts
     (X               : System.Address;   -- [Dim] f32 host in
      Dim             : Integer;
      Top_K           : Integer;
      Intermed        : Integer;
      N_Experts       : Integer;
      Top_Idx         : System.Address;   -- C int[Top_K], 0-based
      Top_W           : System.Address;   -- C float[Top_K]
      Gate_Exp        : GPU_Weight;
      Up_Exp          : GPU_Weight;
      Down_Exp        : GPU_Weight;
      Shared_Gate     : GPU_Weight;
      Shared_Up       : GPU_Weight;
      Shared_Down     : GPU_Weight;
      Shared_Gate_Inp : System.Address;   -- [Dim] f32 host, or Null_Address
      Gate_Inp_Len    : Integer;
      Y               : System.Address);  -- [Dim] f32 host out

   --  Resident delta-net recurrence (Increment 2). Dnet_New allocates the
   --  per-layer recurrent state S_All [NV*KHD, VHD] on the device and returns a
   --  handle >= 0 (or -1 if unavailable / alloc failed). Dnet_Recur runs one
   --  decode step's per-head recurrence + gated RMSNorm against that resident
   --  state — only cq/gate/beta/z go in and o_row comes out; S_All stays on the
   --  device. Oracle: LLM_DeltaNet.Step + the gated norm in LLM_DeltaNet_Blk.
   function Dnet_Available return Boolean;

   function Dnet_New (NV, KHD, VHD : Integer) return Integer;

   procedure Dnet_Recur
     (Handle : Integer;
      CQ     : System.Address;   -- [QO]    conv'd + SiLU qkv (host)
      Gate   : System.Address;   -- [NV]    per-head decay
      Beta   : System.Address;   -- [NV]    per-head beta
      Z      : System.Address;   -- [V_Dim] gate projection
      Norm_W : System.Address;   -- [VHD]   dense norm weight
      O_Row  : System.Address;   -- [V_Dim] output (host)
      NV, KHD, VHD, QO, Q_Dim, N_K_Heads, V_Dim : Integer);

end LLM_Qwen_GPU;
