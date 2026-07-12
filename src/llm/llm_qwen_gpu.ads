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

   --  Resident delta-net layer (Increment 2 / Phase B). Dnet_New allocates the
   --  per-layer device state — recurrent S_All [NV*KHD, VHD] AND the causal-
   --  conv history window [Kernel-1, QO] — and returns a handle >= 0 (or -1 if
   --  unavailable / alloc failed). Dnet_Step then runs the ENTIRE delta-net
   --  decode layer on the device in one call: qkv/alpha/beta/z projections,
   --  causal conv1d + SiLU, per-head decay/beta transform, the gated-delta
   --  recurrence + gated RMSNorm, and the output projection — 1 H2D of x and
   --  1 D2H of the result instead of ~5 blocking per-matvec round-trips.
   --  Oracle: the CPU path in LLM_DeltaNet_Blk.Step.
   --
   --  Projection weights are GPU_Weight (Kind >= 0 K-quant, or -1 meaning the
   --  raw bytes are a dense row-major [out,in] F32 matrix — the caller must
   --  verify F32 for Kind = -1). The small dense tensors (conv/a/dt/norm) are
   --  host F32 addresses + byte lengths; they upload once via the resident
   --  weight cache keyed by host pointer.
   function Dnet_Available return Boolean;

   function Dnet_New (NV, KHD, VHD, QO, Kernel : Integer) return Integer;

   procedure Dnet_Step
     (Handle  : Integer;
      X       : System.Address;   -- [Dim] f32 host in
      Dim     : Integer;
      QKV_W   : GPU_Weight;       -- rows = QO
      Alpha_W : GPU_Weight;       -- rows = NV
      Beta_W  : GPU_Weight;       -- rows = NV
      Gate_W  : GPU_Weight;       -- rows = V_Dim
      Out_W   : GPU_Weight;       -- [Dim, V_Dim]
      Conv_W  : System.Address;   -- [QO, Kernel] f32 host
      Conv_B  : Long_Long_Integer;
      A_W     : System.Address;   -- [NV] f32 host
      A_B     : Long_Long_Integer;
      Dt_W    : System.Address;   -- [NV] f32 host
      Dt_B    : Long_Long_Integer;
      Norm_W  : System.Address;   -- [VHD] f32 host
      Norm_B  : Long_Long_Integer;
      NV, KHD, VHD, QO, Q_Dim, N_K_Heads, V_Dim, Kernel : Integer;
      Y       : System.Address);  -- [Dim] f32 host out

   --  Resident full-attention layer (Phase B2). Fattn_New allocates the
   --  device K/V caches [Max_Len, KVD] + score scratch and returns a handle
   --  (or -1). Fattn_Step runs the whole GQA decode layer on the device:
   --  q(+gate)/k/v projections, per-head QK-RMSNorm + partial RoPE, K/V
   --  append at Pos, causal softmax over the cache, per-dim sigmoid gate,
   --  out projection. Oracle: the CPU path in LLM_FullAttn.Step. The caller
   --  advances its Len after the call (Pos is the 0-based position).
   function Fattn_Available return Boolean;

   function Fattn_New (Max_Len, KVD, NQ : Integer) return Integer;

   procedure Fattn_Step
     (Handle   : Integer;
      X        : System.Address;   -- [Dim] f32 host in
      Dim      : Integer;
      Q_W      : GPU_Weight;       -- rows = NQ*2*HD (query|gate)
      K_W      : GPU_Weight;       -- rows = NKV*HD
      V_W      : GPU_Weight;
      O_W      : GPU_Weight;       -- [Dim, NQ*HD]
      Q_Norm   : System.Address;   -- [HD] f32 host
      QN_B     : Long_Long_Integer;
      K_Norm   : System.Address;   -- [HD] f32 host
      KN_B     : Long_Long_Integer;
      NQ, NKV, HD, Pos : Integer;
      RD       : Integer;          -- rope dim
      Base     : Float;
      Freq_Scale, M_Scale : Float;
      Yarn_On  : Integer;
      Corr_Lo, Corr_Hi : Float;
      FF       : System.Address;   -- [RD/2] f32 host, or Null_Address
      FF_B     : Long_Long_Integer;
      Use_FF, Interleaved, Sec_Total : Integer;
      Y        : System.Address);  -- [Dim] f32 host out

   --  Release a per-generation device state (S_All/conv window or K/V cache).
   --  Without these every request leaks its states' VRAM; slots are reused.
   procedure Dnet_Free  (Handle : Integer);
   procedure Fattn_Free (Handle : Integer);

   --  Phase C — full resident forward chain. Layers are registered once per
   --  loaded model (device weight pointers resolved through the resident
   --  cache); Chain_Forward then runs a whole decode step on the device:
   --  embedding row in, logits out, hidden state never leaves VRAM.
   --  Handles is the address of a C int array: per layer, the generation's
   --  dnet/fattn state handle.
   function Chain_Available return Boolean;
   procedure Chain_Reset;

   procedure Chain_Dnet
     (Attn_Norm : System.Address; AN_B : Long_Long_Integer;
      Post_Norm : System.Address; PN_B : Long_Long_Integer;
      QKV_W, Alpha_W, Beta_W, Gate_W, Out_W : GPU_Weight;
      Conv_W : System.Address; Conv_B : Long_Long_Integer;
      A_W    : System.Address; A_B    : Long_Long_Integer;
      Dt_W   : System.Address; Dt_B   : Long_Long_Integer;
      Norm_W : System.Address; Norm_B : Long_Long_Integer;
      NV, KHD, VHD, QO, Q_Dim, N_K_Heads, V_Dim, Kernel : Integer);

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
      Use_FF, Interleaved, Sec_Total : Integer);

   procedure Chain_MoE
     (Router, Gate_Exp, Up_Exp, Down_Exp,
      Shared_Gate, Shared_Up, Shared_Down : GPU_Weight;
      SGI : System.Address; SGI_B : Long_Long_Integer; SGI_Len : Integer;
      N_Experts, Top_K, Intermed : Integer);

   procedure Chain_Model
     (Embed : System.Address; Embed_B : Long_Long_Integer;
      FNorm : System.Address; FNorm_B : Long_Long_Integer;
      LM    : System.Address; LM_B    : Long_Long_Integer; LM_K : Integer;
      Dim, Vocab : Integer);

   function Chain_Ready return Boolean;

   --  Bind the per-generation state handles and force a fresh graph capture
   --  (device state pointers differ from the previous generation). Call before
   --  the decode loop; Chain_End releases the graph after it.
   procedure Chain_Begin (Handles : System.Address);
   procedure Chain_End;

   procedure Chain_Forward
     (Embed_Row : Integer;      -- 0-based row in the embedding table
      Pos       : Integer;      -- 0-based token position (RoPE / KV append)
      Handles   : System.Address;  -- C int[n_layers] state handles
      Logits    : System.Address); -- [Vocab] f32 host out

   --  Batched decode step for B lanes (continuous batching): the shared-weight
   --  matvecs read each weight once for all B (the throughput win). Rows/Pos
   --  are C int[B]; Handles is C int[B*n_layers] (lane b's layer-li state);
   --  Logits is [B*Vocab] host out.
   function Chain_Batch_Available return Boolean;
   procedure Chain_Forward_Batch
     (B : Integer; Rows, Pos, Handles, Logits : System.Address);

end LLM_Qwen_GPU;
