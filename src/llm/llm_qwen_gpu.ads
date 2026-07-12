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

   --  Fused MoE FFN decode: y[dim] := MoE(x[dim]).
   --
   --  Runs router GEMV -> stable softmax -> greedy top-K -> the K selected
   --  SwiGLU experts (gate/up/down 3D expert slices) -> weighted combine ->
   --  shared expert (with optional sigmoid gate), entirely on the device.
   --  Gate/Up expert blocks are 3D [n_experts, intermed, dim]; Down is
   --  [n_experts, dim, intermed]; per-expert byte stride = Bytes / N_Experts.
   --
   --  Shared_Gate_Inp is the dense-F32 gate-input vector (length Dim) or a
   --  null address with Gate_Inp_Len <= 1 when the layer has no shared gate.
   --
   --  Precondition (caller): every GPU_Weight.Kind >= 0 and Available. If any
   --  weight is non-K-quant the caller must use the CPU path instead.
   procedure MoE_Decode
     (X               : System.Address;   -- [Dim] f32 host in
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
      Shared_Gate_Inp : System.Address;   -- [Dim] f32 host, or Null_Address
      Gate_Inp_Len    : Integer;
      Y               : System.Address);  -- [Dim] f32 host out

end LLM_Qwen_GPU;
