---------------------------------------------------------------------
-- LLM_GPU — optional GPU offload for quantized matvec, loaded at runtime.
--
-- When the environment variable ASPIDA_GPU is set, this dlopen()s the CUDA
-- shim (libaspidagpu.so, path via ASPIDA_GPU_LIB or ./libaspidagpu.so) and
-- routes Q4_K / Q6_K mat-vecs to it. If the library is absent (e.g. a CPU-only
-- build/host), Available returns False and callers fall back to the pure-Ada
-- LLM_Weight.MatVec — so nothing links against CUDA at build time and the CPU
-- engine is completely unaffected.
---------------------------------------------------------------------

with System;

package LLM_GPU is

   --  True iff the GPU shim was found and loaded (ASPIDA_GPU set + lib present).
   function Available return Boolean;

   --  y[out] = W . x[in], with W the still-quantized bytes at W_Addr (Kind:
   --  0=Q4_K, 1=Q6_K, 2=Q5_K, 3=Q3_K, 4=Q2_K). X/Y point to In_Dim / Out_Dim
   --  contiguous C floats.
   procedure MatVec
     (W_Addr  : System.Address;
      W_Bytes : Long_Long_Integer;
      Kind    : Integer;
      In_Dim  : Integer;
      Out_Dim : Integer;
      X       : System.Address;
      Y       : System.Address);

   --  Drop the device-side mirror of the weight whose host bytes start at
   --  Addr (the same address passed as W_Addr to MatVec/MatMul). The shim
   --  caches each distinct host weight pointer's uploaded VRAM copy keyed by
   --  that pointer; when a model is evicted its host bytes are freed and may be
   --  reallocated, so the stale device buffer must be released first — both to
   --  avoid a VRAM leak and to prevent a future model whose bytes land at the
   --  same host address from being served the previous model's weights.
   --
   --  No-op when the GPU is disabled, the shim is absent, or the shim predates
   --  this entry point (older shims simply leak the eviction's VRAM — call it
   --  unconditionally; it is always safe). Idempotent for a given Addr.
   procedure Free_Weight (Addr : System.Address);

   --  True iff the loaded shim also exports the batched matmul entry point.
   function Has_MatMul return Boolean;

   --  Batched: Y[Batch,Out] = X[Batch,In] . W (row-major). One weight read
   --  serves all Batch rows — the continuous-batching throughput primitive.
   procedure MatMul
     (W_Addr  : System.Address;
      W_Bytes : Long_Long_Integer;
      Kind    : Integer;
      In_Dim  : Integer;
      Out_Dim : Integer;
      Batch   : Integer;
      X       : System.Address;
      Y       : System.Address);

end LLM_GPU;
