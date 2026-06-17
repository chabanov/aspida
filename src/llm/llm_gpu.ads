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
   --  0 = Q4_K, 1 = Q6_K). X/Y point to In_Dim / Out_Dim contiguous C floats.
   procedure MatVec
     (W_Addr  : System.Address;
      W_Bytes : Long_Long_Integer;
      Kind    : Integer;
      In_Dim  : Integer;
      Out_Dim : Integer;
      X       : System.Address;
      Y       : System.Address);

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
