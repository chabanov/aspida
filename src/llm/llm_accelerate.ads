---------------------------------------------------------------------
-- LLM_Accelerate — Apple Accelerate.framework BLAS bindings
--
-- Uses cblas_sgemm for FP32 matrix multiplication, which is
-- hardware-optimized on Apple Silicon (AMX + Neon).
--
-- cblas.h API:
--   void cblas_sgemm(
--     CBLAS_ORDER Order,       // 101=RowMajor, 102=ColMajor  
--     CBLAS_TRANSPOSE TransA,  // 111=NoTrans, 112=Trans
--     CBLAS_TRANSPOSE TransB,
--     int M, int N, int K,
--     float alpha,
--     const float *A, int lda,
--     const float *B, int ldb,
--     float beta,
--     float *C, int ldc);
---------------------------------------------------------------------

with LLM_Tensor;

package LLM_Accelerate is

   -- Perform C = alpha*A*B + beta*C using Accelerate BLAS
   -- All matrices in row-major order.
   -- A: M×K, B: K×N, C: M×N
   procedure SGEMM
     (A : LLM_Tensor.Tensor;
      B : LLM_Tensor.Tensor;
      C : in out LLM_Tensor.Tensor;
      Alpha : Float := 1.0;
      Beta  : Float := 0.0;
      Transpose_A : Boolean := False;
      Transpose_B : Boolean := False)
     with Pre => LLM_Tensor.Rank (A) = 2 and
                 LLM_Tensor.Rank (B) = 2 and
                 LLM_Tensor.Rank (C) = 2;

   -- Vector dot product using accelerate
   function Dot (A, B : LLM_Tensor.Tensor) return Float
     with Pre => LLM_Tensor.Numel (A) = LLM_Tensor.Numel (B);

   -- Check if Accelerate.framework is available
   function Is_Available return Boolean;

end LLM_Accelerate;
