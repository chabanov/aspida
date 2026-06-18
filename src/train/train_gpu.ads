---------------------------------------------------------------------
-- Train_GPU — Ada binding to the GPU training shim (libaspidatrain.so),
-- dlopen'd at runtime exactly like LLM_GPU does for inference. Off unless
-- ASPIDA_TRAIN_GPU is set; library path from ASPIDA_TRAIN_LIB (default
-- ./libaspidatrain.so). All buffers are flat row-major FP32.
--
-- Entry points mirror the validated kernels (see gpu/train_shim.cu):
--   MM_Fwd:  C[M,N] = A[M,K] . B[K,N]
--   MM_DA :  dA[M,K] = dC[M,N] . B[K,N]^T
--   MM_DB :  dB[K,N] = A[M,K]^T . dC[M,N]
--   Softmax_Bwd / RMSNorm_Bwd : as in Train.
---------------------------------------------------------------------

with Interfaces.C;

package Train_GPU is

   subtype C_Float is Interfaces.C.C_float;
   type F32_Array is array (Natural range <>) of C_Float;

   --  True once the shim is dlopen'd and the entry points resolved.
   function Available return Boolean;

   Not_Available : exception;   -- raised by the ops if Available is False

   procedure MM_Fwd (A, B : F32_Array; C : out F32_Array; M, K, N : Positive);
   procedure MM_DA  (DC, B : F32_Array; DA : out F32_Array; M, K, N : Positive);
   procedure MM_DB  (A, DC : F32_Array; DB : out F32_Array; M, K, N : Positive);
   procedure Softmax_Bwd
     (P, DP : F32_Array; DS : out F32_Array; R, N : Positive);
   procedure RMSNorm_Bwd
     (X, G, DY : F32_Array; DX, DG : out F32_Array;
      R, D : Positive; Eps : Float := 1.0E-6);

end Train_GPU;
