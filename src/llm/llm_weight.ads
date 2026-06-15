---------------------------------------------------------------------
-- LLM_Weight — a projection weight that is either a dense F32 tensor or
-- a still-quantized GGUF tensor matvec'd on the fly (QMatVec).
--
-- This lets the layer math be written once (MatVec / MatVec_Expert) while
-- unit tests use dense synthetic weights and the real model keeps weights
-- quantized so they fit in RAM. Logical orientation is always [out, in].
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_GGUF;
with System;

package LLM_Weight is

   --  Shared handle to a tensor's raw quantized bytes (model-lifetime; the
   --  model owns these and they are never freed or deep-copied).
   type Byte_Data is access String;

   type Weight is private;

   --  Build from a dense [out, in] tensor (used by tests / small weights).
   function From_Dense (T : LLM_Tensor.Tensor) return Weight;

   --  Build from a still-quantized GGUF tensor; Bytes is the raw tensor data.
   function From_Quant
     (Info : LLM_GGUF.Tensor_Info; Bytes : Byte_Data) return Weight;

   function Present (W : Weight) return Boolean;
   function Rows    (W : Weight) return Integer;   -- output dim
   function Cols    (W : Weight) return Integer;   -- input dim
   function Count   (W : Weight) return Long_Long_Integer;  -- total elements

   --  y[out] = W[out,in] . x[in].
   function MatVec (W : Weight; X : LLM_Tensor.Tensor) return LLM_Tensor.Tensor;

   --  Dequantize a single row (0-based) of a [in, out]-shaped weight, i.e. an
   --  embedding-table lookup without materialising the whole F32 tensor.
   --  Returns a [1, in] tensor.
   function Get_Row (W : Weight; Row : Integer) return LLM_Tensor.Tensor;

   --  Raw quantized-tensor access, for offloading the matvec to a GPU backend
   --  (LLM_GPU). Address/length of the still-quantized bytes and a small kind
   --  code (0 = Q4_K, 1 = Q6_K, -1 = other/dense) the GPU kernels understand.
   function Raw_Address (W : Weight) return System.Address;
   function Raw_Bytes   (W : Weight) return Long_Long_Integer;
   function Kind_Code   (W : Weight) return Integer;

   --  3D expert weight (row-major [n_experts, out_e, in]); matvec one expert.
   function N_Experts (W : Weight) return Integer;
   function Expert_Out (W : Weight) return Integer;   -- per-expert output dim
   function MatVec_Expert
     (W : Weight; E : Integer; X : LLM_Tensor.Tensor) return LLM_Tensor.Tensor;

private

   type Weight is record
      Present_F : Boolean := False;
      Is_Quant  : Boolean := False;
      Dense     : LLM_Tensor.Tensor;            -- when not quant
      Info      : LLM_GGUF.Tensor_Info;         -- when quant
      Bytes     : Byte_Data;                    -- when quant (shared, not freed)
   end record;

end LLM_Weight;
