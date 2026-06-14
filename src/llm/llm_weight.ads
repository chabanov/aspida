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
