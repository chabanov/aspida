---------------------------------------------------------------------
-- LLM_Weights — Binary FP32 weight loader for GPT-2 format
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_Autograd;

package LLM_Weights is

   -- Read a tensor from a binary FP32 file (little-endian, row-major)
   function Load_Tensor (Path : String; Shape : LLM_Tensor.Dims) return LLM_Autograd.Var;

   -- Load a 1D tensor (bias or layer norm weight)
   function Load_Vector (Path : String; N : Integer) return LLM_Autograd.Var;

   -- Load a 2D tensor (weight matrix)
   function Load_Matrix (Path : String; Rows, Cols : Integer) return LLM_Autograd.Var;

end LLM_Weights;
