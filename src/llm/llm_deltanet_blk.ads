---------------------------------------------------------------------
-- LLM_DeltaNet_Blk — full Qwen3-Next gated delta-net layer
--
-- Wires the gated delta rule (LLM_DeltaNet) with the in/out projections,
-- causal conv1d and gating. Forward maps [seq, dim] -> [seq, dim].
-- Projection weights are LLM_Weight (dense for tests, quantized for the
-- real model); conv/a/dt/norm stay dense (small, element-wise).
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_Weight;

package LLM_DeltaNet_Blk is

   type DeltaNet_Layer is record
      QKV_W, Alpha_W, Beta_W, Out_W, Gate_W : LLM_Weight.Weight;
      Conv_W, A_W, Dt_W, Norm_W             : LLM_Tensor.Tensor;
      Dim            : Integer;
      QKV_Out        : Integer;
      N_K_Heads      : Integer;
      N_V_Heads      : Integer;
      Key_Head_Dim   : Integer;
      Value_Head_Dim : Integer;
      V_Dim          : Integer;
   end record;

   function Create
     (QKV_W   : LLM_Weight.Weight;
      Conv_W  : LLM_Tensor.Tensor;
      A_W     : LLM_Tensor.Tensor;
      Dt_W    : LLM_Tensor.Tensor;
      Alpha_W : LLM_Weight.Weight;
      Beta_W  : LLM_Weight.Weight;
      Norm_W  : LLM_Tensor.Tensor;
      Out_W   : LLM_Weight.Weight;
      Gate_W  : LLM_Weight.Weight)
      return DeltaNet_Layer;

   function Forward (L : DeltaNet_Layer; X : LLM_Tensor.Tensor)
      return LLM_Tensor.Tensor;

end LLM_DeltaNet_Blk;
