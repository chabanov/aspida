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

   --  Release the quantized host bytes (and any GPU mirror) of this layer's
   --  projection weights — for Phase 1b model eviction. Idempotent. The dense
   --  conv/a/dt/norm Tensors are controlled and finalize with the block array.
   procedure Free (L : in out DeltaNet_Layer);

   --------------------------------------------------------------------
   -- Incremental decode (one token at a time, O(1) per step).
   --
   -- DNet_State carries the per-head recurrent state S and the causal
   -- conv1d window across decode steps. Step processes a single token
   -- [1, Dim] and returns [1, Dim], mutating the state in place.
   --------------------------------------------------------------------
   type DNet_State is record
      S_All      : LLM_Tensor.Tensor;  -- packed states [N_V_Heads*Key_Head_Dim, Value_Head_Dim]
      Conv_Hist  : LLM_Tensor.Tensor;  -- last (Kernel-1) raw qkv rows [Kernel-1, QKV_Out]
      GPU_Handle : Integer := -1;      -- resident device S_All (Increment 2), or -1 = CPU
   end record;

   function Init_State
     (L : DeltaNet_Layer; Force_Host : Boolean := False)
      return DNet_State;

   function Step (L : DeltaNet_Layer; St : in out DNet_State; X : LLM_Tensor.Tensor)
      return LLM_Tensor.Tensor;

end LLM_DeltaNet_Blk;
