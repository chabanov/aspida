---------------------------------------------------------------------
-- LLM_Dense_FFN — dense SwiGLU feed-forward block (qwen35 dense).
--
-- The dense qwen35 architecture (e.g. DeepReinforce Hura 9B) has
-- no routed experts: each block carries a single SwiGLU MLP
--
--   y = down( silu(gate(x)) * up(x) )
--
-- with the standard GGUF tensor layout (logical shapes = GGUF dims):
--   ffn_gate.weight [intermed, dim]
--   ffn_up.weight   [intermed, dim]
--   ffn_down.weight [dim, intermed]
--
-- Projection weights are kept quantized (LLM_Weight) and matvec'd on the
-- fly, exactly like the MoE / attention paths. This is the dense
-- counterpart of LLM_MoE used when expert_count is absent or zero.
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_Weight;

package LLM_Dense_FFN is

   type Dense_FFN_Layer is record
      Gate_W : LLM_Weight.Weight;   -- [intermed, dim]
      Up_W   : LLM_Weight.Weight;   -- [intermed, dim]
      Down_W : LLM_Weight.Weight;   -- [dim, intermed]
      Dim      : Integer := 0;
      Intermed : Integer := 0;
   end record;

   function Create
     (Gate_W, Up_W, Down_W : LLM_Weight.Weight) return Dense_FFN_Layer;

   -- Forward pass: x [1, dim] -> output [1, dim].
   function Forward (L : Dense_FFN_Layer; X : LLM_Tensor.Tensor)
      return LLM_Tensor.Tensor;

   --  Release the quantized host bytes (and any GPU mirror) of this layer's
   --  projection weights — for model eviction. Idempotent.
   procedure Free (L : in out Dense_FFN_Layer);

end LLM_Dense_FFN;
