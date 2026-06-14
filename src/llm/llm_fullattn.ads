---------------------------------------------------------------------
-- LLM_FullAttn — Qwen3-Next gated full-attention layer (L mod 4 == 3)
--
-- Standard causal GQA with QK-norm and PARTIAL RoPE, plus a per-head
-- output gate packed inside the q projection (q_proj = query | gate).
-- The projection weights are LLM_Weight (dense for tests, quantized for
-- the real model); the small per-head norms stay dense.
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_RoPE;
with LLM_Weight;

package LLM_FullAttn is

   type Full_Attn_Layer is record
      Q_W, K_W, V_W, O_W : LLM_Weight.Weight;
      Q_Norm, K_Norm     : LLM_Tensor.Tensor;
      RoPE        : LLM_RoPE.RoPE_Params;
      Dim         : Integer;
      N_Q_Heads   : Integer;
      N_KV_Heads  : Integer;
      Head_Dim    : Integer;
   end record;

   function Create
     (Q_W, K_W, V_W  : LLM_Weight.Weight;
      Q_Norm, K_Norm : LLM_Tensor.Tensor;
      O_W            : LLM_Weight.Weight;
      RoPE           : LLM_RoPE.RoPE_Params)
      return Full_Attn_Layer;

   function Forward (L : Full_Attn_Layer; X : LLM_Tensor.Tensor)
      return LLM_Tensor.Tensor;

end LLM_FullAttn;
