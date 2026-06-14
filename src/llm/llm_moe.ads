---------------------------------------------------------------------
-- LLM_MoE — Mixture of Experts layer
--
-- Qwen 3.5 MoE architecture:
--   Per block, there are 256 routed experts + 1 shared expert.
--   Router (gate) selects top-k experts per token via softmax gating.
--   Each expert is a standard FFN: Linear → SiLU → Linear.
--   Shared expert always contributes (no routing).
--
--   Final output = sum(router_weight[i] * expert[i](x)) + shared_expert(x)
--
-- Implemented weight layout (row-major logical shapes = GGUF dims reversed):
--   ffn_gate_inp.weight   [n_experts, dim]            — router logits
--   ffn_gate_exps.weight  [n_experts, intermed, dim]  — per-expert gate (3D)
--   ffn_up_exps.weight    [n_experts, intermed, dim]  — per-expert up   (3D)
--   ffn_down_exps.weight  [n_experts, dim, intermed]  — per-expert down (3D)
--
--   ffn_gate_shexp.weight [intermed, dim]      — shared expert gate
--   ffn_up_shexp.weight   [intermed, dim]      — shared expert up
--   ffn_down_shexp.weight [dim, intermed]      — shared expert down
--   ffn_gate_inp_shexp.weight [1, dim]         — shared expert sigmoid gate
--                                                (optional; skipped if absent)
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_Weight;

package LLM_MoE is

   type MoE_Layer is record
      --  Projection weights (dense for tests / quantized for the real model).
      Gate_Inp_W   : LLM_Weight.Weight;   -- router  [n_exp, dim]
      Gate_Exp_W   : LLM_Weight.Weight;   -- 3D [n_exp, intermed, dim]
      Up_W         : LLM_Weight.Weight;   -- 3D [n_exp, intermed, dim]
      Down_W       : LLM_Weight.Weight;   -- 3D [n_exp, dim, intermed]
      Shexp_Gate_W : LLM_Weight.Weight;   -- [intermed, dim]
      Shexp_Up_W   : LLM_Weight.Weight;   -- [intermed, dim]
      Shexp_Down_W : LLM_Weight.Weight;   -- [dim, intermed]
      Shexp_Gate_Inp_W : LLM_Tensor.Tensor;  -- [dim] (dense, element-wise)

      N_Experts : Integer := 256;
      Top_K     : Integer := 8;   -- Qwen uses top-8
      Dim       : Integer;
      Intermed  : Integer;
   end record;

   -- Create MoE layer from weights
   function Create_MoE
     (Gate_Inp_W, Gate_Exp_W, Up_W, Down_W,
      Shexp_Gate_W, Shexp_Up_W, Shexp_Down_W : LLM_Weight.Weight;
      Shexp_Gate_Inp_W : LLM_Tensor.Tensor;
      N_Experts : Integer)
      return MoE_Layer;

   -- Forward pass: x [1, dim] → output [1, dim]
   function Forward (M : MoE_Layer; X : LLM_Tensor.Tensor) return LLM_Tensor.Tensor;

end LLM_MoE;
