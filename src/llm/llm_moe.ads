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
-- Qwen Tensor layout (3D: last dim = num_experts):
--   ffn_gate_exps.weight [dim, intermed, 256] — router gating
--   ffn_up_exps.weight   [dim, intermed, 256] — expert up-projection
--   ffn_down_exps.weight  [intermed, dim, 256] — expert down-projection
--   ffn_gate_inp.weight   [dim, 256]           — router input projection
--
--   ffn_gate_shexp.weight [dim, intermed]      — shared expert gate
--   ffn_up_shexp.weight   [dim, intermed]      — shared expert up
--   ffn_down_shexp.weight  [intermed, dim]      — shared expert down
--   ffn_gate_inp_shexp.weight [dim]             — shared expert router
---------------------------------------------------------------------

with LLM_Tensor;

package LLM_MoE is

   type MoE_Layer is record
      -- Router: input projection + gating
      Gate_Inp_W : LLM_Tensor.Tensor;  -- [dim, 256]
      Gate_Exp_W : LLM_Tensor.Tensor;  -- [dim, intermed, 256]

      -- Routed experts
      Up_W   : LLM_Tensor.Tensor;  -- [dim, intermed, 256]
      Down_W : LLM_Tensor.Tensor;  -- [intermed, dim, 256]

      -- Shared expert
      Shexp_Gate_W     : LLM_Tensor.Tensor;  -- [dim, intermed]
      Shexp_Up_W       : LLM_Tensor.Tensor;  -- [dim, intermed]
      Shexp_Down_W     : LLM_Tensor.Tensor;  -- [intermed, dim]
      Shexp_Gate_Inp_W : LLM_Tensor.Tensor;  -- [dim]

      N_Experts : Integer := 256;
      Top_K     : Integer := 8;   -- Qwen uses top-8
      Dim       : Integer;
      Intermed  : Integer;
   end record;

   -- Create MoE layer from tensors
   function Create_MoE
     (Gate_Inp_W, Gate_Exp_W, Up_W, Down_W,
      Shexp_Gate_W, Shexp_Up_W, Shexp_Down_W, Shexp_Gate_Inp_W : LLM_Tensor.Tensor;
      N_Experts : Integer)
      return MoE_Layer;

   -- Forward pass: x [1, dim] → output [1, dim]
   function Forward (M : MoE_Layer; X : LLM_Tensor.Tensor) return LLM_Tensor.Tensor;

end LLM_MoE;
