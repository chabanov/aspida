---------------------------------------------------------------------
-- LLM_SSM — Mamba State Space Model
--
-- Implements the selective state space (S6) layer used in Mamba
-- and integrated into Qwen 3.5 MoE architecture.
--
-- Mamba equations (simplified per-token):
--   h_t = A * h_{t-1} + B * x_t       (state update)
--   y_t = C * h_t + D * x_t           (output)
--
-- In practice, A is diagonalized and discretized:
--   A_bar = exp(dt * A)
--   B_bar = (A_bar - I) / A * B * dt   (zero-order hold)
--
-- Qwen SSM parameters (per block):
--   ssm_conv1d.weight [4, dim*2]   — 1D convolution kernel
--   ssm_a               [state_dim] — diagonal A (learned)
--   ssm_dt.bias         [state_dim] — delta_t bias
--   ssm_norm.weight     [dim/16]    — normalization
--   ssm_out.weight      [dim*2, dim] — output projection
--   ssm_alpha.weight    [dim, state_dim] — B projection
--   ssm_beta.weight     [dim, state_dim] — C projection
--
-- dim = 2048 (qwen3.embedding_length)
-- state_dim = 32 (ssm_a size)
-- intermediate = 2 * dim = 4096
--
-- Token processing: sequential (SSM is recurrent)
---------------------------------------------------------------------

with LLM_Tensor;

package LLM_SSM is

   -- Parameters for one SSM layer
   type SSM_Params is record
      -- Conv1d: [kernel_size=4, intermediate=2*dim]
      Conv_Weight : LLM_Tensor.Tensor;
      -- A: diagonal state matrix [state_dim]
      A_Diag      : LLM_Tensor.Tensor;
      -- dt bias: [state_dim]
      Dt_Bias     : LLM_Tensor.Tensor;
      -- Gamma: layer norm [dim/16=128]
      Gamma       : LLM_Tensor.Tensor;
      -- Output projection [intermediate, dim]
      Out_Weight  : LLM_Tensor.Tensor;
      -- Alpha (B projection): [dim, state_dim]
      Alpha_W     : LLM_Tensor.Tensor;
      -- Beta (C projection): [dim, state_dim]
      Beta_W      : LLM_Tensor.Tensor;
   end record;

   -- Create SSM parameters from tensors (used by model loader)
   function Create_SSM
     (Conv_W, A_D, Dt_B, Gamma, Out_W, Alpha_W, Beta_W : LLM_Tensor.Tensor)
      return SSM_Params;

   -- Forward pass for SSM over a sequence of tokens
   -- X: input sequence [1, dim] × seq_len (one token at a time)
   -- Returns: output [1, dim] for each token (caller accumulates)
   -- State is maintained internally (h_t across calls)
   --
   -- Implementation outline (per token):
   --   1. Conv1D: apply causal conv over last 4 tokens
   --   2. SiLU activation
   --   3. Project B(x) and C(x)
   --   4. Discretize: dt = softplus(dt_bias + linear(x))
   --   5. Selective scan: h = A_bar * h + B_bar * x
   --   6. Output: y = C * h, then normalize + project
   function Forward
     (P    : SSM_Params;
      X    : LLM_Tensor.Tensor;   -- [dim] single token
      State : in out LLM_Tensor.Tensor  -- [state_dim] hidden state (updated in-place)
     ) return LLM_Tensor.Tensor;  -- [dim] output

   -- Reset state to zeros
   function Init_State (State_Dim : Integer) return LLM_Tensor.Tensor;

end LLM_SSM;
