---------------------------------------------------------------------
-- LLM_Qwen_Blk — Qwen 3.5 Hybrid Transformer Block
--
-- Architecture (Qwen3.5-35B-A3B, qwen35moe):
--
--   Full-Attention layers (every 4th block, L mod 4 = 0):
--     x_norm = RMSNorm(x)
--     q, k, v = split(QKV_Project(x_norm))
--     attn_out = GQA_Attention(q, k, v)
--     x = x + attn_out
--     x_norm2 = RMSNorm(x)
--     x = x + MoE(x_norm2)
--
--   SSM layers (all other blocks):
--     x_norm = RMSNorm(x)
--     q, k, v = split(QKV_Project(x_norm))  [sliding attn]
--     attn_out = GQA_Attention(q, k, v)
--     gate = sigmoid(attn_gate @ x_norm)
--     ssm_out = SSM(x_norm)
--     combined = gate * attn_out + (1 - gate) * ssm_out
--     x = x + combined
--     x_norm2 = RMSNorm(x)
--     x = x + MoE(x_norm2)
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_MoE;
with LLM_SSM;

package LLM_Qwen_Blk is

   type Qwen_Block is tagged record
      -- Is this a full-attention layer? (L mod 4 = 0)
      Is_Full_Attn : Boolean;

      -- Unified QKV projection [dim, heads * head_dim * 3] (includes KV for GQA)
      QKV_W : LLM_Tensor.Tensor;

      -- Attention output gate (sigmoid) [dim, dim] — SSM layers only
      Attn_Gate_W : LLM_Tensor.Tensor;

      -- Output projection [dim, dim]
      O_W : LLM_Tensor.Tensor;

      -- RMSNorm weights (input + post-attn)
      Attn_Norm_W     : LLM_Tensor.Tensor;  -- [dim]
      Post_Attn_Norm_W : LLM_Tensor.Tensor;  -- [dim]

      -- SSM layer (Mamba hybrid block — non full-attn layers)
      SSM : LLM_SSM.SSM_Params;

      -- MoE layer (present on all blocks)
      MoE : LLM_MoE.MoE_Layer;

      -- Hyperparameters
      N_Heads    : Integer;
      N_KV_Heads : Integer;  -- GQA: 2 KV heads vs 16 Q heads
      Dim        : Integer;
   end record;

   -- Create a Qwen hybrid block
   function Create_Qwen_Block
     (QKV_W         : LLM_Tensor.Tensor;
      Attn_Gate_W   : LLM_Tensor.Tensor;
      O_W           : LLM_Tensor.Tensor;
      Attn_Norm_W   : LLM_Tensor.Tensor;
      Post_Attn_Norm_W : LLM_Tensor.Tensor;
      Ssm_Params    : LLM_SSM.SSM_Params;
      Moe_Layer     : LLM_MoE.MoE_Layer;
      Is_Full_Attn  : Boolean;
      Dim           : Integer;
      N_Heads       : Integer;
      N_KV_Heads    : Integer)
      return Qwen_Block;

   -- Forward pass through one block
   -- X: input [1, dim]
   -- Returns: output [1, dim]
   function Forward (B : Qwen_Block; X : LLM_Tensor.Tensor) return LLM_Tensor.Tensor;

   -- Equality (required for Ada.Containers.Vectors)
   function "=" (Left, Right : Qwen_Block) return Boolean;

end LLM_Qwen_Blk;
