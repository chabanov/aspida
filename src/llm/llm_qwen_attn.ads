---------------------------------------------------------------------
-- LLM_Qwen_Attn — Qwen 3.5 GQA Attention (QKV split + GQA repeat)
--
-- Architecture:
--   1. Split QKV: unified QKV weight → Q [dim, n_heads*head_dim]
--                                      K [dim, n_kv_heads*head_dim]
--                                      V [dim, n_kv_heads*head_dim]
--   2. Apply RoPE to Q and K
--   3. GQA: repeat K,V n_repeat = n_heads / n_kv_heads times
--   4. Scores = Q @ K^T / sqrt(head_dim)
--   5. Causal mask (upper triangular → -inf)
--   6. Softmax over last dim
--   7. Output = Softmax(Scores) @ V
--   8. Gate: sigmoid(attn_gate @ x) * output
--   9. Output projection: out_proj @ output
---------------------------------------------------------------------

with LLM_Tensor;   use LLM_Tensor;
with LLM_RoPE;     use LLM_RoPE;

package LLM_Qwen_Attn is

   type Qwen_Attn_Params is record
      -- Unified QKV weight [dim, (n_heads + 2*n_kv_heads)*head_dim]
      QKV_W   : Tensor;
      -- Output projection [dim, dim]
      O_W     : Tensor;
      -- Attention gate weight [dim, dim] (used only on SSM layers)
      Gate_W  : Tensor;
      -- RoPE params (shared across all heads)
      RoPE    : RoPE_Params;
      -- Model dimensions
      Dim       : Integer;
      N_Heads   : Integer;
      N_KV_Heads: Integer;
      Head_Dim  : Integer;
      -- Whether this layer uses gating (SSM layers: yes; full-attn: no)
      Use_Gate  : Boolean;
   end record;

   -- Forward pass over a whole sequence.
   -- X:   input [Seq_Len, dim]
   -- Pos: absolute position of the first row (0-based; used for RoPE)
   -- Returns: attention output [Seq_Len, dim], computed with a causal
   --          mask and softmax-normalised attention weights.
   function Forward (P : Qwen_Attn_Params; X : Tensor; Pos : Integer) return Tensor;

   -- Constructor
   function Create_Qwen_Attn_Params
     (QKV_W, O_W, Gate_W : Tensor;
      RoPE : RoPE_Params;
      Dim, N_Heads, N_KV_Heads, Head_Dim : Integer;
      Use_Gate : Boolean)
      return Qwen_Attn_Params;

end LLM_Qwen_Attn;
