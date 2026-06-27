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

   --  Release the quantized host bytes (and any GPU mirror) of this layer's
   --  projection weights — for Phase 1b model eviction. Idempotent. The dense
   --  Q_Norm/K_Norm Tensors are controlled and finalize with the enclosing
   --  block array; only the byte-data weights need explicit release.
   procedure Free (L : in out Full_Attn_Layer);

   --------------------------------------------------------------------
   -- Incremental decode (KV cache, one token at a time).
   --
   -- Attn_State holds the K/V cache for all positions seen so far. Step
   -- projects the single new token, appends its K/V, attends the new
   -- query against the whole cache and returns [1, Dim].
   --------------------------------------------------------------------
   type Attn_State is record
      K_Cache : LLM_Tensor.Tensor;  -- [Max_Len, N_KV_Heads*Head_Dim]
      V_Cache : LLM_Tensor.Tensor;  -- [Max_Len, N_KV_Heads*Head_Dim]
      Len     : Natural := 0;       -- positions filled so far
   end record;

   function Init_State (L : Full_Attn_Layer; Max_Len : Integer) return Attn_State;

   function Step (L : Full_Attn_Layer; St : in out Attn_State; X : LLM_Tensor.Tensor)
      return LLM_Tensor.Tensor;

   --------------------------------------------------------------------
   -- RMSNorm a head vector then apply RoPE to its first rope_dim dims.
   -- Exposed for integration tests so they can exercise Norm_Rope
   -- directly (Forward calls it internally and discards the result
   -- inside Q/K projections, so a test that only calls Forward can
   -- only observe a softened / attenuated effect through the O-proj).
   --
   -- The Section_Positions parameter feeds the per-section RoPE path
   -- in LLM_RoPE.Apply_Sections; the default is uniform ([Pos, Pos,
   -- Pos, Pos]) which reproduces the legacy text-only rotation exactly.
   --------------------------------------------------------------------
   function Norm_Rope
     (V : LLM_Tensor.Tensor;
      Norm_W : LLM_Tensor.Tensor;
      RoPE : LLM_RoPE.RoPE_Params;
      Pos : Integer;
      Sec : LLM_RoPE.Section_Positions := [others => 0])
      return LLM_Tensor.Tensor;

end LLM_FullAttn;
