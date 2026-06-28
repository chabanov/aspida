---------------------------------------------------------------------
-- LLM_Qwen_Blk — one Qwen3-Next hybrid block
--
-- Each block is RMSNorm -> (full attention | gated delta-net) -> residual
-- -> RMSNorm -> MoE -> residual, over a whole sequence [seq, dim].
-- The layer type is fixed at load time: full attention when L mod 4 == 3,
-- gated delta-net otherwise. The unused layer field is left default.
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_MoE;
with LLM_Dense_FFN;
with LLM_FullAttn;
with LLM_DeltaNet_Blk;

package LLM_Qwen_Blk is

   type Qwen_Block is record
      Is_Full_Attn     : Boolean := False;
      Full             : LLM_FullAttn.Full_Attn_Layer;       -- if Is_Full_Attn
      DNet             : LLM_DeltaNet_Blk.DeltaNet_Layer;    -- otherwise
      --  FFN: routed experts (qwen35moe) when Is_MoE, else a single dense
      --  SwiGLU MLP (qwen35 dense). Exactly one of MoE / Dense is populated.
      Is_MoE           : Boolean := True;
      MoE              : LLM_MoE.MoE_Layer;                   -- if Is_MoE
      Dense            : LLM_Dense_FFN.Dense_FFN_Layer;       -- otherwise
      Attn_Norm_W      : LLM_Tensor.Tensor;
      Post_Attn_Norm_W : LLM_Tensor.Tensor;
      Dim              : Integer := 0;
   end record;

   -- Forward over a whole sequence: X [seq, dim] -> [seq, dim].
   function Forward (B : Qwen_Block; X : LLM_Tensor.Tensor)
      return LLM_Tensor.Tensor;

   --------------------------------------------------------------------
   -- Incremental decode (one token at a time, with cached attn state).
   --------------------------------------------------------------------
   type Block_State is record
      Is_Full : Boolean := False;
      Full_St : LLM_FullAttn.Attn_State;
      DNet_St : LLM_DeltaNet_Blk.DNet_State;
   end record;

   function Init_State (B : Qwen_Block; Max_Len : Integer) return Block_State;

   --  X [1, dim] -> [1, dim].
   function Step (B : Qwen_Block; St : in out Block_State; X : LLM_Tensor.Tensor)
      return LLM_Tensor.Tensor;

end LLM_Qwen_Blk;
