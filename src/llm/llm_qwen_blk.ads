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
with LLM_FullAttn;
with LLM_DeltaNet_Blk;

package LLM_Qwen_Blk is

   type Qwen_Block is record
      Is_Full_Attn     : Boolean := False;
      Full             : LLM_FullAttn.Full_Attn_Layer;       -- if Is_Full_Attn
      DNet             : LLM_DeltaNet_Blk.DeltaNet_Layer;    -- otherwise
      MoE              : LLM_MoE.MoE_Layer;
      Attn_Norm_W      : LLM_Tensor.Tensor;
      Post_Attn_Norm_W : LLM_Tensor.Tensor;
      Dim              : Integer := 0;
   end record;

   -- Forward over a whole sequence: X [seq, dim] -> [seq, dim].
   function Forward (B : Qwen_Block; X : LLM_Tensor.Tensor)
      return LLM_Tensor.Tensor;

end LLM_Qwen_Blk;
