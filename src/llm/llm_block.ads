---------------------------------------------------------------------
-- LLM_Block — One Transformer Block: Attn + MLP, pre-norm residuals
---------------------------------------------------------------------

with LLM_Autograd;
with LLM_Layer;
with LLM_Attention;
with LLM_MLP;

package LLM_Block is

   type Transformer_Block is record
      Attn_Norm : LLM_Layer.LayerNorm_Layer;
      Attn      : LLM_Attention.Attention_Layer;
      MLP_Norm  : LLM_Layer.LayerNorm_Layer;
      MLP       : LLM_MLP.MLP_Layer;
   end record;

   function New_Block (Dim, N_Heads : Integer) return Transformer_Block;

   function Forward (B : Transformer_Block; X : LLM_Autograd.Var) return LLM_Autograd.Var;

end LLM_Block;
