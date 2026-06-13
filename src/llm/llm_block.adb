---------------------------------------------------------------------
-- LLM_Block body
---------------------------------------------------------------------

package body LLM_Block is

   use LLM_Autograd;

   function New_Block (Dim, N_Heads : Integer) return Transformer_Block is
   begin
      return (
         Attn_Norm => LLM_Layer.New_LayerNorm (Dim),
         Attn      => LLM_Attention.New_Attention (Dim, N_Heads),
         MLP_Norm  => LLM_Layer.New_LayerNorm (Dim),
         MLP       => LLM_MLP.New_MLP (Dim)
      );
   end New_Block;

   function Forward (B : Transformer_Block; X : LLM_Autograd.Var) return LLM_Autograd.Var is
      -- Pre-norm + residual
      Normed  : LLM_Autograd.Var := LLM_Layer.Forward (B.Attn_Norm, X);
      Attn_Out : LLM_Autograd.Var := LLM_Attention.Forward (B.Attn, Normed);
      X1       : LLM_Autograd.Var := X + Attn_Out;

      Normed2 : LLM_Autograd.Var := LLM_Layer.Forward (B.MLP_Norm, X1);
      MLP_Out : LLM_Autograd.Var := LLM_MLP.Forward (B.MLP, Normed2);
   begin
      return X1 + MLP_Out;
   end Forward;

end LLM_Block;
