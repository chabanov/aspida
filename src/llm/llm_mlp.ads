---------------------------------------------------------------------
-- LLM_MLP — Feed-Forward Network: Linear → GELU → Linear
---------------------------------------------------------------------

with LLM_Autograd;
with LLM_Layer;

package LLM_MLP is

   type MLP_Layer is record
      FC1 : LLM_Layer.Linear_Layer;  -- dim → 4*dim
      FC2 : LLM_Layer.Linear_Layer;  -- 4*dim → dim
   end record;

   function New_MLP (Dim : Integer) return MLP_Layer;
   function Forward (M : MLP_Layer; X : LLM_Autograd.Var) return LLM_Autograd.Var;

end LLM_MLP;
