---------------------------------------------------------------------
-- LLM_Attention — Multi-Head Scaled Dot-Product Attention
---------------------------------------------------------------------

with LLM_Autograd;
with LLM_Layer;

package LLM_Attention is

   type Attention_Layer is record
      Q_Proj : LLM_Layer.Linear_Layer;  -- dim → dim
      K_Proj : LLM_Layer.Linear_Layer;
      V_Proj : LLM_Layer.Linear_Layer;
      Out_Proj : LLM_Layer.Linear_Layer;
      N_Heads : Integer;
      Head_Dim : Integer;
      Scale : Float;
   end record;

   function New_Attention (Dim, N_Heads : Integer) return Attention_Layer;

   function Forward (A : Attention_Layer; X : LLM_Autograd.Var) return LLM_Autograd.Var;

end LLM_Attention;
