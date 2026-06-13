---------------------------------------------------------------------
-- LLM_MLP body
---------------------------------------------------------------------

package body LLM_MLP is

   function New_MLP (Dim : Integer) return MLP_Layer is
   begin
      return (
         FC1 => LLM_Layer.New_Linear (Dim, 4 * Dim),
         FC2 => LLM_Layer.New_Linear (4 * Dim, Dim)
      );
   end New_MLP;

   function Forward (M : MLP_Layer; X : LLM_Autograd.Var) return LLM_Autograd.Var is
      H : LLM_Autograd.Var := LLM_Layer.Forward (M.FC1, X);
      H_Act : LLM_Autograd.Var := LLM_Autograd.Gelu (H);
   begin
      return LLM_Layer.Forward (M.FC2, H_Act);
   end Forward;

end LLM_MLP;
