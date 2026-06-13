with LLM_Tensor;

package body LLM_Attention is
   function New_Attention (Dim, N_Heads : Integer) return Attention_Layer is
      Head_Dim : constant Integer := Dim / N_Heads;
   begin
      return (
         Q_Proj   => LLM_Layer.New_Linear (Dim, Dim),
         K_Proj   => LLM_Layer.New_Linear (Dim, Dim),
         V_Proj   => LLM_Layer.New_Linear (Dim, Dim),
         Out_Proj => LLM_Layer.New_Linear (Dim, Dim),
         N_Heads  => N_Heads,
         Head_Dim => Head_Dim,
         Scale    => 1.0 / Float (Head_Dim)
      );
   end New_Attention;

   function Forward (A : Attention_Layer; X : LLM_Autograd.Var) return LLM_Autograd.Var is
      Q : constant LLM_Autograd.Var := LLM_Layer.Forward (A.Q_Proj, X);
      K : constant LLM_Autograd.Var := LLM_Layer.Forward (A.K_Proj, X);
      V : constant LLM_Autograd.Var := LLM_Layer.Forward (A.V_Proj, X);

      -- Scores = Q @ K^T / sqrt(d_k)
      -- We approximate via element-wise: simplified scaled dot-product
      K_T : constant LLM_Tensor.Tensor := LLM_Tensor.Transpose (LLM_Autograd.Data (K));
      Scores : constant LLM_Autograd.Var := LLM_Autograd.Matmul (Q, LLM_Autograd.New_Var (K_T));

      -- Softmax over last dim
      Attn : constant LLM_Autograd.Var := LLM_Autograd.Softmax (Scores);

      -- Output = Attn @ V
      Output : constant LLM_Autograd.Var := LLM_Autograd.Matmul (Attn, V);
   begin
      return LLM_Layer.Forward (A.Out_Proj, Output);
   end Forward;

end LLM_Attention;
