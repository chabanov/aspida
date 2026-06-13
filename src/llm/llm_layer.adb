---------------------------------------------------------------------
-- LLM_Layer body — uses public API only
---------------------------------------------------------------------

with Ada.Numerics.Float_Random;
with LLM_Tensor;
with LLM_Autograd;

package body LLM_Layer is

   use LLM_Tensor;
   use LLM_Autograd;

   Gen : Ada.Numerics.Float_Random.Generator;

   function Random_Tensor (S : Dims) return Tensor is
      T : Tensor := New_Tensor (S);
      Std : constant Float := 0.02;
   begin
      for I in 1 .. Numel (T) loop
         Set_Flat (T, I, Std * (Ada.Numerics.Float_Random.Random (Gen) * 2.0 - 1.0));
      end loop;
      return T;
   end Random_Tensor;

   --------------------------------------------------------------------
   -- Linear
   --------------------------------------------------------------------

   function New_Linear (In_Features, Out_Features : Integer) return Linear_Layer is
   begin
      Ada.Numerics.Float_Random.Reset (Gen, 42);
      return (
         W => New_Var (Random_Tensor ((In_Features, Out_Features))),
         B => New_Var (Random_Tensor ((1, Out_Features)))
      );
   end New_Linear;

   function Forward (L : Linear_Layer; X : Var) return Var is
      M : Var := Matmul (X, L.W);
   begin
      return M + L.B;
   end Forward;

   --------------------------------------------------------------------
   -- Embedding
   --------------------------------------------------------------------

   function New_Embedding (Vocab_Size, Dim : Integer) return Embedding_Layer is
   begin
      Ada.Numerics.Float_Random.Reset (Gen, 43);
      return (
         W => New_Var (Random_Tensor ((Vocab_Size, Dim)))
      );
   end New_Embedding;

   function Forward (E : Embedding_Layer; Token_Id : Integer) return Var is
      W_Shape : constant Dims := Shape (Data (E.W));
      Emb : Tensor := New_Tensor ((1 => W_Shape (2)));
   begin
      for I in 1 .. Numel (Emb) loop
         Set_Flat (Emb, I, Get (Data (E.W), (Token_Id + 1, I)));
      end loop;
      return New_Var (Emb);
   end Forward;

   --------------------------------------------------------------------
   -- LayerNorm
   --------------------------------------------------------------------

   function New_LayerNorm (Dim : Integer) return LayerNorm_Layer is
      Gamma_T : Tensor := New_Tensor ((1, Dim));
      Beta_T  : Tensor := New_Tensor ((1, Dim));
   begin
      for I in 1 .. Numel (Gamma_T) loop
         Set_Flat (Gamma_T, I, 1.0);
         Set_Flat (Beta_T, I, 0.0);
      end loop;
      return (
         Gamma => New_Var (Gamma_T),
         Beta  => New_Var (Beta_T)
      );
   end New_LayerNorm;

   function Forward (L : LayerNorm_Layer; X : Var) return Var is
      Normalized : Var := Layer_Norm (X);
   begin
      return Normalized * L.Gamma + L.Beta;
   end Forward;

end LLM_Layer;
