---------------------------------------------------------------------
-- LLM_Layer — NN building blocks: Linear, Embedding, LayerNorm params
---------------------------------------------------------------------

with LLM_Autograd;

package LLM_Layer is

   -- Weight + bias pair (learnable)
   type Linear_Layer is record
      W : LLM_Autograd.Var;  -- (in_features, out_features)
      B : LLM_Autograd.Var;  -- (out_features,)
   end record;

   function New_Linear (In_Features, Out_Features : Integer) return Linear_Layer;
   function Forward (L : Linear_Layer; X : LLM_Autograd.Var) return LLM_Autograd.Var;

   type Embedding_Layer is record
      W : LLM_Autograd.Var;  -- (vocab_size, dim)
   end record;

   function New_Embedding (Vocab_Size, Dim : Integer) return Embedding_Layer;
   function Forward (E : Embedding_Layer; Token_Id : Integer) return LLM_Autograd.Var;

   type LayerNorm_Layer is record
      Gamma : LLM_Autograd.Var;  -- scale
      Beta  : LLM_Autograd.Var;  -- shift
   end record;

   function New_LayerNorm (Dim : Integer) return LayerNorm_Layer;
   function Forward (L : LayerNorm_Layer; X : LLM_Autograd.Var) return LLM_Autograd.Var;

end LLM_Layer;
