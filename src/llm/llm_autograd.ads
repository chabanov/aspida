---------------------------------------------------------------------
-- LLM_Autograd — Automatic differentiation over LLM_Tensor
-- Tracks computation graph, backward pass computes gradients.
---------------------------------------------------------------------

with LLM_Tensor;

package LLM_Autograd is

   -- A differentiable value: data + gradient accumulator
   type Var is private;

   -- Construction
   function New_Var (Data : LLM_Tensor.Tensor) return Var;
   function New_Var_Scalar (Value : Float) return Var;

   -- Access
   function Data (V : Var) return LLM_Tensor.Tensor;
   function Grad (V : Var) return LLM_Tensor.Tensor;

   -- Set gradient (for loss)
   procedure Set_Grad (V : in out Var; G : LLM_Tensor.Tensor);

   -- Zero all tracked gradients
   procedure Zero_Grad (V : in out Var);

   -- Backward pass (accumulates gradients into all tracked vars)
   procedure Backward (V : Var);

   --------------------------------------------------------------------
   -- Ops — all tracked, compute graph edges
   --------------------------------------------------------------------
   function "+" (A, B : Var) return Var;
   function "-" (A, B : Var) return Var;
   function "*" (A, B : Var) return Var;
   function Matmul (A, B : Var) return Var;
   function Relu (A : Var) return Var;
   function Gelu (A : Var) return Var;
   function Softmax (A : Var) return Var;
   function Layer_Norm (A : Var) return Var;
   function Sum (A : Var) return Var;
   function Mean (A : Var) return Var;
   function Reshape (A : Var; New_Shape : LLM_Tensor.Dims) return Var;

   -- Loss
   function Cross_Entropy (Logits, Targets : Var) return Var;

private

   type Op_Kind is (Op_Add, Op_Sub, Op_Mul, Op_Matmul,
                    Op_Relu, Op_Gelu, Op_Softmax, Op_LayerNorm,
                    Op_Sum, Op_Mean, Op_Reshape, Op_CrossEntropy,
                    Op_Leaf);

   type Var_Rec;
   type Var is access Var_Rec;

   type Tensor_Ptr is access LLM_Tensor.Tensor;

   type Var_Rec is record
      Value      : Tensor_Ptr;
      Gradient   : Tensor_Ptr;
      Grad_Zero  : Boolean := True;
      Op         : Op_Kind;
      Left       : Var;
      Right      : Var;
   end record;

end LLM_Autograd;
