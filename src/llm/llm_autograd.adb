---------------------------------------------------------------------
-- LLM_Autograd body — graph traversal for backward pass
-- All Tensor access via public API functions (no prefixed calls)
---------------------------------------------------------------------

package body LLM_Autograd is

   use LLM_Tensor;

   function Copy (T : Tensor) return Tensor is
      R : Tensor := New_Tensor (Shape (T));
   begin
      for I in 1 .. Numel (T) loop
         Set_Flat (R, I, Get_Flat (T, I));
      end loop;
      return R;
   end Copy;

   --------------------------------------------------------------------
   -- Construction
   --------------------------------------------------------------------

   function New_Var (Data : Tensor) return Var is
      V : Var := new Var_Rec;
   begin
      V.Value := new Tensor'(Copy (Data));
      V.Gradient := new Tensor'(New_Tensor (Shape (Data)));
      V.Op := Op_Leaf;
      return V;
   end New_Var;

   function New_Var_Scalar (Value : Float) return Var is
      T : constant Tensor := New_Scalar (Value);
   begin
      return New_Var (T);
   end New_Var_Scalar;

   function Data (V : Var) return Tensor is
   begin
      return V.Value.all;
   end Data;

   function Grad (V : Var) return Tensor is
   begin
      return V.Gradient.all;
   end Grad;

   procedure Set_Grad (V : in out Var; G : Tensor) is
   begin
      if V.Gradient = null then
         V.Gradient := new Tensor'(Copy (G));
      else
         V.Gradient.all := G;
      end if;
   end Set_Grad;

   procedure Zero_Grad (V : in out Var) is
   begin
      if V.Value /= null then
         V.Gradient.all := New_Tensor (Shape (V.Value.all));
      end if;
   end Zero_Grad;

   --------------------------------------------------------------------
   -- Backward
   --------------------------------------------------------------------

   procedure Accumulate (Into : in out Tensor; From : Tensor) is
   begin
      if Numel (Into) = 0 then
         Into := From;
      else
         Into := Into + From;
      end if;
   end Accumulate;

   procedure Backward_Rec (V : Var; Grad_Out : Tensor) is
   begin
      if V.Gradient = null then
         V.Gradient := new Tensor'(Copy (Grad_Out));
      else
         V.Gradient.all := V.Gradient.all + Grad_Out;
      end if;

      case V.Op is
         when Op_Leaf =>
            null;

         when Op_Add =>
            Backward_Rec (V.Left, Grad_Out);
            Backward_Rec (V.Right, Grad_Out);

         when Op_Sub =>
            Backward_Rec (V.Left, Grad_Out);
            Backward_Rec (V.Right, Grad_Out);

         when Op_Mul =>
            Backward_Rec (V.Left, Grad_Out * V.Right.Value.all);
            Backward_Rec (V.Right, Grad_Out * V.Left.Value.all);

         when Op_Matmul =>
            Backward_Rec (V.Left, Matmul (Grad_Out, Transpose (V.Right.Value.all)));
            Backward_Rec (V.Right, Matmul (Transpose (V.Left.Value.all), Grad_Out));

         when Op_Relu | Op_Gelu | Op_Softmax | Op_LayerNorm
            | Op_Sum | Op_Mean | Op_CrossEntropy | Op_Reshape =>
            Backward_Rec (V.Left, Grad_Out);
      end case;
   end Backward_Rec;

   procedure Backward (V : Var) is
      Ones : Tensor := New_Tensor (Shape (V.Value.all));
   begin
      for I in 1 .. Numel (Ones) loop
         Set_Flat (Ones, I, 1.0);
      end loop;
      Backward_Rec (V, Ones);
   end Backward;

   --------------------------------------------------------------------
   -- Ops
   --------------------------------------------------------------------

   function Make_Op (Left, Right : Var; Op : Op_Kind) return Var is
      V : Var := new Var_Rec;
   begin
      V.Op := Op;
      V.Left := Left;
      V.Right := Right;
      return V;
   end Make_Op;

   function "+" (A, B : Var) return Var is
      V : Var := Make_Op (A, B, Op_Add);
   begin
      V.Value := new Tensor'(A.Value.all + B.Value.all);
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end "+";

   function "-" (A, B : Var) return Var is
      V : Var := Make_Op (A, B, Op_Sub);
   begin
      V.Value := new Tensor'(A.Value.all - B.Value.all);
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end "-";

   function "*" (A, B : Var) return Var is
      V : Var := Make_Op (A, B, Op_Mul);
   begin
      V.Value := new Tensor'(A.Value.all * B.Value.all);
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end "*";

   function Matmul (A, B : Var) return Var is
      V : Var := Make_Op (A, B, Op_Matmul);
   begin
      V.Value := new Tensor'(LLM_Tensor.Matmul (A.Value.all, B.Value.all));
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end Matmul;

   function Relu (A : Var) return Var is
      V : Var := Make_Op (A, null, Op_Relu);
   begin
      V.Value := new Tensor'(LLM_Tensor.Relu (A.Value.all));
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end Relu;

   function Gelu (A : Var) return Var is
      V : Var := Make_Op (A, null, Op_Gelu);
   begin
      V.Value := new Tensor'(LLM_Tensor.Gelu (A.Value.all));
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end Gelu;

   function Softmax (A : Var) return Var is
      V : Var := Make_Op (A, null, Op_Softmax);
   begin
      V.Value := new Tensor'(LLM_Tensor.Softmax (A.Value.all));
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end Softmax;

   function Layer_Norm (A : Var) return Var is
      V : Var := Make_Op (A, null, Op_LayerNorm);
   begin
      V.Value := new Tensor'(LLM_Tensor.Layer_Norm (A.Value.all));
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end Layer_Norm;

   function Sum (A : Var) return Var is
      S : constant Float := LLM_Tensor.Sum (A.Value.all);
      V : Var := Make_Op (A, null, Op_Sum);
   begin
      V.Value := new Tensor'(New_Scalar (S));
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end Sum;

   function Mean (A : Var) return Var is
      M : constant Float := LLM_Tensor.Mean (A.Value.all);
      V : Var := Make_Op (A, null, Op_Mean);
   begin
      V.Value := new Tensor'(New_Scalar (M));
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end Mean;

   function Reshape (A : Var; New_Shape : Dims) return Var is
      V : Var := Make_Op (A, null, Op_Reshape);
   begin
      V.Value := new Tensor'(LLM_Tensor.Reshape (A.Value.all, New_Shape));
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end Reshape;

   function Cross_Entropy (Logits, Targets : Var) return Var is
      V : Var := Make_Op (Logits, Targets, Op_CrossEntropy);
   begin
      V.Value := new Tensor'(New_Scalar (0.0));
      V.Gradient := new Tensor'(New_Tensor (Shape (V.Value.all)));
      return V;
   end Cross_Entropy;

end LLM_Autograd;
