---------------------------------------------------------------------
-- LLM_Tensor body — core tensor operations
-- Updated for non-discriminated Tensor type
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Numerics.Generic_Elementary_Functions;
with Ada.Numerics;

package body LLM_Tensor is

   package Float_Math is new Ada.Numerics.Generic_Elementary_Functions (Float);
   use Float_Math;

   --------------------------------------------------------------------
   -- Helpers
   --------------------------------------------------------------------

   function Product (D : Dims) return Integer is
      P : Integer := 1;
   begin
      if D'Length = 0 then
         return 1;
      end if;
      for I in D'Range loop
         P := P * D (I);
      end loop;
      return P;
   end Product;

   function Flat_Index (T : Tensor; Index : Dims) return Integer is
      Idx : Integer := 0;
      Stride : Integer := 1;
   begin
      for D in reverse Index'Range loop
         Idx := Idx + (Index (D) - 1) * Stride;
         Stride := Stride * T.Shape (D);
      end loop;
      return Idx + 1;  -- 1-based
   end Flat_Index;

   --------------------------------------------------------------------
   -- Construction
   --------------------------------------------------------------------

   function New_Tensor (Shape : Dims) return Tensor is
      N : constant Integer := Product (Shape);
      T : Tensor;
   begin
      if N < 1 then
         T.Rank := 0;
         T.Shape := (1 .. 4 => 1);
         T.Data := new Tensor_Data_Rec'(Len => 1, Data => (1 .. 1 => 0.0));
         return T;
      end if;
      for I in Shape'Range loop
         if Shape (I) < 1 then
            T.Rank := 0;
            T.Shape := (1 .. 4 => 1);
            T.Data := new Tensor_Data_Rec'(Len => 1, Data => (1 .. 1 => 0.0));
            return T;
         end if;
      end loop;
      T.Data := new Tensor_Data_Rec'(Len => N, Data => (1 .. N => 0.0));
      T.Rank := Shape'Length;
      T.Shape := (1, 1, 1, 1);
      declare
         D : Dims (1 .. Shape'Length);
      begin
         D := Shape;
         T.Shape (1 .. Shape'Length) := D;
      end;
      return T;
   end New_Tensor;

   function New_Tensor_From (Shape : Dims; Data : String) return Tensor is
      T : Tensor := New_Tensor (Shape);
      pragma Unreferenced (Data);
   begin
      -- TODO: binary float parsing from bytes
      return T;
   end New_Tensor_From;

   function New_Scalar (Value : Float) return Tensor is
      T : Tensor;
   begin
      T.Data := new Tensor_Data_Rec'(Len => 1, Data => (1 .. 1 => Value));
      T.Rank := 0;
      T.Shape := (1, 1, 1, 1);
      return T;
   end New_Scalar;

   --------------------------------------------------------------------
   -- Accessors
   --------------------------------------------------------------------

   function Shape (T : Tensor) return Dims is
   begin
      if T.Rank = 0 then
         return (1 .. 0 => <>);
      end if;
      return T.Shape (1 .. T.Rank);
   end Shape;

   function Rank (T : Tensor) return Natural is
   begin
      return T.Rank;
   end Rank;

   function Numel (T : Tensor) return Integer is
   begin
      return T.Data.Len;
   end Numel;

   function Get (T : Tensor; Index : Dims) return Float is
   begin
      return T.Data.Data (Flat_Index (T, Index));
   end Get;

   procedure Set (T : in out Tensor; Index : Dims; Value : Float) is
   begin
      T.Data.Data (Flat_Index (T, Index)) := Value;
   end Set;

   function Get_Flat (T : Tensor; I : Integer) return Float is
   begin
      return T.Data.Data (I);
   end Get_Flat;

   procedure Set_Flat (T : in out Tensor; I : Integer; Value : Float) is
   begin
      T.Data.Data (I) := Value;
   end Set_Flat;

   --------------------------------------------------------------------
   -- Display
   --------------------------------------------------------------------

   procedure Print (T : Tensor; Name : String := "") is
      use Ada.Text_IO;
      First : Boolean;
   begin
      if Name'Length > 0 then
         Put_Line (Name & " shape=(" &
           (if T.Rank > 0 then Integer'Image (T.Shape (1)) else " scalar") & ")");
      end if;

      if T.Rank = 0 then
         Put_Line (Float'Image (T.Data.Data (1)));
      elsif T.Rank = 1 then
         Put ("[");
         for I in T.Data.Data'Range loop
            if I > T.Data.Data'First then Put (", "); end if;
            Put (Float'Image (T.Data.Data (I)));
         end loop;
         Put_Line ("]");
      elsif T.Rank = 2 then
         Put_Line ("[");
         for R in 1 .. T.Shape (1) loop
            Put ("  [");
            First := True;
            for C in 1 .. T.Shape (2) loop
               if not First then Put (", "); end if;
               First := False;
               Put (Float'Image (Get (T, (R, C))));
            end loop;
            Put_Line ("]");
         end loop;
         Put_Line ("]");
      end if;
   end Print;

   --------------------------------------------------------------------
   -- Element-wise ops
   --------------------------------------------------------------------

   function "+" (A, B : Tensor) return Tensor is
      R : Tensor := New_Tensor (A.Shape);
   begin
      for I in A.Data.Data'Range loop
         R.Data.Data (I) := A.Data.Data (I) + B.Data.Data (I);
      end loop;
      return R;
   end "+";

   function "-" (A, B : Tensor) return Tensor is
      R : Tensor := New_Tensor (A.Shape);
   begin
      for I in A.Data.Data'Range loop
         R.Data.Data (I) := A.Data.Data (I) - B.Data.Data (I);
      end loop;
      return R;
   end "-";

   function "*" (A, B : Tensor) return Tensor is
      R : Tensor := New_Tensor (A.Shape);
   begin
      for I in A.Data.Data'Range loop
         R.Data.Data (I) := A.Data.Data (I) * B.Data.Data (I);
      end loop;
      return R;
   end "*";

   function "/" (A, B : Tensor) return Tensor is
      R : Tensor := New_Tensor (A.Shape);
   begin
      for I in A.Data.Data'Range loop
         R.Data.Data (I) := A.Data.Data (I) / B.Data.Data (I);
      end loop;
      return R;
   end "/";

   --------------------------------------------------------------------
   -- Matmul (M×K) × (K×N) → (M×N)
   --------------------------------------------------------------------

   function Matmul (A, B : Tensor) return Tensor is
      M : constant Integer := A.Shape (1);
      K : constant Integer := A.Shape (2);
      N : constant Integer := B.Shape (2);
      R : Tensor := New_Tensor ((M, N));
   begin
      for I in 1 .. M loop
         for J in 1 .. N loop
            declare
               Sum : Float := 0.0;
            begin
               for L in 1 .. K loop
                  Sum := Sum + Get (A, (I, L)) * Get (B, (L, J));
               end loop;
               Set (R, (I, J), Sum);
            end;
         end loop;
      end loop;
      return R;
   end Matmul;

   --------------------------------------------------------------------
   -- Dot product (1D vectors)
   --------------------------------------------------------------------

   function Dot (A, B : Tensor) return Float is
      Sum : Float := 0.0;
   begin
      for I in A.Data.Data'Range loop
         Sum := Sum + A.Data.Data (I) * B.Data.Data (I);
      end loop;
      return Sum;
   end Dot;

   --------------------------------------------------------------------
   -- 2D Transpose
   --------------------------------------------------------------------

   function Transpose (T : Tensor) return Tensor is
      R : Tensor := New_Tensor ((T.Shape (2), T.Shape (1)));
   begin
      for I in 1 .. T.Shape (1) loop
         for J in 1 .. T.Shape (2) loop
            Set (R, (J, I), Get (T, (I, J)));
         end loop;
      end loop;
      return R;
   end Transpose;

   --------------------------------------------------------------------
   -- Activations
   --------------------------------------------------------------------

   function Relu (T : Tensor) return Tensor is
      R : Tensor := New_Tensor (T.Shape);
   begin
      for I in T.Data.Data'Range loop
         if T.Data.Data (I) > 0.0 then
            R.Data.Data (I) := T.Data.Data (I);
         else
            R.Data.Data (I) := 0.0;
         end if;
      end loop;
      return R;
   end Relu;

   function Gelu (T : Tensor) return Tensor is
      R : Tensor := New_Tensor (T.Shape);
      C : constant Float := Sqrt (2.0 / Ada.Numerics.Pi);
   begin
      for I in T.Data.Data'Range loop
         declare
            X : constant Float := T.Data.Data (I);
            Tanh_Arg : constant Float := C * (X + 0.044715 * X**3);
         begin
            R.Data.Data (I) := 0.5 * X * (1.0 + Float_Math.Tanh (Tanh_Arg));
         end;
      end loop;
      return R;
   end Gelu;

   function Softmax (T : Tensor) return Tensor is
      R : Tensor := New_Tensor (T.Shape);
      Max_Val : Float := T.Data.Data (T.Data.Data'First);
      Sum : Float := 0.0;
   begin
      for I in T.Data.Data'Range loop
         if T.Data.Data (I) > Max_Val then Max_Val := T.Data.Data (I); end if;
      end loop;
      for I in T.Data.Data'Range loop
         R.Data.Data (I) := Exp (T.Data.Data (I) - Max_Val);
         Sum := Sum + R.Data.Data (I);
      end loop;
      for I in T.Data.Data'Range loop
         R.Data.Data (I) := R.Data.Data (I) / Sum;
      end loop;
      return R;
   end Softmax;

   function Layer_Norm (T : Tensor) return Tensor is
      R : Tensor := New_Tensor (T.Shape);
      N : constant Float := Float (T.Data.Len);
      Mean_Val : Float := 0.0;
      Var_Val : Float := 0.0;
   begin
      for I in T.Data.Data'Range loop
         Mean_Val := Mean_Val + T.Data.Data (I);
      end loop;
      Mean_Val := Mean_Val / N;
      for I in T.Data.Data'Range loop
         Var_Val := Var_Val + (T.Data.Data (I) - Mean_Val) ** 2;
      end loop;
      Var_Val := Var_Val / N;
      for I in T.Data.Data'Range loop
         R.Data.Data (I) := (T.Data.Data (I) - Mean_Val) / Sqrt (Var_Val + 1.0e-5);
      end loop;
      return R;
   end Layer_Norm;

   --------------------------------------------------------------------
   -- Reduction
   --------------------------------------------------------------------

   function Sum (T : Tensor) return Float is
      S : Float := 0.0;
   begin
      for I in T.Data.Data'Range loop
         S := S + T.Data.Data (I);
      end loop;
      return S;
   end Sum;

   function Mean (T : Tensor) return Float is
   begin
      return Sum (T) / Float (T.Data.Len);
   end Mean;

   function Max (T : Tensor) return Float is
      M : Float := T.Data.Data (T.Data.Data'First);
   begin
      for I in T.Data.Data'Range loop
         if T.Data.Data (I) > M then M := T.Data.Data (I); end if;
      end loop;
      return M;
   end Max;

   --------------------------------------------------------------------
   -- Reshape
   --------------------------------------------------------------------

   function Reshape (T : Tensor; New_Shape : Dims) return Tensor is
      R : Tensor;
   begin
      R.Data := T.Data;
      R.Rank := New_Shape'Length;
      R.Shape := (1, 1, 1, 1);  -- zero out
      declare
         D : Dims (1 .. New_Shape'Length);
      begin
         D := New_Shape;
         R.Shape (1 .. New_Shape'Length) := D;
      end;
      return R;
   end Reshape;

   function "=" (Left, Right : Tensor) return Boolean is
   begin
      if Numel (Left) /= Numel (Right) then return False; end if;
      for I in 1 .. Numel (Left) loop
         if Get_Flat (Left, I) /= Get_Flat (Right, I) then return False; end if;
      end loop;
      return True;
   end "=";

end LLM_Tensor;
