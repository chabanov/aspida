---------------------------------------------------------------------
-- LLM_Tensor — N-dimensional array with basic linear-algebra ops
-- Pure ADA, no external dependencies. FP32 precision.
---------------------------------------------------------------------

package LLM_Tensor is

   -- Dimension of a tensor. Always rank 1..3 for our use case.
   type Dims is array (Positive range <>) of Positive;

   -- Main tensor type — flat array + shape
   type Tensor is private;

   -- Construction
   function New_Tensor (Shape : Dims) return Tensor;
   function New_Tensor_From (Shape : Dims; Data : String) return Tensor;
   -- Data is binary FP32 bytes (little-endian)

   function New_Scalar (Value : Float) return Tensor;

   -- Accessors
   function Shape (T : Tensor) return Dims;
   function Rank (T : Tensor) return Natural;
   function Numel (T : Tensor) return Integer;
   function Get (T : Tensor; Index : Dims) return Float
     with Pre => Index'Length = Rank (T);
   procedure Set (T : in out Tensor; Index : Dims; Value : Float)
     with Pre => Index'Length = Rank (T);

   -- Element access by flat index (for loops)
   function Get_Flat (T : Tensor; I : Integer) return Float;
   procedure Set_Flat (T : in out Tensor; I : Integer; Value : Float);

   -- Display
   procedure Print (T : Tensor; Name : String := "");

   --------------------------------------------------------------------
   -- Ops — all return new tensors
   --------------------------------------------------------------------
   function "+" (A, B : Tensor) return Tensor;
   function "-" (A, B : Tensor) return Tensor;
   function "*" (A, B : Tensor) return Tensor;   -- element-wise
   function "/" (A, B : Tensor) return Tensor;   -- element-wise

   function Matmul (A, B : Tensor) return Tensor;  -- matrix multiply (2D only)
   function Dot (A, B : Tensor) return Float;      -- dot product (1D only)
   function Transpose (T : Tensor) return Tensor;  -- 2D transpose

   -- Activations
   function Relu (T : Tensor) return Tensor;
   function Gelu (T : Tensor) return Tensor;
   function Softmax (T : Tensor) return Tensor;    -- along last dim
   function Layer_Norm (T : Tensor) return Tensor; -- normalize last dim

   -- Reduction
   function Sum (T : Tensor) return Float;
   function Mean (T : Tensor) return Float;
   function Max (T : Tensor) return Float;

   -- Helpers (public, needed for preconditions)
   function Product (D : Dims) return Integer;

   -- Reshape (same numel)
   function Reshape (T : Tensor; New_Shape : Dims) return Tensor
     with Pre => Numel (T) = Product (New_Shape);

   -- Equality comparison
   function "=" (Left, Right : Tensor) return Boolean;

private

   type Float_Array is array (Integer range <>) of Float;

   type Tensor_Data_Rec (Len : Integer) is record
      Data : Float_Array (1 .. Len);
   end record;

   type Tensor_Data is access Tensor_Data_Rec;

   type Tensor is record
      Data  : Tensor_Data;
      Shape : Dims (1 .. 4);  -- rank up to 4
      Rank  : Natural := 0;
   end record;

end LLM_Tensor;
