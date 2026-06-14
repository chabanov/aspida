---------------------------------------------------------------------
-- LLM_Tensor — N-dimensional array with basic linear-algebra ops
-- Pure ADA, no external dependencies. FP32 precision.
---------------------------------------------------------------------

with Ada.Finalization;

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

   --  Dense matrix-vector product W*x for a row-major weight W [Rows, Cols]
   --  and vector X [Cols] (any rank, read flat), returning Y [1, Rows].
   --  The hot dot loop runs directly over the raw FP32 storage (no Get
   --  indirection, so it vectorises) and is split across the worker pool.
   --  Used for the LM head (vocab x dim) and other large dense matvecs.
   function MatVec_Rows (W : Tensor; X : Tensor) return Tensor;

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

   --  Owning, value-semantics handle for the heap-allocated tensor data:
   --  deep-copies on assignment (Adjust) and frees on scope exit (Finalize),
   --  so tensors neither leak nor alias each other's storage.
   type Data_Handle is new Ada.Finalization.Controlled with record
      Ptr : Tensor_Data := null;
   end record;

   overriding procedure Adjust   (H : in out Data_Handle);
   overriding procedure Finalize (H : in out Data_Handle);

   type Tensor is record
      Data  : Data_Handle;
      Shape : Dims (1 .. 4);  -- rank up to 4
      Rank  : Natural := 0;
   end record;

end LLM_Tensor;
