---------------------------------------------------------------------
-- LLM_Accelerate body — Apple Accelerate.framework C BLAS bindings
---------------------------------------------------------------------

with Ada.Text_IO;

package body LLM_Accelerate is

   use LLM_Tensor;

   -- C enum values for CBLAS
   CblasRowMajor : constant Integer := 101;
   CblasColMajor : constant Integer := 102;
   CblasNoTrans  : constant Integer := 111;
   CblasTrans    : constant Integer := 112;

   -- cblas_sgemm from Accelerate.framework
   -- Link with: -framework Accelerate
   procedure C_SGEMM
     (Order   : Integer;
      TransA  : Integer;
      TransB  : Integer;
      M       : Integer;
      N       : Integer;
      K       : Integer;
      Alpha   : Float;
      A       : System.Address;
      Lda     : Integer;
      B       : System.Address;
      Ldb     : Integer;
      Beta    : Float;
      C       : System.Address;
      Ldc     : Integer)
     with Import => True,
          Convention => C,
          External_Name => "cblas_sgemm";

   -- cblas_sdot
   function C_SDOT
     (N    : Integer;
      X    : System.Address;
      IncX : Integer;
      Y    : System.Address;
      IncY : Integer) return Float
     with Import => True,
          Convention => C,
          External_Name => "cblas_sdot";

   --------------------------------------------------------------------
   -- SGEMM
   --------------------------------------------------------------------

   procedure SGEMM
     (A : Tensor;
      B : Tensor;
      C : in out Tensor;
      Alpha : Float := 1.0;
      Beta  : Float := 0.0;
      Transpose_A : Boolean := False;
      Transpose_B : Boolean := False)
   is
      M       : Integer;
      N       : Integer;
      K       : Integer;
      Lda     : Integer;
      Ldb     : Integer;
      Ldc     : Integer;
      Trans_A : Integer;
      Trans_B : Integer;
   begin
      if Transpose_A then
         M := A.Shape (2);  -- columns become rows
         K := A.Shape (1);  -- rows become inner dim
         Trans_A := CblasTrans;
         Lda := A.Shape (1);
      else
         M := A.Shape (1);
         K := A.Shape (2);
         Trans_A := CblasNoTrans;
         Lda := A.Shape (2);
      end if;

      if Transpose_B then
         -- B is K×N, transposed: N×K
         N := B.Shape (1);
         Trans_B := CblasTrans;
         Ldb := B.Shape (1);
      else
         N := B.Shape (2);
         Trans_B := CblasNoTrans;
         Ldb := B.Shape (2);
      end if;

      Ldc := N;

      -- Get raw float pointers from tensors
      declare
         A_Ptr : constant System.Address := A.Data.Data (A.Data.Data'First)'Address;
         B_Ptr : constant System.Address := B.Data.Data (B.Data.Data'First)'Address;
         C_Ptr : System.Address := C.Data.Data (C.Data.Data'First)'Address;
      begin
         C_SGEMM (CblasRowMajor, Trans_A, Trans_B, M, N, K,
                  Alpha, A_Ptr, Lda,
                  B_Ptr, Ldb,
                  Beta,  C_Ptr, Ldc);
      end;
   end SGEMM;

   --------------------------------------------------------------------
   -- SDOT
   --------------------------------------------------------------------

   function Dot (A, B : Tensor) return Float is
      N : constant Integer := Numel (A);
   begin
      return C_SDOT (N, A.Data.Data (A.Data.Data'First)'Address, 1,
                        B.Data.Data (B.Data.Data'First)'Address, 1);
   end Dot;

   --------------------------------------------------------------------
   -- Is_Available
   --------------------------------------------------------------------

   function Is_Available return Boolean is
   begin
      -- Try a trivial call to see if the library is linked
      declare
         Dummy : Float;
      begin
         Dummy := C_SDOT (0, System.Null_Address, 1, System.Null_Address, 1);
         return True;
      exception
         when others =>
            return False;
      end;
   end Is_Available;

end LLM_Accelerate;
