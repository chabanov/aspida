---------------------------------------------------------------------
-- LLM_Weights body — reads binary FP32 files
---------------------------------------------------------------------

with Ada.Sequential_IO;
with Ada.Text_IO;

package body LLM_Weights is

   use LLM_Tensor;
   use LLM_Autograd;

   package Float_IO is new Ada.Sequential_IO (Float);
   use Float_IO;

   function Load_Tensor (Path : String; Shape : Dims) return Var is
      F : Float_IO.File_Type;
      T : Tensor := New_Tensor (Shape);
      N : constant Integer := Numel (T);
   begin
      Ada.Text_IO.Put_Line ("Loading " & Path & " (" & Integer'Image (N) & " floats)...");
      Float_IO.Open (F, In_File, Path);
      for I in 1 .. N loop
         declare
            V : Float;
         begin
            Float_IO.Read (F, V);
            Set_Flat (T, I, V);
         end;
      end loop;
      Float_IO.Close (F);
      return New_Var (T);
   exception
      when others =>
         Ada.Text_IO.Put_Line ("ERROR loading " & Path);
         return New_Var (New_Tensor (Shape));
   end Load_Tensor;

   function Load_Vector (Path : String; N : Integer) return Var is
   begin
      return Load_Tensor (Path, [1, N]);
   end Load_Vector;

   function Load_Matrix (Path : String; Rows, Cols : Integer) return Var is
   begin
      return Load_Tensor (Path, [Rows, Cols]);
   end Load_Matrix;

end LLM_Weights;
