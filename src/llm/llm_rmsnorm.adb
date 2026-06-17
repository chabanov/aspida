---------------------------------------------------------------------
-- LLM_RMSNorm body
---------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions;

package body LLM_RMSNorm is

   --  Called per token per layer; indices derive from the vector length.
   pragma Suppress (All_Checks);

   function Forward (X : Tensor; Weight : Tensor; Eps : Float := Epsilon)
      return Tensor is
      N : constant Integer := Numel (X);
      -- Compute mean of squares
      Sum_Sq : Float := 0.0;
   begin
      for I in 1 .. N loop
         declare
            Val : constant Float := Get_Flat (X, I);
         begin
            Sum_Sq := Sum_Sq + Val * Val;
         end;
      end loop;

      declare
         Rms : constant Float := Ada.Numerics.Elementary_Functions.Sqrt
           (Sum_Sq / Float (N) + Eps);
         Result : Tensor := New_Tensor ([1, N]);
      begin
         for I in 1 .. N loop
            declare
               W  : constant Float := Get_Flat (Weight, I);
               Val : constant Float := Get_Flat (X, I);
            begin
               Set_Flat (Result, I, (Val / Rms) * W);
            end;
         end loop;
         return Result;
      end;
   end Forward;

end LLM_RMSNorm;
