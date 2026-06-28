---------------------------------------------------------------------
-- LLM_Dense_FFN body — dense SwiGLU MLP.
---------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;
with LLM_Tensor; use LLM_Tensor;
with LLM_Weight; use LLM_Weight;
with LLM_GPU;

package body LLM_Dense_FFN is

   --  Drop a weight's GPU mirror (keyed by host address) THEN its host bytes.
   procedure Drop_W (W : in out LLM_Weight.Weight) is
   begin
      LLM_GPU.Free_Weight (LLM_Weight.Raw_Address (W));
      LLM_Weight.Free_Bytes (W);
   end Drop_W;

   procedure Free (L : in out Dense_FFN_Layer) is
   begin
      Drop_W (L.Gate_W); Drop_W (L.Up_W); Drop_W (L.Down_W);
   end Free;

   --  Hot per-token kernel; indices derive from the layer's own dims.
   pragma Suppress (All_Checks);

   function Create
     (Gate_W, Up_W, Down_W : Weight) return Dense_FFN_Layer
   is
      L : Dense_FFN_Layer;
   begin
      L.Gate_W := Gate_W; L.Up_W := Up_W; L.Down_W := Down_W;
      L.Dim      := Cols (Gate_W);   -- input dim
      L.Intermed := Rows (Gate_W);   -- feed-forward length
      return L;
   end Create;

   function Forward (L : Dense_FFN_Layer; X : Tensor) return Tensor is
      FF   : constant Integer := L.Intermed;
      Gate : constant Tensor := MatVec (L.Gate_W, X);   -- [1, intermed]
      Up   : constant Tensor := MatVec (L.Up_W,   X);   -- [1, intermed]
      Act  : Tensor := New_Tensor ([1, FF]);
   begin
      --  SwiGLU: silu(gate) * up, then down-project.
      for I in 1 .. FF loop
         declare
            G : constant Float := Get_Flat (Gate, I);
         begin
            Set_Flat (Act, I, (G / (1.0 + Exp (-G))) * Get_Flat (Up, I));
         end;
      end loop;
      return MatVec (L.Down_W, Act);   -- [1, dim]
   end Forward;

end LLM_Dense_FFN;
