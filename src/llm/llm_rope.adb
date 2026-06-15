---------------------------------------------------------------------
-- LLM_RoPE body — mRoPE implementation for Qwen 3.5
---------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;

package body LLM_RoPE is

   function Create_Qwen_RoPE
     (Dim       : Integer := 64;
      Freq_Base : Float   := 10_000_000.0;
      Max_Pos   : Integer := 262_144) return RoPE_Params
   is
      P : RoPE_Params;
      S : Tensor := New_Tensor ([1, 4]);
   begin
      P.Dim       := Dim;
      P.Freq_Base := Freq_Base;
      P.Max_Pos   := Max_Pos;
      Set_Flat (S, 1, 11.0);  -- section 0: dims 0-10
      Set_Flat (S, 2, 11.0);  -- section 1: dims 11-21
      Set_Flat (S, 3, 10.0);  -- section 2: dims 22-31
      Set_Flat (S, 4, 0.0);   -- section 3: unused
      P.Sections := S;
      return P;
   end Create_Qwen_RoPE;

   function Apply (P : RoPE_Params; X : Tensor; Pos : Integer) return Tensor is
      Half_Dim : constant Integer := P.Dim / 2;  -- 32 for Qwen
      Result   : Tensor := New_Tensor ([1, P.Dim]);
      Theta    : Float;
      Cos_Val  : Float;
      Sin_Val  : Float;
      X1, X2   : Float;
   begin
      --  NeoX / rotate_half convention (Qwen, Llama): pair dimension i with
      --  i + dim/2 (first half with second half), NOT adjacent (2i, 2i+1).
      --    theta_i = pos / freq_base^(2i/dim)
      --    out[i]          = x[i]*cos - x[i+d/2]*sin
      --    out[i+d/2]      = x[i+d/2]*cos + x[i]*sin
      for I in 0 .. Half_Dim - 1 loop
         Theta := Float (Pos) / (P.Freq_Base ** (Float (2 * I) / Float (P.Dim)));
         Cos_Val := Cos (Theta);
         Sin_Val := Sin (Theta);

         X1 := Get_Flat (X, I + 1);              -- first half
         X2 := Get_Flat (X, I + Half_Dim + 1);   -- second half

         Set_Flat (Result, I + 1,            X1 * Cos_Val - X2 * Sin_Val);
         Set_Flat (Result, I + Half_Dim + 1, X2 * Cos_Val + X1 * Sin_Val);
      end loop;

      return Result;
   end Apply;

end LLM_RoPE;
