---------------------------------------------------------------------
-- LLM_RoPE body — mRoPE implementation for Qwen 3.5
---------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;

package body LLM_RoPE is

   function Create_Qwen_RoPE return RoPE_Params is
      P : RoPE_Params;
      S : Tensor := New_Tensor ([1, 4]);
   begin
      P.Dim       := 64;
      P.Freq_Base := 10_000_000.0;
      P.Max_Pos   := 262_144;
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
      X_Rot    : Float;
      X_Pass   : Float;
   begin
      -- Compute inv_freq: theta_i = pos / (freq_base ^ (2i / dim))
      for I in 0 .. Half_Dim - 1 loop
         Theta := Float (Pos) / (P.Freq_Base ** (Float (2 * I) / Float (P.Dim)));

         Cos_Val := Cos (Theta);
         Sin_Val := Sin (Theta);

         -- Rotate pair (x_2i, x_2i+1):
         --   new_x_2i   = x_2i * cos - x_2i+1 * sin
         --   new_x_2i+1 = x_2i+1 * cos + x_2i * sin
         X_Rot  := Get_Flat (X, 2 * I + 1);
         X_Pass := Get_Flat (X, 2 * I + 2);

         Set_Flat (Result, 2 * I + 1,
           X_Rot * Cos_Val - X_Pass * Sin_Val);
         Set_Flat (Result, 2 * I + 2,
           X_Pass * Cos_Val + X_Rot * Sin_Val);
      end loop;

      return Result;
   end Apply;

end LLM_RoPE;
