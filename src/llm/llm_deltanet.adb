---------------------------------------------------------------------
-- LLM_DeltaNet body — gated delta rule recurrence
---------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;
with LLM_Tensor; use LLM_Tensor;

package body LLM_DeltaNet is

   --  Hot per-token recurrence; indices derive from head dims.
   pragma Suppress (All_Checks);

   function L2_Normalize (T : Tensor) return Tensor is
      N  : constant Integer := Numel (T);
      SS : Float := 0.0;
   begin
      for I in 1 .. N loop
         SS := SS + Get_Flat (T, I) ** 2;
      end loop;
      return R : Tensor := New_Tensor ([1, N]) do
         declare
            Inv : constant Float := 1.0 / (Sqrt (SS) + 1.0e-6);
         begin
            for I in 1 .. N loop
               Set_Flat (R, I, Get_Flat (T, I) * Inv);
            end loop;
         end;
      end return;
   end L2_Normalize;

   function Init_State (Dk, Dv : Integer) return Tensor is
   begin
      return New_Tensor ([Dk, Dv]);
   end Init_State;

   procedure Step
     (S    : in out Tensor;
      Q, K : Tensor;
      V    : Tensor;
      G    : Float;
      Beta : Float;
      O    : out Tensor;
      Base : Integer := 0)
   is
      Dk    : constant Integer := Numel (Q);
      Dv    : constant Integer := Numel (V);
      QN    : constant Tensor := L2_Normalize (Q);
      KN    : constant Tensor := L2_Normalize (K);
      Scale : constant Float := 1.0 / Sqrt (Float (Dk));
      Corr  : array (1 .. Dv) of Float;   -- error correction (the "delta")
   begin
      --  corr[v] = beta * (v[v] - sum_k g*S_old[k,v]*k~[k])
      for Vi in 1 .. Dv loop
         declare
            Retr : Float := 0.0;
         begin
            for Ki in 1 .. Dk loop
               Retr := Retr + G * Get (S, [Base + Ki, Vi]) * Get_Flat (KN, Ki);
            end loop;
            Corr (Vi) := Beta * (Get_Flat (V, Vi) - Retr);
         end;
      end loop;

      --  Gated write: S[k,v] = g*S_old[k,v] + k~[k]*corr[v]
      for Ki in 1 .. Dk loop
         declare
            Kk : constant Float := Get_Flat (KN, Ki);
         begin
            for Vi in 1 .. Dv loop
               Set (S, [Base + Ki, Vi],
                    G * Get (S, [Base + Ki, Vi]) + Kk * Corr (Vi));
            end loop;
         end;
      end loop;

      --  Output: o[v] = (sum_k S[k,v]*q~[k]) / sqrt(Dk)
      for Vi in 1 .. Dv loop
         declare
            Acc : Float := 0.0;
         begin
            for Ki in 1 .. Dk loop
               Acc := Acc + Get (S, [Base + Ki, Vi]) * Get_Flat (QN, Ki);
            end loop;
            Set_Flat (O, Vi, Acc * Scale);
         end;
      end loop;
   end Step;

end LLM_DeltaNet;
