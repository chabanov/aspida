---------------------------------------------------------------------
-- LLM_SSM body — Mamba-style diagonal selective state-space scan
--
-- Per token (called once per sequence position, carrying State):
--   z      = Gamma^T x                       in_proj -> [2*dim]
--   u      = SiLU(z[1..dim])                 SSM input branch
--   gate   = z[dim+1..2*dim]                 gating branch
--   B[n]   = sum_d u[d] * Alpha_W[d,n]       input-dependent B
--   C[n]   = sum_d u[d] * Beta_W[d,n]        input-dependent C
--   dt[d]  = softplus(u[d] + dt_bias)        input-dependent step
--   H[d,n] = exp(dt[d]*A[n]) * H[d,n] + dt[d]*B[n]*u[d]   selective scan
--   y[d]   = (sum_n C[n] * H[d,n]) * SiLU(gate[d])
--   out    = Out_Weight^T y
--
-- The state H is [d_inner, d_state]; it is carried in the in-out State
-- tensor and self-sized on first use. (The causal conv1d branch of full
-- Mamba is omitted; dt's exact bias mapping to the Qwen tensors is
-- provisional pending validation against the real model.)
---------------------------------------------------------------------

with Ada.Numerics.Generic_Elementary_Functions;
with LLM_Tensor; use LLM_Tensor;

package body LLM_SSM is

   package Float_Math is new Ada.Numerics.Generic_Elementary_Functions (Float);
   use Float_Math;

   --------------------------------------------------------------------
   -- Helpers
   --------------------------------------------------------------------

   function Softplus (X : Float) return Float is
   begin
      if X > 20.0 then
         return X;
      elsif X < -20.0 then
         return Exp (X);
      else
         return Log (1.0 + Exp (X));
      end if;
   end Softplus;

   -- SiLU = x * sigmoid(x) = x / (1 + exp(-x))
   function Silu (X : Float) return Float is
   begin
      return X / (1.0 + Exp (-X));
   end Silu;

   --------------------------------------------------------------------
   -- Create SSM params (just store tensors)
   --------------------------------------------------------------------

   function Create_SSM
     (Conv_W, A_D, Dt_B, Gamma, Out_W, Alpha_W, Beta_W : Tensor)
      return SSM_Params
   is
      P : SSM_Params;
   begin
      P.Conv_Weight := Conv_W;
      P.A_Diag      := A_D;
      P.Dt_Bias     := Dt_B;
      P.Gamma       := Gamma;
      P.Out_Weight  := Out_W;
      P.Alpha_W     := Alpha_W;
      P.Beta_W      := Beta_W;
      return P;
   end Create_SSM;

   --------------------------------------------------------------------
   -- Init state (a seed; Forward self-sizes to [d_inner, d_state])
   --------------------------------------------------------------------

   function Init_State (State_Dim : Integer) return Tensor is
   begin
      return New_Tensor ([1, State_Dim]);
   end Init_State;

   --------------------------------------------------------------------
   -- Forward: one selective-scan step over a single token
   --------------------------------------------------------------------

   function Forward
     (P     : SSM_Params;
      X     : Tensor;        -- [dim] single token
      State : in out Tensor   -- [d_inner, d_state] hidden state (in/out)
     ) return Tensor
   is
      Dim      : constant Integer := Numel (X);
      Intermed : constant Integer := 2 * Dim;
      SD       : constant Integer := Numel (P.A_Diag);  -- d_state
   begin
      -- Degenerate / unloaded params: return zeros instead of crashing.
      if SD < 1
        or else Numel (P.Gamma) < Dim * Intermed
        or else Numel (P.Alpha_W) < Dim * SD
        or else Numel (P.Beta_W) < Dim * SD
        or else Numel (P.Out_Weight) < Dim * Dim
      then
         return New_Tensor ([1, Dim]);
      end if;

      -- Ensure the carried state is [Dim, SD]; (re)initialise to zeros on
      -- first use or any shape change.
      if Rank (State) /= 2
        or else Shape (State) (1) /= Dim
        or else Shape (State) (2) /= SD
      then
         State := New_Tensor ([Dim, SD]);
      end if;

      declare
         U    : Tensor := New_Tensor ([1, Dim]);   -- SSM input branch (SiLU)
         Gate : Tensor := New_Tensor ([1, Dim]);   -- gating branch
         B    : Tensor := New_Tensor ([1, SD]);
         C    : Tensor := New_Tensor ([1, SD]);
         Dt   : Tensor := New_Tensor ([1, Dim]);
         Y    : Tensor := New_Tensor ([1, Dim]);
         Dt0  : constant Float :=
           (if Numel (P.Dt_Bias) >= 1 then Get_Flat (P.Dt_Bias, 1) else 0.0);
      begin
         -- in_proj: z = Gamma^T x → split into u (SiLU) and gate.
         for I in 1 .. Dim loop
            declare
               Su : Float := 0.0;
               Sg : Float := 0.0;
            begin
               for J in 1 .. Dim loop
                  Su := Su + Get_Flat (X, J) * Get (P.Gamma, [J, I]);
                  Sg := Sg + Get_Flat (X, J) * Get (P.Gamma, [J, Dim + I]);
               end loop;
               Set_Flat (U, I, Silu (Su));
               Set_Flat (Gate, I, Sg);
            end;
         end loop;

         -- Input-dependent step size per channel.
         for I in 1 .. Dim loop
            Set_Flat (Dt, I, Softplus (Get_Flat (U, I) + Dt0));
         end loop;

         -- Input-dependent B and C per state dimension.
         for N in 1 .. SD loop
            declare
               Sb : Float := 0.0;
               Sc : Float := 0.0;
            begin
               for J in 1 .. Dim loop
                  Sb := Sb + Get_Flat (U, J) * Get (P.Alpha_W, [J, N]);
                  Sc := Sc + Get_Flat (U, J) * Get (P.Beta_W, [J, N]);
               end loop;
               Set_Flat (B, N, Sb);
               Set_Flat (C, N, Sc);
            end;
         end loop;

         -- Selective scan: update H[d,n] and read out y[d] = sum_n C[n] H[d,n].
         for D in 1 .. Dim loop
            declare
               Acc  : Float := 0.0;
               Dt_D : constant Float := Get_Flat (Dt, D);
               U_D  : constant Float := Get_Flat (U, D);
            begin
               for N in 1 .. SD loop
                  declare
                     A_Bar : constant Float := Exp (Dt_D * Get_Flat (P.A_Diag, N));
                     H_New : constant Float :=
                       A_Bar * Get (State, [D, N]) + Dt_D * Get_Flat (B, N) * U_D;
                  begin
                     Set (State, [D, N], H_New);
                     Acc := Acc + Get_Flat (C, N) * H_New;
                  end;
               end loop;
               Set_Flat (Y, D, Acc * Silu (Get_Flat (Gate, D)));
            end;
         end loop;

         -- Output projection: out = Out_Weight^T y.
         return Result : Tensor := New_Tensor ([1, Dim]) do
            for I in 1 .. Dim loop
               declare
                  S : Float := 0.0;
               begin
                  for J in 1 .. Dim loop
                     S := S + Get_Flat (Y, J) * Get (P.Out_Weight, [J, I]);
                  end loop;
                  Set_Flat (Result, I, S);
               end;
            end loop;
         end return;
      end;
   end Forward;

end LLM_SSM;
