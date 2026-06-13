---------------------------------------------------------------------
-- LLM_SSM body — Mamba selective state space implementation
---------------------------------------------------------------------

with Ada.Numerics.Generic_Elementary_Functions;
with Ada.Strings.Fixed;
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
   -- Init state
   --------------------------------------------------------------------

   function Init_State (State_Dim : Integer) return Tensor is
   begin
      return New_Tensor ((1, State_Dim));
   end Init_State;

   --------------------------------------------------------------------
   -- Conv1D: causal convolution with kernel [4, channels]
   --
   -- Input: [channels] single-token vector
   -- Kernel: [4, channels] (kernel_size=4, channels=intermediate=2*dim)
   -- We maintain a buffer of last 4 inputs for causal conv.
   --
   -- For Qwen SSM:
   --   First, project input to intermediate space:
   --     x_inter = x_in * W(norm) [dim → 4096]
   --   Then conv1d(intermediate) with kernel [4, 4096]
   --------------------------------------------------------------------

   -- Apply 1D causal conv to a buffer of past inputs
   -- Conv_Buffer: [4, channels] last 4 inputs (row 1 = oldest, row 4 = newest)
   -- Conv_Weight: [4, channels] convolution kernel
   -- Returns: [channels] convolved output
   function Conv1D
     (Conv_Buffer  : Tensor;  -- [4, channels]
      Conv_Weight  : Tensor;  -- [4, channels]
      Channels     : Integer)
      return Tensor
   is
      Result : Tensor := New_Tensor ((1, Channels));
      Sum    : Float;
   begin
      for C in 1 .. Channels loop
         Sum := 0.0;
         for K in 1 .. 4 loop
            Sum := Sum + Get (Conv_Buffer, (K, C)) * Get (Conv_Weight, (K, C));
         end loop;
         Set_Flat (Result, C, Sum);
      end loop;
      return Result;
   end Conv1D;

   -- Shift buffer: drop oldest, add new, return new buffer
   procedure Shift_Buffer
     (Buf  : in out Tensor;  -- [4, channels]
      New_X : Tensor;         -- [channels]
      Channels : Integer)
   is
   begin
      -- Shift rows up
      for K in 1 .. 3 loop
         for C in 1 .. Channels loop
            Set (Buf, [K, C], Get (Buf, [K + 1, C]));
         end loop;
      end loop;
      -- Set newest row
      for C in 1 .. Channels loop
         Set (Buf, [4, C], Get_Flat (New_X, C));
      end loop;
   end Shift_Buffer;

   --------------------------------------------------------------------
   -- Forward: Mamba selective scan for one token
   --------------------------------------------------------------------

   function Forward
     (P     : SSM_Params;
      X     : Tensor;        -- [dim] single token
      State : in out Tensor   -- [state_dim] hidden state
     ) return Tensor
   is
      Dim        : constant Integer := Numel (X);
      State_Dim  : constant Integer := Numel (State);
      Intermed   : constant Integer := 2 * Dim;
      Kernel_Size : constant Integer := 4;

      -- 1. Linear projection: x → intermediate space via gamma
      Proj : Tensor := New_Tensor ([1, Intermed]);
   begin
      for I in 1 .. Intermed loop
         declare
            Sum : Float := 0.0;
         begin
            for J in 1 .. Dim loop
               Sum := Sum + Get_Flat (X, J) * Get (P.Gamma, (J, I));
            end loop;
            Set_Flat (Proj, I, Sum);
         end;
      end loop;

      -- 2. SiLU activation
      for I in 1 .. Intermed loop
         Set_Flat (Proj, I, Silu (Get_Flat (Proj, I)));
      end loop;

      -- 3. Split: x_ssm (first half), gate (second half)
      declare
         X_Ssm  : Tensor := New_Tensor ([1, Dim]);
         Gate   : Tensor := New_Tensor ([1, Dim]);
      begin
         for I in 1 .. Dim loop
            Set_Flat (X_Ssm, I, Get_Flat (Proj, I));
            Set_Flat (Gate, I, Silu (Get_Flat (Proj, Dim + I)));
         end loop;

         -- 4. B = Alpha_W @ X_Ssm
         declare
            B : Tensor := New_Tensor ([1, State_Dim]);
            C : Tensor := New_Tensor ([1, State_Dim]);
         begin
            for I in 1 .. State_Dim loop
               declare
                  Sum_B : Float := 0.0;
                  Sum_C : Float := 0.0;
               begin
                  for J in 1 .. Dim loop
                     Sum_B := Sum_B + Get_Flat (X_Ssm, J) * Get (P.Alpha_W, (J, I));
                     Sum_C := Sum_C + Get_Flat (X_Ssm, J) * Get (P.Beta_W, (J, I));
                  end loop;
                  Set_Flat (B, I, Sum_B);
                  Set_Flat (C, I, Sum_C);
               end;
            end loop;

            -- 5. Delta = softplus(linear(x_ssm) + dt_bias)
            declare
               Dt : Tensor := New_Tensor ([1, State_Dim]);
            begin
               for I in 1 .. State_Dim loop
                  declare
                     Sum_Dt : Float := 0.0;
                  begin
                     for J in 1 .. Dim loop
                        Sum_Dt := Sum_Dt + Get_Flat (X_Ssm, J)
                          * Get (P.Dt_Bias, (J, I));
                     end loop;
                     Sum_Dt := Sum_Dt + Get_Flat (P.Dt_Bias, I);
                     Set_Flat (Dt, I, Softplus (Sum_Dt));
                  end;
               end loop;

               -- 6. Discretize + update state
               for I in 1 .. State_Dim loop
                  declare
                     A_Bar : constant Float := Exp (Get_Flat (Dt, I)
                       * Get_Flat (P.A_Diag, I));
                     B_Bar : constant Float := Get_Flat (Dt, I) * Get_Flat (B, I);
                     Dt_State : constant Float :=
                       A_Bar * Get_Flat (State, I) + B_Bar * Get_Flat (X_Ssm, I);
                  begin
                     Set_Flat (State, I, Dt_State);
                  end;
               end loop;
            end;

            -- 7. Y = State * C (element-wise), then gate * Y
            declare
               Y : Tensor := New_Tensor ([1, Dim]);
            begin
               for I in 1 .. Dim loop
                  Set_Flat (Y, I, Get_Flat (State, I) * Get_Flat (C, I) * Get_Flat (Gate, I));
               end loop;

               -- 8. Output projection: out_weight @ y
               declare
                  Result : Tensor := New_Tensor ([1, Dim]);
               begin
                  for I in 1 .. Dim loop
                     declare
                        Sum : Float := 0.0;
                     begin
                        for J in 1 .. Dim loop
                           Sum := Sum + Get_Flat (Y, J) * Get (P.Out_Weight, (J, I));
                        end loop;
                        Set_Flat (Result, I, Sum);
                     end;
                  end loop;
                  return Result;
               end;
            end;
         end;
      end;
   end Forward;

end LLM_SSM;
