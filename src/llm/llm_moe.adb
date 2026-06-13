---------------------------------------------------------------------
-- LLM_MoE body — Router + expert FFN computation
---------------------------------------------------------------------

with Ada.Numerics.Generic_Elementary_Functions;
with LLM_Tensor; use LLM_Tensor;

package body LLM_MoE is

   package Float_Math is new Ada.Numerics.Generic_Elementary_Functions (Float);
   use Float_Math;

   --------------------------------------------------------------------
   -- Silu activation
   --------------------------------------------------------------------

   function Silu (X : Float) return Float is
   begin
      return X / (1.0 + Exp (-X));
   end Silu;

   --------------------------------------------------------------------
   -- Create
   --------------------------------------------------------------------

   function Create_MoE
     (Gate_Inp_W, Gate_Exp_W, Up_W, Down_W,
      Shexp_Gate_W, Shexp_Up_W, Shexp_Down_W, Shexp_Gate_Inp_W : Tensor;
      N_Experts : Integer)
      return MoE_Layer
   is
      M : MoE_Layer;
   begin
      M.Gate_Inp_W       := Gate_Inp_W;
      M.Gate_Exp_W       := Gate_Exp_W;
      M.Up_W             := Up_W;
      M.Down_W           := Down_W;
      M.Shexp_Gate_W     := Shexp_Gate_W;
      M.Shexp_Up_W       := Shexp_Up_W;
      M.Shexp_Down_W     := Shexp_Down_W;
      M.Shexp_Gate_Inp_W := Shexp_Gate_Inp_W;

      -- Dim from router gate (shape[2] since gate is [dim, n_experts])
      M.Dim      := Shape (Gate_Inp_W) (2);
      M.N_Experts := N_Experts;
      M.Top_K     := 8;
      -- Intermed: derived from Up_W shape[1], or default 512 for Qwen 3.5
      if Shape (Up_W) (1) > 1 then
         M.Intermed := Shape (Up_W) (1) / M.N_Experts;
      else
         M.Intermed := 512;
      end if;

      return M;
   end Create_MoE;

   --------------------------------------------------------------------
   -- Forward — Top-K gated mixture of experts
   --------------------------------------------------------------------

   function Forward (M : MoE_Layer; X : Tensor) return Tensor is

      Dim      : constant Integer := M.Dim;
      Intermed : constant Integer := M.Intermed;

      -- Router logits: gate_inp @ x → [256]
      Router_Logits : Tensor := New_Tensor ((1, M.N_Experts));

      -- Top-K expert indices and weights (softmax over selected)
      Result : Tensor := New_Tensor ((1, Dim));
      Top_Weights : array (1 .. 8) of Float := (others => 0.0);
      Top_Indices : array (1 .. 8) of Integer := (others => 1);

      -- Shared expert output
      Shared_Out : Tensor := New_Tensor ((1, Dim));

      pragma Unreferenced (Dim, Intermed, Router_Logits, Top_Weights, Top_Indices, Shared_Out);
   begin
      -- 1. Router: compute gate scores
      for E in 1 .. M.N_Experts loop
         declare
            Score : Float := 0.0;
         begin
            for I in 1 .. Dim loop
               Score := Score + Get_Flat (X, I)
                 * Get (M.Gate_Inp_W, (I, E));
            end loop;
            Set_Flat (Router_Logits, E, Score);
         end;
      end loop;

      -- Softmax over experts
      declare
         Max_Logit : Float := Get_Flat (Router_Logits, 1);
         Sum_Exp   : Float := 0.0;
      begin
         for E in 1 .. M.N_Experts loop
            if Get_Flat (Router_Logits, E) > Max_Logit then
               Max_Logit := Get_Flat (Router_Logits, E);
            end if;
         end loop;
         for E in 1 .. M.N_Experts loop
            Sum_Exp := Sum_Exp + Exp (Get_Flat (Router_Logits, E) - Max_Logit);
         end loop;
         for E in 1 .. M.N_Experts loop
            Set_Flat (Router_Logits, E,
              Exp (Get_Flat (Router_Logits, E) - Max_Logit) / Sum_Exp);
         end loop;
      end;

      -- Top-8 selection (simple greedy)
      declare
         Used : array (1 .. M.N_Experts) of Boolean := (others => False);
      begin
         for K in 1 .. M.Top_K loop
            declare
               Best_Idx : Integer := 1;
               Best_Val : Float := -1.0e30;
            begin
               for E in 1 .. M.N_Experts loop
                  if not Used (E) and then Get_Flat (Router_Logits, E) > Best_Val then
                     Best_Val := Get_Flat (Router_Logits, E);
                     Best_Idx := E;
                  end if;
               end loop;
               Top_Indices (K) := Best_Idx;
               Top_Weights (K) := Best_Val;
               Used (Best_Idx) := True;
            end;
         end loop;
      end;

      -- Normalize top weights
      declare
         Sum_W : Float := 0.0;
      begin
         for K in 1 .. M.Top_K loop
            Sum_W := Sum_W + Top_Weights (K);
         end loop;
         for K in 1 .. M.Top_K loop
            Top_Weights (K) := Top_Weights (K) / Sum_W;
         end loop;
      end;

      -- 2. Expert FFN (simplified: weight * x, for each selected expert)
      for K in 1 .. M.Top_K loop
         declare
            E    : constant Integer := Top_Indices (K);
            W    : Float := Top_Weights (K);
            -- Expert MLP: Up projection (select columns for expert E)
            Exp_Out : Tensor := New_Tensor ((1, Dim));
         begin
            for I in 1 .. Dim loop
               declare
                  Acc_Silu : Float := 0.0;
                  Acc_Down : Float := 0.0;
               begin
                  -- Gate: Silu activation
                  for J in 1 .. M.Intermed loop
                     declare
                        Gate_Val : constant Float :=
                          Get (M.Gate_Exp_W, (J + (E - 1) * M.Intermed, 1)) *
                          Get_Flat (X, I);
                        Up_Val : constant Float :=
                          Get (M.Up_W, (J + (E - 1) * M.Intermed, 1)) *
                          Get_Flat (X, I);
                     begin
                        Acc_Silu := Acc_Silu + Silu (Gate_Val + Up_Val);
                     end;
                  end loop;
                  -- Down projection
                  for J in 1 .. M.Intermed loop
                     Acc_Down := Acc_Down + Acc_Silu *
                       Get (M.Down_W, (J + (E - 1) * M.Intermed, 1));
                  end loop;
                  Set_Flat (Exp_Out, I, Acc_Down);
               end;
               Set_Flat (Result, I, Get_Flat (Result, I) + W * Get_Flat (Exp_Out, I));
            end loop;
         end;
      end loop;

      -- 3. Shared expert (simplified)
      for I in 1 .. Dim loop
         declare
            Acc_Silu : Float := 0.0;
            Acc_Down : Float := 0.0;
         begin
            for J in 1 .. M.Intermed loop
               declare
                  Gate_Val : constant Float := Get (M.Shexp_Gate_W, (J, 1)) * Get_Flat (X, I);
                  Up_Val   : constant Float := Get (M.Shexp_Up_W, (J, 1)) * Get_Flat (X, I);
               begin
                  Acc_Silu := Acc_Silu + Silu (Gate_Val + Up_Val);
               end;
            end loop;
            for J in 1 .. M.Intermed loop
               Acc_Down := Acc_Down + Acc_Silu * Get (M.Shexp_Down_W, (J, 1));
            end loop;
            Set_Flat (Result, I, Get_Flat (Result, I) + Acc_Down);
         end;
      end loop;

      return Result;
   end Forward;

end LLM_MoE;
