---------------------------------------------------------------------
-- LLM_MoE body — Router + SwiGLU expert FFN (top-k mixture of experts)
--
-- Implemented weight layout (2D, expert-major — the row-major flatten
-- of the GGUF 3D expert tensors):
--   Gate_Inp_W : [n_experts, dim]              router logits
--   Gate_Exp_W : [n_experts*intermed, dim]     per-expert gate proj
--   Up_W       : [n_experts*intermed, dim]     per-expert up proj
--   Down_W     : [n_experts*dim, intermed]     per-expert down proj
--   Shexp_Gate_W / Shexp_Up_W : [intermed, dim]
--   Shexp_Down_W              : [dim, intermed]
--   Shexp_Gate_Inp_W          : [1, dim]  (optional sigmoid gate)
--
-- One expert FFN is SwiGLU:  y = Down · ( silu(Gate · x) * (Up · x) ).
---------------------------------------------------------------------

with Ada.Numerics.Generic_Elementary_Functions;
with LLM_Tensor; use LLM_Tensor;

package body LLM_MoE is

   package Float_Math is new Ada.Numerics.Generic_Elementary_Functions (Float);
   use Float_Math;

   --------------------------------------------------------------------
   -- Silu activation: x * sigmoid(x)
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

      M.N_Experts := N_Experts;
      --  Router is [n_experts, dim]; the model dim is the second axis.
      M.Dim := Shape (Gate_Inp_W) (2);
      --  Never select more experts than exist.
      M.Top_K := Integer'Min (8, N_Experts);
      --  Gate/Up are [n_experts*intermed, dim]; recover intermed.
      if Shape (Up_W) (1) > 1 then
         M.Intermed := Shape (Up_W) (1) / M.N_Experts;
      else
         M.Intermed := 512;
      end if;

      return M;
   end Create_MoE;

   --------------------------------------------------------------------
   -- One SwiGLU expert E:  y = Down_E · ( silu(Gate_E · x) * (Up_E · x) )
   -- Gate_Base / Down_Base are the row offsets of expert E in the
   -- flattened weight tensors.
   --------------------------------------------------------------------

   procedure SwiGLU
     (Gate_W, Up_W, Down_W : Tensor;
      Gate_Base, Down_Base : Integer;
      X      : Tensor;
      Dim    : Integer;
      Intermed : Integer;
      Y      : out Tensor)
   is
      H : array (1 .. Intermed) of Float;
   begin
      for J in 1 .. Intermed loop
         declare
            G : Float := 0.0;
            U : Float := 0.0;
         begin
            for D in 1 .. Dim loop
               G := G + Get (Gate_W, [Gate_Base + J, D]) * Get_Flat (X, D);
               U := U + Get (Up_W,   [Gate_Base + J, D]) * Get_Flat (X, D);
            end loop;
            H (J) := Silu (G) * U;
         end;
      end loop;

      for I in 1 .. Dim loop
         declare
            Acc : Float := 0.0;
         begin
            for J in 1 .. Intermed loop
               Acc := Acc + Get (Down_W, [Down_Base + I, J]) * H (J);
            end loop;
            Set_Flat (Y, I, Acc);
         end;
      end loop;
   end SwiGLU;

   --------------------------------------------------------------------
   -- Forward — top-k gated mixture of experts + shared expert
   --------------------------------------------------------------------

   function Forward (M : MoE_Layer; X : Tensor) return Tensor is
      Dim      : constant Integer := M.Dim;
      Intermed : constant Integer := M.Intermed;
      N_Exp    : constant Integer := M.N_Experts;
      Top_K    : constant Integer := M.Top_K;

      Router  : array (1 .. N_Exp) of Float := [others => 0.0];
      Top_Idx : array (1 .. Top_K) of Integer := [others => 1];
      Top_W   : array (1 .. Top_K) of Float := [others => 0.0];

      Result  : Tensor := New_Tensor ([1, Dim]);
      Exp_Out : Tensor := New_Tensor ([1, Dim]);
   begin
      ----------------------------------------------------------------
      -- 1. Router logits: gate_inp [n_experts, dim] · x
      ----------------------------------------------------------------
      for E in 1 .. N_Exp loop
         declare
            S : Float := 0.0;
         begin
            for D in 1 .. Dim loop
               S := S + Get (M.Gate_Inp_W, [E, D]) * Get_Flat (X, D);
            end loop;
            Router (E) := S;
         end;
      end loop;

      ----------------------------------------------------------------
      -- 2. Softmax over experts (numerically stable)
      ----------------------------------------------------------------
      declare
         Max_L : Float := Router (1);
         Sum_E : Float := 0.0;
      begin
         for E in 2 .. N_Exp loop
            if Router (E) > Max_L then
               Max_L := Router (E);
            end if;
         end loop;
         for E in 1 .. N_Exp loop
            Router (E) := Exp (Router (E) - Max_L);
            Sum_E := Sum_E + Router (E);
         end loop;
         for E in 1 .. N_Exp loop
            Router (E) := Router (E) / Sum_E;
         end loop;
      end;

      ----------------------------------------------------------------
      -- 3. Greedy top-k selection, then renormalise the chosen weights
      ----------------------------------------------------------------
      declare
         Used  : array (1 .. N_Exp) of Boolean := [others => False];
         Sum_W : Float := 0.0;
      begin
         for K in 1 .. Top_K loop
            declare
               Best_E : Integer := 0;
               Best_V : Float   := Float'First;
            begin
               for E in 1 .. N_Exp loop
                  if not Used (E) and then Router (E) > Best_V then
                     Best_V := Router (E);
                     Best_E := E;
                  end if;
               end loop;
               Top_Idx (K) := Best_E;
               Top_W (K)   := Best_V;
               Used (Best_E) := True;
               Sum_W := Sum_W + Best_V;
            end;
         end loop;
         for K in 1 .. Top_K loop
            Top_W (K) := Top_W (K) / Sum_W;
         end loop;
      end;

      ----------------------------------------------------------------
      -- 4. Run each selected expert and accumulate the weighted output
      ----------------------------------------------------------------
      for K in 1 .. Top_K loop
         declare
            E : constant Integer := Top_Idx (K);
         begin
            SwiGLU (M.Gate_Exp_W, M.Up_W, M.Down_W,
                    (E - 1) * Intermed, (E - 1) * Dim,
                    X, Dim, Intermed, Exp_Out);
            for I in 1 .. Dim loop
               Set_Flat (Result, I,
                 Get_Flat (Result, I) + Top_W (K) * Get_Flat (Exp_Out, I));
            end loop;
         end;
      end loop;

      ----------------------------------------------------------------
      -- 5. Shared expert (always active), with optional sigmoid gate
      ----------------------------------------------------------------
      declare
         Shared_Gate : Float := 1.0;
         H : array (1 .. Intermed) of Float;
      begin
         if Numel (M.Shexp_Gate_Inp_W) > 1 then
            declare
               GS : Float := 0.0;
            begin
               for D in 1 .. Dim loop
                  GS := GS + Get_Flat (M.Shexp_Gate_Inp_W, D) * Get_Flat (X, D);
               end loop;
               Shared_Gate := 1.0 / (1.0 + Exp (-GS));
            end;
         end if;

         for J in 1 .. Intermed loop
            declare
               G : Float := 0.0;
               U : Float := 0.0;
            begin
               for D in 1 .. Dim loop
                  G := G + Get (M.Shexp_Gate_W, [J, D]) * Get_Flat (X, D);
                  U := U + Get (M.Shexp_Up_W,   [J, D]) * Get_Flat (X, D);
               end loop;
               H (J) := Silu (G) * U;
            end;
         end loop;

         for I in 1 .. Dim loop
            declare
               Acc : Float := 0.0;
            begin
               for J in 1 .. Intermed loop
                  Acc := Acc + Get (M.Shexp_Down_W, [I, J]) * H (J);
               end loop;
               Set_Flat (Result, I, Get_Flat (Result, I) + Shared_Gate * Acc);
            end;
         end loop;
      end;

      return Result;
   end Forward;

end LLM_MoE;
