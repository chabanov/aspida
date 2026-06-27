---------------------------------------------------------------------
-- LLM_MoE body — router top-k + SwiGLU experts (via LLM_Weight matvec)
--
-- One expert FFN is SwiGLU:  y = Down . ( silu(Gate . x) * (Up . x) ).
-- Routed experts are 3D weights (per-expert matvec); the shared expert
-- and router are 2D matvecs. Weights are dense (tests) or quantized.
---------------------------------------------------------------------

with Ada.Numerics.Generic_Elementary_Functions;
with LLM_Tensor; use LLM_Tensor;
with LLM_Weight; use LLM_Weight;
with LLM_Pool;
with LLM_GPU;

package body LLM_MoE is

   procedure Drop_W (W : in out LLM_Weight.Weight) is
   begin
      LLM_GPU.Free_Weight (LLM_Weight.Raw_Address (W));
      LLM_Weight.Free_Bytes (W);
   end Drop_W;

   procedure Free (M : in out MoE_Layer) is
   begin
      Drop_W (M.Gate_Inp_W);
      Drop_W (M.Gate_Exp_W);   Drop_W (M.Up_W);   Drop_W (M.Down_W);
      Drop_W (M.Shexp_Gate_W); Drop_W (M.Shexp_Up_W); Drop_W (M.Shexp_Down_W);
   end Free;

   --  Hot per-token kernel (router + 8 experts); indices derive from dims.
   pragma Suppress (All_Checks);

   package Float_Math is new Ada.Numerics.Generic_Elementary_Functions (Float);
   use Float_Math;

   function Silu (X : Float) return Float is (X / (1.0 + Exp (-X)));

   function Create_MoE
     (Gate_Inp_W, Gate_Exp_W, Up_W, Down_W,
      Shexp_Gate_W, Shexp_Up_W, Shexp_Down_W : Weight;
      Shexp_Gate_Inp_W : Tensor;
      N_Experts : Integer)
      return MoE_Layer
   is
      M : MoE_Layer;
   begin
      M.Gate_Inp_W := Gate_Inp_W;  M.Gate_Exp_W := Gate_Exp_W;
      M.Up_W := Up_W;  M.Down_W := Down_W;
      M.Shexp_Gate_W := Shexp_Gate_W;  M.Shexp_Up_W := Shexp_Up_W;
      M.Shexp_Down_W := Shexp_Down_W;  M.Shexp_Gate_Inp_W := Shexp_Gate_Inp_W;

      M.N_Experts := N_Experts;
      M.Dim       := Cols (Gate_Inp_W);
      M.Top_K     := Integer'Min (8, N_Experts);
      M.Intermed  := Expert_Out (Gate_Exp_W);
      return M;
   end Create_MoE;

   function Forward (M : MoE_Layer; X : Tensor) return Tensor is
      Dim      : constant Integer := M.Dim;
      Intermed : constant Integer := M.Intermed;
      N_Exp    : constant Integer := M.N_Experts;
      Top_K    : constant Integer := M.Top_K;

      Router  : array (1 .. N_Exp) of Float;
      Top_Idx : array (1 .. Top_K) of Integer := [others => 1];
      Top_W   : array (1 .. Top_K) of Float := [others => 0.0];

      Result : Tensor := New_Tensor ([1, Dim]);

      --  SwiGLU hidden = silu(gate) * up, given the two [1,intermed] projections.
      function Hidden (Gate_P, Up_P : Tensor) return Tensor is
      begin
         return H : Tensor := New_Tensor ([1, Intermed]) do
            for J in 1 .. Intermed loop
               Set_Flat (H, J, Silu (Get_Flat (Gate_P, J)) * Get_Flat (Up_P, J));
            end loop;
         end return;
      end Hidden;
   begin
      --  1. Router logits.
      declare
         RL : constant Tensor := MatVec (M.Gate_Inp_W, X);
      begin
         for E in 1 .. N_Exp loop
            Router (E) := Get_Flat (RL, E);
         end loop;
      end;

      --  2. Softmax (numerically stable).
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

      --  3. Greedy top-k + renormalise.
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

      --  4. Selected experts — run the (independent) experts across CPUs,
      --     each into its own row of Exp_Y, then reduce.
      declare
         Exp_Y : Tensor := New_Tensor ([Top_K, Dim]);

         type Experts_Op is new LLM_Pool.Parallel_Op with null record;
         overriding procedure Execute (Op : in out Experts_Op; Lo, Hi : Integer) is
         begin
            for K in Lo .. Hi loop
               declare
                  E : constant Integer := Top_Idx (K);
                  H : constant Tensor := Hidden (MatVec_Expert (M.Gate_Exp_W, E, X),
                                                 MatVec_Expert (M.Up_W, E, X));
                  Y : constant Tensor := MatVec_Expert (M.Down_W, E, H);
               begin
                  for I in 1 .. Dim loop
                     Set (Exp_Y, [K, I], Top_W (K) * Get_Flat (Y, I));
                  end loop;
               end;
            end loop;
         end Execute;

         Experts : Experts_Op;
      begin
         LLM_Pool.Run (Experts, 1, Top_K, Min_Grain => 2);
         for K in 1 .. Top_K loop
            for I in 1 .. Dim loop
               Set_Flat (Result, I, Get_Flat (Result, I) + Get (Exp_Y, [K, I]));
            end loop;
         end loop;
      end;

      --  5. Shared expert (optional sigmoid gate).
      declare
         H : constant Tensor := Hidden (MatVec (M.Shexp_Gate_W, X),
                                        MatVec (M.Shexp_Up_W, X));
         Y : constant Tensor := MatVec (M.Shexp_Down_W, H);
         Shared_Gate : Float := 1.0;
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
         for I in 1 .. Dim loop
            Set_Flat (Result, I, Get_Flat (Result, I) + Shared_Gate * Get_Flat (Y, I));
         end loop;
      end;

      return Result;
   end Forward;

end LLM_MoE;
