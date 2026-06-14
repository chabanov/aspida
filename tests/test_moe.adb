---------------------------------------------------------------------
-- Test MoE (LLM_MoE) — router top-k + SwiGLU experts + shared expert
--
-- Cross-checks LLM_MoE.Forward against an independent reference that
-- recomputes the same math (softmax routing, greedy top-k, SwiGLU
-- experts, shared expert) over the same synthetic weights, for both
-- Top_K = 2 (all experts) and Top_K = 1 (selection).
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;
with LLM_Tensor; use LLM_Tensor;
with LLM_Weight;
with LLM_MoE;

procedure Test_MoE is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Condition : Boolean) is
   begin
      if Condition then
         Put_Line ("  PASS: " & Name);
         Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);
         Failed := Failed + 1;
      end if;
   end Assert;

   function Close (A, B : Float; Tol : Float := 1.0e-4) return Boolean is
   begin
      return abs (A - B) < Tol;
   end Close;

   function Silu (X : Float) return Float is
   begin
      return X / (1.0 + Exp (-X));
   end Silu;

   N_Exp    : constant := 2;
   Dim      : constant := 2;
   Intermed : constant := 2;

   --  Deterministic, distinct-per-seed 2D weight filler.
   function Mk (Rows, Cols, Seed : Integer) return Tensor is
      T : Tensor := New_Tensor ([Rows, Cols]);
   begin
      for R in 1 .. Rows loop
         for C in 1 .. Cols loop
            Set (T, [R, C], 0.1 * Float (((R * 7 + C * 3 + Seed) mod 11) - 5));
         end loop;
      end loop;
      return T;
   end Mk;

   --  Deterministic 3D filler (matches the real [expert, *, *] layout).
   function Mk3 (D1, D2, D3, Seed : Integer) return Tensor is
      T : Tensor := New_Tensor ([D1, D2, D3]);
   begin
      for A in 1 .. D1 loop
         for B in 1 .. D2 loop
            for C in 1 .. D3 loop
               Set (T, [A, B, C],
                 0.1 * Float (((A * 13 + B * 7 + C * 3 + Seed) mod 11) - 5));
            end loop;
         end loop;
      end loop;
      return T;
   end Mk3;

   Gate_Inp : constant Tensor := Mk (N_Exp, Dim, 1);                  -- [n_exp, dim]
   Gate_Exp : constant Tensor := Mk3 (N_Exp, Intermed, Dim, 2);       -- [e, ff, dim]
   Up_W     : constant Tensor := Mk3 (N_Exp, Intermed, Dim, 3);       -- [e, ff, dim]
   Down_W   : constant Tensor := Mk3 (N_Exp, Dim, Intermed, 4);       -- [e, dim, ff]
   Sh_Gate  : constant Tensor := Mk (Intermed, Dim, 5);              -- [ff, dim]
   Sh_Up    : constant Tensor := Mk (Intermed, Dim, 6);
   Sh_Down  : constant Tensor := Mk (Dim, Intermed, 7);             -- [dim, ff]
   Sh_GInp  : constant Tensor := New_Tensor ([1, 1]);               -- dummy => no gate

   X : Tensor := New_Tensor ([1, Dim]);

   --  Independent reference re-implementation (mirrors the math in llm_moe.adb).
   type Vec is array (1 .. Dim) of Float;
   type HVec is array (1 .. Intermed) of Float;

   --  Reference for expert E with 3D weights: Gate/Up [e,ff,dim], Down [e,dim,ff].
   function Ref_SwiGLU (Gate, Up, Down : Tensor; E : Integer) return Vec is
      H : HVec;
      Y : Vec := [others => 0.0];
   begin
      for J in 1 .. Intermed loop
         declare
            G : Float := 0.0;
            U : Float := 0.0;
         begin
            for D in 1 .. Dim loop
               G := G + Get (Gate, [E, J, D]) * Get_Flat (X, D);
               U := U + Get (Up,   [E, J, D]) * Get_Flat (X, D);
            end loop;
            H (J) := Silu (G) * U;
         end;
      end loop;
      for I in 1 .. Dim loop
         for J in 1 .. Intermed loop
            Y (I) := Y (I) + Get (Down, [E, I, J]) * H (J);
         end loop;
      end loop;
      return Y;
   end Ref_SwiGLU;

   --  Reference for the (2D) shared expert: Gate/Up [ff,dim], Down [dim,ff].
   function Ref_Shared (Gate, Up, Down : Tensor) return Vec is
      H : HVec;
      Y : Vec := [others => 0.0];
   begin
      for J in 1 .. Intermed loop
         declare
            G : Float := 0.0;
            U : Float := 0.0;
         begin
            for D in 1 .. Dim loop
               G := G + Get (Gate, [J, D]) * Get_Flat (X, D);
               U := U + Get (Up,   [J, D]) * Get_Flat (X, D);
            end loop;
            H (J) := Silu (G) * U;
         end;
      end loop;
      for I in 1 .. Dim loop
         for J in 1 .. Intermed loop
            Y (I) := Y (I) + Get (Down, [I, J]) * H (J);
         end loop;
      end loop;
      return Y;
   end Ref_Shared;

   function Reference (Top_K : Integer) return Vec is
      P     : array (1 .. N_Exp) of Float;
      Used  : array (1 .. N_Exp) of Boolean := [others => False];
      Max_L : Float := Float'First;
      Sum_E : Float := 0.0;
      Sum_W : Float := 0.0;
      Out_V : Vec := [others => 0.0];
   begin
      --  Router softmax
      for E in 1 .. N_Exp loop
         declare
            S : Float := 0.0;
         begin
            for D in 1 .. Dim loop
               S := S + Get (Gate_Inp, [E, D]) * Get_Flat (X, D);
            end loop;
            P (E) := S;
            if S > Max_L then
               Max_L := S;
            end if;
         end;
      end loop;
      for E in 1 .. N_Exp loop
         P (E) := Exp (P (E) - Max_L);
         Sum_E := Sum_E + P (E);
      end loop;
      for E in 1 .. N_Exp loop
         P (E) := P (E) / Sum_E;
      end loop;

      --  Greedy top-k + renormalise, accumulating the experts
      declare
         Idx : array (1 .. Top_K) of Integer;
         Wgt : array (1 .. Top_K) of Float;
      begin
         for K in 1 .. Top_K loop
            declare
               Best_E : Integer := 0;
               Best_V : Float := Float'First;
            begin
               for E in 1 .. N_Exp loop
                  if not Used (E) and then P (E) > Best_V then
                     Best_V := P (E);
                     Best_E := E;
                  end if;
               end loop;
               Idx (K) := Best_E;
               Wgt (K) := Best_V;
               Used (Best_E) := True;
               Sum_W := Sum_W + Best_V;
            end;
         end loop;
         for K in 1 .. Top_K loop
            declare
               E : constant Integer := Idx (K);
               W : constant Float := Wgt (K) / Sum_W;
               Y : constant Vec := Ref_SwiGLU (Gate_Exp, Up_W, Down_W, E);
            begin
               for I in 1 .. Dim loop
                  Out_V (I) := Out_V (I) + W * Y (I);
               end loop;
            end;
         end loop;
      end;

      --  Shared expert (dummy gate input => ungated)
      declare
         Y : constant Vec := Ref_Shared (Sh_Gate, Sh_Up, Sh_Down);
      begin
         for I in 1 .. Dim loop
            Out_V (I) := Out_V (I) + Y (I);
         end loop;
      end;

      return Out_V;
   end Reference;

   M : LLM_MoE.MoE_Layer :=
     LLM_MoE.Create_MoE
       (LLM_Weight.From_Dense (Gate_Inp), LLM_Weight.From_Dense (Gate_Exp),
        LLM_Weight.From_Dense (Up_W), LLM_Weight.From_Dense (Down_W),
        LLM_Weight.From_Dense (Sh_Gate), LLM_Weight.From_Dense (Sh_Up),
        LLM_Weight.From_Dense (Sh_Down), Sh_GInp, N_Exp);

begin
   Put_Line ("=== MoE Test Suite ===");
   New_Line;

   Set_Flat (X, 1, 0.7);
   Set_Flat (X, 2, -0.2);

   --  Sanity: Create_MoE recovered the dimensions.
   Assert ("dim recovered", M.Dim = Dim);
   Assert ("intermed recovered", M.Intermed = Intermed);
   Assert ("top_k clamped to n_experts", M.Top_K = N_Exp);

   --  Test 1: Top_K = 2 (both experts) — full SwiGLU + routing + shared.
   declare
      Expected : constant Vec := Reference (2);
      Got      : constant Tensor := LLM_MoE.Forward (M, X);
   begin
      for I in 1 .. Dim loop
         Assert ("topk=2 out dim" & Integer'Image (I),
           Close (Get_Flat (Got, I), Expected (I)));
      end loop;
   end;

   --  Test 2: Top_K = 1 — only the highest-probability expert is used.
   M.Top_K := 1;
   declare
      Expected : constant Vec := Reference (1);
      Got      : constant Tensor := LLM_MoE.Forward (M, X);
   begin
      for I in 1 .. Dim loop
         Assert ("topk=1 out dim" & Integer'Image (I),
           Close (Get_Flat (Got, I), Expected (I)));
      end loop;
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_MoE;
