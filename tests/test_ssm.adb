---------------------------------------------------------------------
-- Test LLM_SSM — diagonal selective state-space scan
--
-- Verifies recurrence properties without a real model:
--   A. Determinism: the same input prefix yields the same output.
--   B. State carry: a token's output depends on the carried history
--      (output differs from the same token run with a fresh state).
--   C. Zero input -> zero output.
--   D. Decay: under zero input the state magnitude strictly decreases
--      (A_bar = exp(dt*A) with A < 0).
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_Tensor; use LLM_Tensor;
with LLM_SSM;

procedure Test_SSM is
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

   function Close (A, B : Float; Tol : Float := 1.0e-5) return Boolean is
   begin
      return abs (A - B) < Tol;
   end Close;

   Dim : constant := 2;
   SD  : constant := 2;

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

   --  A < 0 so the discretised A_bar = exp(dt*A) is a contraction.
   function Make_A return Tensor is
      A : Tensor := New_Tensor ([1, SD]);
   begin
      Set_Flat (A, 1, -0.5);
      Set_Flat (A, 2, -1.0);
      return A;
   end Make_A;

   P : constant LLM_SSM.SSM_Params :=
     LLM_SSM.Create_SSM
       (Conv_W  => New_Tensor ([1, 1]),          -- unused
        A_D     => Make_A,                        -- [1, SD]
        Dt_B    => New_Tensor ([1, SD]),          -- zeros
        Gamma   => Mk (Dim, 2 * Dim, 2),          -- in_proj [dim, 2*dim]
        Out_W   => Mk (Dim, Dim, 3),              -- [dim, dim]
        Alpha_W => Mk (Dim, SD, 4),               -- B proj [dim, SD]
        Beta_W  => Mk (Dim, SD, 5));              -- C proj [dim, SD]

   procedure Advance (X : Tensor; S : in out Tensor) is
      R : constant Tensor := LLM_SSM.Forward (P, X, S);
      pragma Unreferenced (R);
   begin
      null;
   end Advance;

   function Mag (S : Tensor) return Float is
      M : Float := 0.0;
   begin
      for D in 1 .. Dim loop
         for N in 1 .. SD loop
            M := M + abs Get (S, [D, N]);
         end loop;
      end loop;
      return M;
   end Mag;

   X1, X2 : Tensor := New_Tensor ([1, Dim]);
   X0     : constant Tensor := New_Tensor ([1, Dim]);  -- all zeros

begin
   Put_Line ("=== SSM Test Suite ===");
   New_Line;

   Set_Flat (X1, 1, 3.0);  Set_Flat (X1, 2, -2.5);
   Set_Flat (X2, 1, 1.5);  Set_Flat (X2, 2, 0.9);
   --  X0 stays zero.

   -- A. Determinism: same prefix -> same output.
   declare
      Sa : Tensor := LLM_SSM.Init_State (SD);
      Sb : Tensor := LLM_SSM.Init_State (SD);
      Y2a, Y2b : Tensor;
   begin
      Advance (X1, Sa);
      Y2a := LLM_SSM.Forward (P, X2, Sa);
      Advance (X1, Sb);
      Y2b := LLM_SSM.Forward (P, X2, Sb);
      for I in 1 .. Dim loop
         Assert ("determinism dim" & Integer'Image (I),
           Close (Get_Flat (Y2a, I), Get_Flat (Y2b, I)));
      end loop;
   end;

   -- B. State carry affects the output.
   declare
      Sh : Tensor := LLM_SSM.Init_State (SD);
      Sf : Tensor := LLM_SSM.Init_State (SD);
      Yh, Yf : Tensor;
      Differs : Boolean := False;
   begin
      Advance (X1, Sh);                         -- history
      Yh := LLM_SSM.Forward (P, X2, Sh);
      Yf := LLM_SSM.Forward (P, X2, Sf);         -- fresh state
      for I in 1 .. Dim loop
         if not Close (Get_Flat (Yh, I), Get_Flat (Yf, I)) then
            Differs := True;
         end if;
      end loop;
      Assert ("history changes output", Differs);
   end;

   -- C. Zero input -> zero output.
   declare
      Sz : Tensor := LLM_SSM.Init_State (SD);
      Yz : constant Tensor := LLM_SSM.Forward (P, X0, Sz);
   begin
      for I in 1 .. Dim loop
         Assert ("zero in -> zero out dim" & Integer'Image (I),
           Close (Get_Flat (Yz, I), 0.0));
      end loop;
   end;

   -- D. State decays under zero input after a nonzero seed.
   declare
      St : Tensor := LLM_SSM.Init_State (SD);
      M1, M2, M3 : Float;
   begin
      Advance (X1, St);   M1 := Mag (St);
      Advance (X0, St);   M2 := Mag (St);
      Advance (X0, St);   M3 := Mag (St);
      Assert ("seed state non-zero", M1 > 1.0e-6);
      Assert ("decay step 1", M2 < M1);
      Assert ("decay step 2", M3 < M2);
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_SSM;
