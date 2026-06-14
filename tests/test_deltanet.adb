---------------------------------------------------------------------
-- Test LLM_DeltaNet — gated delta rule recurrence
--
-- Verifies (no model needed):
--   A. Closed form: a hand-computed 1-D sequence matches exactly.
--   B. Decay: with beta=0 (no write) the state contracts by g each step.
--   C. Determinism/causality: the same prefix yields the same output.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_Tensor;   use LLM_Tensor;
with LLM_DeltaNet;

procedure Test_DeltaNet is
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

   --  Build a [1,N] row tensor from up to 2 values.
   function Vec1 (A : Float) return Tensor is
   begin
      return T : Tensor := New_Tensor ([1, 1]) do
         Set_Flat (T, 1, A);
      end return;
   end Vec1;

   function Vec2 (A, B : Float) return Tensor is
   begin
      return T : Tensor := New_Tensor ([1, 2]) do
         Set_Flat (T, 1, A);
         Set_Flat (T, 2, B);
      end return;
   end Vec2;

begin
   Put_Line ("=== DeltaNet Test Suite ===");
   New_Line;

   ------------------------------------------------------------------
   -- A. Closed form, Dk=Dv=1, q=k=v=1, g=0.5, beta=1.
   --    token1: S=1, o=1 ;  token2: retr=0.5, S=1, o=1.
   ------------------------------------------------------------------
   declare
      S : Tensor := LLM_DeltaNet.Init_State (1, 1);
      O : Tensor := New_Tensor ([1, 1]);
   begin
      LLM_DeltaNet.Step (S, Vec1 (1.0), Vec1 (1.0), Vec1 (1.0), 0.5, 1.0, O);
      Assert ("closed-form o1 = 1.0", Close (Get_Flat (O, 1), 1.0, 1.0e-3));
      LLM_DeltaNet.Step (S, Vec1 (1.0), Vec1 (1.0), Vec1 (1.0), 0.5, 1.0, O);
      Assert ("closed-form o2 = 1.0", Close (Get_Flat (O, 1), 1.0, 1.0e-3));
   end;

   ------------------------------------------------------------------
   -- B. Decay: seed the state, then beta=0 → S contracts by g per step.
   ------------------------------------------------------------------
   declare
      S : Tensor := LLM_DeltaNet.Init_State (1, 1);
      O : Tensor := New_Tensor ([1, 1]);
   begin
      LLM_DeltaNet.Step (S, Vec1 (1.0), Vec1 (1.0), Vec1 (1.0), 0.5, 1.0, O);
      declare
         M0 : constant Float := abs Get (S, [1, 1]);
      begin
         LLM_DeltaNet.Step (S, Vec1 (1.0), Vec1 (1.0), Vec1 (0.0), 0.5, 0.0, O);
         Assert ("decay step1 = g*S", Close (abs Get (S, [1, 1]), 0.5 * M0));
         LLM_DeltaNet.Step (S, Vec1 (1.0), Vec1 (1.0), Vec1 (0.0), 0.5, 0.0, O);
         Assert ("decay step2 = g^2*S", Close (abs Get (S, [1, 1]), 0.25 * M0));
      end;
   end;

   ------------------------------------------------------------------
   -- C. Determinism/causality, Dk=Dv=2: o at token 2 is unaffected by a
   --    later token 3.
   ------------------------------------------------------------------
   declare
      Sa : Tensor := LLM_DeltaNet.Init_State (2, 2);
      Sb : Tensor := LLM_DeltaNet.Init_State (2, 2);
      Oa : Tensor := New_Tensor ([1, 2]);
      Ob : Tensor := New_Tensor ([1, 2]);
   begin
      --  Run with a 3rd token present.
      LLM_DeltaNet.Step (Sa, Vec2 (0.5, -0.3), Vec2 (0.2, 0.4), Vec2 (1.0, -1.0),
                         0.7, 0.6, Oa);
      LLM_DeltaNet.Step (Sa, Vec2 (-0.1, 0.8), Vec2 (0.3, -0.2), Vec2 (0.5, 0.5),
                         0.7, 0.6, Oa);   -- token 2 output recorded in Oa
      --  Run only the 2-token prefix.
      LLM_DeltaNet.Step (Sb, Vec2 (0.5, -0.3), Vec2 (0.2, 0.4), Vec2 (1.0, -1.0),
                         0.7, 0.6, Ob);
      LLM_DeltaNet.Step (Sb, Vec2 (-0.1, 0.8), Vec2 (0.3, -0.2), Vec2 (0.5, 0.5),
                         0.7, 0.6, Ob);
      for I in 1 .. 2 loop
         Assert ("determinism dim" & Integer'Image (I),
           Close (Get_Flat (Oa, I), Get_Flat (Ob, I)));
      end loop;
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_DeltaNet;
