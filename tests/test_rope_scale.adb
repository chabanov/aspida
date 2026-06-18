------------------------------------------------------------------------
-- test_rope_scale — linear RoPE scaling (Position Interpolation).
--  * Freq_Scale 1.0 is a byte-for-byte no-op (regression guard);
--  * Set_Linear_Scale(2.0) halves the angle, so the scaled rotation at
--    position 4 equals the unscaled rotation at position 2.
------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Numerics.Elementary_Functions; use Ada.Numerics.Elementary_Functions;
with LLM_RoPE;
with LLM_Tensor; use LLM_Tensor;

procedure Test_RoPE_Scale is
   Pass : Boolean := True;
   Dim  : constant := 8;

   procedure Check (Name : String; Cond : Boolean) is
   begin
      if Cond then Put_Line ("  PASS: " & Name);
      else Put_Line ("  FAIL: " & Name); Pass := False; end if;
   end Check;

   function Close (A, B : Tensor) return Boolean is
   begin
      for I in 1 .. Dim loop
         if abs (Get_Flat (A, I) - Get_Flat (B, I)) > 1.0e-5 then
            return False;
         end if;
      end loop;
      return True;
   end Close;

   X : Tensor := New_Tensor ([1, Dim]);
begin
   Put_Line ("=== Linear RoPE scaling (Position Interpolation) ===");
   for I in 1 .. Dim loop
      Set_Flat (X, I, 0.1 * Float (I) - 0.3);
   end loop;

   declare
      Base : constant LLM_RoPE.RoPE_Params :=
        LLM_RoPE.Create_Qwen_RoPE (Dim, 10_000.0, 1000);
      Scaled, NoOp : LLM_RoPE.RoPE_Params := Base;
   begin
      LLM_RoPE.Set_Linear_Scale (Scaled, 2.0);   -- Freq_Scale := 0.5
      LLM_RoPE.Set_Linear_Scale (NoOp,   1.0);   -- no-op

      Check ("factor 1.0 is a no-op",
             Close (LLM_RoPE.Apply (NoOp, X, 4), LLM_RoPE.Apply (Base, X, 4)));

      Check ("factor 2.0: scaled@4 == unscaled@2",
             Close (LLM_RoPE.Apply (Scaled, X, 4), LLM_RoPE.Apply (Base, X, 2)));

      Check ("factor 2.0: scaled@8 == unscaled@4",
             Close (LLM_RoPE.Apply (Scaled, X, 8), LLM_RoPE.Apply (Base, X, 4)));

      Check ("scaled differs from unscaled at the same position",
             not Close (LLM_RoPE.Apply (Scaled, X, 4), LLM_RoPE.Apply (Base, X, 4)));
   end;

   --  NTK-aware base scaling.
   declare
      Base : constant LLM_RoPE.RoPE_Params :=
        LLM_RoPE.Create_Qwen_RoPE (Dim, 10_000.0, 1000);
      NTK1, NTK4 : LLM_RoPE.RoPE_Params := Base;
      Expect : constant Float :=
        10_000.0 * 4.0 ** (Float (Dim) / Float (Dim - 2));
   begin
      LLM_RoPE.Set_NTK_Scale (NTK1, 1.0);   -- no-op
      LLM_RoPE.Set_NTK_Scale (NTK4, 4.0);
      Check ("NTK factor 1.0 leaves base unchanged",
             abs (NTK1.Freq_Base - 10_000.0) < 1.0e-3);
      Check ("NTK factor 4.0 scales base by 4^(d/(d-2))",
             abs (NTK4.Freq_Base - Expect) < 1.0);
      Check ("NTK factor 1.0 is a no-op rotation",
             Close (LLM_RoPE.Apply (NTK1, X, 5), LLM_RoPE.Apply (Base, X, 5)));
      Check ("NTK factor 4.0 changes the rotation",
             not Close (LLM_RoPE.Apply (NTK4, X, 5), LLM_RoPE.Apply (Base, X, 5)));
   end;

   --  Full YaRN.
   declare
      Base : constant LLM_RoPE.RoPE_Params :=
        LLM_RoPE.Create_Qwen_RoPE (Dim, 10_000.0, 1000);
      Y1, Y2 : LLM_RoPE.RoPE_Params := Base;
   begin
      LLM_RoPE.Set_Yarn_Scale (Y1, 1.0, 256);    -- no-op (factor <= 1)
      LLM_RoPE.Set_Yarn_Scale (Y2, 4.0, 256);

      Check ("YaRN factor 1.0 is a no-op (== standard RoPE)",
             (not Y1.Yarn_On)
             and then Close (LLM_RoPE.Apply (Y1, X, 7), LLM_RoPE.Apply (Base, X, 7)));

      Check ("YaRN enabled at factor 4.0", Y2.Yarn_On);
      Check ("YaRN M_Scale = 1 + 0.1*ln(factor)",
             abs (Y2.M_Scale - (1.0 + 0.1 * Log (4.0))) < 1.0e-5);
      Check ("YaRN correction dims in range [0, Dim-1] and ordered",
             Y2.Corr_Low >= 0.0 and then Y2.Corr_High <= Float (Dim - 1)
             and then Y2.Corr_Low <= Y2.Corr_High);
      Check ("YaRN factor 4.0 changes the rotation",
             not Close (LLM_RoPE.Apply (Y2, X, 7), LLM_RoPE.Apply (Base, X, 7)));
      --  YaRN extrapolates high-freq dims (no position scaling there), so it
      --  must NOT equal pure linear PI, which scales every dim uniformly.
      declare
         Lin : LLM_RoPE.RoPE_Params := Base;
      begin
         LLM_RoPE.Set_Linear_Scale (Lin, 4.0);
         Check ("YaRN differs from pure-linear (ramp is active)",
                not Close (LLM_RoPE.Apply (Y2, X, 7), LLM_RoPE.Apply (Lin, X, 7)));
      end;
   end;

   New_Line;
   if Pass then Put_Line ("RESULT: PASS"); else Put_Line ("RESULT: FAIL"); end if;
end Test_RoPE_Scale;
