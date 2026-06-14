---------------------------------------------------------------------
-- Test LLM_Qwen_Blk — assembled hybrid block (integration)
--
-- Builds a small delta-net block (RMSNorm -> delta-net -> residual ->
-- RMSNorm -> MoE -> residual) from synthetic weights and checks the
-- assembled block is causal and shape-preserving. Validates the block
-- wiring (layer dispatch, residuals, per-row norm/MoE).
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_Tensor; use LLM_Tensor;
with LLM_Weight;
with LLM_MoE;
with LLM_DeltaNet_Blk;
with LLM_Qwen_Blk;

procedure Test_Block is
   use Ada.Text_IO;

   Passed : Natural := 0;
   Failed : Natural := 0;

   procedure Assert (Name : String; Condition : Boolean) is
   begin
      if Condition then
         Put_Line ("  PASS: " & Name);  Passed := Passed + 1;
      else
         Put_Line ("  FAIL: " & Name);  Failed := Failed + 1;
      end if;
   end Assert;

   function Close (A, B : Float; Tol : Float := 1.0e-5) return Boolean is
     (abs (A - B) < Tol);

   Dim : constant := 4;
   HD  : constant := 2;   -- delta-net head dim
   NK  : constant := 2;
   NV  : constant := 4;
   QO  : constant := NK * HD + NK * HD + NV * HD;  -- 16
   N_Exp    : constant := 2;
   Intermed : constant := 2;

   function Mk2 (R, C, Seed : Integer) return Tensor is
      T : Tensor := New_Tensor ([R, C]);
   begin
      for A in 1 .. R loop
         for B in 1 .. C loop
            Set (T, [A, B], 0.1 * Float (((A * 7 + B * 3 + Seed) mod 11) - 5));
         end loop;
      end loop;
      return T;
   end Mk2;

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

   function Mk1 (N, Seed : Integer) return Tensor is
      T : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop
         Set_Flat (T, I, 0.5 + 0.1 * Float (((I * 5 + Seed) mod 7)));
      end loop;
      return T;
   end Mk1;

   function W (T : Tensor) return LLM_Weight.Weight is (LLM_Weight.From_Dense (T));

   Blk : LLM_Qwen_Blk.Qwen_Block;

begin
   Put_Line ("=== Block Integration Test Suite ===");
   New_Line;

   Blk.Is_Full_Attn     := False;
   Blk.Dim              := Dim;
   Blk.Attn_Norm_W      := Mk1 (Dim, 1);
   Blk.Post_Attn_Norm_W := Mk1 (Dim, 2);
   Blk.DNet := LLM_DeltaNet_Blk.Create
     (QKV_W   => W (Mk2 (QO, Dim, 1)),
      Conv_W  => Mk2 (QO, 4, 2),
      A_W     => Mk1 (NV, 3),
      Dt_W    => Mk1 (NV, 4),
      Alpha_W => W (Mk2 (NV, Dim, 5)),
      Beta_W  => W (Mk2 (NV, Dim, 6)),
      Norm_W  => Mk1 (HD, 7),
      Out_W   => W (Mk2 (Dim, NV * HD, 8)),
      Gate_W  => W (Mk2 (NV * HD, Dim, 9)));
   Blk.MoE := LLM_MoE.Create_MoE
     (W (Mk2 (N_Exp, Dim, 1)),
      W (Mk3 (N_Exp, Intermed, Dim, 2)),
      W (Mk3 (N_Exp, Intermed, Dim, 3)),
      W (Mk3 (N_Exp, Dim, Intermed, 4)),
      W (Mk2 (Intermed, Dim, 5)),
      W (Mk2 (Intermed, Dim, 6)),
      W (Mk2 (Dim, Intermed, 7)),
      New_Tensor ([1, 1]),
      N_Exp);

   declare
      X3 : Tensor := New_Tensor ([3, Dim]);
      X2 : Tensor := New_Tensor ([2, Dim]);
      O3, O2 : Tensor;
   begin
      for T in 1 .. 3 loop
         for I in 1 .. Dim loop
            Set (X3, [T, I], 0.3 * Float (T) - 0.2 * Float (I));
         end loop;
      end loop;
      for T in 1 .. 2 loop
         for I in 1 .. Dim loop
            Set (X2, [T, I], Get (X3, [T, I]));
         end loop;
      end loop;

      O3 := LLM_Qwen_Blk.Forward (Blk, X3);
      O2 := LLM_Qwen_Blk.Forward (Blk, X2);

      Assert ("block output shape [3, dim]",
        Rank (O3) = 2 and then Shape (O3) (1) = 3 and then Shape (O3) (2) = Dim);
      for T in 1 .. 2 loop
         for I in 1 .. Dim loop
            Assert ("block causal: pos" & Integer'Image (T) & " dim" & Integer'Image (I),
              Close (Get (O3, [T, I]), Get (O2, [T, I])));
         end loop;
      end loop;
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");
   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Block;
