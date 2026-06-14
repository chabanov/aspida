---------------------------------------------------------------------
-- Test LLM_DeltaNet_Blk — full gated delta-net layer
--
-- Validates the whole layer (in-proj, causal conv, per-head β/g, delta
-- recurrence, gated RMSNorm, out-proj) on a small synthetic layer:
--   A. Output shape is [seq, dim].
--   B. Causality: appending a later token does not change earlier
--      outputs (proves conv1d causality + recurrence carry are correct).
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_Tensor; use LLM_Tensor;
with LLM_Weight;
with LLM_DeltaNet_Blk;

procedure Test_DeltaNet_Blk is
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

   --  Small config: dim=4, head_dim=2, 2 k-heads, 4 v-heads (repeat 2).
   Dim : constant := 4;
   HD  : constant := 2;   -- key/value head dim
   NK  : constant := 2;   -- k-heads
   NV  : constant := 4;   -- v-heads
   QO  : constant := NK * HD + NK * HD + NV * HD;  -- q+k+v = 4+4+8 = 16
   VD  : constant := NV * HD;                      -- 8

   function Mk2 (Rows, Cols, Seed : Integer) return Tensor is
      T : Tensor := New_Tensor ([Rows, Cols]);
   begin
      for R in 1 .. Rows loop
         for C in 1 .. Cols loop
            Set (T, [R, C], 0.1 * Float (((R * 7 + C * 3 + Seed) mod 11) - 5));
         end loop;
      end loop;
      return T;
   end Mk2;

   function Mk1 (N, Seed : Integer) return Tensor is
      T : Tensor := New_Tensor ([1, N]);
   begin
      for I in 1 .. N loop
         Set_Flat (T, I, 0.1 * Float (((I * 5 + Seed) mod 9) - 4));
      end loop;
      return T;
   end Mk1;

   function W (T : Tensor) return LLM_Weight.Weight is (LLM_Weight.From_Dense (T));

   Layer : constant LLM_DeltaNet_Blk.DeltaNet_Layer :=
     LLM_DeltaNet_Blk.Create
       (QKV_W   => W (Mk2 (QO, Dim, 1)),   -- [qkv_out, dim]
        Conv_W  => Mk2 (QO, 4, 2),         -- [qkv_out, kernel]
        A_W     => Mk1 (NV, 3),            -- [n_v_heads]
        Dt_W    => Mk1 (NV, 4),            -- [n_v_heads]
        Alpha_W => W (Mk2 (NV, Dim, 5)),   -- [n_v_heads, dim]
        Beta_W  => W (Mk2 (NV, Dim, 6)),   -- [n_v_heads, dim]
        Norm_W  => Mk1 (HD, 7),            -- [value_head_dim]
        Out_W   => W (Mk2 (Dim, VD, 8)),   -- [dim, v_dim]
        Gate_W  => W (Mk2 (VD, Dim, 9)));  -- [v_dim, dim]

begin
   Put_Line ("=== DeltaNet Layer Test Suite ===");
   New_Line;

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

      O3 := LLM_DeltaNet_Blk.Forward (Layer, X3);
      O2 := LLM_DeltaNet_Blk.Forward (Layer, X2);

      Assert ("output shape [3, dim]",
        Rank (O3) = 2 and then Shape (O3) (1) = 3 and then Shape (O3) (2) = Dim);

      --  Causality: positions 1..2 identical with or without token 3.
      for T in 1 .. 2 loop
         for I in 1 .. Dim loop
            Assert ("causal: pos" & Integer'Image (T) & " dim" & Integer'Image (I),
              Close (Get (O3, [T, I]), Get (O2, [T, I])));
         end loop;
      end loop;
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_DeltaNet_Blk;
