---------------------------------------------------------------------
-- Test LLM_FullAttn — Qwen3-Next gated full-attention layer
--
-- Small synthetic layer (dim=4, head_dim=4, 2 q-heads, 1 kv-head,
-- rope_dim=2). Validates output shape and causality (a later token does
-- not change earlier outputs) through the whole layer: q/gate split,
-- QK-norm, partial RoPE, causal GQA softmax, sigmoid gate, o-proj.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_Tensor; use LLM_Tensor;
with LLM_RoPE;
with LLM_Weight;
with LLM_FullAttn;

procedure Test_FullAttn is
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

   Dim : constant := 4;
   HD  : constant := 4;   -- head dim
   NQ  : constant := 2;   -- q heads
   NKV : constant := 1;   -- kv heads
   QO  : constant := NQ * 2 * HD;   -- 16 (query + gate)
   KO  : constant := NKV * HD;      -- 4
   AO  : constant := NQ * HD;       -- 8

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
         Set_Flat (T, I, 0.5 + 0.1 * Float (((I * 5 + Seed) mod 7)));
      end loop;
      return T;
   end Mk1;

   function Make_RoPE return LLM_RoPE.RoPE_Params is
      R : LLM_RoPE.RoPE_Params;
   begin
      R.Dim       := 2;        -- partial: rotate 2 of head_dim 4
      R.Freq_Base := 10_000.0;
      R.Max_Pos   := 1_000;
      R.Sections  := New_Tensor ([1, 4]);
      return R;
   end Make_RoPE;

   function W (T : Tensor) return LLM_Weight.Weight is (LLM_Weight.From_Dense (T));

   Layer : constant LLM_FullAttn.Full_Attn_Layer :=
     LLM_FullAttn.Create
       (Q_W    => W (Mk2 (QO, Dim, 1)),   -- [n_q*2*head_dim, dim]
        K_W    => W (Mk2 (KO, Dim, 2)),
        V_W    => W (Mk2 (KO, Dim, 3)),
        Q_Norm => Mk1 (HD, 4),
        K_Norm => Mk1 (HD, 5),
        O_W    => W (Mk2 (Dim, AO, 6)),   -- [dim, n_q*head_dim]
        RoPE   => Make_RoPE);

begin
   Put_Line ("=== Full-Attention Layer Test Suite ===");
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

      O3 := LLM_FullAttn.Forward (Layer, X3);
      O2 := LLM_FullAttn.Forward (Layer, X2);

      Assert ("output shape [3, dim]",
        Rank (O3) = 2 and then Shape (O3) (1) = 3 and then Shape (O3) (2) = Dim);

      for T in 1 .. 2 loop
         for I in 1 .. Dim loop
            Assert ("causal: pos" & Integer'Image (T) & " dim" & Integer'Image (I),
              Close (Get (O3, [T, I]), Get (O2, [T, I])));
         end loop;
      end loop;

      --  Incremental decode (KV cache) == batched Forward.
      declare
         St : LLM_FullAttn.Attn_State := LLM_FullAttn.Init_State (Layer, 3);
      begin
         for T in 1 .. 3 loop
            declare
               Xt : Tensor := New_Tensor ([1, Dim]);
            begin
               for I in 1 .. Dim loop
                  Set_Flat (Xt, I, Get (X3, [T, I]));
               end loop;
               declare
                  Ot : constant Tensor := LLM_FullAttn.Step (Layer, St, Xt);
               begin
                  for I in 1 .. Dim loop
                     Assert ("incremental==batched: pos" & Integer'Image (T)
                       & " dim" & Integer'Image (I),
                       Close (Get_Flat (Ot, I), Get (O3, [T, I])));
                  end loop;
               end;
            end;
         end loop;
      end;
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_FullAttn;
