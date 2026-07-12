---------------------------------------------------------------------
-- LLM_DeltaNet_Blk body
---------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;
with LLM_Tensor; use LLM_Tensor;
with LLM_Weight; use LLM_Weight;
with LLM_DeltaNet;
with LLM_GPU;
with LLM_Qwen_GPU;

package body LLM_DeltaNet_Blk is

   procedure Drop_W (W : in out LLM_Weight.Weight) is
   begin
      LLM_GPU.Free_Weight (LLM_Weight.Raw_Address (W));
      LLM_Weight.Free_Bytes (W);
   end Drop_W;

   procedure Free (L : in out DeltaNet_Layer) is
   begin
      Drop_W (L.QKV_W);  Drop_W (L.Alpha_W); Drop_W (L.Beta_W);
      Drop_W (L.Out_W);  Drop_W (L.Gate_W);
   end Free;

   --  Hot per-token kernel; loop indices derive from the layer's own dims, so
   --  range/overflow checks are redundant. Suppressed for the inner math.
   pragma Suppress (All_Checks);

   function Silu (X : Float) return Float is (X / (1.0 + Exp (-X)));

   function Softplus (X : Float) return Float is
   begin
      if X > 20.0 then
         return X;
      elsif X < -20.0 then
         return Exp (X);
      else
         return Log (1.0 + Exp (X));
      end if;
   end Softplus;

   function Create
     (QKV_W   : Weight;
      Conv_W  : Tensor;
      A_W     : Tensor;
      Dt_W    : Tensor;
      Alpha_W : Weight;
      Beta_W  : Weight;
      Norm_W  : Tensor;
      Out_W   : Weight;
      Gate_W  : Weight)
      return DeltaNet_Layer
   is
      L : DeltaNet_Layer;
   begin
      L.QKV_W := QKV_W;  L.Conv_W := Conv_W;  L.A_W := A_W;  L.Dt_W := Dt_W;
      L.Alpha_W := Alpha_W;  L.Beta_W := Beta_W;  L.Norm_W := Norm_W;
      L.Out_W := Out_W;  L.Gate_W := Gate_W;

      L.Dim            := Cols (QKV_W);
      L.QKV_Out        := Rows (QKV_W);
      L.N_V_Heads      := Numel (A_W);
      L.Value_Head_Dim := Numel (Norm_W);
      L.Key_Head_Dim   := L.Value_Head_Dim;
      L.V_Dim          := L.N_V_Heads * L.Value_Head_Dim;
      declare
         Q_Dim : constant Integer := (L.QKV_Out - L.V_Dim) / 2;
      begin
         L.N_K_Heads := Q_Dim / L.Key_Head_Dim;
      end;
      return L;
   end Create;

   function Forward (L : DeltaNet_Layer; X : Tensor) return Tensor is
      Seq    : constant Integer := Shape (X) (1);
      Dim    : constant Integer := L.Dim;
      QO     : constant Integer := L.QKV_Out;
      NV     : constant Integer := L.N_V_Heads;
      KHD    : constant Integer := L.Key_Head_Dim;
      VHD    : constant Integer := L.Value_Head_Dim;
      Q_Dim  : constant Integer := L.N_K_Heads * KHD;
      V_Dim  : constant Integer := L.V_Dim;
      Kernel : constant Integer := Shape (L.Conv_W) (2);

      CQ    : Tensor := New_Tensor ([Seq, QO]);     -- conv'd + SiLU qkv
      O_Full : Tensor := New_Tensor ([Seq, V_Dim]); -- gated-normed head outputs
      Out_T : Tensor := New_Tensor ([Seq, Dim]);

      type State_Arr is array (1 .. NV) of Tensor;
      S : State_Arr;

      function Row (T : Integer) return Tensor is
         R : Tensor := New_Tensor ([1, Dim]);
      begin
         for I in 1 .. Dim loop
            Set_Flat (R, I, Get (X, [T, I]));
         end loop;
         return R;
      end Row;
   begin
      --  1. In-projection (qkv) per token, then causal conv1d + SiLU.
      declare
         QKV : Tensor := New_Tensor ([Seq, QO]);
      begin
         for T in 1 .. Seq loop
            declare
               P : constant Tensor := MatVec (L.QKV_W, Row (T));
            begin
               for O in 1 .. QO loop
                  Set (QKV, [T, O], Get_Flat (P, O));
               end loop;
            end;
         end loop;

         for T in 1 .. Seq loop
            for C in 1 .. QO loop
               declare
                  Acc : Float := 0.0;
               begin
                  for K in 1 .. Kernel loop
                     declare
                        Src : constant Integer := T - Kernel + K;
                     begin
                        if Src >= 1 then
                           Acc := Acc + Get (QKV, [Src, C]) * Get (L.Conv_W, [C, K]);
                        end if;
                     end;
                  end loop;
                  Set (CQ, [T, C], Silu (Acc));
               end;
            end loop;
         end loop;
      end;

      --  2. Per-head delta recurrence over the sequence.
      for H in 1 .. NV loop
         S (H) := LLM_DeltaNet.Init_State (KHD, VHD);
      end loop;

      for T in 1 .. Seq loop
         declare
            Xt   : constant Tensor := Row (T);
            AR   : constant Tensor := MatVec (L.Alpha_W, Xt);   -- [1, NV]
            BR   : constant Tensor := MatVec (L.Beta_W, Xt);    -- [1, NV]
            Z    : constant Tensor := MatVec (L.Gate_W, Xt);    -- [1, V_Dim]
            Beta : array (1 .. NV) of Float;
            Gate : array (1 .. NV) of Float;
         begin
            for H in 1 .. NV loop
               Beta (H) := 1.0 / (1.0 + Exp (-Get_Flat (BR, H)));
               --  ssm_a is stored pre-transformed as -exp(A_log), so the
               --  per-step log-decay is g = ssm_a * softplus(alpha + dt_bias)
               --  and the multiplicative decay is exp(g). (Verified against
               --  llama: gate-0 = MUL(a_softplus, ssm_a), decay = exp(gate).)
               Gate (H) := Exp (Get_Flat (L.A_W, H)
                                * Softplus (Get_Flat (AR, H) + Get_Flat (L.Dt_W, H)));
            end loop;

            for H in 1 .. NV loop
               declare
                  K_Head : constant Integer := (H - 1) mod L.N_K_Heads + 1;
                  Q_Vec  : Tensor := New_Tensor ([1, KHD]);
                  K_Vec  : Tensor := New_Tensor ([1, KHD]);
                  V_Vec  : Tensor := New_Tensor ([1, VHD]);
                  O_Vec  : Tensor := New_Tensor ([1, VHD]);
               begin
                  for D in 1 .. KHD loop
                     Set_Flat (Q_Vec, D, Get (CQ, [T, (K_Head - 1) * KHD + D]));
                     Set_Flat (K_Vec, D, Get (CQ, [T, Q_Dim + (K_Head - 1) * KHD + D]));
                  end loop;
                  for D in 1 .. VHD loop
                     Set_Flat (V_Vec, D, Get (CQ, [T, 2 * Q_Dim + (H - 1) * VHD + D]));
                  end loop;

                  LLM_DeltaNet.Step (S (H), Q_Vec, K_Vec, V_Vec,
                                     Gate (H), Beta (H), O_Vec);

                  --  Gated RMSNorm over VHD dims, store into O_Full.
                  declare
                     SS : Float := 0.0;
                  begin
                     for D in 1 .. VHD loop
                        SS := SS + Get_Flat (O_Vec, D) ** 2;
                     end loop;
                     declare
                        Rms : constant Float := Sqrt (SS / Float (VHD) + 1.0e-6);
                     begin
                        for D in 1 .. VHD loop
                           Set (O_Full, [T, (H - 1) * VHD + D],
                                (Get_Flat (O_Vec, D) / Rms) * Get_Flat (L.Norm_W, D)
                                * Silu (Get_Flat (Z, (H - 1) * VHD + D)));
                        end loop;
                     end;
                  end;
               end;
            end loop;
         end;
      end loop;

      --  3. Output projection: out[t] = Out_W . O_Full[t].
      for T in 1 .. Seq loop
         declare
            O_Row : Tensor := New_Tensor ([1, V_Dim]);
         begin
            for J in 1 .. V_Dim loop
               Set_Flat (O_Row, J, Get (O_Full, [T, J]));
            end loop;
            declare
               Ot : constant Tensor := MatVec (L.Out_W, O_Row);
            begin
               for D in 1 .. Dim loop
                  Set (Out_T, [T, D], Get_Flat (Ot, D));
               end loop;
            end;
         end;
      end loop;

      return Out_T;
   end Forward;

   --------------------------------------------------------------------
   -- Incremental decode
   --------------------------------------------------------------------

   --  A projection is GPU-eligible when the resident kernels can read it:
   --  K-quant (Kind_Code >= 0) or raw F32 (dense kernel, Kind -1).
   function GPU_OK (W : LLM_Weight.Weight) return Boolean is
     (Kind_Code (W) >= 0 or else LLM_Weight.Is_F32 (W));

   --  Describe a weight for the resident step (Kind -1 = dense F32 bytes).
   function GW (W : LLM_Weight.Weight) return LLM_Qwen_GPU.GPU_Weight is
     (Addr  => Raw_Address (W),
      Bytes => Raw_Bytes (W),
      Kind  => Kind_Code (W));

   function Init_State (L : DeltaNet_Layer) return DNet_State is
      Kernel : constant Integer := Shape (L.Conv_W) (2);
   begin
      return St : DNet_State do
         St.S_All     := New_Tensor ([L.N_V_Heads * L.Key_Head_Dim,
                                      L.Value_Head_Dim]);
         St.Conv_Hist := New_Tensor ([Integer'Max (1, Kernel - 1), L.QKV_Out]);
         --  Phase B: keep the recurrent state AND conv window resident on the
         --  device, and run the whole layer there — but only when every
         --  projection is a format the resident kernels read.
         if LLM_Qwen_GPU.Dnet_Available
           and then GPU_OK (L.QKV_W) and then GPU_OK (L.Alpha_W)
           and then GPU_OK (L.Beta_W) and then GPU_OK (L.Gate_W)
           and then GPU_OK (L.Out_W)
         then
            St.GPU_Handle := LLM_Qwen_GPU.Dnet_New
              (L.N_V_Heads, L.Key_Head_Dim, L.Value_Head_Dim,
               L.QKV_Out, Kernel);
         end if;
      end return;
   end Init_State;

   function Step (L : DeltaNet_Layer; St : in out DNet_State; X : Tensor)
      return Tensor
   is
      Dim    : constant Integer := L.Dim;
      QO     : constant Integer := L.QKV_Out;
      NV     : constant Integer := L.N_V_Heads;
      KHD    : constant Integer := L.Key_Head_Dim;
      VHD    : constant Integer := L.Value_Head_Dim;
      Q_Dim  : constant Integer := L.N_K_Heads * KHD;
      Kernel : constant Integer := Shape (L.Conv_W) (2);
   begin
      --  Phase B: the whole layer on the device in ONE call — projections,
      --  conv (resident window), recurrence (resident S_All), out-projection.
      --  The small dense tensors upload once (cached by host pointer).
      if St.GPU_Handle >= 0 then
         return Out_T : constant Tensor := New_Tensor ([1, Dim]) do
            LLM_Qwen_GPU.Dnet_Step
              (Handle  => St.GPU_Handle,
               X       => Data_Address (X),
               Dim     => Dim,
               QKV_W   => GW (L.QKV_W),
               Alpha_W => GW (L.Alpha_W),
               Beta_W  => GW (L.Beta_W),
               Gate_W  => GW (L.Gate_W),
               Out_W   => GW (L.Out_W),
               Conv_W  => Data_Address (L.Conv_W),
               Conv_B  => Long_Long_Integer (Numel (L.Conv_W)) * 4,
               A_W     => Data_Address (L.A_W),
               A_B     => Long_Long_Integer (Numel (L.A_W)) * 4,
               Dt_W    => Data_Address (L.Dt_W),
               Dt_B    => Long_Long_Integer (Numel (L.Dt_W)) * 4,
               Norm_W  => Data_Address (L.Norm_W),
               Norm_B  => Long_Long_Integer (Numel (L.Norm_W)) * 4,
               NV      => NV,
               KHD     => KHD,
               VHD     => VHD,
               QO      => QO,
               Q_Dim   => Q_Dim,
               N_K_Heads => L.N_K_Heads,
               V_Dim   => L.V_Dim,
               Kernel  => Kernel,
               Y       => Data_Address (Out_T));
         end return;
      end if;

      declare
      QKV   : constant Tensor := MatVec (L.QKV_W, X);   -- [1, QO]
      CQ    : Tensor := New_Tensor ([1, QO]);           -- conv'd + SiLU qkv
      AR    : constant Tensor := MatVec (L.Alpha_W, X);
      BR    : constant Tensor := MatVec (L.Beta_W, X);
      Z     : constant Tensor := MatVec (L.Gate_W, X);
      O_Row : Tensor := New_Tensor ([1, L.V_Dim]);
      Beta  : array (1 .. NV) of Float;
      Gate  : array (1 .. NV) of Float;
      begin
      --  1. Causal conv1d + SiLU using the running window: weight tap K maps
      --     to source position (t - Kernel + K); taps < Kernel come from the
      --     history rows, tap Kernel is the current token. Unseen history is
      --     zero (Conv_Hist is zero-initialised), matching the batched path.
      for C in 1 .. QO loop
         declare
            Acc : Float := Get_Flat (QKV, C) * Get (L.Conv_W, [C, Kernel]);
         begin
            for K in 1 .. Kernel - 1 loop
               Acc := Acc + Get (St.Conv_Hist, [K, C]) * Get (L.Conv_W, [C, K]);
            end loop;
            Set_Flat (CQ, C, Silu (Acc));
         end;
      end loop;

      --  Advance the history: drop the oldest row, append the current qkv.
      for K in 1 .. Kernel - 2 loop
         for C in 1 .. QO loop
            Set (St.Conv_Hist, [K, C], Get (St.Conv_Hist, [K + 1, C]));
         end loop;
      end loop;
      if Kernel >= 2 then
         for C in 1 .. QO loop
            Set (St.Conv_Hist, [Kernel - 1, C], Get_Flat (QKV, C));
         end loop;
      end if;

      --  2. Per-head decay/beta, then one delta step advancing the state.
      for H in 1 .. NV loop
         Beta (H) := 1.0 / (1.0 + Exp (-Get_Flat (BR, H)));
         Gate (H) := Exp (Get_Flat (L.A_W, H)
                          * Softplus (Get_Flat (AR, H) + Get_Flat (L.Dt_W, H)));
      end loop;

      for H in 1 .. NV loop
         declare
            K_Head  : constant Integer := (H - 1) mod L.N_K_Heads + 1;
            Q_Vec   : Tensor := New_Tensor ([1, KHD]);
            K_Vec   : Tensor := New_Tensor ([1, KHD]);
            V_Vec   : Tensor := New_Tensor ([1, VHD]);
            O_Vec   : Tensor := New_Tensor ([1, VHD]);
            Base    : constant Integer := (H - 1) * KHD;  -- row offset in S_All
         begin
            for D in 1 .. KHD loop
               Set_Flat (Q_Vec, D, Get (CQ, [1, (K_Head - 1) * KHD + D]));
               Set_Flat (K_Vec, D, Get (CQ, [1, Q_Dim + (K_Head - 1) * KHD + D]));
            end loop;
            for D in 1 .. VHD loop
               Set_Flat (V_Vec, D, Get (CQ, [1, 2 * Q_Dim + (H - 1) * VHD + D]));
            end loop;

            --  Advance this head's state in place (rows Base+1..Base+KHD of
            --  the packed S_All), avoiding an unpack/repack copy per step.
            LLM_DeltaNet.Step (St.S_All, Q_Vec, K_Vec, V_Vec,
                               Gate (H), Beta (H), O_Vec, Base => Base);

            --  Gated RMSNorm over VHD dims into the output row.
            declare
               SS : Float := 0.0;
            begin
               for D in 1 .. VHD loop
                  SS := SS + Get_Flat (O_Vec, D) ** 2;
               end loop;
               declare
                  Rms : constant Float := Sqrt (SS / Float (VHD) + 1.0e-6);
               begin
                  for D in 1 .. VHD loop
                     Set_Flat (O_Row, (H - 1) * VHD + D,
                          (Get_Flat (O_Vec, D) / Rms) * Get_Flat (L.Norm_W, D)
                          * Silu (Get_Flat (Z, (H - 1) * VHD + D)));
                  end loop;
               end;
            end;
         end;
      end loop;

      --  3. Output projection.
      declare
         Ot     : constant Tensor := MatVec (L.Out_W, O_Row);
         Out_T  : Tensor := New_Tensor ([1, Dim]);
      begin
         for D in 1 .. Dim loop
            Set_Flat (Out_T, D, Get_Flat (Ot, D));
         end loop;
         return Out_T;
      end;
      end;
   end Step;

end LLM_DeltaNet_Blk;
