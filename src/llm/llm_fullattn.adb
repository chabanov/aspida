---------------------------------------------------------------------
-- LLM_FullAttn body
---------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;
with LLM_Tensor; use LLM_Tensor;
with LLM_RMSNorm;
with LLM_Weight; use LLM_Weight;
with LLM_RoPE;   use LLM_RoPE;

package body LLM_FullAttn is

   --  Hot per-token kernel; indices derive from the layer's own dims.
   pragma Suppress (All_Checks);

   function Sigmoid (X : Float) return Float is (1.0 / (1.0 + Exp (-X)));

   function Create
     (Q_W, K_W, V_W  : Weight;
      Q_Norm, K_Norm : Tensor;
      O_W            : Weight;
      RoPE           : LLM_RoPE.RoPE_Params)
      return Full_Attn_Layer
   is
      L : Full_Attn_Layer;
   begin
      L.Q_W := Q_W;  L.K_W := K_W;  L.V_W := V_W;  L.O_W := O_W;
      L.Q_Norm := Q_Norm;  L.K_Norm := K_Norm;  L.RoPE := RoPE;
      L.Dim        := Cols (Q_W);
      L.Head_Dim   := Numel (Q_Norm);
      L.N_Q_Heads  := Rows (Q_W) / (2 * L.Head_Dim);
      L.N_KV_Heads := Rows (K_W) / L.Head_Dim;
      return L;
   end Create;

   --  RMSNorm a head vector, then rotate its first rope_dim dims (partial RoPE).
   function Norm_Rope
     (V : Tensor; Norm_W : Tensor; RoPE : LLM_RoPE.RoPE_Params; Pos : Integer;
      Sec : LLM_RoPE.Section_Positions := [others => 0])
      return Tensor
   is
      HD     : constant Integer := Numel (V);
      RD     : constant Integer := RoPE.Dim;
      Normed : constant Tensor := LLM_RMSNorm.Forward (V, Norm_W);
      Sub    : Tensor := New_Tensor ([1, RD]);
      --  When the caller didn't pass Sec (default = [0,0,0,0]), we
      --  reproduce the legacy text-only rotation: every section uses
      --  the same Pos. The sentinel works because legitimate
      --  Section_Positions are non-zero on at least the time axis.
      Eff_Sec : constant LLM_RoPE.Section_Positions :=
        (if Sec = LLM_RoPE.Section_Positions'(others => 0)
         then LLM_RoPE.Uniform_Positions (Pos)
         else Sec);
   begin
      for I in 1 .. RD loop
         Set_Flat (Sub, I, Get_Flat (Normed, I));
      end loop;
      declare
         Rot : constant Tensor := LLM_RoPE.Apply_Sections (RoPE, Sub, Eff_Sec);
      begin
         return R : Tensor := New_Tensor ([1, HD]) do
            for I in 1 .. RD loop
               Set_Flat (R, I, Get_Flat (Rot, I));
            end loop;
            for I in RD + 1 .. HD loop
               Set_Flat (R, I, Get_Flat (Normed, I));
            end loop;
         end return;
      end;
   end Norm_Rope;

   function Forward (L : Full_Attn_Layer; X : Tensor) return Tensor is
      Seq     : constant Integer := Shape (X) (1);
      Dim     : constant Integer := L.Dim;
      NQ      : constant Integer := L.N_Q_Heads;
      NKV     : constant Integer := L.N_KV_Heads;
      HD      : constant Integer := L.Head_Dim;
      Rep     : constant Integer := NQ / NKV;
      Scale   : constant Float := 1.0 / Sqrt (Float (HD));
      Att_Dim : constant Integer := NQ * HD;

      Q_All : Tensor := New_Tensor ([Seq, NQ * HD]);
      G_All : Tensor := New_Tensor ([Seq, NQ * HD]);   -- raw gate
      K_All : Tensor := New_Tensor ([Seq, NKV * HD]);
      V_All : Tensor := New_Tensor ([Seq, NKV * HD]);
      Attn  : Tensor := New_Tensor ([Seq, Att_Dim]);
      Out_T : Tensor := New_Tensor ([Seq, Dim]);

      function Row (T : Integer) return Tensor is
         R : Tensor := New_Tensor ([1, Dim]);
      begin
         for I in 1 .. Dim loop
            Set_Flat (R, I, Get (X, [T, I]));
         end loop;
         return R;
      end Row;
   begin
      ----------------------------------------------------------------
      -- 1. Project q(+gate), k, v per position; QK-norm + partial RoPE.
      ----------------------------------------------------------------
      for T in 1 .. Seq loop
         declare
            Xt : constant Tensor := Row (T);
            QG : constant Tensor := MatVec (L.Q_W, Xt);   -- [1, n_q*2*head_dim]
            Kt : constant Tensor := MatVec (L.K_W, Xt);   -- [1, n_kv*head_dim]
            Vt : constant Tensor := MatVec (L.V_W, Xt);
         begin
            for H in 1 .. NQ loop
               declare
                  Base : constant Integer := (H - 1) * 2 * HD;
                  Qh   : Tensor := New_Tensor ([1, HD]);
               begin
                  for D in 1 .. HD loop
                     Set_Flat (Qh, D, Get_Flat (QG, Base + D));
                     Set (G_All, [T, (H - 1) * HD + D], Get_Flat (QG, Base + HD + D));
                  end loop;
                  declare
                     QR : constant Tensor := Norm_Rope (Qh, L.Q_Norm, L.RoPE, T - 1);
                  begin
                     for D in 1 .. HD loop
                        Set (Q_All, [T, (H - 1) * HD + D], Get_Flat (QR, D));
                     end loop;
                  end;
               end;
            end loop;

            for J in 1 .. NKV loop
               declare
                  Kh : Tensor := New_Tensor ([1, HD]);
               begin
                  for D in 1 .. HD loop
                     Set_Flat (Kh, D, Get_Flat (Kt, (J - 1) * HD + D));
                     Set (V_All, [T, (J - 1) * HD + D], Get_Flat (Vt, (J - 1) * HD + D));
                  end loop;
                  declare
                     KR : constant Tensor := Norm_Rope (Kh, L.K_Norm, L.RoPE, T - 1);
                  begin
                     for D in 1 .. HD loop
                        Set (K_All, [T, (J - 1) * HD + D], Get_Flat (KR, D));
                     end loop;
                  end;
               end;
            end loop;
         end;
      end loop;

      ----------------------------------------------------------------
      -- 2. Causal GQA softmax attention, then per-head sigmoid gate.
      ----------------------------------------------------------------
      for T in 1 .. Seq loop
         for H in 1 .. NQ loop
            declare
               KVH    : constant Integer := (H - 1) / Rep + 1;
               Q_Off  : constant Integer := (H - 1) * HD;
               KV_Off : constant Integer := (KVH - 1) * HD;
               Scores : array (1 .. T) of Float;
               Max_S  : Float := Float'First;
               Sum_E  : Float := 0.0;
            begin
               for SS in 1 .. T loop
                  declare
                     Dot : Float := 0.0;
                  begin
                     for D in 1 .. HD loop
                        Dot := Dot
                          + Get (Q_All, [T, Q_Off + D]) * Get (K_All, [SS, KV_Off + D]);
                     end loop;
                     Scores (SS) := Dot * Scale;
                     if Scores (SS) > Max_S then
                        Max_S := Scores (SS);
                     end if;
                  end;
               end loop;
               for SS in 1 .. T loop
                  Scores (SS) := Exp (Scores (SS) - Max_S);
                  Sum_E := Sum_E + Scores (SS);
               end loop;

               for D in 1 .. HD loop
                  declare
                     Acc : Float := 0.0;
                  begin
                     for SS in 1 .. T loop
                        Acc := Acc + (Scores (SS) / Sum_E) * Get (V_All, [SS, KV_Off + D]);
                     end loop;
                     Set (Attn, [T, Q_Off + D],
                          Acc * Sigmoid (Get (G_All, [T, Q_Off + D])));
                  end;
               end loop;
            end;
         end loop;
      end loop;

      ----------------------------------------------------------------
      -- 3. Output projection: out[t] = O_W . attn[t]
      ----------------------------------------------------------------
      for T in 1 .. Seq loop
         declare
            A_Row : Tensor := New_Tensor ([1, Att_Dim]);
         begin
            for J in 1 .. Att_Dim loop
               Set_Flat (A_Row, J, Get (Attn, [T, J]));
            end loop;
            declare
               Ot : constant Tensor := MatVec (L.O_W, A_Row);
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

   function Init_State (L : Full_Attn_Layer; Max_Len : Integer) return Attn_State is
      KV_Dim : constant Integer := L.N_KV_Heads * L.Head_Dim;
   begin
      return St : Attn_State do
         St.K_Cache := New_Tensor ([Integer'Max (1, Max_Len), KV_Dim]);
         St.V_Cache := New_Tensor ([Integer'Max (1, Max_Len), KV_Dim]);
         St.Len     := 0;
      end return;
   end Init_State;

   function Step (L : Full_Attn_Layer; St : in out Attn_State; X : Tensor)
      return Tensor
   is
      Dim     : constant Integer := L.Dim;
      NQ      : constant Integer := L.N_Q_Heads;
      NKV     : constant Integer := L.N_KV_Heads;
      HD      : constant Integer := L.Head_Dim;
      Rep     : constant Integer := NQ / NKV;
      Scale   : constant Float := 1.0 / Sqrt (Float (HD));
      Att_Dim : constant Integer := NQ * HD;
      Pos     : constant Integer := St.Len;        -- 0-based abs position for RoPE

      QG    : constant Tensor := MatVec (L.Q_W, X);   -- [1, n_q*2*head_dim]
      Kt    : constant Tensor := MatVec (L.K_W, X);   -- [1, n_kv*head_dim]
      Vt    : constant Tensor := MatVec (L.V_W, X);
      Q_All : Tensor := New_Tensor ([1, NQ * HD]);
      G_All : Tensor := New_Tensor ([1, NQ * HD]);    -- raw gate
      Attn  : Tensor := New_Tensor ([1, Att_Dim]);
      Row   : constant Integer := St.Len + 1;         -- cache row for this token
   begin
      --  Guard the KV cache bound (checks are suppressed in this unit): the
      --  cache is sized to Max_Len at Init_State, so decoding past it would be
      --  a silent out-of-bounds write. Fail loudly instead.
      if Row > Shape (St.K_Cache) (1) then
         raise Constraint_Error
           with "FullAttn: KV cache overflow at position" & Integer'Image (Row)
                & " (max" & Integer'Image (Shape (St.K_Cache) (1)) & ")";
      end if;

      --  1. Project q(+gate), QK-norm + partial RoPE; append k, v to the cache.
      for H in 1 .. NQ loop
         declare
            Base : constant Integer := (H - 1) * 2 * HD;
            Qh   : Tensor := New_Tensor ([1, HD]);
         begin
            for D in 1 .. HD loop
               Set_Flat (Qh, D, Get_Flat (QG, Base + D));
               Set (G_All, [1, (H - 1) * HD + D], Get_Flat (QG, Base + HD + D));
            end loop;
            declare
               QR : constant Tensor := Norm_Rope (Qh, L.Q_Norm, L.RoPE, Pos);
            begin
               for D in 1 .. HD loop
                  Set (Q_All, [1, (H - 1) * HD + D], Get_Flat (QR, D));
               end loop;
            end;
         end;
      end loop;

      for J in 1 .. NKV loop
         declare
            Kh : Tensor := New_Tensor ([1, HD]);
         begin
            for D in 1 .. HD loop
               Set_Flat (Kh, D, Get_Flat (Kt, (J - 1) * HD + D));
               Set (St.V_Cache, [Row, (J - 1) * HD + D],
                    Get_Flat (Vt, (J - 1) * HD + D));
            end loop;
            declare
               KR : constant Tensor := Norm_Rope (Kh, L.K_Norm, L.RoPE, Pos);
            begin
               for D in 1 .. HD loop
                  Set (St.K_Cache, [Row, (J - 1) * HD + D], Get_Flat (KR, D));
               end loop;
            end;
         end;
      end loop;

      St.Len := Row;

      --  2. Causal GQA softmax over all cached positions, then sigmoid gate.
      for H in 1 .. NQ loop
         declare
            KVH    : constant Integer := (H - 1) / Rep + 1;
            Q_Off  : constant Integer := (H - 1) * HD;
            KV_Off : constant Integer := (KVH - 1) * HD;
            Scores : array (1 .. St.Len) of Float;
            Max_S  : Float := Float'First;
            Sum_E  : Float := 0.0;
         begin
            for SS in 1 .. St.Len loop
               declare
                  Dot : Float := 0.0;
               begin
                  for D in 1 .. HD loop
                     Dot := Dot + Get (Q_All, [1, Q_Off + D])
                                  * Get (St.K_Cache, [SS, KV_Off + D]);
                  end loop;
                  Scores (SS) := Dot * Scale;
                  if Scores (SS) > Max_S then
                     Max_S := Scores (SS);
                  end if;
               end;
            end loop;
            for SS in 1 .. St.Len loop
               Scores (SS) := Exp (Scores (SS) - Max_S);
               Sum_E := Sum_E + Scores (SS);
            end loop;

            for D in 1 .. HD loop
               declare
                  Acc : Float := 0.0;
               begin
                  for SS in 1 .. St.Len loop
                     Acc := Acc + (Scores (SS) / Sum_E)
                                  * Get (St.V_Cache, [SS, KV_Off + D]);
                  end loop;
                  Set (Attn, [1, Q_Off + D],
                       Acc * Sigmoid (Get (G_All, [1, Q_Off + D])));
               end;
            end loop;
         end;
      end loop;

      --  3. Output projection.
      declare
         Ot    : constant Tensor := MatVec (L.O_W, Attn);
         Out_T : Tensor := New_Tensor ([1, Dim]);
      begin
         for D in 1 .. Dim loop
            Set_Flat (Out_T, D, Get_Flat (Ot, D));
         end loop;
         return Out_T;
      end;
   end Step;

end LLM_FullAttn;
