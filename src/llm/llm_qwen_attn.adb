---------------------------------------------------------------------
-- LLM_Qwen_Attn body — causal GQA attention with RoPE + softmax
--
-- Processes a whole sequence X [Seq_Len, Dim] at once and returns
-- [Seq_Len, Dim]. Each query position attends causally to all key
-- positions <= itself, with softmax-normalised scores (the previous
-- implementation was single-token and skipped softmax entirely).
--
-- Pos is the absolute position of the first row of X (0-based), used
-- for RoPE; for a fresh prompt it is 0.
---------------------------------------------------------------------

with Ada.Numerics.Elementary_Functions;
use Ada.Numerics.Elementary_Functions;

package body LLM_Qwen_Attn is

   function Create_Qwen_Attn_Params
     (QKV_W, O_W, Gate_W : Tensor;
      RoPE : RoPE_Params;
      Dim, N_Heads, N_KV_Heads, Head_Dim : Integer;
      Use_Gate : Boolean)
      return Qwen_Attn_Params
   is
      P : Qwen_Attn_Params;
   begin
      P.QKV_W := QKV_W;  P.O_W := O_W;  P.Gate_W := Gate_W;
      P.RoPE := RoPE;  P.Dim := Dim;  P.N_Heads := N_Heads;
      P.N_KV_Heads := N_KV_Heads;  P.Head_Dim := Head_Dim;
      P.Use_Gate := Use_Gate;
      return P;
   end Create_Qwen_Attn_Params;

   function Forward (P : Qwen_Attn_Params; X : Tensor; Pos : Integer) return Tensor is
      Seq_Len : constant Integer := Shape (X) (1);
      D       : constant Integer := P.Dim;
      H_Dim   : constant Integer := P.Head_Dim;
      N_Rep   : constant Integer := P.N_Heads / P.N_KV_Heads;  -- GQA repeat factor
      Scale   : constant Float   := 1.0 / Sqrt (Float (H_Dim));

      -- QKV_W is [dim, (n_heads + 2*n_kv_heads) * head_dim]
      Total_Dim : constant Integer := (P.N_Heads + 2 * P.N_KV_Heads) * H_Dim;

      -- Per-position projected Q/K/V, laid out [Seq_Len, heads * head_dim]
      Q_All : Tensor := New_Tensor ([Seq_Len, P.N_Heads * H_Dim]);
      K_All : Tensor := New_Tensor ([Seq_Len, P.N_KV_Heads * H_Dim]);
      V_All : Tensor := New_Tensor ([Seq_Len, P.N_KV_Heads * H_Dim]);

      Output : Tensor := New_Tensor ([Seq_Len, D]);
   begin
      ----------------------------------------------------------------
      -- 1. Project QKV for every position, split into heads, and apply
      --    RoPE to the Q and K heads (V is not rotated).
      ----------------------------------------------------------------
      for T in 1 .. Seq_Len loop
         declare
            QKV_Out : Tensor := New_Tensor ([1, Total_Dim]);
         begin
            for J in 1 .. Total_Dim loop
               declare
                  Sum : Float := 0.0;
               begin
                  for I in 1 .. D loop
                     Sum := Sum + Get (X, [T, I]) * Get (P.QKV_W, [I, J]);
                  end loop;
                  Set_Flat (QKV_Out, J, Sum);
               end;
            end loop;

            -- Q heads (with RoPE at absolute position Pos + T - 1)
            for H in 1 .. P.N_Heads loop
               declare
                  Head : Tensor := New_Tensor ([1, H_Dim]);
               begin
                  for I in 1 .. H_Dim loop
                     Set_Flat (Head, I, Get_Flat (QKV_Out, (H - 1) * H_Dim + I));
                  end loop;
                  Head := LLM_RoPE.Apply (P.RoPE, Head, Pos + T - 1);
                  for I in 1 .. H_Dim loop
                     Set (Q_All, [T, (H - 1) * H_Dim + I], Get_Flat (Head, I));
                  end loop;
               end;
            end loop;

            -- K, V heads
            for H in 1 .. P.N_KV_Heads loop
               declare
                  K_Off : constant Integer := P.N_Heads * H_Dim + (H - 1) * H_Dim;
                  V_Off : constant Integer :=
                    (P.N_Heads + P.N_KV_Heads) * H_Dim + (H - 1) * H_Dim;
                  K_Head : Tensor := New_Tensor ([1, H_Dim]);
               begin
                  for I in 1 .. H_Dim loop
                     Set_Flat (K_Head, I, Get_Flat (QKV_Out, K_Off + I));
                  end loop;
                  K_Head := LLM_RoPE.Apply (P.RoPE, K_Head, Pos + T - 1);
                  for I in 1 .. H_Dim loop
                     Set (K_All, [T, (H - 1) * H_Dim + I], Get_Flat (K_Head, I));
                     Set (V_All, [T, (H - 1) * H_Dim + I], Get_Flat (QKV_Out, V_Off + I));
                  end loop;
               end;
            end loop;
         end;
      end loop;

      ----------------------------------------------------------------
      -- 2. Causal scaled-dot-product attention with softmax.
      --    For each query position T and head QH, attend over keys
      --    1 .. T only (causal mask), normalise with softmax, then
      --    take the weighted sum of the V vectors.
      ----------------------------------------------------------------
      declare
         Attn_Heads : Tensor := New_Tensor ([Seq_Len, P.N_Heads * H_Dim]);
      begin
         for T in 1 .. Seq_Len loop
            for QH in 1 .. P.N_Heads loop
               declare
                  KV_H   : constant Integer := (QH - 1) / N_Rep + 1;  -- GQA group
                  Q_Off  : constant Integer := (QH - 1) * H_Dim;
                  KV_Off : constant Integer := (KV_H - 1) * H_Dim;
                  Scores : array (1 .. T) of Float;
                  Max_S  : Float := Float'First;
                  Sum_E  : Float := 0.0;
               begin
                  -- Raw scaled scores over the causal window 1 .. T
                  for S in 1 .. T loop
                     declare
                        Dot : Float := 0.0;
                     begin
                        for I in 1 .. H_Dim loop
                           Dot := Dot
                             + Get (Q_All, [T, Q_Off + I])
                               * Get (K_All, [S, KV_Off + I]);
                        end loop;
                        Scores (S) := Dot * Scale;
                        if Scores (S) > Max_S then
                           Max_S := Scores (S);
                        end if;
                     end;
                  end loop;

                  -- Softmax (numerically stable: subtract the max first)
                  for S in 1 .. T loop
                     Scores (S) := Exp (Scores (S) - Max_S);
                     Sum_E := Sum_E + Scores (S);
                  end loop;

                  -- Weighted sum of V over the causal window
                  for I in 1 .. H_Dim loop
                     declare
                        Acc : Float := 0.0;
                     begin
                        for S in 1 .. T loop
                           Acc := Acc
                             + (Scores (S) / Sum_E) * Get (V_All, [S, KV_Off + I]);
                        end loop;
                        Set (Attn_Heads, [T, Q_Off + I], Acc);
                     end;
                  end loop;
               end;
            end loop;
         end loop;

         ----------------------------------------------------------------
         -- 3. Output projection: attn_heads[T] @ O_W [n_heads*head_dim, dim]
         ----------------------------------------------------------------
         for T in 1 .. Seq_Len loop
            for I in 1 .. D loop
               declare
                  Sum : Float := 0.0;
               begin
                  for J in 1 .. P.N_Heads * H_Dim loop
                     Sum := Sum + Get (Attn_Heads, [T, J]) * Get (P.O_W, [J, I]);
                  end loop;
                  Set (Output, [T, I], Sum);
               end;
            end loop;
         end loop;
      end;

      ----------------------------------------------------------------
      -- 4. Attention output gate (SSM layers only): element-wise
      --    sigmoid gate computed from the layer input.
      --       out[i] := out[i] * sigmoid((X @ Gate_W)[i])
      ----------------------------------------------------------------
      if P.Use_Gate then
         for T in 1 .. Seq_Len loop
            for I in 1 .. D loop
               declare
                  Gate_Sum : Float := 0.0;
                  Gate_Val : Float;
               begin
                  for J in 1 .. D loop
                     Gate_Sum := Gate_Sum + Get (X, [T, J]) * Get (P.Gate_W, [J, I]);
                  end loop;
                  Gate_Val := 1.0 / (1.0 + Exp (-Gate_Sum));
                  Set (Output, [T, I], Get (Output, [T, I]) * Gate_Val);
               end;
            end loop;
         end loop;
      end if;

      return Output;
   end Forward;

end LLM_Qwen_Attn;
