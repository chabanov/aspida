---------------------------------------------------------------------
-- LLM_Qwen_Attn body — GQA + RoPE + gate
--
-- Single-token inference (KV-cache not needed for basic forward pass).
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
      N_Rep  : constant Integer := P.N_Heads / P.N_KV_Heads;  -- 8 for Qwen
      D      : constant Integer := P.Dim;
      H_Dim  : constant Integer := P.Head_Dim;
      Scale  : constant Float := 1.0 / Sqrt (Float (H_Dim));

      -- Allocate head tensors [1, head_dim]
      Q : array (1 .. P.N_Heads) of Tensor;
      K : array (1 .. P.N_KV_Heads) of Tensor;
      V : array (1 .. P.N_KV_Heads) of Tensor;

      -- Project X through QKV weight
      -- QKV_W is [dim, total_dim] where total_dim = (n_heads + 2*n_kv_heads)*head_dim
      Total_Dim : constant Integer := (P.N_Heads + 2 * P.N_KV_Heads) * H_Dim;
      QKV_Out   : Tensor := New_Tensor ([1, Total_Dim]);
   begin
      -- QKV projection (simplified: dense matmul dim×Total_Dim)
      for J in 1 .. Total_Dim loop
         declare
            Sum : Float := 0.0;
         begin
            for I in 1 .. D loop
               Sum := Sum + Get_Flat (X, I) * Get (P.QKV_W, [I, J]);
            end loop;
            Set_Flat (QKV_Out, J, Sum);
         end;
      end loop;

      -- Split QKV output into Q, K, V
      for H in 1 .. P.N_Heads loop
         Q (H) := New_Tensor ([1, H_Dim]);
         for I in 1 .. H_Dim loop
            Set_Flat (Q (H), I,
              Get_Flat (QKV_Out, (H - 1) * H_Dim + I));
         end loop;
      end loop;

      for H in 1 .. P.N_KV_Heads loop
         K (H) := New_Tensor ([1, H_Dim]);
         V (H) := New_Tensor ([1, H_Dim]);
         declare
            K_Off : constant Integer := P.N_Heads * H_Dim + (H - 1) * H_Dim;
            V_Off : constant Integer := P.N_Heads * H_Dim + P.N_KV_Heads * H_Dim + (H - 1) * H_Dim;
         begin
            for I in 1 .. H_Dim loop
               Set_Flat (K (H), I, Get_Flat (QKV_Out, K_Off + I));
               Set_Flat (V (H), I, Get_Flat (QKV_Out, V_Off + I));
            end loop;
         end;
      end loop;

      -- Apply RoPE to Q and K
      for H in 1 .. P.N_Heads loop
         Q (H) := LLM_RoPE.Apply (P.RoPE, Q (H), Pos);
      end loop;
      for H in 1 .. P.N_KV_Heads loop
         K (H) := LLM_RoPE.Apply (P.RoPE, K (H), Pos);
      end loop;

      -- Attention: for each Q head, compute scores with repeated KV
      declare
         Attn_Out : Tensor := New_Tensor ([1, P.N_Heads * H_Dim]);
      begin
         for QH in 1 .. P.N_Heads loop
            declare
               KV_H : constant Integer := (QH - 1) / N_Rep + 1;
               Scores : Tensor := New_Tensor ([1, 1]);
               Score  : Float := 0.0;
            begin
               -- Single-token: dot product Q @ K^T (just one token for now)
               for I in 1 .. H_Dim loop
                  Score := Score + Get_Flat (Q (QH), I) * Get_Flat (K (KV_H), I);
               end loop;
               Score := Score * Scale;

               -- Softmax would go here (single token = trivial)
               -- Output = score * V
               for I in 1 .. H_Dim loop
                  Set_Flat (Attn_Out, (QH - 1) * H_Dim + I,
                    Score * Get_Flat (V (KV_H), I));
               end loop;
            end;
         end loop;

         -- Output projection: attn_out @ O_W^T
         declare
            Output : Tensor := New_Tensor ([1, D]);
            Sum : Float;
         begin
            for I in 1 .. D loop
               Sum := 0.0;
               for J in 1 .. P.N_Heads * H_Dim loop
                  Sum := Sum + Get_Flat (Attn_Out, J) * Get (P.O_W, [J, I]);
               end loop;
               Set_Flat (Output, I, Sum);
            end loop;

            -- Gate (if applicable)
            if P.Use_Gate then
               -- gate = sigmoid(Gate_W @ X)
               declare
                  Gate_Sum : Float := 0.0;
                  Gate_Val : Float;
               begin
                  for I in 1 .. D loop
                     Gate_Sum := Gate_Sum + Get_Flat (X, I) * Get (P.Gate_W, [I, 1]);
                  end loop;
                  Gate_Val := 1.0 / (1.0 + Exp (-Gate_Sum));
                  for I in 1 .. D loop
                     Set_Flat (Output, I, Gate_Val * Get_Flat (Output, I) +
                       (1.0 - Gate_Val) * Get_Flat (X, I));
                  end loop;
               end;
            end if;

            return Output;
         end;
      end;
   end Forward;

end LLM_Qwen_Attn;
