---------------------------------------------------------------------
-- Test causal GQA attention (LLM_Qwen_Attn)
--
-- Validates the two defining properties of causal self-attention
-- without needing a real model:
--   A. Causal masking: a query position's output is unchanged when
--      later tokens are appended to the sequence.
--   B. Softmax averaging: with zero query weights all scores are equal,
--      so attention is uniform and the output is the projection of the
--      mean of the V vectors over the causal window.
---------------------------------------------------------------------

with Ada.Text_IO;
with Ada.Command_Line;
with LLM_Tensor; use LLM_Tensor;
with LLM_RoPE;
with LLM_Qwen_Attn;

procedure Test_Attention is
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

   --  Tiny GQA config: 2 query heads sharing 1 KV head (repeat = 2).
   Dim       : constant := 4;
   N_Heads   : constant := 2;
   N_KV      : constant := 1;
   H_Dim     : constant := 2;
   Total_Dim : constant := (N_Heads + 2 * N_KV) * H_Dim;  -- 8

   --  RoPE sized to the head dim so Apply touches exactly the head.
   function Make_RoPE return LLM_RoPE.RoPE_Params is
      R : LLM_RoPE.RoPE_Params;
   begin
      R.Dim       := H_Dim;
      R.Freq_Base := 10_000.0;
      R.Max_Pos   := 1_000;
      R.Sections  := New_Tensor ([1, 4]);
      return R;
   end Make_RoPE;

   --  Deterministic output projection [n_heads*head_dim, dim].
   function Make_OW return Tensor is
      O : Tensor := New_Tensor ([N_Heads * H_Dim, Dim]);
   begin
      for I in 1 .. N_Heads * H_Dim loop
         for J in 1 .. Dim loop
            Set (O, [I, J], 0.1 * Float ((I * 3 + J) mod 7 - 3));
         end loop;
      end loop;
      return O;
   end Make_OW;

   function Make_Params (QKV : Tensor) return LLM_Qwen_Attn.Qwen_Attn_Params is
   begin
      return LLM_Qwen_Attn.Create_Qwen_Attn_Params
        (QKV, Make_OW, New_Tensor ([Dim, Dim]), Make_RoPE,
         Dim, N_Heads, N_KV, H_Dim, False);
   end Make_Params;

begin
   Put_Line ("=== Attention Test Suite ===");
   New_Line;

   ------------------------------------------------------------------
   -- Test A: causal masking — appending a 3rd token must not change
   --         the outputs at positions 1 and 2.
   ------------------------------------------------------------------
   declare
      QKV : Tensor := New_Tensor ([Dim, Total_Dim]);
      P   : LLM_Qwen_Attn.Qwen_Attn_Params;
      X3  : Tensor := New_Tensor ([3, Dim]);
      X2  : Tensor := New_Tensor ([2, Dim]);
      O3, O2 : Tensor;
   begin
      for I in 1 .. Dim loop
         for J in 1 .. Total_Dim loop
            Set (QKV, [I, J], 0.1 * Float ((I * 5 + J * 2) mod 9 - 4));
         end loop;
      end loop;
      P := Make_Params (QKV);

      for T in 1 .. 3 loop
         for I in 1 .. Dim loop
            Set (X3, [T, I], Float (T) + 0.1 * Float (I));
         end loop;
      end loop;
      for T in 1 .. 2 loop
         for I in 1 .. Dim loop
            Set (X2, [T, I], Get (X3, [T, I]));
         end loop;
      end loop;

      O3 := LLM_Qwen_Attn.Forward (P, X3, 0);
      O2 := LLM_Qwen_Attn.Forward (P, X2, 0);

      for T in 1 .. 2 loop
         for I in 1 .. Dim loop
            Assert ("causal: pos" & Integer'Image (T) & " dim" & Integer'Image (I),
              Close (Get (O3, [T, I]), Get (O2, [T, I]), 1.0e-5));
         end loop;
      end loop;
   end;

   ------------------------------------------------------------------
   -- Test B: softmax averaging — zero query weights ⇒ all scores 0 ⇒
   --         uniform attention ⇒ each head outputs mean(V1, V2), and
   --         the projected result must match the hand-computed value.
   ------------------------------------------------------------------
   declare
      QKV : Tensor := New_Tensor ([Dim, Total_Dim]);
      OW  : constant Tensor := Make_OW;
      P   : LLM_Qwen_Attn.Qwen_Attn_Params;
      X   : Tensor := New_Tensor ([2, Dim]);
      O   : Tensor;
      V1, V2, Mean_V : array (1 .. H_Dim) of Float := [others => 0.0];
      Exp_Out        : array (1 .. Dim) of Float := [others => 0.0];
   begin
      for I in 1 .. Dim loop
         for J in 1 .. Total_Dim loop
            if J <= N_Heads * H_Dim then
               Set (QKV, [I, J], 0.0);                       -- zero Q columns
            else
               Set (QKV, [I, J], 0.1 * Float ((I * 5 + J * 2) mod 9 - 4));
            end if;
         end loop;
      end loop;
      P := Make_Params (QKV);

      for T in 1 .. 2 loop
         for I in 1 .. Dim loop
            Set (X, [T, I], Float (T) + 0.1 * Float (I));
         end loop;
      end loop;

      --  V columns start after Q (n_heads*head_dim) and K (n_kv*head_dim).
      for I in 1 .. H_Dim loop
         declare
            V_Col : constant Integer := (N_Heads + N_KV) * H_Dim + I;
         begin
            for K in 1 .. Dim loop
               V1 (I) := V1 (I) + Get (X, [1, K]) * Get (QKV, [K, V_Col]);
               V2 (I) := V2 (I) + Get (X, [2, K]) * Get (QKV, [K, V_Col]);
            end loop;
            Mean_V (I) := (V1 (I) + V2 (I)) / 2.0;
         end;
      end loop;

      --  Both query heads share KV head 1, so attn_heads[pos 2] = [Mean_V, Mean_V].
      for I in 1 .. Dim loop
         for QH in 1 .. N_Heads loop
            for K in 1 .. H_Dim loop
               Exp_Out (I) := Exp_Out (I)
                 + Mean_V (K) * Get (OW, [(QH - 1) * H_Dim + K, I]);
            end loop;
         end loop;
      end loop;

      O := LLM_Qwen_Attn.Forward (P, X, 0);
      for I in 1 .. Dim loop
         Assert ("uniform-avg: out dim" & Integer'Image (I),
           Close (Get (O, [2, I]), Exp_Out (I), 1.0e-4));
      end loop;
   end;

   New_Line;
   Put_Line ("Results:" & Passed'Image & " passed," & Failed'Image & " failed.");

   if Failed > 0 then
      Ada.Command_Line.Set_Exit_Status (1);
   end if;
end Test_Attention;
