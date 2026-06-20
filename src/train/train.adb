---------------------------------------------------------------------
-- Train body — forward/backward primitives, AdamW, PRNG.
---------------------------------------------------------------------

with Interfaces;                            use Interfaces;
with Ada.Numerics.Long_Elementary_Functions;
use  Ada.Numerics.Long_Elementary_Functions;

package body Train is

   --------------------------------------------------------------------
   --  PRNG — 64-bit LCG; Uniform returns the top 53 bits as [0,1).
   --------------------------------------------------------------------
   function Seeded (Seed : Long_Float) return RNG is
      G : RNG;
   begin
      --  fold the seed into the state; any value is fine, it just decorrelates.
      G.S := 16#853C49E6748FEA9B# xor Unsigned_64 (abs (Long_Float'Truncation (Seed)) + 1.0);
      return G;
   end Seeded;

   function Uniform (G : in out RNG) return Real is
      Two53 : constant Real := 9_007_199_254_740_992.0;   -- 2**53
   begin
      G.S := G.S * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407;
      return Real (Shift_Right (G.S, 11)) / Two53;
   end Uniform;

   procedure Init_Glorot (W : out Matrix; G : in out RNG) is
      Fan_In  : constant Real := Real (W'Length (1));
      Fan_Out : constant Real := Real (W'Length (2));
      Scale   : constant Real := Sqrt (6.0 / (Fan_In + Fan_Out));
   begin
      for I in W'Range (1) loop
         for J in W'Range (2) loop
            W (I, J) := (2.0 * Uniform (G) - 1.0) * Scale;   -- U(-scale, scale)
         end loop;
      end loop;
   end Init_Glorot;

   --------------------------------------------------------------------
   --  Linear
   --------------------------------------------------------------------
   procedure Linear_Forward (X, W, B : Matrix; Y : out Matrix) is
      Acc : Real;
   begin
      for R in 1 .. X'Length (1) loop
         for O in 1 .. W'Length (2) loop
            Acc := B (1, O);
            for I in 1 .. X'Length (2) loop
               Acc := Acc + X (R, I) * W (I, O);
            end loop;
            Y (R, O) := Acc;
         end loop;
      end loop;
   end Linear_Forward;

   procedure Linear_Backward_NoDX (X, W, DY : Matrix; DW, DB : out Matrix) is
      N   : constant Positive := X'Length (1);
      Inp : constant Positive := X'Length (2);
      Ou  : constant Positive := W'Length (2);
      Acc : Real;
   begin
      --  dW[In,Ou] = X^T[In,N] . dY[N,Ou]
      for I in 1 .. Inp loop
         for O in 1 .. Ou loop
            Acc := 0.0;
            for R in 1 .. N loop
               Acc := Acc + X (R, I) * DY (R, O);
            end loop;
            DW (I, O) := Acc;
         end loop;
      end loop;
      --  dB[1,Ou] = column sums of dY
      for O in 1 .. Ou loop
         Acc := 0.0;
         for R in 1 .. N loop
            Acc := Acc + DY (R, O);
         end loop;
         DB (1, O) := Acc;
      end loop;
   end Linear_Backward_NoDX;

   procedure Linear_Backward
     (X, W, DY : Matrix; DX, DW, DB : out Matrix)
   is
      N   : constant Positive := X'Length (1);
      Inp : constant Positive := X'Length (2);
      Ou  : constant Positive := W'Length (2);
      Acc : Real;
   begin
      --  dX[N,In] = dY[N,Ou] . W^T[Ou,In]
      for R in 1 .. N loop
         for I in 1 .. Inp loop
            Acc := 0.0;
            for O in 1 .. Ou loop
               Acc := Acc + DY (R, O) * W (I, O);
            end loop;
            DX (R, I) := Acc;
         end loop;
      end loop;
      Linear_Backward_NoDX (X, W, DY, DW, DB);
   end Linear_Backward;

   procedure Linear_NB_Forward (X, W : Matrix; Y : out Matrix) is
      Acc : Real;
   begin
      for R in 1 .. X'Length (1) loop
         for O in 1 .. W'Length (2) loop
            Acc := 0.0;
            for I in 1 .. X'Length (2) loop
               Acc := Acc + X (R, I) * W (I, O);
            end loop;
            Y (R, O) := Acc;
         end loop;
      end loop;
   end Linear_NB_Forward;

   procedure Linear_NB_Backward (X, W, DY : Matrix; DX, DW : out Matrix) is
      N   : constant Positive := X'Length (1);
      Inp : constant Positive := X'Length (2);
      Ou  : constant Positive := W'Length (2);
      Acc : Real;
   begin
      for R in 1 .. N loop
         for I in 1 .. Inp loop
            Acc := 0.0;
            for O in 1 .. Ou loop Acc := Acc + DY (R, O) * W (I, O); end loop;
            DX (R, I) := Acc;
         end loop;
      end loop;
      for I in 1 .. Inp loop
         for O in 1 .. Ou loop
            Acc := 0.0;
            for R in 1 .. N loop Acc := Acc + X (R, I) * DY (R, O); end loop;
            DW (I, O) := Acc;
         end loop;
      end loop;
   end Linear_NB_Backward;

   --------------------------------------------------------------------
   --  SiLU
   --------------------------------------------------------------------
   function Sigmoid (X : Real) return Real is (1.0 / (1.0 + Exp (-X)));

   procedure SiLU_Forward (X : Matrix; Y : out Matrix) is
   begin
      for I in X'Range (1) loop
         for J in X'Range (2) loop
            Y (I, J) := X (I, J) * Sigmoid (X (I, J));
         end loop;
      end loop;
   end SiLU_Forward;

   procedure SiLU_Backward (X, DY : Matrix; DX : out Matrix) is
      S : Real;
   begin
      for I in X'Range (1) loop
         for J in X'Range (2) loop
            S := Sigmoid (X (I, J));
            --  d/dx [x*sigmoid(x)] = sigmoid + x*sigmoid*(1-sigmoid)
            DX (I, J) := DY (I, J) * (S + X (I, J) * S * (1.0 - S));
         end loop;
      end loop;
   end SiLU_Backward;

   procedure Fake_Quant_Forward (X : Matrix; Bits : Positive; Y : out Matrix) is
      Q_Max : constant Real := Real (2 ** (Bits - 1) - 1);   -- e.g. 127 for 8-bit
      Amax  : Real := 0.0;
   begin
      for I in X'Range (1) loop
         for J in X'Range (2) loop
            if abs (X (I, J)) > Amax then Amax := abs (X (I, J)); end if;
         end loop;
      end loop;
      if Amax = 0.0 or else Q_Max = 0.0 then
         Y := X;                       -- degenerate: nothing to quantize
         return;
      end if;
      declare
         Scale : constant Real := Amax / Q_Max;     -- per-tensor symmetric step
      begin
         for I in X'Range (1) loop
            for J in X'Range (2) loop
               --  round-to-nearest onto the grid, clamp, then dequantize
               declare
                  Q : Real := Real'Rounding (X (I, J) / Scale);
               begin
                  if Q > Q_Max then Q := Q_Max; end if;
                  if Q < -Q_Max then Q := -Q_Max; end if;
                  Y (I, J) := Q * Scale;
               end;
            end loop;
         end loop;
      end;
   end Fake_Quant_Forward;

   procedure Fake_Quant_Forward_Blocked
     (X : Matrix; Bits : Positive; Block : Positive; Y : out Matrix)
   is
      Q_Max : constant Real := Real (2 ** (Bits - 1) - 1);
      --  Treat the matrix as a flat row-major stream of elements; quantize each
      --  contiguous Block-window with its own symmetric scale.
      Rows : constant Integer := X'Length (1);
      Cols : constant Integer := X'Length (2);
      N    : constant Natural := Rows * Cols;
      function Flat (I : Natural) return Real is
         --  0-based flat index I -> X element (row-major).
         R : constant Integer := X'First (1) + I / Cols;
         C : constant Integer := X'First (2) + I mod Cols;
      begin
         return X (R, C);
      end Flat;
   begin
      if Block >= N or else Q_Max = 0.0 then
         Fake_Quant_Forward (X, Bits, Y);     -- degenerate -> per-tensor
         return;
      end if;
      for B0 in 0 .. (N + Block - 1) / Block - 1 loop
         declare
            Lo   : constant Natural := B0 * Block;
            Hi   : constant Natural := Integer'Min (Lo + Block, N) - 1;
            Amax : Real := 0.0;
            Scale : Real;
         begin
            for I in Lo .. Hi loop
               if abs Flat (I) > Amax then Amax := abs Flat (I); end if;
            end loop;
            if Amax = 0.0 then
               for I in Lo .. Hi loop
                  Y (X'First (1) + I / Cols, X'First (2) + I mod Cols) := 0.0;
               end loop;
            else
               Scale := Amax / Q_Max;
               for I in Lo .. Hi loop
                  declare
                     Q : Real := Real'Rounding (Flat (I) / Scale);
                     R : constant Integer := X'First (1) + I / Cols;
                     C : constant Integer := X'First (2) + I mod Cols;
                  begin
                     if Q > Q_Max then Q := Q_Max; end if;
                     if Q < -Q_Max then Q := -Q_Max; end if;
                     Y (R, C) := Q * Scale;
                  end;
               end loop;
            end if;
         end;
      end loop;
   end Fake_Quant_Forward_Blocked;

   procedure Fake_Quant_Backward (DY : Matrix; DX : out Matrix) is
   begin
      DX := DY;   -- straight-through estimator: rounding treated as identity
   end Fake_Quant_Backward;

   --------------------------------------------------------------------
   --  RMSNorm
   --------------------------------------------------------------------
   procedure RMSNorm_Forward
     (X, Gamma : Matrix; Y : out Matrix; Eps : Real := 1.0E-6)
   is
      D  : constant Positive := X'Length (2);
      MS, RI : Real;
   begin
      for R in 1 .. X'Length (1) loop
         MS := 0.0;
         for J in 1 .. D loop MS := MS + X (R, J) * X (R, J); end loop;
         RI := 1.0 / Sqrt (MS / Real (D) + Eps);
         for J in 1 .. D loop
            Y (R, J) := X (R, J) * RI * Gamma (1, J);
         end loop;
      end loop;
   end RMSNorm_Forward;

   procedure RMSNorm_Backward
     (X, Gamma, DY : Matrix; DX, DGamma : out Matrix; Eps : Real := 1.0E-6)
   is
      D  : constant Positive := X'Length (2);
      MS, RI, RI3, C, DN, N : Real;
   begin
      DGamma := [others => [others => 0.0]];
      for R in 1 .. X'Length (1) loop
         MS := 0.0;
         for J in 1 .. D loop MS := MS + X (R, J) * X (R, J); end loop;
         RI  := 1.0 / Sqrt (MS / Real (D) + Eps);
         RI3 := RI * RI * RI;
         C := 0.0;
         for J in 1 .. D loop
            N  := X (R, J) * RI;                 -- normalized (pre-gain)
            DN := DY (R, J) * Gamma (1, J);
            DGamma (1, J) := DGamma (1, J) + DY (R, J) * N;
            C := C + DN * X (R, J);
         end loop;
         for J in 1 .. D loop
            DN := DY (R, J) * Gamma (1, J);
            DX (R, J) := RI * DN - (RI3 / Real (D)) * X (R, J) * C;
         end loop;
      end loop;
   end RMSNorm_Backward;

   --------------------------------------------------------------------
   --  Single-head causal self-attention
   --------------------------------------------------------------------
   procedure Attention_Forward (Q, K, V : Matrix; O : out Matrix; A : out Matrix)
   is
      T     : constant Positive := Q'Length (1);
      D     : constant Positive := Q'Length (2);
      Scale : constant Real := 1.0 / Sqrt (Real (D));
      S     : Vector (1 .. T);
      M, Sum, Dot, Acc : Real;
   begin
      A := [others => [others => 0.0]];
      for I in 1 .. T loop
         M := Real'First;
         for J in 1 .. I loop
            Dot := 0.0;
            for E in 1 .. D loop Dot := Dot + Q (I, E) * K (J, E); end loop;
            S (J) := Dot * Scale;
            if S (J) > M then M := S (J); end if;
         end loop;
         Sum := 0.0;
         for J in 1 .. I loop
            S (J) := Exp (S (J) - M);
            Sum := Sum + S (J);
         end loop;
         for J in 1 .. I loop A (I, J) := S (J) / Sum; end loop;
         for E in 1 .. D loop
            Acc := 0.0;
            for J in 1 .. I loop Acc := Acc + A (I, J) * V (J, E); end loop;
            O (I, E) := Acc;
         end loop;
      end loop;
   end Attention_Forward;

   procedure Attention_Backward
     (Q, K, V, A, DOut : Matrix; DQ, DK, DV : out Matrix)
   is
      T     : constant Positive := Q'Length (1);
      D     : constant Positive := Q'Length (2);
      Scale : constant Real := 1.0 / Sqrt (Real (D));
      Acc, DotSum, V0, DS : Real;
   begin
      DQ := [others => [others => 0.0]];
      DK := [others => [others => 0.0]];
      --  dV[j] = sum_{i>=j} A[i,j] * dO[i]
      for J in 1 .. T loop
         for E in 1 .. D loop
            Acc := 0.0;
            for I in J .. T loop Acc := Acc + A (I, J) * DOut (I, E); end loop;
            DV (J, E) := Acc;
         end loop;
      end loop;
      --  per query row i: softmax backward through the scores, then to Q,K
      for I in 1 .. T loop
         declare
            DA : Vector (1 .. I);
         begin
            DotSum := 0.0;
            for J in 1 .. I loop
               V0 := 0.0;
               for E in 1 .. D loop V0 := V0 + DOut (I, E) * V (J, E); end loop;
               DA (J) := V0;
               DotSum := DotSum + A (I, J) * DA (J);
            end loop;
            for J in 1 .. I loop
               DS := A (I, J) * (DA (J) - DotSum);    -- softmax Jacobian
               for E in 1 .. D loop
                  DQ (I, E) := DQ (I, E) + Scale * DS * K (J, E);
                  DK (J, E) := DK (J, E) + Scale * DS * Q (I, E);
               end loop;
            end loop;
         end;
      end loop;
   end Attention_Backward;

   --------------------------------------------------------------------
   --  Multi-head causal self-attention (H heads over disjoint column slices)
   --------------------------------------------------------------------
   procedure MHA_Forward
     (Q, K, V : Matrix; H : Positive; O : out Matrix; A : out Matrix)
   is
      T     : constant Positive := Q'Length (1);
      D     : constant Positive := Q'Length (2);
      HD    : constant Positive := D / H;
      Scale : constant Real := 1.0 / Sqrt (Real (HD));
      S     : Vector (1 .. T);
      M, Sum, Dot, Acc : Real;
   begin
      A := [others => [others => 0.0]];
      for Hi in 1 .. H loop
         declare
            C0 : constant Natural := (Hi - 1) * HD;   -- column offset
            R0 : constant Natural := (Hi - 1) * T;     -- A-row offset
         begin
            for I in 1 .. T loop
               M := Real'First;
               for J in 1 .. I loop
                  Dot := 0.0;
                  for E in 1 .. HD loop Dot := Dot + Q (I, C0 + E) * K (J, C0 + E); end loop;
                  S (J) := Dot * Scale;
                  if S (J) > M then M := S (J); end if;
               end loop;
               Sum := 0.0;
               for J in 1 .. I loop S (J) := Exp (S (J) - M); Sum := Sum + S (J); end loop;
               for J in 1 .. I loop A (R0 + I, J) := S (J) / Sum; end loop;
               for E in 1 .. HD loop
                  Acc := 0.0;
                  for J in 1 .. I loop Acc := Acc + A (R0 + I, J) * V (J, C0 + E); end loop;
                  O (I, C0 + E) := Acc;
               end loop;
            end loop;
         end;
      end loop;
   end MHA_Forward;

   procedure MHA_Backward
     (Q, K, V, A : Matrix; H : Positive; DOut : Matrix; DQ, DK, DV : out Matrix)
   is
      T     : constant Positive := Q'Length (1);
      D     : constant Positive := Q'Length (2);
      HD    : constant Positive := D / H;
      Scale : constant Real := 1.0 / Sqrt (Real (HD));
      Acc, DotSum, V0, DS : Real;
   begin
      DQ := [others => [others => 0.0]];
      DK := [others => [others => 0.0]];
      for Hi in 1 .. H loop
         declare
            C0 : constant Natural := (Hi - 1) * HD;
            R0 : constant Natural := (Hi - 1) * T;
         begin
            for J in 1 .. T loop
               for E in 1 .. HD loop
                  Acc := 0.0;
                  for I in J .. T loop Acc := Acc + A (R0 + I, J) * DOut (I, C0 + E); end loop;
                  DV (J, C0 + E) := Acc;
               end loop;
            end loop;
            for I in 1 .. T loop
               declare
                  DA : Vector (1 .. I);
               begin
                  DotSum := 0.0;
                  for J in 1 .. I loop
                     V0 := 0.0;
                     for E in 1 .. HD loop V0 := V0 + DOut (I, C0 + E) * V (J, C0 + E); end loop;
                     DA (J) := V0;
                     DotSum := DotSum + A (R0 + I, J) * DA (J);
                  end loop;
                  for J in 1 .. I loop
                     DS := A (R0 + I, J) * (DA (J) - DotSum);
                     for E in 1 .. HD loop
                        DQ (I, C0 + E) := DQ (I, C0 + E) + Scale * DS * K (J, C0 + E);
                        DK (J, C0 + E) := DK (J, C0 + E) + Scale * DS * Q (I, C0 + E);
                     end loop;
                  end loop;
               end;
            end loop;
         end;
      end loop;
   end MHA_Backward;

   --------------------------------------------------------------------
   --  RoPE (interleaved), per head
   --------------------------------------------------------------------
   procedure RoPE_Forward (X : Matrix; H : Positive; Base : Real; Y : out Matrix)
   is
      T  : constant Positive := X'Length (1);
      D  : constant Positive := X'Length (2);
      HD : constant Positive := D / H;
      Pos, Theta, C, Sn, A, B : Real;
   begin
      Y := X;     -- (odd dims, if any, pass through unchanged)
      for P in 1 .. T loop
         Pos := Real (P - 1);
         for Hi in 1 .. H loop
            declare
               C0 : constant Natural := (Hi - 1) * HD;
            begin
               for I in 0 .. HD / 2 - 1 loop
                  Theta := Pos / (Base ** (Real (2 * I) / Real (HD)));
                  C := Cos (Theta); Sn := Sin (Theta);
                  A := X (P, C0 + 2 * I + 1);
                  B := X (P, C0 + 2 * I + 2);
                  Y (P, C0 + 2 * I + 1) := A * C - B * Sn;
                  Y (P, C0 + 2 * I + 2) := A * Sn + B * C;
               end loop;
            end;
         end loop;
      end loop;
   end RoPE_Forward;

   procedure RoPE_Backward (DY : Matrix; H : Positive; Base : Real; DX : out Matrix)
   is
      T  : constant Positive := DY'Length (1);
      D  : constant Positive := DY'Length (2);
      HD : constant Positive := D / H;
      Pos, Theta, C, Sn, A, B : Real;
   begin
      DX := DY;
      for P in 1 .. T loop
         Pos := Real (P - 1);
         for Hi in 1 .. H loop
            declare
               C0 : constant Natural := (Hi - 1) * HD;
            begin
               for I in 0 .. HD / 2 - 1 loop
                  Theta := Pos / (Base ** (Real (2 * I) / Real (HD)));
                  C := Cos (Theta); Sn := Sin (Theta);
                  A := DY (P, C0 + 2 * I + 1);
                  B := DY (P, C0 + 2 * I + 2);
                  DX (P, C0 + 2 * I + 1) := A * C + B * Sn;     -- R^T
                  DX (P, C0 + 2 * I + 2) := -A * Sn + B * C;
               end loop;
            end;
         end loop;
      end loop;
   end RoPE_Backward;

   --------------------------------------------------------------------
   --  Row softmax (numerically stable)
   --------------------------------------------------------------------
   procedure Softmax_Rows (Logits : Matrix; P : out Matrix) is
      M, Sum : Real;
   begin
      for R in 1 .. Logits'Length (1) loop
         M := Logits (R, 1);
         for K in 2 .. Logits'Length (2) loop
            if Logits (R, K) > M then M := Logits (R, K); end if;
         end loop;
         Sum := 0.0;
         for K in 1 .. Logits'Length (2) loop
            P (R, K) := Exp (Logits (R, K) - M);
            Sum := Sum + P (R, K);
         end loop;
         for K in 1 .. Logits'Length (2) loop
            P (R, K) := P (R, K) / Sum;
         end loop;
      end loop;
   end Softmax_Rows;

   --------------------------------------------------------------------
   --  KL(teacher || softmax(logits)), mean over rows.
   --------------------------------------------------------------------
   function KL_Loss (Logits, Teacher_P : Matrix) return Real is
      N    : constant Positive := Logits'Length (1);
      V    : constant Positive := Logits'Length (2);
      P    : Matrix (1 .. N, 1 .. V);
      Loss : Real := 0.0;
      T    : Real;
   begin
      Softmax_Rows (Logits, P);
      for R in 1 .. N loop
         for K in 1 .. V loop
            T := Teacher_P (R, K);
            if T > 0.0 then
               --  P comes from softmax (always > 0 in exact arithmetic), but a
               --  very peaked distribution can underflow P to 0.0 in float, and
               --  Log (0.0) = -inf would make the whole loss +inf and stick the
               --  optimiser. Floor P at a tiny positive value before the log.
               Loss := Loss + T * (Log (T)
                 - Log (Real'Max (P (R, K), Real'Small)));
            end if;
         end loop;
      end loop;
      return Loss / Real (N);
   end KL_Loss;

   procedure KL_Backward (Logits, Teacher_P : Matrix; DLogits : out Matrix) is
      N : constant Positive := Logits'Length (1);
      V : constant Positive := Logits'Length (2);
      P : Matrix (1 .. N, 1 .. V);
   begin
      Softmax_Rows (Logits, P);
      --  d/dlogit_k of mean_r KL(t_r || softmax) = (p_k - t_k)/N
      for R in 1 .. N loop
         for K in 1 .. V loop
            DLogits (R, K) := (P (R, K) - Teacher_P (R, K)) / Real (N);
         end loop;
      end loop;
   end KL_Backward;

   --------------------------------------------------------------------
   --  Cross-entropy to hard labels, mean over rows.
   --------------------------------------------------------------------
   function CE_Loss (Logits : Matrix; Target : Label_Array) return Real is
      N    : constant Positive := Logits'Length (1);
      V    : constant Positive := Logits'Length (2);
      P    : Matrix (1 .. N, 1 .. V);
      Loss : Real := 0.0;
   begin
      Softmax_Rows (Logits, P);
      for R in 1 .. N loop
         Loss := Loss - Log (P (R, Target (Target'First + R - 1) + 1));
      end loop;
      return Loss / Real (N);
   end CE_Loss;

   procedure CE_Backward (Logits : Matrix; Target : Label_Array; DLogits : out Matrix) is
      N   : constant Positive := Logits'Length (1);
      V   : constant Positive := Logits'Length (2);
      P   : Matrix (1 .. N, 1 .. V);
      Tgt : Natural;
   begin
      Softmax_Rows (Logits, P);
      for R in 1 .. N loop
         Tgt := Target (Target'First + R - 1) + 1;   -- 1-based class index
         for K in 1 .. V loop
            DLogits (R, K) :=
              (P (R, K) - (if K = Tgt then 1.0 else 0.0)) / Real (N);
         end loop;
      end loop;
   end CE_Backward;

   --------------------------------------------------------------------
   --  Token embedding
   --------------------------------------------------------------------
   procedure Embed_Forward (E : Matrix; Tokens : Label_Array; X : out Matrix) is
      D : constant Positive := E'Length (2);
   begin
      for T in 1 .. Tokens'Length loop
         for J in 1 .. D loop
            X (T, J) := E (Tokens (Tokens'First + T - 1) + 1, J);  -- 0-based id
         end loop;
      end loop;
   end Embed_Forward;

   procedure Embed_Backward (Tokens : Label_Array; DX : Matrix; DE : out Matrix) is
      D   : constant Positive := DE'Length (2);
      Tok : Positive;
   begin
      DE := [others => [others => 0.0]];
      for T in 1 .. Tokens'Length loop
         Tok := Tokens (Tokens'First + T - 1) + 1;
         for J in 1 .. D loop
            DE (Tok, J) := DE (Tok, J) + DX (T, J);   -- scatter-add
         end loop;
      end loop;
   end Embed_Backward;

   --------------------------------------------------------------------
   --  AdamW
   --------------------------------------------------------------------
   function New_Adam (Rows, Cols : Positive) return Adam is
      St : Adam (Rows, Cols);
   begin
      return St;     -- M, V default to 0; T = 0
   end New_Adam;

   procedure Adam_Step
     (W : in out Matrix; G : Matrix; St : in out Adam;
      LR   : Real := 1.0E-2;  B1 : Real := 0.9;  B2 : Real := 0.999;
      Eps  : Real := 1.0E-8;  WD : Real := 0.0;  Clip : Real := 0.0)
   is
      BC1, BC2, MH, VH, Gr : Real;
   begin
      St.T := St.T + 1;
      BC1 := 1.0 - B1 ** St.T;
      BC2 := 1.0 - B2 ** St.T;
      for I in W'Range (1) loop
         for J in W'Range (2) loop
            Gr := G (I, J);
            --  Sanitize a NaN/Inf gradient BEFORE it enters the moment
            --  estimates. Otherwise a single NaN (from a broken backward pass
            --  or an overflow) makes M/V sticky-NaN for the rest of training
            --  (B1*M + (1-B1)*NaN = NaN), and the weights go NaN permanently.
            --  NaN fails 'Valid; +/-Inf fall outside the finite range. Zero
            --  such entries so the optimiser keeps stepping on the rest.
            if (not Gr'Valid)
              or else Gr > Real'Last or else Gr < -Real'Last
            then
               Gr := 0.0;
            end if;
            if Clip > 0.0 then               -- gradient clipping (stabilizer)
               if Gr >  Clip then Gr :=  Clip; end if;
               if Gr < -Clip then Gr := -Clip; end if;
            end if;
            St.M (I, J) := B1 * St.M (I, J) + (1.0 - B1) * Gr;
            St.V (I, J) := B2 * St.V (I, J) + (1.0 - B2) * Gr * Gr;
            MH := St.M (I, J) / BC1;
            VH := St.V (I, J) / BC2;
            --  decoupled weight decay (AdamW)
            W (I, J) := W (I, J) - LR * (MH / (Sqrt (VH) + Eps) + WD * W (I, J));
         end loop;
      end loop;
   end Adam_Step;

   procedure Write_Adam
     (S : access Ada.Streams.Root_Stream_Type'Class; A : Adam) is
   begin
      Integer'Write (S, A.T);
      for I in A.M'Range (1) loop
         for J in A.M'Range (2) loop Real'Write (S, A.M (I, J)); end loop;
      end loop;
      for I in A.V'Range (1) loop
         for J in A.V'Range (2) loop Real'Write (S, A.V (I, J)); end loop;
      end loop;
   end Write_Adam;

   procedure Read_Adam
     (S : access Ada.Streams.Root_Stream_Type'Class; A : in out Adam) is
      Ti : Integer;
   begin
      Integer'Read (S, Ti);
      A.T := Ti;
      for I in A.M'Range (1) loop
         for J in A.M'Range (2) loop Real'Read (S, A.M (I, J)); end loop;
      end loop;
      for I in A.V'Range (1) loop
         for J in A.V'Range (2) loop Real'Read (S, A.V (I, J)); end loop;
      end loop;
   end Read_Adam;

end Train;
