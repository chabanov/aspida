---------------------------------------------------------------------
-- Train — from-scratch training primitives for Aspida (no third-party).
--
-- The inference engine is forward-only; training needs gradients. This is
-- the foundation: differentiable ops as explicit forward/backward pairs
-- (hand-coded reverse mode for a fixed architecture), an AdamW optimizer,
-- and the building blocks of the teacher->student logit-distillation loss
-- (row-softmax + KL, plus cross-entropy). FP64 master math on CPU keeps the
-- backward pass verifiable by finite differences (see test_train); the GPU
-- phase will mirror these in FP32/BF16.
--
-- Convention: every Matrix is 1-based, row-major [Rows x Cols]. A "batch" of
-- N token positions with D features is a Matrix [N x D]. A bias/row vector is
-- a Matrix [1 x K].
---------------------------------------------------------------------

private with Interfaces;
with Ada.Streams;

package Train is

   subtype Real is Long_Float;                         -- FP64 master precision

   type Vector is array (Positive range <>) of Real;
   type Matrix is array (Positive range <>, Positive range <>) of Real;

   --------------------------------------------------------------------
   --  Deterministic PRNG (own LCG) — reproducible init & tests.
   --------------------------------------------------------------------
   type RNG is private;
   function  Seeded (Seed : Long_Float) return RNG;
   function  Uniform (G : in out RNG) return Real;     -- [0, 1)
   --  Glorot/Xavier-uniform fill for a [fan_in x fan_out] weight.
   procedure Init_Glorot (W : out Matrix; G : in out RNG);

   --------------------------------------------------------------------
   --  Linear:  Y[N,O] = X[N,I] . W[I,O] + B[1,O]
   --------------------------------------------------------------------
   procedure Linear_Forward (X, W, B : Matrix; Y : out Matrix)
     with Pre => X'Length (2) = W'Length (1)
                 and then Y'Length (1) = X'Length (1)
                 and then Y'Length (2) = W'Length (2)
                 and then B'Length (1) = 1
                 and then B'Length (2) = W'Length (2);

   --  Given upstream dY[N,O], produce input/param grads (overwritten, not
   --  accumulated). DX may be ignored by the first layer.
   procedure Linear_Backward
     (X, W, DY : Matrix; DX, DW, DB : out Matrix)
     with Pre => DY'Length (1) = X'Length (1)
                 and then DY'Length (2) = W'Length (2)
                 and then X'Length (2) = W'Length (1);

   --  Same, but skips the input gradient (first layer / embedding input).
   procedure Linear_Backward_NoDX (X, W, DY : Matrix; DW, DB : out Matrix)
     with Pre => DY'Length (1) = X'Length (1)
                 and then DY'Length (2) = W'Length (2)
                 and then X'Length (2) = W'Length (1);

   --  Bias-free linear (Llama-style): Y[N,O] = X[N,I] . W[I,O].
   procedure Linear_NB_Forward  (X, W : Matrix; Y : out Matrix);
   procedure Linear_NB_Backward (X, W, DY : Matrix; DX, DW : out Matrix);

   --------------------------------------------------------------------
   --  SiLU (swish): Y = X * sigmoid(X), elementwise.
   --------------------------------------------------------------------
   procedure SiLU_Forward  (X : Matrix; Y : out Matrix);
   procedure SiLU_Backward (X, DY : Matrix; DX : out Matrix);

   --------------------------------------------------------------------
   --  RMSNorm with a per-feature gain (Llama-style):
   --     y[r,j] = x[r,j] / sqrt(mean_j(x[r]^2) + Eps) * gamma[j]
   --  Gamma and its gradient DGamma are row vectors [1 x D].
   --------------------------------------------------------------------
   procedure RMSNorm_Forward
     (X, Gamma : Matrix; Y : out Matrix; Eps : Real := 1.0E-6);
   procedure RMSNorm_Backward
     (X, Gamma, DY : Matrix; DX, DGamma : out Matrix; Eps : Real := 1.0E-6);

   --------------------------------------------------------------------
   --  Single-head causal self-attention, scale 1/sqrt(D).
   --  Q,K,V,O are [T x D]; A is the [T x T] lower-triangular attention-weight
   --  matrix produced by the forward pass and consumed by the backward pass.
   --------------------------------------------------------------------
   procedure Attention_Forward  (Q, K, V : Matrix; O : out Matrix; A : out Matrix);
   procedure Attention_Backward
     (Q, K, V, A, DOut : Matrix; DQ, DK, DV : out Matrix);

   --  Multi-head causal self-attention (H heads, head_dim = D/H, scale
   --  1/sqrt(head_dim)). Q,K,V,O are [T x D]; A is [H*T x T] — head h's
   --  attention weights occupy rows (h-1)*T+1 .. h*T. H=1 reduces exactly to
   --  Attention_Forward/Backward.
   procedure MHA_Forward
     (Q, K, V : Matrix; H : Positive; O : out Matrix; A : out Matrix);
   procedure MHA_Backward
     (Q, K, V, A : Matrix; H : Positive; DOut : Matrix; DQ, DK, DV : out Matrix);

   --  RoPE (interleaved/NORM convention, matching the Llama inference path):
   --  per head (head_dim = D/H), rotate adjacent dim pairs (2i,2i+1) of row p
   --  by angle (p-1)/Base^(2i/head_dim). No parameters; backward is the
   --  conjugate rotation. Y/DX may alias-free out-params.
   procedure RoPE_Forward  (X : Matrix; H : Positive; Base : Real; Y : out Matrix);
   procedure RoPE_Backward (DY : Matrix; H : Positive; Base : Real; DX : out Matrix);

   --------------------------------------------------------------------
   --  Row-softmax + losses over logits[N,V].
   --------------------------------------------------------------------
   procedure Softmax_Rows (Logits : Matrix; P : out Matrix);

   --  Distillation loss: mean over rows of KL(teacher_p || softmax(logits)).
   --  Teacher_P rows must be probability distributions (sum 1, >= 0).
   function  KL_Loss     (Logits, Teacher_P : Matrix) return Real;
   procedure KL_Backward (Logits, Teacher_P : Matrix; DLogits : out Matrix);

   --  Cross-entropy to hard labels (Target(r) = 0-based class id), mean rows.
   type Label_Array is array (Positive range <>) of Natural;
   function  CE_Loss     (Logits : Matrix; Target : Label_Array) return Real;
   procedure CE_Backward (Logits : Matrix; Target : Label_Array; DLogits : out Matrix);

   --------------------------------------------------------------------
   --  Token embedding lookup: X[t,:] = E[Tokens(t), :]  (E is [Vocab x D],
   --  Tokens are 0-based ids). Backward scatter-adds into DE[Vocab x D].
   --------------------------------------------------------------------
   procedure Embed_Forward  (E : Matrix; Tokens : Label_Array; X : out Matrix);
   procedure Embed_Backward (Tokens : Label_Array; DX : Matrix; DE : out Matrix);

   --------------------------------------------------------------------
   --  AdamW (decoupled weight decay). One state per parameter matrix.
   --------------------------------------------------------------------
   type Adam (Rows, Cols : Positive) is private;
   function  New_Adam (Rows, Cols : Positive) return Adam;
   --  Clip > 0 clamps each gradient element to [-Clip, Clip] before the
   --  moment update (a simple stabilizer for deeper models).
   procedure Adam_Step
     (W : in out Matrix; G : Matrix; St : in out Adam;
      LR   : Real := 1.0E-2;  B1 : Real := 0.9;  B2 : Real := 0.999;
      Eps  : Real := 1.0E-8;  WD : Real := 0.0;  Clip : Real := 0.0);

   --  Serialize / restore the optimizer moments (M, V) and step count (T) so a
   --  long training run can resume without losing Adam state. The matrices are
   --  written row-major; the reader must target an Adam of identical Rows/Cols.
   procedure Write_Adam
     (S : access Ada.Streams.Root_Stream_Type'Class; A : Adam);
   procedure Read_Adam
     (S : access Ada.Streams.Root_Stream_Type'Class; A : in out Adam);

private

   type RNG is record
      S : Interfaces.Unsigned_64 := 16#853C49E6748FEA9B#;
   end record;

   type Adam (Rows, Cols : Positive) is record
      M : Matrix (1 .. Rows, 1 .. Cols) := [others => [others => 0.0]];
      V : Matrix (1 .. Rows, 1 .. Cols) := [others => [others => 0.0]];
      T : Natural := 0;
   end record;

end Train;
