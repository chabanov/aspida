---------------------------------------------------------------------
-- LLM_RoPE — Rotary Position Embedding (mRoPE for Qwen 3.5)
--
-- Qwen 3.5 uses multimodal RoPE (mRoPE) with sectioned dimensions:
--   sections = [11, 11, 10, 0]
--   rope_dim = dim_of_first_3_sections = 11 + 11 + 10 = 32 (per head)
--   freq_base = 10_000_000.0
--
-- For text-only inference, we use the 1D positional encoding
-- applied to the first rope_dim dimensions of each head.
--
-- The precomputed cos/sin cache avoids recomputation per token.
---------------------------------------------------------------------

with LLM_Tensor; use LLM_Tensor;

package LLM_RoPE is

   type RoPE_Params is record
      Dim        : Integer; -- rope dimension (e.g. 64 for Qwen)
      Freq_Base  : Float;   -- 1e7 for Qwen 3.5
      Max_Pos    : Integer; -- max context (262144)
      Sections   : Tensor;  -- [1, 4] = [11, 11, 10, 0] (set 0 for unused)
      Use_FF     : Boolean := False;  -- divide theta by per-freq factors?
      Freq_Factors : Tensor;          -- [1, Dim/2] proportional-RoPE divisors
      Interleaved : Boolean := False; -- pair adjacent dims (2i,2i+1) instead of
                                      --  split-half (i,i+d/2). Llama GGUF Q/K
                                      --  weights are permuted for this layout;
                                      --  gemma/qwen use split-half (False).
      Freq_Scale : Float := 1.0;      -- linear position interpolation (PI):
                                      --  theta := pos*Freq_Scale/base^(...).
                                      --  1.0 = none; 1/factor extends context
                                      --  (== llama.cpp rope_freq_scale).
      --  YaRN (NTK-by-parts): per-dimension blend of extrapolation (theta) and
      --  interpolation (Freq_Scale*theta) across [Corr_Low, Corr_High], plus an
      --  attention-temperature M_Scale on cos/sin. M_Scale=1 & Yarn_On=False is
      --  exactly standard RoPE (no-op).
      Yarn_On   : Boolean := False;
      Corr_Low  : Float   := 0.0;
      Corr_High : Float   := 0.0;
      M_Scale   : Float   := 1.0;
   end record;

   -- Switch a params record to interleaved (NORM) rotation. Call for Llama,
   -- whose converter permutes Q/K weights so adjacent pairs rotate together.
   procedure Set_Interleaved (P : in out RoPE_Params; On : Boolean := True);

   -- Create RoPE params. Defaults are the Qwen 3.5 values; the loader passes
   -- the GGUF metadata values when present so other configs work unchanged.
   function Create_Qwen_RoPE
     (Dim       : Integer := 64;
      Freq_Base : Float   := 10_000_000.0;
      Max_Pos   : Integer := 262_144) return RoPE_Params;

   -- Enable proportional RoPE: theta_i := theta_i / FF(i)  (Gemma full-attn
   -- layers use this with rope_freqs.weight). FF must hold Dim/2 values.
   procedure Set_Freq_Factors (P : in out RoPE_Params; FF : Tensor);

   -- Linear RoPE scaling (Position Interpolation). Factor > 1 stretches the
   -- trained context by that ratio (Freq_Scale := 1/Factor). Factor <= 1 is a
   -- no-op. This is the well-understood baseline; YaRN is a future refinement.
   procedure Set_Linear_Scale (P : in out RoPE_Params; Factor : Float);

   -- NTK-aware RoPE scaling: scale the frequency base instead of positions,
   -- base' := base * Factor**(Dim/(Dim-2)). Preserves high-frequency detail
   -- better than linear PI, so it degrades less when extending context. No-op
   -- for Factor <= 1. (A documented stepping stone toward full YaRN.)
   procedure Set_NTK_Scale (P : in out RoPE_Params; Factor : Float);

   -- Full YaRN scaling (per the llama.cpp rope_yarn reference): extrapolate
   -- high-frequency dims, interpolate low-frequency dims, ramp between the
   -- correction dims derived from Beta_Fast/Beta_Slow, and apply the attention
   -- temperature M_Scale = 1 + 0.1*ln(Factor). N_Ctx_Orig is the model's
   -- original trained context. No-op for Factor <= 1.
   procedure Set_Yarn_Scale
     (P : in out RoPE_Params; Factor : Float; N_Ctx_Orig : Integer;
      Beta_Fast : Float := 32.0; Beta_Slow : Float := 1.0);

   -- Apply rotary embedding to input tensor
   -- X: query or key tensor [1, head_dim] (single head, single token)
   -- Pos: token position (0-based)
   -- Returns: rotated tensor [1, head_dim]
   function Apply (P : RoPE_Params; X : Tensor; Pos : Integer) return Tensor;

end LLM_RoPE;
