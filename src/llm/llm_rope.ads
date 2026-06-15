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
   end record;

   -- Create RoPE params. Defaults are the Qwen 3.5 values; the loader passes
   -- the GGUF metadata values when present so other configs work unchanged.
   function Create_Qwen_RoPE
     (Dim       : Integer := 64;
      Freq_Base : Float   := 10_000_000.0;
      Max_Pos   : Integer := 262_144) return RoPE_Params;

   -- Apply rotary embedding to input tensor
   -- X: query or key tensor [1, head_dim] (single head, single token)
   -- Pos: token position (0-based)
   -- Returns: rotated tensor [1, head_dim]
   function Apply (P : RoPE_Params; X : Tensor; Pos : Integer) return Tensor;

end LLM_RoPE;
