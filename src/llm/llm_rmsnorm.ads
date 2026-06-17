---------------------------------------------------------------------
-- LLM_RMSNorm — Root Mean Square Layer Normalization
--
-- Formula:  out = x * w / sqrt(mean(x^2) + epsilon)
-- Used by: Qwen 3.5, Llama, Mistral, etc.
-- w: [dim] — learnable weight vector (gamma in PyTorch)
-- epsilon: small constant for numerical stability
---------------------------------------------------------------------

with LLM_Tensor; use LLM_Tensor;

package LLM_RMSNorm is

   --  Default epsilon. NOTE: the correct value is model-specific — read it
   --  from GGUF (<arch>.attention.layer_norm_rms_epsilon) and pass it to
   --  Forward. gemma uses 1e-6; Llama-3 / Qwen use 1e-5. A wrong epsilon
   --  silently degrades output (compounds over layers; English tolerates it,
   --  other languages collapse).
   Epsilon : constant Float := 1.0e-6;

   -- Apply RMSNorm to a vector (single token, dim elements)
   -- X: input vector [1, dim]
   -- Weight: learnable scale [dim]
   -- Eps: model's RMS epsilon (defaults to the legacy 1e-6)
   -- Returns: normalized vector [1, dim]
   function Forward (X : Tensor; Weight : Tensor; Eps : Float := Epsilon)
      return Tensor
     with Pre => Numel (X) = Numel (Weight);

end LLM_RMSNorm;
