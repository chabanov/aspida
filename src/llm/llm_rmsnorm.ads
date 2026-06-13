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

   Epsilon : constant Float := 1.0e-6;

   -- Apply RMSNorm to a vector (single token, dim elements)
   -- X: input vector [1, dim]
   -- Weight: learnable scale [dim]
   -- Returns: normalized vector [1, dim]
   function Forward (X : Tensor; Weight : Tensor) return Tensor
     with Pre => Numel (X) = Numel (Weight);

end LLM_RMSNorm;
