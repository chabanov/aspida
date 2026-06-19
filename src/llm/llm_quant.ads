---------------------------------------------------------------------
-- LLM_Quant — FP32 -> GGML quantization (the inverse of LLM_Dequant). Lets the
-- training engine export a real quantized GGUF (e.g. a QAT-trained model) that
-- the inference engine serves directly, instead of only F32.
--
-- Q8_0 layout (block_q8_0, 32 elements): f16 scale d + 32 x int8 qs, 34 bytes.
-- Dequant is x[i] = d * qs[i] (see LLM_Dequant.Dequant_Q8_0).
---------------------------------------------------------------------

with LLM_Tensor; use LLM_Tensor;

package LLM_Quant is

   --  Encode a tensor as ggml Q8_0 raw bytes. Numel(X) must be a multiple of 32
   --  (the block size); real weight rows are block-aligned.
   function Quantize_Q8_0 (X : Tensor) return String
     with Pre => Numel (X) mod 32 = 0;

   --  Encode a tensor as ggml Q4_0 raw bytes (4-bit, ~8x smaller than F32):
   --  32-element blocks, f16 scale + 16 packed nibbles. Inverse of
   --  LLM_Dequant.Dequant_Q4_0.
   function Quantize_Q4_0 (X : Tensor) return String
     with Pre => Numel (X) mod 32 = 0;

   --  Encode a tensor as ggml Q5_0 raw bytes (legacy 5-bit, symmetric):
   --  32-element blocks, f16 scale + 4-byte qh (5th bit of each element) + 16
   --  packed low nibbles. Dequant is x = d*(q-16), q in 0..31. Inverse of
   --  LLM_Dequant.Dequant_Q5_0.
   function Quantize_Q5_0 (X : Tensor) return String
     with Pre => Numel (X) mod 32 = 0;

   --  Encode a tensor as ggml Q4_K raw bytes (4-bit K-quant, the common
   --  community format): 256-element super-blocks with per-32 affine
   --  (6-bit scale + 6-bit min) under a shared f16 d/dmin. Inverse of
   --  LLM_Dequant.Dequant_Q4_K. Numel(X) must be a multiple of 256.
   function Quantize_Q4_K (X : Tensor) return String
     with Pre => Numel (X) mod 256 = 0;

   --  Encode a tensor as ggml Q5_K raw bytes (5-bit K-quant; what Q5_K_M mixes
   --  use for most weights): like Q4_K (256-elem super-blocks, per-32 affine
   --  6-bit scale+min under shared f16 d/dmin) but each quant is 5-bit, the
   --  high bit packed into a 32-byte qh. Inverse of LLM_Dequant.Dequant_Q5_K.
   --  Numel(X) must be a multiple of 256.
   function Quantize_Q5_K (X : Tensor) return String
     with Pre => Numel (X) mod 256 = 0;

   --  Encode a tensor as ggml Q6_K raw bytes (6-bit K-quant, the high-fidelity
   --  format *_K_M mixes use for sensitive tensors like output.weight):
   --  256-element super-blocks = 128 B low-nibbles + 32 B high-2-bits + 16
   --  int8 per-16 scales + f16 d. Value = d * sc[i/16] * (q - 32), q in 0..63.
   --  Inverse of LLM_Dequant.Dequant_Q6_K. Numel(X) must be a multiple of 256.
   function Quantize_Q6_K (X : Tensor) return String
     with Pre => Numel (X) mod 256 = 0;

   --  IEEE-754 binary16 (half) encode of a finite Float into 2 little-endian
   --  bytes. Round-to-nearest; denormals flush to zero, overflow to inf.
   procedure F32_To_F16 (X : Float; Lo, Hi : out Character);

end LLM_Quant;
