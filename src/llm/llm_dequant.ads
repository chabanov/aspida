---------------------------------------------------------------------
-- LLM_Dequant — Q5_K_M and other GGML quantization → FP32
--
-- Q5_K_M is the format used by parts of the Qwen 3.5 model.
-- Layout per Q5_K super-block of 256 elements (block_q5_K in llama.cpp):
--   - d:      FP16 (2 bytes)   super-block scale
--   - dmin:   FP16 (2 bytes)   super-block min
--   - scales: 12 bytes         8 × 6-bit (scale, min) packed
--   - qh:     32 bytes         1 high bit per element (256 / 8)
--   - qs:     128 bytes        4 low bits per element (256 / 2)
--
-- Total per 256-element super-block: 2 + 2 + 12 + 32 + 128 = 176 bytes
--   → 5.5 bits/weight, 176/1024 ≈ 17.2% of FP32.
--
-- Reference: ggml-quants.c from llama.cpp
---------------------------------------------------------------------

with LLM_Tensor;
with LLM_GGUF;

package LLM_Dequant is

   -- Q8_K: 256-element super-block, 1 f16 scale per 256 el, 1 byte per el (int8 qs)
   procedure Dequant_Q8_K (X : String; Q : out LLM_Tensor.Tensor; N : Natural)
     with Pre => N mod 256 = 0;

   -- Q6_K: 256-element super-block (Q6_K_M variant)
   procedure Dequant_Q6_K (X : String; Q : out LLM_Tensor.Tensor; N : Natural)
     with Pre => N mod 256 = 0;

   -- Q4_K: 256-element super-block (Q4_K_M variant)
   procedure Dequant_Q4_K (X : String; Q : out LLM_Tensor.Tensor; N : Natural)
     with Pre => N mod 256 = 0;

   -- Q5_K: legacy, already implemented
   procedure Dequant_Q5_K (X : String; Q : out LLM_Tensor.Tensor; N : Natural)
     with Pre => N mod 256 = 0;

   -- Dequantize any GGML type. Allocates and returns the tensor.
   function Dequantize
     (Info : LLM_GGUF.Tensor_Info;
      Raw  : String)
      return LLM_Tensor.Tensor;

   -- Get the number of FP32 elements this tensor produces after dequantization
   function Dequant_Num_Elements (Info : LLM_GGUF.Tensor_Info) return Natural;

   -- Streaming quantized matrix-vector: y[out] = W[out,in] . x[in], where W is
   -- the (still-quantized) 2D tensor described by Info and held raw in Raw.
   -- Dequantizes ONE output row at a time, so the full F32 weight is never
   -- materialised (peak extra memory = one row). X is [1, in]; result [1, out].
   function QMatVec
     (Info : LLM_GGUF.Tensor_Info;
      Raw  : String;
      X    : LLM_Tensor.Tensor)
      return LLM_Tensor.Tensor;

end LLM_Dequant;
