// qwen_resident.cu — resident-GPU decode blocks for the Qwen/MoE backend.
//
// Increment 1 (the fused MoE experts) is
// IMPLEMENTED in gpu_matvec.cu (co-located with the K-quant warp kernels +
// g_wcache it reuses), exported as `aspida_gpu_moe_experts` and bound from
// src/llm/llm_qwen_gpu.ad[sb]. This file is kept as the design anchor / home
// for the NEXT increments (resident delta-net + full-attention, then the
// end-to-end resident forward + CUDA graph), which is where the real speedup
// to ollama parity comes from.
//
// EMPIRICAL FINDING (RTX 6000 Ada, Hura Q4_K_M, 200-token decode):
//   per-matvec MoE  : 2.12 tok/s (GPU ~4%)
//   fused resident experts (Increment 1): 2.61 tok/s (GPU ~2%)  => ~1.23x
// The GPU stays ~2-4% utilised in BOTH — so the MoE matvec host<->device
// round-trips were NOT the dominant cost. The bottleneck is the CPU-side work
// between the matvecs: the delta-net recurrence (27/36 layers), the full-attn
// softmax over the KV cache (9/36 layers), the per-matvec attention
// projections, the norms, and the dense-F32 LM head. Reaching parity therefore
// requires making the WHOLE forward resident (Increments 2-3), not just the MoE.
//
// Build (single-nvcc model; --fmad=false keeps quant math bit-exact vs the
// SPARK CPU oracle, -arch=native keeps it portable):
//   nvcc -O3 --fmad=false -arch=native -shared -Xcompiler -fPIC \
//        gpu/gpu_matvec.cu -o libaspidagpu.so
