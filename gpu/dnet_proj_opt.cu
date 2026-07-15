// dnet_proj_opt.cu — delta-net prefill projection optimization (dnet lever).
//
// FINDING (microbench gpu/bench_dnet_proj.cu, real hura Q8_0 dims, P=256):
//   The 4 delta-net input projections in aspida_gpu_chain_prefill run as 4
//   separate k_q8_wmma launches:
//       qkv (dim=2048 -> qo=8192)      11.5 ms  x30
//       alpha (dim -> nv=32)            2.3 ms  x30   <- occupancy-starved
//       beta  (dim -> nv=32)            2.3 ms  x30   <- occupancy-starved
//       gate  (dim -> v_dim=4096)       6.9 ms  x30
//                                   -----------
//                              total  ~22.5 ms  x30
//   alpha/beta cost is absurd for a 2048x32 matmul: k_q8_wmma grids (out/16,
//   B/16) = (2,16) = 32 blocks = 32 warps on a 142-SM GPU -> latency bound.
//
//   The loader ALREADY concatenates qkv|alpha|beta|gate into one contiguous
//   Q8_0 weight L.proj (proj_out = qo+2*nv+v_dim = 12352), used by the decode
//   path but NOT by chunked prefill. Issuing ONE k_q8_wmma over the fused
//   weight absorbs alpha/beta as 4 extra row-tiles in a fully-occupied grid:
//       fused single launch            16.9 ms  x30   (1.33x)
//   A trivially-cheap scatter (below) splits the fused [P,proj_out] output back
//   into the separate qkv/ar/br/z buffers the downstream kernels expect:
//       proj_scatter                    0.3 ms  x30
//   NET fused+scatter = 17.2 ms x30  vs  22.5 ms x30  ->  SAVE 5.3 ms/chunk.
//
//   (Multi-warp/wider-tile GEMM rewrites were all SLOWER — the single-warp
//    k_q8_wmma is already the right primitive here; see bench LEVER 3.)
//
// This file holds the scatter kernel. The chain_prefill wiring is a small patch
// to gpu_matvec.cu (see the PATCH PROPOSAL in the lever report).
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Split fused input-projection output comb[P, proj] (row layout per position:
//   [ qkv(qo) | alpha(nv) | beta(nv) | z(v_dim) ] )
// into the separate contiguous buffers the delta-net kernels read.
extern "C" __global__ void k_proj_scatter(
        const float* __restrict__ comb, int P, int proj,
        float* __restrict__ qkv, int qo,
        float* __restrict__ ar,  float* __restrict__ br, int nv,
        float* __restrict__ z,   int vdim) {
    size_t gid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t) P * proj) return;
    int t = (int)(gid / proj), c = (int)(gid % proj);
    float v = comb[(size_t) t * proj + c];
    if      (c < qo)        qkv[(size_t) t * qo   + c]              = v;
    else if (c < qo + nv)   ar [(size_t) t * nv   + (c - qo)]       = v;
    else if (c < qo + 2*nv) br [(size_t) t * nv   + (c - qo - nv)]  = v;
    else                    z  [(size_t) t * vdim + (c - qo - 2*nv)]= v;
}
