// bench_moe_real.cu — standalone MoE prefill microbench at REAL hura dims.
//   qwen35moe: dim=2048 intermed=512 n_exp=256 top_k=8 (+1 shared) P=256, Q8_0.
// Replicates verbatim the grouped tensor-core path AND the per-token warp path
// from gpu/gpu_matvec.cu, plus a candidate optimized grouped path, and times
// all three with cudaEvents + validates correctness (per-token = reference).
//
// build (on GPU box):
//   nvcc -O3 --fmad=false -arch=native bench_moe_real.cu -o /tmp/bench_moe_real
// run under the bench lock:
//   flock /tmp/aspida_bench.lock -c /tmp/bench_moe_real
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#define DIM      2048
#define INTERMED 512
#define N_EXP    256
#define TOP_K    8
#ifndef PB
#define PB 256
#endif
#define P        PB
#define MOE_MAXK 8
#define PCH      (PB < 256 ? 256 : PB)

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// ---------------- verbatim helpers -------------------------------------------
__device__ __forceinline__ float f16(const uint8_t *p){ __half h; *reinterpret_cast<uint16_t*>(&h)=(uint16_t)p[0]|((uint16_t)p[1]<<8); return __half2float(h);}
__device__ __forceinline__ float warp_reduce(float a){
  for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffffu,a,o); return a;
}
__device__ __forceinline__ float wrow_q8_0(const uint8_t *r, const float *x, int in, int lane) {
  int nb = in / 32; float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const uint8_t *blk = r + (size_t) b * 34; float d = f16(blk);
    const int8_t *qs = (const int8_t *) (blk + 2);
    acc += d * (float) qs[lane] * x[b * 32 + lane];
  }
  acc = warp_reduce(acc);
  return __shfl_sync(0xffffffffu, acc, 0);
}
struct MoeRoute { int idx[MOE_MAXK]; float w[MOE_MAXK + 1]; };

// ---------------- verbatim GROUPED path (path A) -----------------------------
__global__ void k_moe_group(const MoeRoute *__restrict__ route_b, int Pn, int top_k,
                            int *__restrict__ mg_pos, int *__restrict__ mg_k,
                            int *__restrict__ mg_cnt, int Pstride) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= Pn) return;
    const MoeRoute *route = route_b + p;
    for (int k = 0; k < top_k; ++k) {
        int e = route->idx[k];
        int s = atomicAdd(&mg_cnt[e], 1);
        if (s < Pstride) { mg_pos[(size_t) e * Pstride + s] = p; mg_k[(size_t) e * Pstride + s] = k; }
    }
}
__global__ void k_moe_gu_grouped(
    const uint8_t *__restrict__ gdw, const uint8_t *__restrict__ udw,
    const float *__restrict__ x_b, float *__restrict__ h_b,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k,
    const int *__restrict__ mg_cnt,
    long g_bpe, long u_bpe, int dim, int intermed, int Pstride,
    int top_k, int Pn, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : Pn;
    int m0 = blockIdx.y * 16; if (m0 >= cnt) return;
    int n0 = blockIdx.x * 16;
    int lane = threadIdx.x & 31;
    int nb = dim / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *gW = (mode == 0) ? gdw + (size_t) e * g_bpe : gdw;
    const uint8_t *uW = (mode == 0) ? udw + (size_t) e * u_bpe : udw;
    __shared__ half As[16 * 32];
    __shared__ half Gs[32 * 16], Us[32 * 16];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cg, cu;
    nvcuda::wmma::fill_fragment(cg, 0.0f); nvcuda::wmma::fill_fragment(cu, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int m = 0; m < 16; ++m) {
            int gm = m0 + m;
            int p = (gm < cnt) ? ((mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm) : -1;
            float xv = (p >= 0) ? x_b[(size_t) p * dim + kb * 32 + lane] : 0.0f;
            As[(size_t) m * 32 + lane] = __float2half(xv);
        }
        for (int n = 0; n < 16; ++n) {
            int gn = n0 + n;
            const uint8_t *blG = gW + (size_t) gn * bpr + (size_t) kb * 34;
            float dG = f16(blG); const int8_t *qG = (const int8_t *) (blG + 2);
            Gs[(size_t) lane * 16 + n] = __float2half(dG * (float) qG[lane]);
            const uint8_t *blU = uW + (size_t) gn * bpr + (size_t) kb * 34;
            float dU = f16(blU); const int8_t *qU = (const int8_t *) (blU + 2);
            Us[(size_t) lane * 16 + n] = __float2half(dU * (float) qU[lane]);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bg, bu;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, 32);
            nvcuda::wmma::load_matrix_sync(bg, Gs + (size_t) k16 * 16 * 16, 16);
            nvcuda::wmma::load_matrix_sync(bu, Us + (size_t) k16 * 16 * 16, 16);
            nvcuda::wmma::mma_sync(cg, af, bg, cg);
            nvcuda::wmma::mma_sync(cu, af, bu, cu);
        }
        __syncwarp();
    }
    __shared__ float Cg[16 * 16], Cu[16 * 16];
    nvcuda::wmma::store_matrix_sync(Cg, cg, 16, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(Cu, cu, 16, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < 256; idx += 32) {
        int m = idx / 16, n = idx % 16, gm = m0 + m;
        if (gm < cnt && (n0 + n) < intermed) {
            int p = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
            int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
            float g = Cg[idx], u = Cu[idx];
            h_b[((size_t) p * (MOE_MAXK + 1) + k) * intermed + n0 + n] = (g / (1.f + expf(-g))) * u;
        }
    }
}
__global__ void k_moe_down_grouped(
    const uint8_t *__restrict__ ddw, const float *__restrict__ h_b, float *__restrict__ d_buf,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k, const int *__restrict__ mg_cnt,
    long d_bpe, int intermed, int dim, int Pstride, int top_k, int Pn, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : Pn;
    int m0 = blockIdx.y * 16; if (m0 >= cnt) return;
    int n0 = blockIdx.x * 16;
    int lane = threadIdx.x & 31;
    int nb = intermed / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *W = (mode == 0) ? ddw + (size_t) e * d_bpe : ddw;
    __shared__ half As[16 * 32];
    __shared__ half Bs[32 * 16];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cd;
    nvcuda::wmma::fill_fragment(cd, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int m = 0; m < 16; ++m) {
            int gm = m0 + m; float hv = 0.0f;
            if (gm < cnt) {
                int p = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
                int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
                hv = h_b[((size_t) p * (MOE_MAXK + 1) + k) * intermed + kb * 32 + lane];
            }
            As[(size_t) m * 32 + lane] = __float2half(hv);
        }
        for (int n = 0; n < 16; ++n) {
            const uint8_t *bl = W + (size_t) (n0 + n) * bpr + (size_t) kb * 34;
            float d = f16(bl); const int8_t *q = (const int8_t *) (bl + 2);
            Bs[(size_t) lane * 16 + n] = __float2half(d * (float) q[lane]);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, 32);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * 16, 16);
            nvcuda::wmma::mma_sync(cd, af, bf, cd);
        }
        __syncwarp();
    }
    __shared__ float Cs[16 * 16];
    nvcuda::wmma::store_matrix_sync(Cs, cd, 16, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < 256; idx += 32) {
        int m = idx / 16, n = idx % 16, gm = m0 + m;
        if (gm < cnt && (n0 + n) < dim) {
            int p = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
            int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
            d_buf[((size_t) p * (MOE_MAXK + 1) + k) * dim + n0 + n] = Cs[idx];
        }
    }
}
__global__ void k_moe_combine(const float *__restrict__ d_buf, const MoeRoute *__restrict__ route_b,
                              float *__restrict__ y_b, int top_k, int dim, int Pn) {
    int p = blockIdx.y; if (p >= Pn) return;
    int d = blockIdx.x * blockDim.x + threadIdx.x; if (d >= dim) return;
    const MoeRoute *r = route_b + p;
    const float *D = d_buf + (size_t) p * (MOE_MAXK + 1) * dim;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) acc += r->w[k] * D[(size_t) k * dim + d];
    acc += r->w[MOE_MAXK] * D[(size_t) top_k * dim + d];
    y_b[(size_t) p * dim + d] = acc;
}

// ---------------- verbatim PER-TOKEN path (path B = reference) ----------------
__global__ void k_moe_gu_p_b(const uint8_t *__restrict__ gdw, const uint8_t *__restrict__ udw,
                             const uint8_t *__restrict__ sgdw, const uint8_t *__restrict__ sudw,
                             const float *__restrict__ x_b, float *__restrict__ h_b,
                             const MoeRoute *__restrict__ route_b, int top_k, int dim, int intermed,
                             long g_bpe, long u_bpe, int B) {
    int total = (top_k + 1) * intermed;
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int bl = gw / total, wid = gw % total;
    if (bl >= B) return;
    const MoeRoute *route = route_b + bl;
    const float *x = x_b + (size_t) bl * dim;
    float *h = h_b + (size_t) bl * (MOE_MAXK + 1) * intermed;
    int k = wid / intermed, r = wid % intermed;
    const uint8_t *grow, *urow;
    if (k < top_k) {
        int e = route->idx[k];
        grow = gdw + (size_t) e * g_bpe + (size_t) r * (dim / 32) * 34;
        urow = udw + (size_t) e * u_bpe + (size_t) r * (dim / 32) * 34;
    } else {
        grow = sgdw + (size_t) r * (dim / 32) * 34;
        urow = sudw + (size_t) r * (dim / 32) * 34;
    }
    float g = wrow_q8_0(grow, x, dim, lane);
    float u = wrow_q8_0(urow, x, dim, lane);
    if (lane == 0) h[(size_t) k * intermed + r] = (g / (1.f + expf(-g))) * u;
}
__global__ void k_moe_down_p_b(const uint8_t *__restrict__ ddw, const uint8_t *__restrict__ sddw,
                               const float *__restrict__ h_b, float *__restrict__ y_b,
                               const MoeRoute *__restrict__ route_b, int top_k, int intermed, int dim,
                               long d_bpe, int B) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int bl = gw / dim, wid = gw % dim;
    if (bl >= B) return;
    const MoeRoute *route = route_b + bl;
    const float *h = h_b + (size_t) bl * (MOE_MAXK + 1) * intermed;
    size_t bpr_d = (size_t) (intermed / 32) * 34;
    size_t bpr_s = (size_t) (intermed / 32) * 34;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        const uint8_t *row = ddw + (size_t) route->idx[k] * d_bpe + (size_t) wid * bpr_d;
        acc += route->w[k] * wrow_q8_0(row, h + (size_t) k * intermed, intermed, lane);
    }
    acc += route->w[MOE_MAXK]
           * wrow_q8_0(sddw + (size_t) wid * bpr_s, h + (size_t) top_k * intermed, intermed, lane);
    if (lane == 0) y_b[(size_t) bl * dim + wid] = acc;
}

// ============================================================================
// ---------------- OPTIMIZED grouped path (candidate) -------------------------
// Same math/precision as k_moe_gu_grouped (FP16 wmma) but:
//   (1) WPB warps per block  -> higher occupancy (was 1 warp/block)
//   (2) internal loop over token-tiles (m0) -> grid.y removed, so the
//       15/16 empty y-blocks per expert (avg ~8 tok/expert) vanish, and it
//       stays correct for adversarial buckets of any size.
//   (3) each warp owns a distinct 16-wide n-tile -> no weight-read redundancy.
// Tiled: ONE block per expert (grid.z), WPB warps each owning distinct n-tiles
// (grid.x strides them). The x-tile (As) is staged ONCE in shared and REUSED by
// all warps (x is independent of the output column) -> less shared/warp, so
// occupancy rises past the 1-warp-block ceiling. Token-tile loop kills grid.y.
template<int WPB>
__global__ void k_moe_gu_grouped_opt(
    const uint8_t *__restrict__ gdw, const uint8_t *__restrict__ udw,
    const float *__restrict__ x_b, float *__restrict__ h_b,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k,
    const int *__restrict__ mg_cnt,
    long g_bpe, long u_bpe, int dim, int intermed, int Pstride,
    int top_k, int Pn, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : Pn;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * WPB + warp) * 16;
    int nb = dim / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *gW = (mode == 0) ? gdw + (size_t) e * g_bpe : gdw;
    const uint8_t *uW = (mode == 0) ? udw + (size_t) e * u_bpe : udw;
    __shared__ half As[16 * 32];                       // shared x-tile (all warps)
    __shared__ int  sp[16];                            // positions for m0 tile
    __shared__ half Gs[WPB][32 * 16], Us[WPB][32 * 16];
    __shared__ float Cg[WPB][16 * 16], Cu[WPB][16 * 16];
    for (int m0 = 0; m0 < cnt; m0 += 16) {
        if (warp == 0) for (int m = lane; m < 16; m += 32) {
            int gm = m0 + m;
            sp[m] = (gm < cnt) ? ((mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm) : -1;
        }
        __syncthreads();
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cg, cu;
        nvcuda::wmma::fill_fragment(cg, 0.0f); nvcuda::wmma::fill_fragment(cu, 0.0f);
        for (int kb = 0; kb < nb; ++kb) {
            // cooperative staged x load: 16 rows x 32 cols, all threads
            for (int idx = threadIdx.x; idx < 16 * 32; idx += blockDim.x) {
                int m = idx >> 5, l = idx & 31; int p = sp[m];
                As[(size_t) m * 32 + l] = (p >= 0) ? __float2half(x_b[(size_t) p * dim + kb * 32 + l]) : __float2half(0.0f);
            }
            for (int n = 0; n < 16; ++n) {
                int gn = n0 + n;
                const uint8_t *blG = gW + (size_t) gn * bpr + (size_t) kb * 34;
                float dG = f16(blG); const int8_t *qG = (const int8_t *) (blG + 2);
                Gs[warp][(size_t) lane * 16 + n] = __float2half(dG * (float) qG[lane]);
                const uint8_t *blU = uW + (size_t) gn * bpr + (size_t) kb * 34;
                float dU = f16(blU); const int8_t *qU = (const int8_t *) (blU + 2);
                Us[warp][(size_t) lane * 16 + n] = __float2half(dU * (float) qU[lane]);
            }
            __syncthreads();
            #pragma unroll
            for (int k16 = 0; k16 < 2; ++k16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bg, bu;
                nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, 32);
                nvcuda::wmma::load_matrix_sync(bg, Gs[warp] + (size_t) k16 * 16 * 16, 16);
                nvcuda::wmma::load_matrix_sync(bu, Us[warp] + (size_t) k16 * 16 * 16, 16);
                nvcuda::wmma::mma_sync(cg, af, bg, cg);
                nvcuda::wmma::mma_sync(cu, af, bu, cu);
            }
            __syncthreads();
        }
        nvcuda::wmma::store_matrix_sync(Cg[warp], cg, 16, nvcuda::wmma::mem_row_major);
        nvcuda::wmma::store_matrix_sync(Cu[warp], cu, 16, nvcuda::wmma::mem_row_major);
        for (int idx = lane; idx < 256; idx += 32) {
            int m = idx / 16, n = idx % 16;
            int p = sp[m];
            if (p >= 0 && (n0 + n) < intermed) {
                int gm = m0 + m;
                int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
                float g = Cg[warp][idx], u = Cu[warp][idx];
                h_b[((size_t) p * (MOE_MAXK + 1) + k) * intermed + n0 + n] = (g / (1.f + expf(-g))) * u;
            }
        }
        __syncthreads();
    }
}
template<int WPB>
__global__ void k_moe_down_grouped_opt(
    const uint8_t *__restrict__ ddw, const float *__restrict__ h_b, float *__restrict__ d_buf,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k, const int *__restrict__ mg_cnt,
    long d_bpe, int intermed, int dim, int Pstride, int top_k, int Pn, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : Pn;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n0 = (blockIdx.x * WPB + warp) * 16;
    int nb = intermed / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *W = (mode == 0) ? ddw + (size_t) e * d_bpe : ddw;
    __shared__ half As[16 * 32];               // shared h-tile (all warps)
    __shared__ int  sp[16], sk[16];
    __shared__ half Bs[WPB][32 * 16];
    __shared__ float Cs[WPB][16 * 16];
    for (int m0 = 0; m0 < cnt; m0 += 16) {
        if (warp == 0) for (int m = lane; m < 16; m += 32) {
            int gm = m0 + m;
            if (gm < cnt) { sp[m] = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
                            sk[m] = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k; }
            else { sp[m] = -1; sk[m] = 0; }
        }
        __syncthreads();
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cd;
        nvcuda::wmma::fill_fragment(cd, 0.0f);
        for (int kb = 0; kb < nb; ++kb) {
            for (int idx = threadIdx.x; idx < 16 * 32; idx += blockDim.x) {
                int m = idx >> 5, l = idx & 31; int p = sp[m]; float hv = 0.0f;
                if (p >= 0) hv = h_b[((size_t) p * (MOE_MAXK + 1) + sk[m]) * intermed + kb * 32 + l];
                As[(size_t) m * 32 + l] = __float2half(hv);
            }
            for (int n = 0; n < 16; ++n) {
                const uint8_t *bl = W + (size_t) (n0 + n) * bpr + (size_t) kb * 34;
                float d = f16(bl); const int8_t *q = (const int8_t *) (bl + 2);
                Bs[warp][(size_t) lane * 16 + n] = __float2half(d * (float) q[lane]);
            }
            __syncthreads();
            #pragma unroll
            for (int k16 = 0; k16 < 2; ++k16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
                nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, 32);
                nvcuda::wmma::load_matrix_sync(bf, Bs[warp] + (size_t) k16 * 16 * 16, 16);
                nvcuda::wmma::mma_sync(cd, af, bf, cd);
            }
            __syncthreads();
        }
        nvcuda::wmma::store_matrix_sync(Cs[warp], cd, 16, nvcuda::wmma::mem_row_major);
        for (int idx = lane; idx < 256; idx += 32) {
            int m = idx / 16, n = idx % 16;
            int p = sp[m];
            if (p >= 0 && (n0 + n) < dim)
                d_buf[((size_t) p * (MOE_MAXK + 1) + sk[m]) * dim + n0 + n] = Cs[warp][idx];
        }
        __syncthreads();
    }
}

// ---------------- SCALAR grouped path (candidate 2) --------------------------
// One warp per output column; reads that column's Q8 weight ROW fully
// CONTIGUOUSLY (best coalescing) exactly once, and reuses it across all of the
// expert's bucketed tokens (x served from L2). No shared mem -> max occupancy.
// TM = token accumulators held in registers; loops token-tiles for any cnt.
template<int TM>
__global__ void k_moe_gu_grouped_scalar(
    const uint8_t *__restrict__ gdw, const uint8_t *__restrict__ udw,
    const float *__restrict__ x_b, float *__restrict__ h_b,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k,
    const int *__restrict__ mg_cnt,
    long g_bpe, long u_bpe, int dim, int intermed, int Pstride,
    int top_k, int Pn, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : Pn;
    if (cnt == 0) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int r = blockIdx.x * (blockDim.x >> 5) + warp;   // output column (intermed)
    if (r >= intermed) return;
    int nb = dim / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *gW = ((mode == 0) ? gdw + (size_t) e * g_bpe : gdw) + (size_t) r * bpr;
    const uint8_t *uW = ((mode == 0) ? udw + (size_t) e * u_bpe : udw) + (size_t) r * bpr;
    for (int m0 = 0; m0 < cnt; m0 += TM) {
        int pos[TM]; int nt = 0;
        #pragma unroll
        for (int t = 0; t < TM; ++t) {
            int gm = m0 + t;
            pos[t] = (gm < cnt) ? ((mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm) : -1;
            if (pos[t] >= 0) nt = t + 1;
        }
        float ga[TM], ua[TM];
        #pragma unroll
        for (int t = 0; t < TM; ++t) { ga[t] = 0.f; ua[t] = 0.f; }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *gb = gW + (size_t) b * 34; float dg = f16(gb); float wg = dg * (float) ((const int8_t *) (gb + 2))[lane];
            const uint8_t *ub = uW + (size_t) b * 34; float du = f16(ub); float wu = du * (float) ((const int8_t *) (ub + 2))[lane];
            int col = b * 32 + lane;
            #pragma unroll
            for (int t = 0; t < TM; ++t) if (t < nt) {
                float xv = x_b[(size_t) pos[t] * dim + col];
                ga[t] += wg * xv; ua[t] += wu * xv;
            }
        }
        #pragma unroll
        for (int t = 0; t < TM; ++t) if (t < nt) {
            float g = warp_reduce(ga[t]), u = warp_reduce(ua[t]);
            if (lane == 0) {
                int gm = m0 + t;
                int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
                h_b[((size_t) pos[t] * (MOE_MAXK + 1) + k) * intermed + r] = (g / (1.f + expf(-g))) * u;
            }
        }
    }
}
template<int TM>
__global__ void k_moe_down_grouped_scalar(
    const uint8_t *__restrict__ ddw, const float *__restrict__ h_b, float *__restrict__ d_buf,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k, const int *__restrict__ mg_cnt,
    long d_bpe, int intermed, int dim, int Pstride, int top_k, int Pn, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : Pn;
    if (cnt == 0) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int r = blockIdx.x * (blockDim.x >> 5) + warp;   // output column (dim)
    if (r >= dim) return;
    int nb = intermed / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *W = ((mode == 0) ? ddw + (size_t) e * d_bpe : ddw) + (size_t) r * bpr;
    for (int m0 = 0; m0 < cnt; m0 += TM) {
        int pos[TM], kk[TM]; int nt = 0;
        #pragma unroll
        for (int t = 0; t < TM; ++t) {
            int gm = m0 + t;
            if (gm < cnt) { pos[t] = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
                            kk[t] = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k; nt = t + 1; }
            else pos[t] = -1;
        }
        float da[TM];
        #pragma unroll
        for (int t = 0; t < TM; ++t) da[t] = 0.f;
        for (int b = 0; b < nb; ++b) {
            const uint8_t *wb = W + (size_t) b * 34; float d = f16(wb); float wv = d * (float) ((const int8_t *) (wb + 2))[lane];
            int col = b * 32 + lane;
            #pragma unroll
            for (int t = 0; t < TM; ++t) if (t < nt) {
                float hv = h_b[((size_t) pos[t] * (MOE_MAXK + 1) + kk[t]) * intermed + col];
                da[t] += wv * hv;
            }
        }
        #pragma unroll
        for (int t = 0; t < TM; ++t) if (t < nt) {
            float acc = warp_reduce(da[t]);
            if (lane == 0) d_buf[((size_t) pos[t] * (MOE_MAXK + 1) + kk[t]) * dim + r] = acc;
        }
    }
}

static void fill_q8(std::vector<uint8_t>&w, size_t nrows, int in, std::mt19937&rng){
    // Q8_0: per 32 values -> 2-byte fp16 scale + 32 int8. row = (in/32) blocks.
    int nb = in/32; w.resize(nrows*(size_t)nb*34);
    std::uniform_int_distribution<int> qd(-127,127);
    std::uniform_real_distribution<float> sd(0.005f,0.02f);
    for(size_t r=0;r<nrows;r++) for(int b=0;b<nb;b++){
        uint8_t*blk=&w[(r*nb+b)*34];
        __half h=__float2half(sd(rng)); uint16_t hb=*reinterpret_cast<uint16_t*>(&h);
        blk[0]=hb&0xff; blk[1]=hb>>8;
        for(int j=0;j<32;j++) ((int8_t*)(blk+2))[j]=(int8_t)qd(rng);
    }
}
static float tmax(cudaEvent_t a,cudaEvent_t b){float m;cudaEventElapsedTime(&m,a,b);return m;}

int main(){
    std::mt19937 rng(1234);
    const long g_bpe=(long)INTERMED*(DIM/32)*34;   // per-expert gate bytes
    const long u_bpe=g_bpe;
    const long d_bpe=(long)DIM*(INTERMED/32)*34;    // per-expert down bytes
    printf("dims dim=%d intermed=%d n_exp=%d top_k=%d P=%d | g_bpe=%ld d_bpe=%ld exp_weights=%.1fMB\n",
        DIM,INTERMED,N_EXP,TOP_K,P,g_bpe,d_bpe,(2.0*g_bpe+d_bpe)*N_EXP/1e6);

    // host weights
    std::vector<uint8_t> hg,hu,hd,hsg,hsu,hsd;
    fill_q8(hg,(size_t)N_EXP*INTERMED,DIM,rng);
    fill_q8(hu,(size_t)N_EXP*INTERMED,DIM,rng);
    fill_q8(hd,(size_t)N_EXP*DIM,INTERMED,rng);
    fill_q8(hsg,INTERMED,DIM,rng);
    fill_q8(hsu,INTERMED,DIM,rng);
    fill_q8(hsd,DIM,INTERMED,rng);

    // host x + route
    std::vector<float> hx((size_t)P*DIM);
    std::uniform_real_distribution<float> xd(-1.f,1.f);
    for(auto&v:hx) v=xd(rng)*0.5f;
    std::vector<MoeRoute> hr(P);
    std::uniform_int_distribution<int> ed(0,N_EXP-1);
    for(int p=0;p<P;p++){
        int used[N_EXP]={0}; float wsum=0;
        for(int k=0;k<TOP_K;k++){int e; do{e=ed(rng);}while(used[e]); used[e]=1;
            hr[p].idx[k]=e; hr[p].w[k]=0.5f+0.5f*xd(rng)*0.5f; wsum+=hr[p].w[k];}
        for(int k=0;k<TOP_K;k++) hr[p].w[k]/=wsum;
        hr[p].w[MOE_MAXK]=0.3f;
    }

    // device
    uint8_t *dg,*du,*dd,*dsg,*dsu,*dsd;
    CK(cudaMalloc(&dg,hg.size())); CK(cudaMalloc(&du,hu.size())); CK(cudaMalloc(&dd,hd.size()));
    CK(cudaMalloc(&dsg,hsg.size())); CK(cudaMalloc(&dsu,hsu.size())); CK(cudaMalloc(&dsd,hsd.size()));
    CK(cudaMemcpy(dg,hg.data(),hg.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(du,hu.data(),hu.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd,hd.data(),hd.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsg,hsg.data(),hsg.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsu,hsu.data(),hsu.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsd,hsd.data(),hsd.size(),cudaMemcpyHostToDevice));
    float *dx; CK(cudaMalloc(&dx,hx.size()*4)); CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
    MoeRoute *dr; CK(cudaMalloc(&dr,P*sizeof(MoeRoute))); CK(cudaMemcpy(dr,hr.data(),P*sizeof(MoeRoute),cudaMemcpyHostToDevice));

    int *mg_pos,*mg_k,*mg_cnt;
    CK(cudaMalloc(&mg_pos,(size_t)N_EXP*PCH*4)); CK(cudaMalloc(&mg_k,(size_t)N_EXP*PCH*4)); CK(cudaMalloc(&mg_cnt,N_EXP*4));
    float *dhb,*dmoe,*dy_g,*dy_p; float *dhb_p;
    CK(cudaMalloc(&dhb,(size_t)P*(MOE_MAXK+1)*INTERMED*4));
    CK(cudaMalloc(&dhb_p,(size_t)P*(MOE_MAXK+1)*INTERMED*4));
    CK(cudaMalloc(&dmoe,(size_t)P*(MOE_MAXK+1)*DIM*4));
    CK(cudaMalloc(&dy_g,(size_t)P*DIM*4));
    CK(cudaMalloc(&dy_p,(size_t)P*DIM*4));

    size_t freeB,totB; cudaMemGetInfo(&freeB,&totB);
    printf("VRAM after alloc: free=%zuMB\n",freeB/1048576);

    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0,e1,e2,e3; cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);cudaEventCreate(&e3);

    auto run_grouped=[&](float*yout){
        cudaMemsetAsync(mg_cnt,0,N_EXP*4,st);
        k_moe_group<<<(P+255)/256,256,0,st>>>(dr,P,TOP_K,mg_pos,mg_k,mg_cnt,PCH);
        dim3 grR((INTERMED+15)/16,(PCH+15)/16,N_EXP);
        k_moe_gu_grouped<<<grR,32,0,st>>>(dg,du,dx,dhb,mg_pos,mg_k,mg_cnt,g_bpe,u_bpe,DIM,INTERMED,PCH,TOP_K,P,0);
        dim3 grSh((INTERMED+15)/16,(P+15)/16,1);
        k_moe_gu_grouped<<<grSh,32,0,st>>>(dsg,dsu,dx,dhb,mg_pos,mg_k,mg_cnt,0,0,DIM,INTERMED,PCH,TOP_K,P,1);
        dim3 grD((DIM+15)/16,(PCH+15)/16,N_EXP);
        k_moe_down_grouped<<<grD,32,0,st>>>(dd,dhb,dmoe,mg_pos,mg_k,mg_cnt,d_bpe,INTERMED,DIM,PCH,TOP_K,P,0);
        dim3 grDs((DIM+15)/16,(P+15)/16,1);
        k_moe_down_grouped<<<grDs,32,0,st>>>(dsd,dhb,dmoe,mg_pos,mg_k,mg_cnt,0,INTERMED,DIM,PCH,TOP_K,P,1);
        dim3 grC((DIM+255)/256,P,1);
        k_moe_combine<<<grC,256,0,st>>>(dmoe,dr,yout,TOP_K,DIM,P);
    };
    auto run_pertok=[&](float*yout){
        int total=(TOP_K+1)*INTERMED;
        int b1=((size_t)P*total*32+255)/256, b2=((size_t)P*DIM*32+255)/256;
        k_moe_gu_p_b<<<b1,256,0,st>>>(dg,du,dsg,dsu,dx,dhb_p,dr,TOP_K,DIM,INTERMED,g_bpe,u_bpe,P);
        k_moe_down_p_b<<<b2,256,0,st>>>(dd,dsd,dhb_p,yout,dr,TOP_K,INTERMED,DIM,d_bpe,P);
    };

    // warmup
    for(int i=0;i<5;i++){ run_grouped(dy_g); run_pertok(dy_p); }
    CK(cudaStreamSynchronize(st));

    // correctness: per-token (B) is reference
    std::vector<float> yg(P*DIM),yp(P*DIM);
    CK(cudaMemcpy(yg.data(),dy_g,yg.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(yp.data(),dy_p,yp.size()*4,cudaMemcpyDeviceToHost));
    double maxrel=0,maxabs=0; for(size_t i=0;i<yg.size();i++){
        double a=yp[i],b=yg[i],d=fabs(a-b); if(d>maxabs)maxabs=d;
        double rel=d/(fabs(a)+1e-6); if(rel>maxrel)maxrel=rel;
    }
    printf("CORRECTNESS grouped vs per-token: max_abs=%.3e max_rel=%.3e\n",maxabs,maxrel);

    const int IT=40;
    auto time_it=[&](const char*name, auto fn, float*yout){
        float best=1e30f,sum=0;
        for(int rep=0;rep<2;rep++){
            cudaEventRecord(e0,st);
            for(int i=0;i<IT;i++) fn(yout);
            cudaEventRecord(e1,st); cudaEventSynchronize(e1);
            float ms=tmax(e0,e1)/IT; if(ms<best)best=ms; sum+=ms;
            printf("  %-16s run%d: %.3f ms/iter\n",name,rep,ms);
        }
        printf("  >> %-13s best=%.3f ms  (per whole-model x40 layers = %.1f ms)\n",name,best,best*40);
        return best;
    };
    // split timing for grouped: gu vs down vs group/combine
    auto time_split=[&](){
        cudaMemsetAsync(mg_cnt,0,N_EXP*4,st);
        // group
        cudaEventRecord(e0,st);
        for(int i=0;i<IT;i++){ cudaMemsetAsync(mg_cnt,0,N_EXP*4,st); k_moe_group<<<(P+255)/256,256,0,st>>>(dr,P,TOP_K,mg_pos,mg_k,mg_cnt,PCH);}
        cudaEventRecord(e1,st); cudaEventSynchronize(e1);
        // gu (routed+shared)
        dim3 grR((INTERMED+15)/16,(PCH+15)/16,N_EXP), grSh((INTERMED+15)/16,(P+15)/16,1);
        cudaEventRecord(e1,st);
        for(int i=0;i<IT;i++){
            k_moe_gu_grouped<<<grR,32,0,st>>>(dg,du,dx,dhb,mg_pos,mg_k,mg_cnt,g_bpe,u_bpe,DIM,INTERMED,PCH,TOP_K,P,0);
            k_moe_gu_grouped<<<grSh,32,0,st>>>(dsg,dsu,dx,dhb,mg_pos,mg_k,mg_cnt,0,0,DIM,INTERMED,PCH,TOP_K,P,1);
        }
        cudaEventRecord(e2,st); cudaEventSynchronize(e2);
        // down + combine
        dim3 grD((DIM+15)/16,(PCH+15)/16,N_EXP), grDs((DIM+15)/16,(P+15)/16,1), grC((DIM+255)/256,P,1);
        cudaEventRecord(e2,st);
        for(int i=0;i<IT;i++){
            k_moe_down_grouped<<<grD,32,0,st>>>(dd,dhb,dmoe,mg_pos,mg_k,mg_cnt,d_bpe,INTERMED,DIM,PCH,TOP_K,P,0);
            k_moe_down_grouped<<<grDs,32,0,st>>>(dsd,dhb,dmoe,mg_pos,mg_k,mg_cnt,0,INTERMED,DIM,PCH,TOP_K,P,1);
            k_moe_combine<<<grC,256,0,st>>>(dmoe,dr,dy_g,TOP_K,DIM,P);
        }
        cudaEventRecord(e3,st); cudaEventSynchronize(e3);
        printf("SPLIT grouped: group=%.3f  gu=%.3f  down+combine=%.3f ms/iter\n",
            tmax(e0,e1)/IT, tmax(e1,e2)/IT, tmax(e2,e3)/IT);
    };

    // optimized grouped: WPB warps/block, internal token-tile loop (no grid.y)
#ifndef OWPB
#define OWPB 2
#endif
    const int WPB=OWPB;
    float *dhb_o; CK(cudaMalloc(&dhb_o,(size_t)P*(MOE_MAXK+1)*INTERMED*4));
    float *dmoe_o; CK(cudaMalloc(&dmoe_o,(size_t)P*(MOE_MAXK+1)*DIM*4));
    float *dy_o; CK(cudaMalloc(&dy_o,(size_t)P*DIM*4));
    auto run_opt=[&](float*yout){
        cudaMemsetAsync(mg_cnt,0,N_EXP*4,st);
        k_moe_group<<<(P+255)/256,256,0,st>>>(dr,P,TOP_K,mg_pos,mg_k,mg_cnt,PCH);
        dim3 grR(INTERMED/(16*WPB),1,N_EXP);
        k_moe_gu_grouped_opt<WPB><<<grR,WPB*32,0,st>>>(dg,du,dx,dhb_o,mg_pos,mg_k,mg_cnt,g_bpe,u_bpe,DIM,INTERMED,PCH,TOP_K,P,0);
        dim3 grSh(INTERMED/(16*WPB),1,1);
        k_moe_gu_grouped_opt<WPB><<<grSh,WPB*32,0,st>>>(dsg,dsu,dx,dhb_o,mg_pos,mg_k,mg_cnt,0,0,DIM,INTERMED,PCH,TOP_K,P,1);
        dim3 grD(DIM/(16*WPB),1,N_EXP);
        k_moe_down_grouped_opt<WPB><<<grD,WPB*32,0,st>>>(dd,dhb_o,dmoe_o,mg_pos,mg_k,mg_cnt,d_bpe,INTERMED,DIM,PCH,TOP_K,P,0);
        dim3 grDs(DIM/(16*WPB),1,1);
        k_moe_down_grouped_opt<WPB><<<grDs,WPB*32,0,st>>>(dsd,dhb_o,dmoe_o,mg_pos,mg_k,mg_cnt,0,INTERMED,DIM,PCH,TOP_K,P,1);
        dim3 grC((DIM+255)/256,P,1);
        k_moe_combine<<<grC,256,0,st>>>(dmoe_o,dr,yout,TOP_K,DIM,P);
    };
    // diagnostic: ORIGINAL kernel but grid.y=G (cap tokens/expert at G*16) to
    // isolate empty-y-block overhead. Also print max bucket count.
    { std::vector<int> hc(N_EXP); cudaMemcpy(hc.data(),mg_cnt,N_EXP*4,cudaMemcpyDeviceToHost);
      int mx=0,ne=0; for(int e=0;e<N_EXP;e++){ if(hc[e]>mx)mx=hc[e]; if(hc[e])ne++; }
      printf("bucket stats: max_cnt=%d nonempty_experts=%d/%d (grid.y needs >= %d tiles)\n",mx,ne,N_EXP,(mx+15)/16); }
    auto run_gyN=[&](float*yout,int G){
        cudaMemsetAsync(mg_cnt,0,N_EXP*4,st);
        k_moe_group<<<(P+255)/256,256,0,st>>>(dr,P,TOP_K,mg_pos,mg_k,mg_cnt,PCH);
        dim3 grR((INTERMED+15)/16,G,N_EXP);
        k_moe_gu_grouped<<<grR,32,0,st>>>(dg,du,dx,dhb,mg_pos,mg_k,mg_cnt,g_bpe,u_bpe,DIM,INTERMED,PCH,TOP_K,P,0);
        dim3 grSh((INTERMED+15)/16,(P+15)/16,1);
        k_moe_gu_grouped<<<grSh,32,0,st>>>(dsg,dsu,dx,dhb,mg_pos,mg_k,mg_cnt,0,0,DIM,INTERMED,PCH,TOP_K,P,1);
        dim3 grD((DIM+15)/16,G,N_EXP);
        k_moe_down_grouped<<<grD,32,0,st>>>(dd,dhb,dmoe,mg_pos,mg_k,mg_cnt,d_bpe,INTERMED,DIM,PCH,TOP_K,P,0);
        dim3 grDs((DIM+15)/16,(P+15)/16,1);
        k_moe_down_grouped<<<grDs,32,0,st>>>(dsd,dhb,dmoe,mg_pos,mg_k,mg_cnt,0,INTERMED,DIM,PCH,TOP_K,P,1);
        dim3 grC((DIM+255)/256,P,1);
        k_moe_combine<<<grC,256,0,st>>>(dmoe,dr,yout,TOP_K,DIM,P);
    };
    for(int i=0;i<5;i++) run_opt(dy_o);
    CK(cudaStreamSynchronize(st));
    std::vector<float> yo(P*DIM);
    CK(cudaMemcpy(yo.data(),dy_o,yo.size()*4,cudaMemcpyDeviceToHost));
    double orel=0,oabs=0; for(size_t i=0;i<yo.size();i++){
        double a=yg[i],b=yo[i],d=fabs(a-b); if(d>oabs)oabs=d;
        double rel=d/(fabs(a)+1e-6); if(rel>orel)orel=rel;
    }
    printf("CORRECTNESS opt vs grouped-baseline: max_abs=%.3e max_rel=%.3e\n",oabs,orel);

    // scalar grouped path
    const int SW=4;               // warps per block
    const int TM=8;               // token accumulators
    float *dhb_s; CK(cudaMalloc(&dhb_s,(size_t)P*(MOE_MAXK+1)*INTERMED*4));
    float *dmoe_s; CK(cudaMalloc(&dmoe_s,(size_t)P*(MOE_MAXK+1)*DIM*4));
    float *dy_s; CK(cudaMalloc(&dy_s,(size_t)P*DIM*4));
    auto run_scalar=[&](float*yout){
        cudaMemsetAsync(mg_cnt,0,N_EXP*4,st);
        k_moe_group<<<(P+255)/256,256,0,st>>>(dr,P,TOP_K,mg_pos,mg_k,mg_cnt,PCH);
        dim3 grR(INTERMED/SW,1,N_EXP);
        k_moe_gu_grouped_scalar<TM><<<grR,SW*32,0,st>>>(dg,du,dx,dhb_s,mg_pos,mg_k,mg_cnt,g_bpe,u_bpe,DIM,INTERMED,PCH,TOP_K,P,0);
        dim3 grSh(INTERMED/SW,1,1);
        k_moe_gu_grouped_scalar<TM><<<grSh,SW*32,0,st>>>(dsg,dsu,dx,dhb_s,mg_pos,mg_k,mg_cnt,0,0,DIM,INTERMED,PCH,TOP_K,P,1);
        dim3 grD(DIM/SW,1,N_EXP);
        k_moe_down_grouped_scalar<TM><<<grD,SW*32,0,st>>>(dd,dhb_s,dmoe_s,mg_pos,mg_k,mg_cnt,d_bpe,INTERMED,DIM,PCH,TOP_K,P,0);
        dim3 grDs(DIM/SW,1,1);
        k_moe_down_grouped_scalar<TM><<<grDs,SW*32,0,st>>>(dsd,dhb_s,dmoe_s,mg_pos,mg_k,mg_cnt,0,INTERMED,DIM,PCH,TOP_K,P,1);
        dim3 grC((DIM+255)/256,P,1);
        k_moe_combine<<<grC,256,0,st>>>(dmoe_s,dr,yout,TOP_K,DIM,P);
    };
    for(int i=0;i<5;i++) run_scalar(dy_s);
    CK(cudaStreamSynchronize(st));
    std::vector<float> ys(P*DIM);
    CK(cudaMemcpy(ys.data(),dy_s,ys.size()*4,cudaMemcpyDeviceToHost));
    double srel=0,sabs=0,sprel=0; for(size_t i=0;i<ys.size();i++){
        double d=fabs(ys[i]-yg[i]); if(d>sabs)sabs=d; double rel=d/(fabs(yg[i])+1e-6); if(rel>srel)srel=rel;
        double dp=fabs(ys[i]-yp[i])/(fabs(yp[i])+1e-6); if(dp>sprel)sprel=dp;
    }
    printf("CORRECTNESS scalar vs grouped-baseline: max_abs=%.3e max_rel=%.3e (vs per-token max_rel=%.3e)\n",sabs,srel,sprel);

    printf("\n== TIMINGS (1 layer, P=256) ==\n");
    float bg=time_it("grouped",run_grouped,dy_g);
    float bp=time_it("per-token",run_pertok,dy_p);
    float bo=time_it("opt-grouped",run_opt,dy_o);
    float bs=time_it("scalar-grouped",run_scalar,dy_s);
    printf("  (scalar speedup %.2fx vs baseline)\n",bg/bs);
    float bg1=time_it("grouped-gy1",[&](float*y){run_gyN(y,1);},dy_o);
    float bg2=time_it("grouped-gy2",[&](float*y){run_gyN(y,2);},dy_o);
    printf("  (gy1 speedup %.2fx, gy2 speedup %.2fx vs baseline)\n",bg/bg1,bg/bg2);
    time_split();
    printf("\nbaseline grouped ms/layer=%.3f -> x40=%.1f ms (measured server moe~155ms)\n",bg,bg*40);
    printf("OPT      grouped ms/layer=%.3f -> x40=%.1f ms  (speedup %.2fx)\n",bo,bo*40,bg/bo);
    (void)bp;
    return 0;
}
