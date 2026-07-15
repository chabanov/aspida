// bench_mmq.cu — int8 MMQ feasibility vs the fp16-dequant k_q8_wmma baseline.
// Hypothesis: keeping Q8_0 weights as int8 + on-the-fly int8 activation quant +
// int8 tensor-core (wmma s8) / dp4a math is faster than dequant->fp16->wmma,
// because it (a) skips the per-element fp16 dequant that dominates and (b) runs
// on the 2x-rate int8 tensor pipeline. Correctness target = k_q8_wmma OUTPUT on
// the SAME weights (activation int8 rounding is the added error source).
//
// build (box): nvcc -O3 --fmad=false -arch=native bench_mmq.cu -o /tmp/bench_mmq
// run:         flock /tmp/aspida_bench.lock -c /tmp/bench_mmq
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define WMM_TM 16
#define WMM_TN 16
#define WMM_TK 32

__device__ __forceinline__ float f16(const uint8_t *p){ __half h; *reinterpret_cast<uint16_t*>(&h)=(uint16_t)p[0]|((uint16_t)p[1]<<8); return __half2float(h);}

// ---------------- baseline: verbatim k_q8_wmma (fp16 dequant) ----------------
__global__ void k_q8_wmma(const uint8_t *__restrict__ w, const float *__restrict__ x,
                          float *__restrict__ y, int in, int out, int B) {
    int n0 = blockIdx.x * WMM_TN, m0 = blockIdx.y * WMM_TM;
    int lane = threadIdx.x & 31;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ half As[WMM_TM * WMM_TK];
    __shared__ half Bs[WMM_TK * WMM_TN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
    nvcuda::wmma::fill_fragment(cf, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int n = 0; n < WMM_TN; ++n) {
            int gn = n0 + n;
            const uint8_t *bl = w + (size_t) gn * bpr + (size_t) kb * 34;
            float d = f16(bl); const int8_t *qs = (const int8_t *) (bl + 2);
            Bs[(size_t) lane * WMM_TN + n] = __float2half(d * (float) qs[lane]);
        }
        for (int m = 0; m < WMM_TM; ++m) {
            int gm = m0 + m;
            float xv = (gm < B) ? x[(size_t) gm * in + kb * 32 + lane] : 0.0f;
            As[(size_t) m * WMM_TK + lane] = __float2half(xv);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, WMM_TK);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * WMM_TN, WMM_TN);
            nvcuda::wmma::mma_sync(cf, af, bf, cf);
        }
        __syncwarp();
    }
    __shared__ float Cs[WMM_TM * WMM_TN];
    nvcuda::wmma::store_matrix_sync(Cs, cf, WMM_TN, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
        int m = idx / WMM_TN, n = idx % WMM_TN, gm = m0 + m, gn = n0 + n;
        if (gm < B && gn < out) y[(size_t) gm * out + gn] = Cs[idx];
    }
}

// ---------------- on-the-fly activation int8 quant (per 32-block) ------------
// qx[b*in + k] = round(x/scale); sx[b*nb + kb] = scale = maxabs/127. One kernel,
// its cost is AMORTIZED across every consumer of x in the layer (report note).
__global__ void k_quant_x(const float *__restrict__ x, int8_t *__restrict__ qx,
                          float *__restrict__ sx, int in, int B) {
    int b = blockIdx.y; int kb = blockIdx.x * blockDim.x / 32 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31; int nb = in / 32;
    if (b >= B || kb >= nb) return;
    float v = x[(size_t) b * in + kb * 32 + lane];
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_down_sync(0xffffffffu, a, o));
    a = __shfl_sync(0xffffffffu, a, 0);
    float scale = a / 127.0f; float inv = (a > 0.f) ? 127.0f / a : 0.f;
    qx[(size_t) b * in + kb * 32 + lane] = (int8_t) __float2int_rn(v * inv);
    if (lane == 0) sx[(size_t) b * nb + kb] = scale;
}

// ---------------- MMQ variant A: int8 tensor-core (wmma s8) ------------------
// Same tiling as k_q8_wmma but A,B stay int8; per-32-block int32 partial is
// scaled by sx[row]*sw[col] and accumulated in fp32 (scales differ per row/col,
// so accumulation must break at each 32-block boundary).
__global__ void k_q8_mma_s8(const uint8_t *__restrict__ w, const int8_t *__restrict__ qx,
                            const float *__restrict__ sx, float *__restrict__ y,
                            int in, int out, int B) {
    int n0 = blockIdx.x * WMM_TN, m0 = blockIdx.y * WMM_TM;
    int lane = threadIdx.x & 31;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ signed char As[WMM_TM * WMM_TK];   // qx tile [16 M][32 K]
    __shared__ signed char Bs[WMM_TK * WMM_TN];   // qw tile [32 K][16 N]
    __shared__ float swsh[WMM_TN], sxsh[WMM_TM];
    __shared__ int   Ci[WMM_TM * WMM_TN];
    __shared__ float Cacc[WMM_TM * WMM_TN];
    for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) Cacc[idx] = 0.f;
    __syncwarp();
    for (int kb = 0; kb < nb; ++kb) {
        for (int n = 0; n < WMM_TN; ++n) {
            const uint8_t *bl = w + (size_t) (n0 + n) * bpr + (size_t) kb * 34;
            if (lane == 0) swsh[n] = f16(bl);
            Bs[(size_t) lane * WMM_TN + n] = ((const int8_t *) (bl + 2))[lane];
        }
        for (int m = 0; m < WMM_TM; ++m) {
            int gm = m0 + m;
            As[(size_t) m * WMM_TK + lane] = (gm < B) ? qx[(size_t) gm * in + kb * 32 + lane] : (signed char) 0;
            if (lane == 0) sxsh[m] = (gm < B) ? sx[(size_t) gm * nb + kb] : 0.f;
        }
        __syncwarp();
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> ci;
        nvcuda::wmma::fill_fragment(ci, 0);
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, WMM_TK);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * WMM_TN, WMM_TN);
            nvcuda::wmma::mma_sync(ci, af, bf, ci);
        }
        nvcuda::wmma::store_matrix_sync(Ci, ci, WMM_TN, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
            int m = idx / WMM_TN, n = idx % WMM_TN;
            Cacc[idx] += (float) Ci[idx] * sxsh[m] * swsh[n];
        }
        __syncwarp();
    }
    for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
        int m = idx / WMM_TN, n = idx % WMM_TN, gm = m0 + m, gn = n0 + n;
        if (gm < B && gn < out) y[(size_t) gm * out + gn] = Cacc[idx];
    }
}

// ---------------- SPEED PROBE: int8-TC ceiling, NO per-block scale break -----
// Numerically WRONG (ignores Q8 per-block scales; accumulates raw int32 across
// all K, one scale at end). Purpose: isolate whether the int8 tensor-core tiling
// itself beats fp16 wmma, i.e. whether the scale-break is the whole overhead.
__global__ void k_q8_mma_s8_nobreak(const uint8_t *__restrict__ w, const int8_t *__restrict__ qx,
                            float *__restrict__ y, int in, int out, int B) {
    int n0 = blockIdx.x * WMM_TN, m0 = blockIdx.y * WMM_TM;
    int lane = threadIdx.x & 31;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ signed char As[WMM_TM * WMM_TK];
    __shared__ signed char Bs[WMM_TK * WMM_TN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, int> ci;
    nvcuda::wmma::fill_fragment(ci, 0);
    for (int kb = 0; kb < nb; ++kb) {
        for (int n = 0; n < WMM_TN; ++n) {
            const uint8_t *bl = w + (size_t) (n0 + n) * bpr + (size_t) kb * 34;
            Bs[(size_t) lane * WMM_TN + n] = ((const int8_t *) (bl + 2))[lane];
        }
        for (int m = 0; m < WMM_TM; ++m) {
            int gm = m0 + m;
            As[(size_t) m * WMM_TK + lane] = (gm < B) ? qx[(size_t) gm * in + kb * 32 + lane] : (signed char) 0;
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, signed char, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, signed char, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, WMM_TK);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * WMM_TN, WMM_TN);
            nvcuda::wmma::mma_sync(ci, af, bf, ci);
        }
        __syncwarp();
    }
    __shared__ int Cs[WMM_TM * WMM_TN];
    nvcuda::wmma::store_matrix_sync(Cs, ci, WMM_TN, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
        int m = idx / WMM_TN, n = idx % WMM_TN, gm = m0 + m, gn = n0 + n;
        if (gm < B && gn < out) y[(size_t) gm * out + gn] = (float) Cs[idx] * 1e-4f;
    }
}

// ---------------- MMQ variant B: dp4a (no tensor cores) ----------------------
// Tiled BM x BN outputs per block; qx/qw tiles staged in shared; dp4a int32
// per-block partials scaled by sx*sw. BM=16 rows, one warp per output row-group.
// Warp w owns output column-tile of 32 cols; each lane accumulates for 1 col of
// its 32 across all BM rows via dp4a over the K blocks it strides.
// Simpler: one warp computes a full 16x32 tile? Keep it plain: block=8 warps,
// warp w handles output columns [w*4 .. w*4+3]-ish. Use straightforward
// warp-per-(mtile,ncol) with weight reuse across the 16 rows via shared qw.
#define DBM 16
#define DBN 32
__global__ void k_q8_dp4a(const uint8_t *__restrict__ w, const int8_t *__restrict__ qx,
                          const float *__restrict__ sx, float *__restrict__ y,
                          int in, int out, int B) {
    int n0 = blockIdx.x * DBN, m0 = blockIdx.y * DBM;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;   // 8 warps
    // each thread computes outputs for column (n0 + threadIdx.x%DBN)? Use:
    int nloc = threadIdx.x % DBN; int gn = n0 + nloc;        // this thread's column
    int rowgrp = threadIdx.x / DBN;                          // 0..(TPB/DBN-1)
    int nrg = blockDim.x / DBN;
    __shared__ __align__(16) int8_t Xs[DBM * 32];   // one K-block of qx for 16 rows
    __shared__ float  Sx[DBM];
    float acc[DBM];
    #pragma unroll
    for (int m = 0; m < DBM; ++m) acc[m] = 0.f;
    const int32_t *qxp; (void) qxp;
    for (int kb = 0; kb < nb; ++kb) {
        // stage qx block [16 rows][32] into shared, plus sx[row]
        for (int idx = threadIdx.x; idx < DBM * 32; idx += blockDim.x) {
            int m = idx / 32, l = idx % 32; int gm = m0 + m;
            Xs[idx] = (gm < B) ? qx[(size_t) gm * in + kb * 32 + l] : (int8_t) 0;
        }
        if (threadIdx.x < DBM) { int gm = m0 + threadIdx.x; Sx[threadIdx.x] = (gm < B) ? sx[(size_t) gm * nb + kb] : 0.f; }
        __syncthreads();
        if (gn < out) {
            const uint8_t *bl = w + (size_t) gn * bpr + (size_t) kb * 34;
            float sw = f16(bl);
            const uint8_t *qw = (const uint8_t *) (bl + 2);   // int8 payload, offset +2 (unaligned)
            // pack 32 int8 weights into 8 int32 byte-wise (payload is not 4-aligned)
            int wpk[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                wpk[j] = (int) qw[j*4] | ((int) qw[j*4+1] << 8) | ((int) qw[j*4+2] << 16) | ((int) qw[j*4+3] << 24);
            for (int m = rowgrp; m < DBM; m += nrg) {
                const int8_t *xr = Xs + m * 32;
                int dot = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) dot = __dp4a(wpk[j], *(const int *) (xr + j * 4), dot);
                acc[m] += (float) dot * sw * Sx[m];
            }
        }
        __syncthreads();
    }
    if (gn < out) {
        for (int m = rowgrp; m < DBM; m += nrg) {
            int gm = m0 + m; if (gm < B) y[(size_t) gm * out + gn] = acc[m];
        }
    }
}

// ============================================================================
static void fill_q8(std::vector<uint8_t>&w, size_t nrows, int in, std::mt19937&rng){
    int nb=in/32; w.resize(nrows*(size_t)nb*34);
    std::uniform_int_distribution<int> qd(-127,127);
    std::uniform_real_distribution<float> sd(0.005f,0.02f);
    for(size_t r=0;r<nrows;r++) for(int b=0;b<nb;b++){
        uint8_t*blk=&w[(r*nb+b)*34];
        __half h=__float2half(sd(rng)); uint16_t hb=*reinterpret_cast<uint16_t*>(&h);
        blk[0]=hb&0xff; blk[1]=hb>>8;
        for(int j=0;j<32;j++) ((int8_t*)(blk+2))[j]=(int8_t)qd(rng);
    }
}
static float tel(cudaEvent_t a,cudaEvent_t b){float m;cudaEventElapsedTime(&m,a,b);return m;}

struct Shape{const char*name;int in;int out;int B;};

int main(){
    std::mt19937 rng(99);
    Shape shapes[]={ {"dproj in2048->out8192 B512",2048,8192,512},
                     {"gate  in2048->out512  B512",2048,512,512},
                     {"gate  in2048->out512  B16 ",2048,512,16} };
    size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("VRAM free=%zuMB\n",fb/1048576);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for(Shape S:shapes){
        int in=S.in,out=S.out,B=S.B,nb=in/32;
        std::vector<uint8_t> hw; fill_q8(hw,out,in,rng);
        std::vector<float> hx((size_t)B*in);
        std::uniform_real_distribution<float> xd(-1.f,1.f);
        for(auto&v:hx) v=xd(rng)*0.6f;
        uint8_t*dw; CK(cudaMalloc(&dw,hw.size())); CK(cudaMemcpy(dw,hw.data(),hw.size(),cudaMemcpyHostToDevice));
        float*dx; CK(cudaMalloc(&dx,hx.size()*4)); CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
        int8_t*dqx; CK(cudaMalloc(&dqx,(size_t)B*in));
        float*dsx; CK(cudaMalloc(&dsx,(size_t)B*nb*4));
        float*dyw,*dym,*dyd; CK(cudaMalloc(&dyw,(size_t)B*out*4)); CK(cudaMalloc(&dym,(size_t)B*out*4)); CK(cudaMalloc(&dyd,(size_t)B*out*4));

        auto qx=[&](){ dim3 g((nb+7)/8,B); k_quant_x<<<g,256,0,st>>>(dx,dqx,dsx,in,B); };
        auto base=[&](){ dim3 g((out+15)/16,(B+15)/16); k_q8_wmma<<<g,32,0,st>>>(dw,dx,dyw,in,out,B); };
        auto mma=[&](){ dim3 g((out+15)/16,(B+15)/16); k_q8_mma_s8<<<g,32,0,st>>>(dw,dqx,dsx,dym,in,out,B); };
        auto mmanb=[&](){ dim3 g((out+15)/16,(B+15)/16); k_q8_mma_s8_nobreak<<<g,32,0,st>>>(dw,dqx,dym,in,out,B); };
        auto dp4a=[&](){ dim3 g((out+DBN-1)/DBN,(B+DBM-1)/DBM); k_q8_dp4a<<<g,256,0,st>>>(dw,dqx,dsx,dyd,in,out,B); };

        for(int i=0;i<5;i++){ base(); qx(); mma(); dp4a(); }
        CK(cudaStreamSynchronize(st));
        std::vector<float> yw(B*out),ym(B*out),yd(B*out);
        CK(cudaMemcpy(yw.data(),dyw,yw.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ym.data(),dym,ym.size()*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yd.data(),dyd,yd.size()*4,cudaMemcpyDeviceToHost));
        double refmax=0; for(size_t i=0;i<yw.size();i++) refmax=fmax(refmax,fabs(yw[i]));
        double thr=0.05*refmax;   // filter near-zero refs (denominator artifacts)
        double mrel=0,drel=0,md=0,dd=0; for(size_t i=0;i<yw.size();i++){
            double a=fabs(ym[i]-yw[i]); if(a>md)md=a;
            double b=fabs(yd[i]-yw[i]); if(b>dd)dd=b;
            if(fabs(yw[i])>thr){ double r=a/fabs(yw[i]); if(r>mrel)mrel=r;
                                 double s=b/fabs(yw[i]); if(s>drel)drel=s; }
        }
        printf("  [refmax=%.3f, rel over |ref|>0.05*max]\n",refmax);

        const int IT=40;
        auto timeit=[&](auto fn){ float best=1e30f; for(int r=0;r<2;r++){ cudaEventRecord(e0,st);
            for(int i=0;i<IT;i++) fn(); cudaEventRecord(e1,st); cudaEventSynchronize(e1);
            float ms=tel(e0,e1)/IT; if(ms<best)best=ms;} return best; };
        float tb_=timeit(base), tq=timeit(qx), tm=timeit(mma), tmnb=timeit(mmanb), td=timeit(dp4a);

        printf("\n== %s ==\n",S.name);
        printf("  k_q8_wmma (fp16 dequant): %.4f ms\n",tb_);
        printf("  quant_x (amortized/layer): %.4f ms\n",tq);
        printf("  mma.s8  : %.4f ms  (+qx=%.4f)  rel=%.2e absmax=%.2e  speedup %.2fx (%.2fx w/qx)\n",
               tm,tm+tq,mrel,md,tb_/tm,tb_/(tm+tq));
        printf("  mma.s8 NOBREAK (wrong, speed ceiling): %.4f ms  speedup %.2fx  <- pure int8-TC tiling vs fp16\n",
               tmnb,tb_/tmnb);
        printf("  dp4a    : %.4f ms  (+qx=%.4f)  rel=%.2e absmax=%.2e  speedup %.2fx (%.2fx w/qx)\n",
               td,td+tq,drel,dd,tb_/td,tb_/(td+tq));
        cudaFree(dw);cudaFree(dx);cudaFree(dqx);cudaFree(dsx);cudaFree(dyw);cudaFree(dym);cudaFree(dyd);
    }
    return 0;
}
