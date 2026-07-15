// bench_pipeline.cu — push the fp16 k_q8_wmma past its ~25%-of-peak ceiling by
// raising arithmetic intensity + pipelining data movement. All variants keep
// the SAME math/dequant/accumulation ORDER as k_q8_wmma -> must be BIT-EXACT.
//
// Baseline k_q8_wmma: 1 warp = one 16x16 output tile; the weight n-tile is
// re-loaded+re-dequantized for EVERY 16-row m-tile (B/16 times). At B=512 that
// is 32x redundant weight traffic+dequant from L2.
// multiM: one block owns a weight n-tile and WPB m-tiles; weight block is staged
// & dequantized ONCE per kb into shared, reused by all WPB warps -> WPBx less
// weight L2 traffic AND WPBx less dequant ALU. cp.async double-buffers it.
//
// build (box): nvcc -O3 --fmad=false -arch=native bench_pipeline.cu -o /tmp/bp
// run:         flock /tmp/aspida_bench.lock -c /tmp/bp
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define WMM_TM 16
#define WMM_TN 16
#define WMM_TK 32

__device__ __forceinline__ float f16(const uint8_t *p){ __half h; *reinterpret_cast<uint16_t*>(&h)=(uint16_t)p[0]|((uint16_t)p[1]<<8); return __half2float(h);}

// ---------------- baseline: verbatim k_q8_wmma ------------------------------
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

// ---------------- multiM: shared weight tile, WPB m-tiles/block -------------
// Bit-exact: identical dequant values + identical per-tile K-accumulation order.
template<int WPB>
__global__ void k_q8_wmma_multiM(const uint8_t *__restrict__ w, const float *__restrict__ x,
                                 float *__restrict__ y, int in, int out, int B) {
    int n0 = blockIdx.x * WMM_TN;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int m0 = (blockIdx.y * WPB + warp) * WMM_TM;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ half Bs[WMM_TK * WMM_TN];        // weight tile, shared by all warps
    __shared__ half As[WPB][WMM_TM * WMM_TK];   // per-warp x tile
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
    nvcuda::wmma::fill_fragment(cf, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        // cooperative weight dequant (once per block): Bs[k][n] = W[n0+n, kb*32+k]
        for (int idx = threadIdx.x; idx < WMM_TK * WMM_TN; idx += blockDim.x) {
            int k = idx / WMM_TN, n = idx % WMM_TN;
            const uint8_t *bl = w + (size_t) (n0 + n) * bpr + (size_t) kb * 34;
            Bs[idx] = __float2half(f16(bl) * (float) ((const int8_t *) (bl + 2))[k]);
        }
        for (int m = 0; m < WMM_TM; ++m) {
            int gm = m0 + m;
            float xv = (gm < B) ? x[(size_t) gm * in + kb * 32 + lane] : 0.0f;
            As[warp][(size_t) m * WMM_TK + lane] = __float2half(xv);
        }
        __syncthreads();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As[warp] + k16 * 16, WMM_TK);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * WMM_TN, WMM_TN);
            nvcuda::wmma::mma_sync(cf, af, bf, cf);
        }
        __syncthreads();
    }
    __shared__ float Cs[WPB][WMM_TM * WMM_TN];
    nvcuda::wmma::store_matrix_sync(Cs[warp], cf, WMM_TN, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
        int m = idx / WMM_TN, n = idx % WMM_TN, gm = m0 + m, gn = n0 + n;
        if (gm < B && gn < out) y[(size_t) gm * out + gn] = Cs[warp][idx];
    }
}

// ---------------- multiM + cp.async double-buffered weight bytes ------------
// Prefetch the raw 34B weight blocks of kb+1 into shared while dequantizing+
// computing kb. Weight payload is not 16B-aligned so we cp.async per-column
// 34B (as 32B+2B via two async copies is messy) -> instead stage the whole
// 16-col slab (16*34=544B) with 16B cp.async chunks from a padded layout is not
// possible on the raw tensor; so we cp.async the CONTIGUOUS per-row stripe.
// Simpler + still bit-exact: double-buffer the fp16-dequantized Bs is not async
// (dequant is compute). We instead async-prefetch the raw bytes into a byte
// staging buffer, then dequant from shared. Copies are 16B-aligned by copying
// full 34B blocks rounded: we copy 48 bytes/col (over-read into next block, safe
// since bpr rows are contiguous and we never read the pad).
template<int WPB>
__global__ void k_q8_wmma_multiM_async(const uint8_t *__restrict__ w, const float *__restrict__ x,
                                       float *__restrict__ y, int in, int out, int B) {
    int n0 = blockIdx.x * WMM_TN;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int m0 = (blockIdx.y * WPB + warp) * WMM_TM;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ half Bs[WMM_TK * WMM_TN];
    __shared__ half As[WPB][WMM_TM * WMM_TK];
    __shared__ uint8_t Wr[2][WMM_TN * 34];   // double-buffered raw weight blocks (16 cols x 34B)
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
    nvcuda::wmma::fill_fragment(cf, 0.0f);
    auto prefetch = [&](int kb, int buf) {
        // 16 cols * 34B = 544B; copy in 8B chunks (68 chunks) across the block
        for (int c = threadIdx.x; c < WMM_TN * 34 / 8; c += blockDim.x) {
            int col = (c * 8) / 34;  // approximate; instead copy per-column below
            (void) col;
        }
    };
    (void) prefetch;
    // per-column async prefetch helper: each of 16 cols copied by a chosen warp/lane group
    auto load_raw = [&](int kb, int buf) {
        for (int n = warp; n < WMM_TN; n += WPB) {
            const uint8_t *src = w + (size_t) (n0 + n) * bpr + (size_t) kb * 34;
            uint8_t *dst = Wr[buf] + n * 34;
            // 34B: copy 32B (lane<8 -> 4B each) async + tail 2B
            if (lane < 8) __pipeline_memcpy_async(dst + lane * 4, src + lane * 4, 4);
            if (lane == 8) { dst[32] = src[32]; dst[33] = src[33]; }
        }
        __pipeline_commit();
    };
    load_raw(0, 0);
    for (int kb = 0; kb < nb; ++kb) {
        int cur = kb & 1, nxt = (kb + 1) & 1;
        if (kb + 1 < nb) load_raw(kb + 1, nxt);
        __pipeline_wait_prior(kb + 1 < nb ? 1 : 0);
        __syncthreads();
        for (int idx = threadIdx.x; idx < WMM_TK * WMM_TN; idx += blockDim.x) {
            int k = idx / WMM_TN, n = idx % WMM_TN;
            const uint8_t *bl = Wr[cur] + n * 34;
            Bs[idx] = __float2half(f16(bl) * (float) ((const int8_t *) (bl + 2))[k]);
        }
        for (int m = 0; m < WMM_TM; ++m) {
            int gm = m0 + m;
            float xv = (gm < B) ? x[(size_t) gm * in + kb * 32 + lane] : 0.0f;
            As[warp][(size_t) m * WMM_TK + lane] = __float2half(xv);
        }
        __syncthreads();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As[warp] + k16 * 16, WMM_TK);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * WMM_TN, WMM_TN);
            nvcuda::wmma::mma_sync(cf, af, bf, cf);
        }
        __syncthreads();
    }
    __shared__ float Cs[WPB][WMM_TM * WMM_TN];
    nvcuda::wmma::store_matrix_sync(Cs[warp], cf, WMM_TN, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
        int m = idx / WMM_TN, n = idx % WMM_TN, gm = m0 + m, gn = n0 + n;
        if (gm < B && gn < out) y[(size_t) gm * out + gn] = Cs[warp][idx];
    }
}

// ---------------- register-tiled: MT x NT output fragments per warp ---------
// Raises mma:load ratio: load MT A-frags + NT B-frags per kb, issue MT*NT mma.
// Weight dequant amortized over MT m-tiles; x load amortized over NT n-tiles.
// Bit-exact: each output element accumulates the same K in the same order.
template<int MT, int NT>
__global__ void k_q8_reg(const uint8_t *__restrict__ w, const float *__restrict__ x,
                         float *__restrict__ y, int in, int out, int B) {
    int n0 = blockIdx.x * (WMM_TN * NT), m0 = blockIdx.y * (WMM_TM * MT);
    int lane = threadIdx.x & 31;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ half As[MT][WMM_TM * WMM_TK];
    __shared__ half Bs[NT][WMM_TK * WMM_TN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf[MT][NT];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j) nvcuda::wmma::fill_fragment(cf[i][j], 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        // NOTE: no tail guard here (branch in this hot loop collapses perf ~3x).
        // Caller must ensure out % (16*NT) == 0 (true for all real projection
        // shapes: qkv/o=2048, dproj/lm-head large, router=256 — all %64==0).
        #pragma unroll
        for (int j = 0; j < NT; ++j) for (int n = 0; n < WMM_TN; ++n) {
            const uint8_t *bl = w + (size_t) (n0 + j * WMM_TN + n) * bpr + (size_t) kb * 34;
            Bs[j][(size_t) lane * WMM_TN + n] = __float2half(f16(bl) * (float) ((const int8_t *) (bl + 2))[lane]);
        }
        #pragma unroll
        for (int i = 0; i < MT; ++i) for (int m = 0; m < WMM_TM; ++m) {
            int gm = m0 + i * WMM_TM + m;
            float xv = (gm < B) ? x[(size_t) gm * in + kb * 32 + lane] : 0.0f;
            As[i][(size_t) m * WMM_TK + lane] = __float2half(xv);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af[MT];
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf[NT];
            #pragma unroll
            for (int i = 0; i < MT; ++i) nvcuda::wmma::load_matrix_sync(af[i], As[i] + k16 * 16, WMM_TK);
            #pragma unroll
            for (int j = 0; j < NT; ++j) nvcuda::wmma::load_matrix_sync(bf[j], Bs[j] + (size_t) k16 * 16 * WMM_TN, WMM_TN);
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j) nvcuda::wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
        }
        __syncwarp();
    }
    __shared__ float Cs[WMM_TM * WMM_TN];
    #pragma unroll
    for (int i = 0; i < MT; ++i) for (int j = 0; j < NT; ++j) {
        __syncwarp();
        nvcuda::wmma::store_matrix_sync(Cs, cf[i][j], WMM_TN, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
            int m = idx / WMM_TN, n = idx % WMM_TN;
            int gm = m0 + i * WMM_TM + m, gn = n0 + j * WMM_TN + n;
            if (gm < B && gn < out) y[(size_t) gm * out + gn] = Cs[idx];
        }
    }
}

// ---------------- diagnostic: fp16 weights, NO Q8 dequant -------------------
// Isolates the cost of the per-kb Q8 dequant. Same tiling/accumulation as
// baseline but weight is already half in global (2B/val, read directly).
__global__ void k_f16_wmma(const half *__restrict__ w, const float *__restrict__ x,
                           float *__restrict__ y, int in, int out, int B) {
    int n0 = blockIdx.x * WMM_TN, m0 = blockIdx.y * WMM_TM;
    int lane = threadIdx.x & 31;
    int nb = in / 32; size_t wrow = in;   // half elements per weight row
    __shared__ half As[WMM_TM * WMM_TK];
    __shared__ half Bs[WMM_TK * WMM_TN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
    nvcuda::wmma::fill_fragment(cf, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int n = 0; n < WMM_TN; ++n) {
            int gn = n0 + n;
            Bs[(size_t) lane * WMM_TN + n] = w[(size_t) gn * wrow + kb * 32 + lane];
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
    std::mt19937 rng(7);
    Shape shapes[]={ {"qkv    out8192 B1024 -> reg4x4 2048blk",2048,8192,1024},
                     {"o_proj out2048 B1024 -> reg4x4 512blk (SUSPECT)",2048,2048,1024},
                     {"dkt    out512  B1024 -> reg2x2 512blk (SUSPECT)",2048,512,1024},
                     {"dcomb  out12352 B1024-> reg4x4 3088blk",2048,12352,1024} };
    size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("VRAM free=%zuMB\n",fb/1048576);
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int WPB=8;

    for(Shape S:shapes){
        int in=S.in,out=S.out,B=S.B;
        std::vector<uint8_t> hw; fill_q8(hw,out,in,rng);
        std::vector<float> hx((size_t)B*in);
        std::uniform_real_distribution<float> xd(-1.f,1.f);
        for(auto&v:hx) v=xd(rng)*0.6f;
        // fp16 weights (dequanted host-side from the SAME Q8 for the diagnostic)
        int nb=in/32; std::vector<half> hwf((size_t)out*in);
        for(int r=0;r<out;r++) for(int b=0;b<nb;b++){ const uint8_t*blk=&hw[((size_t)r*nb+b)*34];
            __half hs; *reinterpret_cast<uint16_t*>(&hs)=(uint16_t)blk[0]|((uint16_t)blk[1]<<8); float d=__half2float(hs);
            for(int j=0;j<32;j++) hwf[(size_t)r*in+b*32+j]=__float2half(d*(float)((const int8_t*)(blk+2))[j]); }
        uint8_t*dw; CK(cudaMalloc(&dw,hw.size())); CK(cudaMemcpy(dw,hw.data(),hw.size(),cudaMemcpyHostToDevice));
        half*dwf; CK(cudaMalloc(&dwf,hwf.size()*2)); CK(cudaMemcpy(dwf,hwf.data(),hwf.size()*2,cudaMemcpyHostToDevice));
        float*dx; CK(cudaMalloc(&dx,hx.size()*4)); CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
        float*y0,*y1,*y2; CK(cudaMalloc(&y0,(size_t)B*out*4)); CK(cudaMalloc(&y1,(size_t)B*out*4)); CK(cudaMalloc(&y2,(size_t)B*out*4));

        int gy=(B+15)/(16*WPB); if(gy<1)gy=1;
        auto base =[&](){ dim3 g((out+15)/16,(B+15)/16); k_q8_wmma<<<g,32,0,st>>>(dw,dx,y0,in,out,B); };
        auto mm   =[&](){ dim3 g((out+15)/16,gy); k_q8_wmma_multiM<WPB><<<g,WPB*32,0,st>>>(dw,dx,y1,in,out,B); };
        auto f16w =[&](){ dim3 g((out+15)/16,(B+15)/16); k_f16_wmma<<<g,32,0,st>>>(dwf,dx,y2,in,out,B); };
        // register-tiled Q8 (bit-exact). guarded to shapes divisible by tile.
        auto reg=[&](int MT,int NT,float*yr){
            dim3 g((out+16*NT-1)/(16*NT),(B+16*MT-1)/(16*MT));
            if(MT==2&&NT==2) k_q8_reg<2,2><<<g,32,0,st>>>(dw,dx,yr,in,out,B);
            else if(MT==2&&NT==4) k_q8_reg<2,4><<<g,32,0,st>>>(dw,dx,yr,in,out,B);
            else if(MT==4&&NT==4) k_q8_reg<4,4><<<g,32,0,st>>>(dw,dx,yr,in,out,B);
            else if(MT==1&&NT==4) k_q8_reg<1,4><<<g,32,0,st>>>(dw,dx,yr,in,out,B);
        };

        const int IT=40;
        auto timeit=[&](auto fn){ float best=1e30f; for(int r=0;r<2;r++){ cudaEventRecord(e0,st);
            for(int i=0;i<IT;i++) fn(); cudaEventRecord(e1,st); cudaEventSynchronize(e1);
            float ms=tel(e0,e1)/IT; if(ms<best)best=ms;} return best; };
        double wbytes=(double)out*(in/32)*34;
        double flops=2.0*B*out*in;
        auto tfl=[&](double ms){ return flops/(ms*1e-3)/1e12; };
        auto diff=[&](float*yr){ std::vector<float> t(B*out); CK(cudaMemcpy(t.data(),yr,t.size()*4,cudaMemcpyDeviceToHost));
            std::vector<float> a(B*out); CK(cudaMemcpy(a.data(),y0,a.size()*4,cudaMemcpyDeviceToHost));
            double d=0; for(size_t i=0;i<a.size();i++) d=fmax(d,fabs(a[i]-t[i])); return d; };

        for(int i=0;i<5;i++){ base(); mm(); f16w(); }
        CK(cudaStreamSynchronize(st));
        double d1=diff(y1), d2=diff(y2);
        float tb0=timeit(base), tb1=timeit(mm), tb2=timeit(f16w);

        printf("\n== %s ==  (weight Q8=%.1fMB, %.1f GFLOP)\n",S.name,wbytes/1e6,flops/1e9);
        printf("  baseline k_q8_wmma      : %.4f ms  %.1f TFLOP/s\n",tb0,tfl(tb0));
        printf("  multiM  (WPB=%d)         : %.4f ms  %.1f TFLOP/s  %.2fx  exact_absdiff=%.2e\n",WPB,tb1,tfl(tb1),tb0/tb1,d1);
        printf("  f16w (NO dequant diag)  : %.4f ms  %.1f TFLOP/s  %.2fx  vs-q8-absdiff=%.2e\n",tb2,tfl(tb2),tb0/tb2,d2);
        int tiles[][2]={{1,4},{2,2},{2,4},{4,4}};
        for(auto&t:tiles){ int MT=t[0],NT=t[1];
            if(B<16*MT || out%(16*NT)){ printf("  reg %dx%d               : (skip: needs out%%%d==0, B>=%d)\n",MT,NT,16*NT,16*MT); continue; }
            for(int i=0;i<5;i++) reg(MT,NT,y1);
            CK(cudaStreamSynchronize(st)); double dr=diff(y1);
            float tr=timeit([&](){ reg(MT,NT,y1); });
            printf("  reg %dx%d (Q8)           : %.4f ms  %.1f TFLOP/s  %.2fx  exact_absdiff=%.2e\n",MT,NT,tr,tfl(tr),tb0/tr,dr);
        }
        cudaFree(dw);cudaFree(dx);cudaFree(y0);cudaFree(y1);cudaFree(y2);
    }
    return 0;
}
