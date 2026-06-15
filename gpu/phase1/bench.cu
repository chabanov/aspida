// Stream B · Phase 1 — per-token throughput benchmark at Llama-3.3-70B scale.
// Times the DOMINANT decode cost (the 7 Q4_K projections per layer x 80 layers
// + the output projection) with the validated k_matvec kernel, weights resident
// in VRAM. Synthetic weights (content irrelevant to timing). RMSNorm/RoPE/attn
// are small and excluded here (validated separately in ops.cu).
//
// Answers: roughly how many tok/s the current (un-tiled) GPU kernels give vs the
// CPU engine's ~0.1 tok/s. Build:  nvcc -O3 --fmad=false -arch=native bench.cu -o bench
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

__device__ __forceinline__ float f16(const uint8_t *p) {
    __half h; *reinterpret_cast<uint16_t *>(&h) = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
    return __half2float(h);
}
__device__ __forceinline__ void gsm(const uint8_t *sc, int j, int *d, int *m) {
    if (j < 4) { *d = sc[j] & 63; *m = sc[j + 4] & 63; }
    else { *d = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4); *m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4); }
}
__device__ void deq_block(const uint8_t *b, float *out) {
    float d = f16(b), dmin = f16(b + 2);
    const uint8_t *sc = b + 4, *qs = b + 16;
    for (int g = 0; g < 4; ++g) {
        int s1,m1,s2,m2; gsm(sc,2*g,&s1,&m1); gsm(sc,2*g+1,&s2,&m2);
        float d1=d*s1, mm1=dmin*m1, d2=d*s2, mm2=dmin*m2;
        const uint8_t *q = qs + g*32;
        for (int l=0;l<32;++l) out[64*g+l]    = d1*(q[l]&0x0F)-mm1;
        for (int l=0;l<32;++l) out[64*g+32+l] = d2*(q[l]>>4)-mm2;
    }
}
__global__ void k_matvec(const uint8_t *w, const float *x, float *y, int in_dim, int out_dim) {
    int o = blockIdx.x*blockDim.x + threadIdx.x; if (o>=out_dim) return;
    int nblk = in_dim/256; size_t bpr=(size_t)nblk*144; const uint8_t*row=w+(size_t)o*bpr;
    float tmp[256], acc=0.0f;
    for (int b=0;b<nblk;++b){ deq_block(row+(size_t)b*144,tmp); int base=b*256; for(int l=0;l<256;++l) acc+=tmp[l]*x[base+l]; }
    y[o]=acc;
}

static uint8_t *dweight(long in, long out) {       // Q4_K device buffer for [in,out]
    size_t bytes = (size_t)(in/256)*144*out; uint8_t *p; cudaMalloc(&p, bytes); cudaMemset(p, 0xAB, bytes); return p;
}
static void mv(uint8_t *w, float *x, float *y, int in, int out) {
    k_matvec<<<(out+127)/128,128>>>(w,x,y,in,out);
}

int main() {
    const int DIM=8192, FFN=28672, KV=1024, VOCAB=128256, L=80;
    // one layer's weights (reused across layers — identical compute) + output.
    uint8_t *wq=dweight(DIM,DIM), *wk=dweight(DIM,KV), *wv=dweight(DIM,KV), *wo=dweight(DIM,DIM);
    uint8_t *wg=dweight(DIM,FFN), *wu=dweight(DIM,FFN), *wd=dweight(FFN,DIM), *wout=dweight(DIM,VOCAB);
    float *x, *y, *yk, *yf;
    cudaMalloc(&x, FFN*sizeof(float)); cudaMalloc(&y, DIM*sizeof(float));
    cudaMalloc(&yk, KV*sizeof(float)); cudaMalloc(&yf, FFN*sizeof(float));
    cudaMemset(x,1,FFN*sizeof(float));

    auto token = [&](){
        for (int l=0; l<L; ++l) {
            mv(wq,x,y,DIM,DIM); mv(wk,x,yk,DIM,KV); mv(wv,x,yk,DIM,KV); mv(wo,x,y,DIM,DIM);
            mv(wg,x,yf,DIM,FFN); mv(wu,x,yf,DIM,FFN); mv(wd,yf,y,FFN,DIM);
        }
        mv(wout,x,y,DIM,VOCAB);
    };

    token(); cudaDeviceSynchronize();             // warmup
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int N=5; cudaEventRecord(t0);
    for (int i=0;i<N;++i) token();
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1); ms/=N;
    printf("Llama-70B Q4_K decode (matmul-only), current un-tiled k_matvec:\n");
    printf("  per-token: %.1f ms  ->  %.2f tok/s   (CPU engine ~0.1 tok/s)\n", ms, 1000.0/ms);
    printf("  note: matvec only; attention/norms excluded (minor). Tiling/batched GEMM is the next perf lever.\n");
    return 0;
}
