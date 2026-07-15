// fattn_kvhalf.cu — K/V-precision lever for the GQA-2 full-attn fast path at
// REAL hura dims (hd=256, nq=16, nkv=2, rep=8). At long context the K/V cache
// (pos×kvd×4B per layer) far exceeds the 48MB L2, so attend streams K/V from
// HBM. Storing the cache as fp16 (or Q8_0: 32-val int8 blocks + fp16 scale) and
// dequantizing during the cooperative tile-load — shared stays fp32, so the
// attend arithmetic is IDENTICAL to the fp32 GQA-2 kernel; only the HBM read
// shrinks (fp16 = 1/2, q8 = ~1/3.76). Baseline = fp32 GQA-2.
//   nvcc -O3 -arch=native fattn_kvhalf.cu -o fk && ./fk
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
using std::vector;

// ---- fp32 oracle: one block per (t,h), threads=hd ------------------------
__global__ void k_fattn_attend_chunk(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn, int nq, int nkv, int hd, int kvd, int pos_start) {
    int t = blockIdx.x / nq, h = blockIdx.x % nq, d = threadIdx.x;
    int len = pos_start + t + 1;
    int rep = nq / nkv, kvh = h / rep;
    int q_off = h * hd, kv_off = kvh * hd, att = nq * hd;
    const float *q = q_all + (size_t) t * att + q_off;
    float scale = rsqrtf((float) hd);
    __shared__ float red[256];
    __shared__ float qsh[256];
    qsh[d] = q[d];
    __syncthreads();
    float m = -3.402823466e38f, l = 0.f, acc = 0.f;
    for (int s = 0; s < len; ++s) {
        const float *k = Kc + (size_t) s * kvd + kv_off;
        float part = qsh[d] * k[d];
        red[d] = part; __syncthreads();
        for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (d < o) red[d] += red[d + o]; __syncthreads(); }
        float dot = red[0] * scale;
        __syncthreads();
        float m_new = fmaxf(m, dot);
        float corr = expf(m - m_new), p = expf(dot - m_new);
        l = l * corr + p;
        acc = acc * corr + p * Vc[(size_t) s * kvd + kv_off + d];
        m = m_new;
    }
    float g = g_all[(size_t) t * att + q_off + d];
    attn[(size_t) t * att + q_off + d] = (acc / l) * (1.f / (1.f + expf(-g)));
}

// ---- GQA-2 with templated K/V storage. KVMODE 0=fp32, 1=fp16, 2=q8_0.
// Kc/Vc are reinterpreted per mode; Ks/Vs are the Q8 scale arrays (half,
// one per 32-value block along kvd), unused for mode 0/1. Attend math is fp32
// and byte-identical to the fp32 kernel — only the tile-load dequant differs.
template<int HPB, int RQ, int KVMODE>
__global__ void k_gqa_kv(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const void *__restrict__ Kc, const void *__restrict__ Vc,
    const half *__restrict__ Ks, const half *__restrict__ Vs,
    float *__restrict__ attn, int nq, int nkv, int hd, int kvd, int pos_start, int P, int TK) {
    int hg = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nwarp = blockDim.x >> 5;
    int t = blockIdx.x * nwarp + warp;
    int rep = nq / nkv;
    int h_base = hg * HPB, kvh = h_base / rep;
    int kv_off = kvh * hd, att = nq * hd, kvb = kvd >> 5;   // blocks/row for q8
    extern __shared__ float shkv[];
    float *Ksh = shkv, *Vsh = shkv + (size_t) TK * hd;
    float q[HPB][RQ], acc[HPB][RQ], m[HPB], l[HPB];
    bool active = t < P;
    int len = active ? (pos_start + t + 1) : 0;
    int t_last = min(P - 1, (blockIdx.x + 1) * nwarp - 1);
    int len_max = pos_start + t_last + 1;
    #pragma unroll
    for (int hh = 0; hh < HPB; ++hh) {
        const float *qp = q_all + (size_t) t * att + (h_base + hh) * hd;
        #pragma unroll
        for (int j = 0; j < RQ; ++j) { q[hh][j] = active ? qp[lane + 32 * j] : 0.f; acc[hh][j] = 0.f; }
        m[hh] = -3.402823466e38f; l[hh] = 0.f;
    }
    float scale = rsqrtf((float) hd);
    for (int s0 = 0; s0 < len_max; s0 += TK) {
        int tk = min(TK, len_max - s0);
        for (int i = threadIdx.x; i < tk * hd; i += blockDim.x) {
            int srow = i / hd, sd = i - srow * hd;
            int r = s0 + srow, col = kv_off + sd;
            size_t g = (size_t) r * kvd + col;
            float kf, vf;
            if (KVMODE == 0) { kf = ((const float *) Kc)[g]; vf = ((const float *) Vc)[g]; }
            else if (KVMODE == 1) { kf = __half2float(((const half *) Kc)[g]); vf = __half2float(((const half *) Vc)[g]); }
            else { size_t sc = (size_t) r * kvb + (col >> 5);
                   kf = (float) ((const signed char *) Kc)[g] * __half2float(Ks[sc]);
                   vf = (float) ((const signed char *) Vc)[g] * __half2float(Vs[sc]); }
            Ksh[(size_t) srow * hd + sd] = kf; Vsh[(size_t) srow * hd + sd] = vf;
        }
        __syncthreads();
        int smax = min(tk, len - s0);
        for (int s = 0; s < smax; ++s) {
            const float *ks = Ksh + (size_t) s * hd + lane;
            const float *vs = Vsh + (size_t) s * hd + lane;
            float kreg[RQ], vreg[RQ];
            #pragma unroll
            for (int j = 0; j < RQ; ++j) { kreg[j] = ks[32 * j]; vreg[j] = vs[32 * j]; }
            #pragma unroll
            for (int hh = 0; hh < HPB; ++hh) {
                float part = 0.f;
                #pragma unroll
                for (int j = 0; j < RQ; ++j) part += q[hh][j] * kreg[j];
                for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
                float dot = part * scale;
                float m_new = fmaxf(m[hh], dot);
                float corr = expf(m[hh] - m_new), p = expf(dot - m_new);
                l[hh] = l[hh] * corr + p;
                #pragma unroll
                for (int j = 0; j < RQ; ++j) acc[hh][j] = acc[hh][j] * corr + p * vreg[j];
                m[hh] = m_new;
            }
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int hh = 0; hh < HPB; ++hh) {
            const float *gp = g_all + (size_t) t * att + (h_base + hh) * hd;
            float *op = attn + (size_t) t * att + (h_base + hh) * hd;
            #pragma unroll
            for (int j = 0; j < RQ; ++j) {
                float g = gp[lane + 32 * j];
                op[lane + 32 * j] = (acc[hh][j] / l[hh]) * (1.f / (1.f + expf(-g)));
            }
        }
    }
}

// ---- host-side quantization kernels ---------------------------------------
__global__ void k_to_fp16(const float *in, half *out, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}
// Q8_0: one thread per 32-value block. scale = max|v|/127, q = round(v/scale).
__global__ void k_to_q8(const float *in, signed char *q, half *sc, size_t nblk) {
    size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const float *v = in + b * 32;
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    float d = amax / 127.f, id = (d > 0.f) ? 1.f / d : 0.f;
    sc[b] = __float2half(d);
    for (int i = 0; i < 32; ++i) { int qi = (int) lrintf(v[i] * id); q[b * 32 + i] = (signed char) max(-127, min(127, qi)); }
}

static float* dup(const vector<float>&h){float*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}

int main(int argc, char** argv){
    int hd=256,nq=16,nkv=2,P=512;
    if(argc>=5){hd=atoi(argv[1]);nq=atoi(argv[2]);nkv=atoi(argv[3]);P=atoi(argv[4]);}
    int kvd=nkv*hd,att=nq*hd,RQ=hd/32,kvb=kvd/32;
    printf("== dims hd=%d nq=%d nkv=%d P=%d (RQ=%d) ==\n",hd,nq,nkv,P,RQ);
    if(RQ!=8){printf("bench specialized for hd=256\n");return 1;}
    int TK=16, TQW=16;                        // GQA-2 serving config
    size_t shm=(size_t)2*TK*hd*4;
    std::mt19937 rng(3);std::normal_distribution<float> nd(0,1);
    int positions[]={12288,25344,37376};
    for(int pi=0;pi<3;++pi){
        int ps=positions[pi];int len=ps+P;
        auto rnd=[&](size_t n){vector<float>x(n);for(auto&e:x)e=nd(rng)*0.3f;return x;};
        vector<float> qa=rnd((size_t)P*att),ga=rnd((size_t)P*att),K=rnd((size_t)len*kvd),V=rnd((size_t)len*kvd);
        float*dqa=dup(qa),*dga=dup(ga),*dK=dup(K),*dV=dup(V),*dref,*dw;
        CK(cudaMalloc(&dref,(size_t)P*att*4));CK(cudaMalloc(&dw,(size_t)P*att*4));
        // fp16 + q8 copies
        size_t nkv_el=(size_t)len*kvd, nblk=nkv_el/32;
        half *K16,*V16; CK(cudaMalloc(&K16,nkv_el*2));CK(cudaMalloc(&V16,nkv_el*2));
        signed char *Kq,*Vq; half *Ksc,*Vsc;
        CK(cudaMalloc(&Kq,nkv_el));CK(cudaMalloc(&Vq,nkv_el));
        CK(cudaMalloc(&Ksc,nblk*2));CK(cudaMalloc(&Vsc,nblk*2));
        k_to_fp16<<<(nkv_el+255)/256,256>>>(dK,K16,nkv_el);
        k_to_fp16<<<(nkv_el+255)/256,256>>>(dV,V16,nkv_el);
        k_to_q8<<<(nblk+255)/256,256>>>(dK,Kq,Ksc,nblk);
        k_to_q8<<<(nblk+255)/256,256>>>(dV,Vq,Vsc,nblk);
        CK(cudaDeviceSynchronize());
        // oracle
        k_fattn_attend_chunk<<<(size_t)nq*P,hd>>>(dqa,dga,dK,dV,dref,nq,nkv,hd,kvd,ps);
        CK(cudaDeviceSynchronize());
        vector<float> a(P*att);CK(cudaMemcpy(a.data(),dref,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        dim3 grid((P+TQW-1)/TQW, nq/2);
        printf("pos=%5d len=%5d:\n",ps,len);
        struct Run{const char*name;int mode;};
        Run runs[]={{"fp32",0},{"fp16",1},{"q8_0",2}};
        for(auto&r:runs){
            auto launch=[&](float*o){
                if(r.mode==0) k_gqa_kv<2,8,0><<<grid,TQW*32,shm>>>(dqa,dga,dK,dV,nullptr,nullptr,o,nq,nkv,hd,kvd,ps,P,TK);
                else if(r.mode==1) k_gqa_kv<2,8,1><<<grid,TQW*32,shm>>>(dqa,dga,K16,V16,nullptr,nullptr,o,nq,nkv,hd,kvd,ps,P,TK);
                else k_gqa_kv<2,8,2><<<grid,TQW*32,shm>>>(dqa,dga,Kq,Vq,Ksc,Vsc,o,nq,nkv,hd,kvd,ps,P,TK);
            };
            CK(cudaMemset(dw,0,(size_t)P*att*4));
            launch(dw);CK(cudaDeviceSynchronize());
            cudaError_t e=cudaGetLastError(); if(e){printf("  %s ERR: %s\n",r.name,cudaGetErrorString(e));continue;}
            vector<float> b(P*att);CK(cudaMemcpy(b.data(),dw,(size_t)P*att*4,cudaMemcpyDeviceToHost));
            float mx=0,denom=0;for(size_t i=0;i<a.size();++i){mx=std::max(mx,fabsf(a[i]-b[i]));denom=std::max(denom,fabsf(a[i]));}
            for(int i=0;i<5;++i)launch(dw);CK(cudaDeviceSynchronize());
            auto time_it=[&](){cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);int N=40;
                cudaEventRecord(e0);for(int i=0;i<N;++i)launch(dw);cudaEventRecord(e1);CK(cudaEventSynchronize(e1));
                float t=0;cudaEventElapsedTime(&t,e0,e1);cudaEventDestroy(e0);cudaEventDestroy(e1);return t/N;};
            float t1=time_it(),t2=time_it();
            printf("  %s  %.3f / %.3f ms/chunk  max|err|=%.2e (rel~%.1e)\n",r.name,t1,t2,mx,mx/(denom+1e-9f));
        }
        cudaFree(dqa);cudaFree(dga);cudaFree(dK);cudaFree(dV);cudaFree(dref);cudaFree(dw);
        cudaFree(K16);cudaFree(V16);cudaFree(Kq);cudaFree(Vq);cudaFree(Ksc);cudaFree(Vsc);
    }
    return 0;
}
