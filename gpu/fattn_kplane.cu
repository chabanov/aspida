// fattn_kplane.cu — key-per-lane attend restructure for the full-attn fast path
// at REAL hura dims (hd=256, nq=16, nkv=2, rep=8). Baseline = current GQA-2
// (query-per-warp): per key per head it pays 8 QK FMAs + a 5-step shfl butterfly
// + 2 expf (corr,p) + 8 PV FMAs — and critically issues the softmax expf once
// per KEY (all 32 lanes redundantly in SIMT). Key-per-lane assigns lane=key: a
// warp processes a 32-key tile, each lane computes ITS key's full dot privately
// (no butterfly), and the softmax expf is issued once per 32 keys (each lane a
// distinct key) → ~32x fewer SFU issues + ~80x fewer shuffles. Same fp32 FMA
// count. K/V held transposed in shared ([d][key], pad 33) so QK and PV are
// bank-conflict-free; O accumulated in registers (RQ dims/lane), online softmax
// per 32-key tile.
//   nvcc -O3 -arch=native fattn_kplane.cu -o fkp && ./fkp
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
using std::vector;

// ---- fp32 oracle ---------------------------------------------------------
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

// ---- GQA-2 baseline (query-per-warp, verbatim) ---------------------------
template<int HPB, int RQ>
__global__ void k_fattn_tile_gqa(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn, int nq, int nkv, int hd, int kvd, int pos_start, int P, int TK) {
    int hg = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nwarp = blockDim.x >> 5;
    int t = blockIdx.x * nwarp + warp;
    int rep = nq / nkv;
    int h_base = hg * HPB, kvh = h_base / rep;
    int kv_off = kvh * hd, att = nq * hd;
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
            Ksh[(size_t) srow * hd + sd] = Kc[(size_t)(s0 + srow) * kvd + kv_off + sd];
            Vsh[(size_t) srow * hd + sd] = Vc[(size_t)(s0 + srow) * kvd + kv_off + sd];
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

// ---- key-per-lane. lane=key over a 32-key tile; warp=1 query; one head/block.
// FAST=1 uses __expf (fast-math SFU); FAST=0 uses expf (exact). NW warps/block.
#define PAD 33   // shared stride for the 32-key axis (33 => conflict-free)
template<int RQ, int NW, int FAST>
__global__ void k_fattn_kplane(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn, int nq, int nkv, int hd, int kvd, int pos_start, int P) {
    const int TK = 32;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.y;
    int t = blockIdx.x * NW + warp;
    int rep = nq / nkv, kvh = h / rep, kv_off = kvh * hd, att = nq * hd;
    extern __shared__ float sh[];
    float *Ksh = sh;                         // [hd][PAD]
    float *Vsh = Ksh + (size_t) hd * PAD;    // [hd][PAD]
    float *Qsh = Vsh + (size_t) hd * PAD;    // [NW][hd]
    float *Psh = Qsh + (size_t) NW * hd;     // [NW][32]
    float *qme = Qsh + (size_t) warp * hd;
    float *pme = Psh + (size_t) warp * 32;
    bool active = t < P;
    for (int d = lane; d < hd; d += 32) qme[d] = active ? q_all[(size_t) t * att + h * hd + d] : 0.f;
    float O[RQ];
    #pragma unroll
    for (int j = 0; j < RQ; ++j) O[j] = 0.f;
    float m = -3.402823466e38f, l = 0.f, scale = rsqrtf((float) hd);
    int len = active ? (pos_start + t + 1) : 0;
    int t_last = min(P - 1, (blockIdx.x + 1) * NW - 1);
    int len_max = pos_start + t_last + 1;
    __syncthreads();     // Q ready
    for (int k0 = 0; k0 < len_max; k0 += TK) {
        int kn = min(TK, len_max - k0);
        //  cooperative TRANSPOSED load: Ksh[d][key] = K[k0+key][kv_off+d].
        //  HBM read is coalesced in d (consecutive threads); shared write is
        //  stride-PAD (conflict-free).
        for (int i = threadIdx.x; i < TK * hd; i += blockDim.x) {
            int key = i / hd, d = i - key * hd;
            float kf = 0.f, vf = 0.f;
            if (key < kn) { size_t g = (size_t)(k0 + key) * kvd + kv_off + d; kf = Kc[g]; vf = Vc[g]; }
            Ksh[(size_t) d * PAD + key] = kf;
            Vsh[(size_t) d * PAD + key] = vf;
        }
        __syncthreads();
        //  QK: this warp's query vs key=lane. Full dot, no butterfly.
        bool kact = active && lane < kn && (k0 + lane) < len;
        float dot = 0.f;
        for (int d = 0; d < hd; ++d) dot += qme[d] * Ksh[(size_t) d * PAD + lane];
        dot *= scale;
        float dv = kact ? dot : -3.402823466e38f;
        //  tile max + running-max online softmax (2 warp reduces / 32 keys)
        float tmax = dv;
        for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
        if (tmax > -3.0e38f) {
            float m_new = fmaxf(m, tmax);
            float corr = FAST ? __expf(m - m_new) : expf(m - m_new);
            float p = kact ? (FAST ? __expf(dv - m_new) : expf(dv - m_new)) : 0.f;
            pme[lane] = p;
            float tsum = p;
            for (int o = 16; o > 0; o >>= 1) tsum += __shfl_xor_sync(0xffffffffu, tsum, o);
            l = l * corr + tsum;
            #pragma unroll
            for (int j = 0; j < RQ; ++j) O[j] *= corr;
            m = m_new;
            __syncwarp();
            //  PV: lane owns dims {lane, lane+32, ...}. O[d] += sum_k p[k]*V[d][k].
            #pragma unroll
            for (int j = 0; j < RQ; ++j) {
                int d = lane + 32 * j;
                float acc = 0.f;
                for (int k = 0; k < 32; ++k) acc += pme[k] * Vsh[(size_t) d * PAD + k];
                O[j] += acc;
            }
        }
        __syncthreads();     // before next tile overwrites Ksh/Vsh
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < RQ; ++j) {
            int d = lane + 32 * j;
            float g = g_all[(size_t) t * att + h * hd + d];
            attn[(size_t) t * att + h * hd + d] = (O[j] / l) * (1.f / (1.f + expf(-g)));
        }
    }
}

static float* dup(const vector<float>&h){float*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}

int main(int argc, char** argv){
    int hd=256,nq=16,nkv=2,P=512;
    if(argc>=5){hd=atoi(argv[1]);nq=atoi(argv[2]);nkv=atoi(argv[3]);P=atoi(argv[4]);}
    int kvd=nkv*hd,att=nq*hd,RQ=hd/32;
    printf("== dims hd=%d nq=%d nkv=%d P=%d (RQ=%d) ==\n",hd,nq,nkv,P,RQ);
    if(RQ!=8){printf("bench specialized for hd=256\n");return 1;}
    const int NW=8;
    int TK_b=16, TQW=16;                                  // GQA-2 baseline config
    size_t shm_b=(size_t)2*TK_b*hd*4;
    size_t shm_k=((size_t)2*hd*PAD + (size_t)NW*hd + (size_t)NW*32)*4;   // kplane
    printf("shared: gqa2=%zuB  kplane=%zuB (NW=%d)\n",shm_b,shm_k,NW);
    cudaFuncSetAttribute(k_fattn_kplane<8,NW,0>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shm_k);
    cudaFuncSetAttribute(k_fattn_kplane<8,NW,1>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shm_k);
    std::mt19937 rng(3);std::normal_distribution<float> nd(0,1);
    int positions[]={12288,25344,37376};
    for(int pi=0;pi<3;++pi){
        int ps=positions[pi];int len=ps+P;
        auto rnd=[&](size_t n){vector<float>x(n);for(auto&e:x)e=nd(rng)*0.3f;return x;};
        vector<float> qa=rnd((size_t)P*att),ga=rnd((size_t)P*att),K=rnd((size_t)len*kvd),V=rnd((size_t)len*kvd);
        float*dqa=dup(qa),*dga=dup(ga),*dK=dup(K),*dV=dup(V),*dref,*dw;
        CK(cudaMalloc(&dref,(size_t)P*att*4));CK(cudaMalloc(&dw,(size_t)P*att*4));
        k_fattn_attend_chunk<<<(size_t)nq*P,hd>>>(dqa,dga,dK,dV,dref,nq,nkv,hd,kvd,ps);
        CK(cudaDeviceSynchronize());
        vector<float> a(P*att);CK(cudaMemcpy(a.data(),dref,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        printf("pos=%5d len=%5d:\n",ps,len);
        struct Run{const char*name;int kind;};
        Run runs[]={{"gqa2-fp32     ",0},{"kplane-exact  ",1},{"kplane-fastmth",2}};
        for(auto&r:runs){
            dim3 grid_b((P+TQW-1)/TQW, nq/2);
            dim3 grid_k((P+NW-1)/NW, nq);
            auto launch=[&](float*o){
                if(r.kind==0) k_fattn_tile_gqa<2,8><<<grid_b,TQW*32,shm_b>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P,TK_b);
                else if(r.kind==1) k_fattn_kplane<8,NW,0><<<grid_k,NW*32,shm_k>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P);
                else k_fattn_kplane<8,NW,1><<<grid_k,NW*32,shm_k>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P);
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
    }
    return 0;
}
