// fattn_gqa.cu — GQA K/V-reuse lever for the tiled full-attn kernel at REAL
// hura dims (hd=256, nq=16, nkv=2, rep=8). The serving tiled kernel launches one
// independent block per q-head (grid.y=nq=16), so each kv-head's K/V is streamed
// from HBM rep=8 times. This bench tests a block-per-(query-tile, kv-head) kernel
// that loads each K/V tile into shared ONCE and attends all HPB group q-heads
// against it — cutting K/V HBM traffic by up to HPB×.
//   nvcc -O3 -arch=native -Xptxas -v fattn_gqa.cu -o fg && ./fg [hd nq nkv P]
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
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

// ---- tiled serving default (baseline, verbatim from gpu_matvec.cu) --------
__global__ void k_fattn_attend_tile(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn, int nq, int nkv, int hd, int kvd, int pos_start, int P, int TK) {
    int h = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nwarp = blockDim.x >> 5;
    int t = blockIdx.x * nwarp + warp;
    int rep = nq / nkv, kvh = h / rep;
    int kv_off = kvh * hd, att = nq * hd;
    int RQ = hd >> 5;
    extern __shared__ float shkv[];
    float *Ksh = shkv, *Vsh = shkv + (size_t) TK * hd;
    float q[8], acc[8];
    bool active = t < P;
    int len = active ? (pos_start + t + 1) : 0;
    int t_last = min(P - 1, (blockIdx.x + 1) * nwarp - 1);
    int len_max = pos_start + t_last + 1;
    const float *qp = q_all + (size_t) t * att + h * hd;
    #pragma unroll
    for (int j = 0; j < 8; ++j) { q[j] = 0.f; acc[j] = 0.f; }
    for (int j = 0; j < RQ; ++j) q[j] = active ? qp[lane + 32 * j] : 0.f;
    float scale = rsqrtf((float) hd), m = -3.402823466e38f, l = 0.f;
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
            float part = 0.f;
            for (int j = 0; j < RQ; ++j) part += q[j] * Ksh[(size_t) s * hd + lane + 32 * j];
            for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
            float dot = part * scale;
            float m_new = fmaxf(m, dot);
            float corr = expf(m - m_new), p = expf(dot - m_new);
            l = l * corr + p;
            for (int j = 0; j < RQ; ++j) acc[j] = acc[j] * corr + p * Vsh[(size_t) s * hd + lane + 32 * j];
            m = m_new;
        }
        __syncthreads();
    }
    if (active) {
        const float *gp = g_all + (size_t) t * att + h * hd;
        float *op = attn + (size_t) t * att + h * hd;
        for (int j = 0; j < RQ; ++j) {
            float g = gp[lane + 32 * j];
            op[lane + 32 * j] = (acc[j] / l) * (1.f / (1.f + expf(-g)));
        }
    }
}

// ---- GQA K/V-reuse variant. block = (query-tile, head-group). Each warp = one
// query position; it carries online-softmax state for HPB group heads at once,
// all sharing one Ksh/Vsh tile (same kv-head). K/V HBM traffic drops HPB×.
// HPB must divide rep so all HPB heads map to one kvh. Shared = 2*TK*hd floats
// (unchanged from baseline) — occupancy is register-bound here, not shared-bound.
template<int HPB, int RQ>
__global__ void k_fattn_tile_gqa(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn, int nq, int nkv, int hd, int kvd, int pos_start, int P, int TK) {
    int hg = blockIdx.y;                     // head-group index
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

static float* dup(const vector<float>&h){float*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}

int main(int argc, char** argv){
    int hd=256,nq=16,nkv=2,P=256;
    if(argc>=5){hd=atoi(argv[1]);nq=atoi(argv[2]);nkv=atoi(argv[3]);P=atoi(argv[4]);}
    int kvd=nkv*hd,att=nq*hd,rep=nq/nkv,RQ=hd/32;
    printf("== dims hd=%d nq=%d nkv=%d P=%d (rep=%d RQ=%d) ==\n",hd,nq,nkv,P,rep,RQ);
    if(RQ!=8){printf("this bench specialized for hd=256 (RQ=8)\n");return 1;}
    int TK=16, TQW=32;
    size_t shm=(size_t)2*TK*hd*4;
    std::mt19937 rng(3);std::normal_distribution<float> nd(0,1);
    int positions[]={0,4096,12288,25344};
    // launch kinds: 0 tiled | 2/4/8 gqa HPB.  nw = query-warps/block, sized so
    // per-block regs (HPB-dependent) leave room for >=1 (ideally >=2) blocks/SM.
    struct Run{const char*name;int hpb;int nw;};
    Run runs[]={{"tiled  ",0,32},
                {"gqa2/8 ",2,8},{"gqa2/12",2,12},{"gqa2/16",2,16},{"gqa2/24",2,24},
                {"gqa4/8 ",4,8},{"gqa4/12",4,12},{"gqa8/4 ",8,4}};
    for(int pi=0;pi<4;++pi){
        int ps=positions[pi];int len=ps+P;
        auto rnd=[&](size_t n){vector<float>x(n);for(auto&e:x)e=nd(rng)*0.3f;return x;};
        vector<float> qa=rnd((size_t)P*att),ga=rnd((size_t)P*att),K=rnd((size_t)len*kvd),V=rnd((size_t)len*kvd);
        float*dqa=dup(qa),*dga=dup(ga),*dK=dup(K),*dV=dup(V),*dref,*dw;
        CK(cudaMalloc(&dref,(size_t)P*att*4));CK(cudaMalloc(&dw,(size_t)P*att*4));
        k_fattn_attend_chunk<<<(size_t)nq*P,hd>>>(dqa,dga,dK,dV,dref,nq,nkv,hd,kvd,ps);
        CK(cudaDeviceSynchronize());
        vector<float> a(P*att);CK(cudaMemcpy(a.data(),dref,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        printf("pos=%5d len=%5d:\n",ps,len);
        for(auto&r:runs){
            int nw=r.nw;
            dim3 grid((P+nw-1)/nw, r.hpb? nq/r.hpb : nq);
            auto launch=[&](float*o){
                if(r.hpb==0) k_fattn_attend_tile<<<grid,nw*32,shm>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P,TK);
                else if(r.hpb==2) k_fattn_tile_gqa<2,8><<<grid,nw*32,shm>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P,TK);
                else if(r.hpb==4) k_fattn_tile_gqa<4,8><<<grid,nw*32,shm>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P,TK);
                else k_fattn_tile_gqa<8,8><<<grid,nw*32,shm>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P,TK);
            };
            CK(cudaMemset(dw,0,(size_t)P*att*4));
            launch(dw);CK(cudaDeviceSynchronize());
            cudaError_t e=cudaGetLastError(); if(e){printf("  %s LAUNCH ERR: %s\n",r.name,cudaGetErrorString(e));continue;}
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
