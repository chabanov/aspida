// fattn baseline bench: current k_fattn_attend_tile (default) vs k_fattn_attend_chunk
// (bit-exact oracle), correctness + timing across positions. Establishes the target
// the tensor-core version must beat.
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
using std::vector;
__global__ void k_fattn_attend_tile(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn,
    int nq, int nkv, int hd, int kvd, int pos_start, int P, int TK) {
    int h = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nwarp = blockDim.x >> 5;
    int t = blockIdx.x * nwarp + warp;      //  query position in chunk
    int rep = nq / nkv, kvh = h / rep;
    int kv_off = kvh * hd, att = nq * hd;
    int RQ = hd >> 5;                       //  q/acc registers per lane (<= 8)
    extern __shared__ float shkv[];         //  K tile [TK][hd], then V tile
    float *Ksh = shkv, *Vsh = shkv + (size_t) TK * hd;
    float q[8], acc[8];
    bool active = t < P;
    int len = active ? (pos_start + t + 1) : 0;
    int t_last = min(P - 1, (blockIdx.x + 1) * nwarp - 1);
    int len_max = pos_start + t_last + 1;   //  block-wide causal bound
    const float *qp = q_all + (size_t) t * att + h * hd;
    #pragma unroll
    for (int j = 0; j < 8; ++j) { q[j] = 0.f; acc[j] = 0.f; }
    for (int j = 0; j < RQ; ++j) q[j] = active ? qp[lane + 32 * j] : 0.f;
    float scale = rsqrtf((float) hd), m = -3.402823466e38f, l = 0.f;
    for (int s0 = 0; s0 < len_max; s0 += TK) {
        int tk = min(TK, len_max - s0);
        //  Cooperative tile load: ALL warps participate (idle query warps
        //  included — they must reach the __syncthreads below).
        for (int i = threadIdx.x; i < tk * hd; i += blockDim.x) {
            int srow = i / hd, sd = i - srow * hd;
            Ksh[(size_t) srow * hd + sd] = Kc[(size_t)(s0 + srow) * kvd + kv_off + sd];
            Vsh[(size_t) srow * hd + sd] = Vc[(size_t)(s0 + srow) * kvd + kv_off + sd];
        }
        __syncthreads();
        int smax = min(tk, len - s0);       //  warp-local causal bound
        for (int s = 0; s < smax; ++s) {
            float part = 0.f;
            for (int j = 0; j < RQ; ++j) part += q[j] * Ksh[(size_t) s * hd + lane + 32 * j];
            //  Butterfly reduce: every lane ends with the full dot product.
            for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
            float dot = part * scale;
            float m_new = fmaxf(m, dot);
            float corr = expf(m - m_new);
            float p = expf(dot - m_new);
            l = l * corr + p;
            for (int j = 0; j < RQ; ++j)
                acc[j] = acc[j] * corr + p * Vsh[(size_t) s * hd + lane + 32 * j];
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
__global__ void k_fattn_attend_chunk(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn,
    int nq, int nkv, int hd, int kvd, int pos_start) {
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
        // block reduce (hd threads) -> dot in red[0], broadcast
        red[d] = part; __syncthreads();
        for (int o = blockDim.x / 2; o > 0; o >>= 1) {
            if (d < o) red[d] += red[d + o];
            __syncthreads();
        }
        float dot = red[0] * scale;
        __syncthreads();
        float m_new = fmaxf(m, dot);
        float corr = expf(m - m_new);
        float p = expf(dot - m_new);
        l = l * corr + p;
        acc = acc * corr + p * Vc[(size_t) s * kvd + kv_off + d];
        m = m_new;
    }
    float g = g_all[(size_t) t * att + q_off + d];
    attn[(size_t) t * att + q_off + d] = (acc / l) * (1.f / (1.f + expf(-g)));
}
static float* dup(const vector<float>&h){float*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}
int main(){
    int hd=128,nq=32,nkv=8,P=256; int kvd=nkv*hd,att=nq*hd;
    std::mt19937 rng(3);std::normal_distribution<float> nd(0,1);
    int positions[]={0,4096,12288,25344};
    for(int pi=0;pi<4;++pi){
        int ps=positions[pi]; int len=ps+P;
        auto rnd=[&](int n){vector<float>x(n);for(auto&e:x)e=nd(rng);return x;};
        vector<float> qa=rnd(P*att),ga=rnd(P*att),K=rnd((size_t)len*kvd),V=rnd((size_t)len*kvd);
        float*dqa=dup(qa),*dga=dup(ga),*dK=dup(K),*dV=dup(V);
        float*dref,*dtile;CK(cudaMalloc(&dref,(size_t)P*att*4));CK(cudaMalloc(&dtile,(size_t)P*att*4));
        int TK=(hd<=128)?32:16,TQW=32; size_t shm=(size_t)2*TK*hd*4;
        dim3 grid((P+TQW-1)/TQW,nq);
        auto run_tile=[&](float*out){k_fattn_attend_tile<<<grid,TQW*32,shm>>>(dqa,dga,dK,dV,out,nq,nkv,hd,kvd,ps,P,TK);};
        auto run_ref =[&](float*out){k_fattn_attend_chunk<<<(size_t)nq*P,hd>>>(dqa,dga,dK,dV,out,nq,nkv,hd,kvd,ps);};
        run_ref(dref);run_tile(dtile);CK(cudaDeviceSynchronize());
        vector<float> a(P*att),b(P*att);CK(cudaMemcpy(a.data(),dref,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(),dtile,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        float mx=0;for(size_t i=0;i<a.size();++i)mx=std::max(mx,fabsf(a[i]-b[i]));
        // time tile
        for(int i=0;i<3;++i)run_tile(dtile);CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);int N=30;
        cudaEventRecord(e0);for(int i=0;i<N;++i)run_tile(dtile);cudaEventRecord(e1);CK(cudaEventSynchronize(e1));
        float t=0;cudaEventElapsedTime(&t,e0,e1);
        printf("pos_start=%5d len=%5d  tile=%.3f ms/chunk  (tile-vs-oracle max=%.2e)\n",ps,len,t/N,mx);
        cudaFree(dqa);cudaFree(dga);cudaFree(dK);cudaFree(dV);cudaFree(dref);cudaFree(dtile);
    }
    return 0;
}
