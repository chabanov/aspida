// fattn_wmma3.cu — occupancy-aware WMMA FlashAttention (FA-2 style) for the
// full-attn fast path at REAL hura dims (hd=256, nq=16, nkv=2, rep=8). The v1
// wmma died two ways: 1 warp/block @42.5KB shared = 2 warps/SM, and a serial
// per-lane O-rescale over hd=256. This kernel fixes BOTH:
//   * O kept in registers as accumulator fragments across key-tiles; the online-
//     softmax rescale is applied IN-PLACE to the fragments via the known sm_70+
//     m16n16k16 fp32-accumulator row map (no shared roundtrip).
//   * fp16 K/V tile in shared, loaded ONCE per block-iter and shared across W
//     query-warps (amortize) — Q per-warp in shared (fp16), reloaded to frags.
// Baseline = GQA-2 fp32. Bar: >=1.4x at pos>=25k, rel err <=5e-4.
//   nvcc -O3 -arch=native -Xptxas -v fattn_wmma3.cu -o fw3 && ./fw3
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
using std::vector;
using namespace nvcuda;

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
    __shared__ float red[256]; __shared__ float qsh[256];
    qsh[d] = q[d]; __syncthreads();
    float m = -3.402823466e38f, l = 0.f, acc = 0.f;
    for (int s = 0; s < len; ++s) {
        const float *k = Kc + (size_t) s * kvd + kv_off;
        float part = qsh[d] * k[d];
        red[d] = part; __syncthreads();
        for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (d < o) red[d] += red[d + o]; __syncthreads(); }
        float dot = red[0] * scale; __syncthreads();
        float m_new = fmaxf(m, dot);
        float corr = expf(m - m_new), p = expf(dot - m_new);
        l = l * corr + p; acc = acc * corr + p * Vc[(size_t) s * kvd + kv_off + d]; m = m_new;
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
    int hg = blockIdx.y; int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nwarp = blockDim.x >> 5; int t = blockIdx.x * nwarp + warp;
    int rep = nq / nkv; int h_base = hg * HPB, kvh = h_base / rep;
    int kv_off = kvh * hd, att = nq * hd;
    extern __shared__ float shkv[]; float *Ksh = shkv, *Vsh = shkv + (size_t) TK * hd;
    float q[HPB][RQ], acc[HPB][RQ], m[HPB], l[HPB];
    bool active = t < P; int len = active ? (pos_start + t + 1) : 0;
    int t_last = min(P - 1, (blockIdx.x + 1) * nwarp - 1); int len_max = pos_start + t_last + 1;
    #pragma unroll
    for (int hh = 0; hh < HPB; ++hh) { const float *qp = q_all + (size_t) t * att + (h_base + hh) * hd;
        #pragma unroll
        for (int j = 0; j < RQ; ++j) { q[hh][j] = active ? qp[lane + 32 * j] : 0.f; acc[hh][j] = 0.f; }
        m[hh] = -3.402823466e38f; l[hh] = 0.f; }
    float scale = rsqrtf((float) hd);
    for (int s0 = 0; s0 < len_max; s0 += TK) { int tk = min(TK, len_max - s0);
        for (int i = threadIdx.x; i < tk * hd; i += blockDim.x) { int srow = i / hd, sd = i - srow * hd;
            Ksh[(size_t) srow * hd + sd] = Kc[(size_t)(s0 + srow) * kvd + kv_off + sd];
            Vsh[(size_t) srow * hd + sd] = Vc[(size_t)(s0 + srow) * kvd + kv_off + sd]; }
        __syncthreads(); int smax = min(tk, len - s0);
        for (int s = 0; s < smax; ++s) { const float *ks = Ksh + (size_t) s * hd + lane; const float *vs = Vsh + (size_t) s * hd + lane;
            float kreg[RQ], vreg[RQ];
            #pragma unroll
            for (int j = 0; j < RQ; ++j) { kreg[j] = ks[32 * j]; vreg[j] = vs[32 * j]; }
            #pragma unroll
            for (int hh = 0; hh < HPB; ++hh) { float part = 0.f;
                #pragma unroll
                for (int j = 0; j < RQ; ++j) part += q[hh][j] * kreg[j];
                for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
                float dot = part * scale; float m_new = fmaxf(m[hh], dot);
                float corr = expf(m[hh] - m_new), p = expf(dot - m_new); l[hh] = l[hh] * corr + p;
                #pragma unroll
                for (int j = 0; j < RQ; ++j) acc[hh][j] = acc[hh][j] * corr + p * vreg[j]; m[hh] = m_new; } }
        __syncthreads(); }
    if (active) {
        #pragma unroll
        for (int hh = 0; hh < HPB; ++hh) { const float *gp = g_all + (size_t) t * att + (h_base + hh) * hd;
            float *op = attn + (size_t) t * att + (h_base + hh) * hd;
            #pragma unroll
            for (int j = 0; j < RQ; ++j) { float g = gp[lane + 32 * j]; op[lane + 32 * j] = (acc[hh][j] / l[hh]) * (1.f / (1.f + expf(-g))); } } }
}

// row of accumulator element e for this lane (sm_70+ m16n16k16 f32 layout):
//   row = (lane>>2) + ((e & 2) ? 8 : 0)   (e=0,1,4,5 -> low 8 rows; 2,3,6,7 -> high)
__device__ __forceinline__ int accrow(int lane, int e) { return (lane >> 2) + ((e & 2) ? 8 : 0); }

// ---- occupancy-aware WMMA attend. W query-warps/block share one fp16 K/V tile.
// One warp = one 16-query tile for head h. O in registers (HT accumulator frags),
// rescaled in-place per key-tile. Q per-warp in shared (fp16). GATE=1 applies the
// sigmoid gate (real path); rel err vs fp32 oracle reported by caller.
template<int W>
__global__ void k_fattn_wmma3(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn, int nq, int nkv, int hd, int kvd, int pos_start, int P) {
    const int HT = 16;                    // hd/16 (hd=256)
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.y, qtile = blockIdx.x * W + warp, q0 = qtile * 16;
    int rep = nq / nkv, kvh = h / rep, kv_off = kvh * hd, att = nq * hd;
    extern __shared__ char smem[];
    half *Ksh = (half *) smem;                       // [16*hd] block-shared
    half *Vsh = Ksh + 16 * hd;                       // [16*hd] block-shared
    half *Qb  = Vsh + 16 * hd;                        // [W][16*hd]
    float *Sb = (float *) (Qb + (size_t) W * 16 * hd);   // [W][16*16]
    half  *Pb = (half *) (Sb + (size_t) W * 16 * 16);    // [W][16*16]
    float *Cb = (float *) (Pb + (size_t) W * 16 * 16);   // [W][16] corr
    float *Lb = Cb + (size_t) W * 16;                    // [W][16] l  (also m via reg)
    half *Qsh = Qb + (size_t) warp * 16 * hd;
    float *Ssh = Sb + (size_t) warp * 16 * 16;
    half *Psh = Pb + (size_t) warp * 16 * 16;
    float *Csh = Cb + (size_t) warp * 16;
    float *Lsh = Lb + (size_t) warp * 16;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O[HT];
    #pragma unroll
    for (int n = 0; n < HT; ++n) wmma::fill_fragment(O[n], 0.f);
    // load this warp's Q tile into shared (fp16)
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd;
        Qsh[i] = (q0 + r < P) ? __float2half(q_all[(size_t)(q0 + r) * att + h * hd + d]) : __float2half(0.f); }
    float m = -3.402823466e38f, l = 0.f;         // per-row (lane<16 owns row=lane)
    float scale = rsqrtf((float) hd);
    int qmax = min(16, P - q0);
    //  Block-UNIFORM key bound: every warp in the block must run the SAME number
    //  of key-tiles or the __syncthreads() below diverges (warps have different
    //  q0). Warps whose own rows are exhausted just see all-masked keys (corr=1,
    //  p=0 → O unchanged).
    int qlast_blk = min(P - 1, (int)((blockIdx.x + 1) * W * 16 - 1));
    int len_max = pos_start + qlast_blk + 1;
    __syncthreads();
    for (int k0 = 0; k0 < len_max; k0 += 16) {
        int kn = min(16, len_max - k0);
        // block-cooperative fp16 K/V tile load (once, shared across warps)
        for (int i = threadIdx.x; i < 16 * hd; i += blockDim.x) { int r = i / hd, d = i % hd;
            half kf = (r < kn) ? __float2half(Kc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f);
            half vf = (r < kn) ? __float2half(Vc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f);
            Ksh[i] = kf; Vsh[i] = vf; }
        __syncthreads();
        // QK: S = Q @ K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> Sf;
        wmma::fill_fragment(Sf, 0.f);
        #pragma unroll
        for (int kt = 0; kt < HT; ++kt) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
            wmma::load_matrix_sync(af, Qsh + kt * 16, hd);
            wmma::load_matrix_sync(bf, Ksh + kt * 16, hd);       // col_major => K^T
            wmma::mma_sync(Sf, af, bf, Sf);
        }
        wmma::store_matrix_sync(Ssh, Sf, 16, wmma::mem_row_major);
        __syncwarp();
        // softmax: lanes 0..15 each own a query row
        float corr = 1.f;
        if (lane < 16) {
            int qpos = pos_start + q0 + lane; float rmax = m;
            float s[16];
            #pragma unroll
            for (int k = 0; k < 16; ++k) { float sv = (k < kn && (k0 + k) <= qpos) ? Ssh[lane * 16 + k] * scale : -3.402823466e38f; s[k] = sv; rmax = fmaxf(rmax, sv); }
            float m_new = rmax; corr = (m > -3.0e38f) ? __expf(m - m_new) : 0.f;
            float lnew = l * corr;
            #pragma unroll
            for (int k = 0; k < 16; ++k) { float p = (s[k] > -3.0e38f) ? __expf(s[k] - m_new) : 0.f; Psh[lane * 16 + k] = __float2half(p); lnew += p; }
            m = (m_new > -3.0e38f) ? m_new : m; l = lnew; Csh[lane] = corr;
        }   // lanes >= 16 own no query row; rows 0..15 fully written above.
        __syncwarp();
        // rescale O frags in-place by corr[row]  (no shared roundtrip)
        #pragma unroll
        for (int n = 0; n < HT; ++n) {
            #pragma unroll
            for (int e = 0; e < O[n].num_elements; ++e) O[n].x[e] *= Csh[accrow(lane, e)];
        }
        // PV: O += P @ V
        #pragma unroll
        for (int n = 0; n < HT; ++n) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
            wmma::load_matrix_sync(af, Psh, 16);
            wmma::load_matrix_sync(bf, Vsh + n * 16, hd);
            wmma::mma_sync(O[n], af, bf, O[n]);
        }
        __syncthreads();      // before next K/V tile overwrites Ksh/Vsh
    }
    if (lane < 16) Lsh[lane] = l;
    __syncwarp();
    // epilogue: O[n]/l[row] * sigmoid(gate), write out
    #pragma unroll
    for (int n = 0; n < HT; ++n) {
        #pragma unroll
        for (int e = 0; e < O[n].num_elements; ++e) O[n].x[e] /= Lsh[accrow(lane, e)];
        wmma::store_matrix_sync(Ssh, O[n], 16, wmma::mem_row_major);   // reuse Ssh as 16x16 scratch
        __syncwarp();
        for (int i = lane; i < 16 * 16; i += 32) { int r = i / 16, c = i % 16; int d = n * 16 + c;
            if (q0 + r < P) { float g = g_all[(size_t)(q0 + r) * att + h * hd + d];
                attn[(size_t)(q0 + r) * att + h * hd + d] = Ssh[i] * (1.f / (1.f + expf(-g))); } }
        __syncwarp();
    }
}

static float* dup(const vector<float>&h){float*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}

template<int W> size_t shm_w3(int hd){ return (size_t)(2*16*hd + (size_t)W*16*hd)*2 + (size_t)(W*16*16)*4 + (size_t)(W*16*16)*2 + (size_t)(W*16 + W*16)*4; }

int main(int argc, char** argv){
    int hd=256,nq=16,nkv=2,P=512;
    if(argc>=5){hd=atoi(argv[1]);nq=atoi(argv[2]);nkv=atoi(argv[3]);P=atoi(argv[4]);}
    int kvd=nkv*hd,att=nq*hd,RQ=hd/32;
    printf("== dims hd=%d nq=%d nkv=%d P=%d (RQ=%d) ==\n",hd,nq,nkv,P,RQ);
    if(RQ!=8){printf("bench specialized for hd=256\n");return 1;}
    int TK_b=16, TQW=16; size_t shm_b=(size_t)2*TK_b*hd*4;
    size_t s4=shm_w3<4>(hd), s6=shm_w3<6>(hd), s8=shm_w3<8>(hd);
    printf("shared: gqa2=%zu  w3<4>=%zu  w3<6>=%zu  w3<8>=%zu (max dyn ~99KB)\n",shm_b,s4,s6,s8);
    cudaFuncSetAttribute(k_fattn_wmma3<4>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)s4);
    cudaFuncSetAttribute(k_fattn_wmma3<6>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)s6);
    if(s8<=99*1024) cudaFuncSetAttribute(k_fattn_wmma3<8>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)s8);
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
        Run runs[]={{"gqa2 ",0},{"w3<4>",4},{"w3<6>",6},{"w3<8>",8}};
        for(auto&r:runs){
            if(r.kind==8 && s8>99*1024){printf("  w3<8> SKIP (shared %zu>99KB)\n",s8);continue;}
            dim3 gb((P+TQW-1)/TQW, nq/2);
            int Wv=r.kind? r.kind:1;
            dim3 gk((P+16*Wv-1)/(16*Wv), nq);
            size_t shk = r.kind==4? s4 : r.kind==6? s6 : s8;
            auto launch=[&](float*o){
                if(r.kind==0) k_fattn_tile_gqa<2,8><<<gb,TQW*32,shm_b>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P,TK_b);
                else if(r.kind==4) k_fattn_wmma3<4><<<gk,4*32,s4>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P);
                else if(r.kind==6) k_fattn_wmma3<6><<<gk,6*32,s6>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P);
                else k_fattn_wmma3<8><<<gk,8*32,s8>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P);
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
