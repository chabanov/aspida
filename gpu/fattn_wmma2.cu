// fattn_wmma2.cu — attention-lever bench at the REAL hura Qwen3.6-35B-A3B dims.
//   GGUF: head_count=16, head_count_kv=2, key/value_length=256, rope dim=64.
//   => nq=16, nkv=2, hd=256, rep=8. (Standalone fattn_wmma.cu assumed 128/32/8.)
// Compares: fp32 oracle (k_fattn_attend_chunk) | tiled serving default
//   (k_fattn_attend_tile) | wmma baseline (k_fattn_wmma) | deepening variants.
//   nvcc -O3 -arch=native fattn_wmma2.cu -o fw2 && ./fw2 [hd nq nkv P]
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

// ---- fp32 oracle: one block per (t,h), threads=hd -------------------------
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

// ---- tiled serving default (verbatim from gpu_matvec.cu) ------------------
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

// ---- wmma baseline (verbatim from gpu_matvec.cu k_fattn_wmma) -------------
__global__ void k_fattn_wmma(const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc, float *__restrict__ attn,
    int nq, int nkv, int hd, int kvd, int pos_start, int P) {
    int qt = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    int rep = nq / nkv, kvh = h / rep, kv_off = kvh * hd, att = nq * hd, HT = hd / 16, q0 = qt * 16;
    extern __shared__ char smem[];
    half *Qsh = (half *) smem;
    half *Ksh = Qsh + 16 * hd, *Vsh = Ksh + 16 * hd, *Psh = Vsh + 16 * hd;
    float *Osh = (float *) (Psh + 16 * 16);
    float *Ssh = Osh + 16 * hd;
    __shared__ float m[16], l[16];
    int qmax = min(16, P - q0);
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd;
        Qsh[i] = (q0 + r < P) ? __float2half(q_all[(size_t)(q0 + r) * att + h * hd + d]) : __float2half(0.f);
        Osh[i] = 0.f; }
    if (lane < 16) { m[lane] = -3.402823466e38f; l[lane] = 0.f; }
    __syncwarp();
    float scale = rsqrtf((float) hd);
    int len_max = pos_start + q0 + qmax;
    for (int k0 = 0; k0 < len_max; k0 += 16) {
        int kn = min(16, len_max - k0);
        for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd;
            Ksh[i] = (r < kn) ? __float2half(Kc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f);
            Vsh[i] = (r < kn) ? __float2half(Vc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f); }
        __syncwarp();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
        wmma::fill_fragment(cf, 0.f);
        for (int kt = 0; kt < HT; ++kt) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
            wmma::load_matrix_sync(af, Qsh + kt * 16, hd);
            wmma::load_matrix_sync(bf, Ksh + kt * 16, hd);
            wmma::mma_sync(cf, af, bf, cf);
        }
        wmma::store_matrix_sync(Ssh, cf, 16, wmma::mem_row_major);
        __syncwarp();
        if (lane < 16) {
            if (lane < qmax) {
                int qpos = pos_start + q0 + lane; float rmax = m[lane]; float s[16];
                for (int k = 0; k < 16; ++k) { float sv = (k < kn && (k0 + k) <= qpos) ? Ssh[lane * 16 + k] * scale : -3.402823466e38f; s[k] = sv; rmax = fmaxf(rmax, sv); }
                float corr = expf(m[lane] - rmax), lnew = l[lane] * corr;
                for (int k = 0; k < 16; ++k) { float p = (s[k] > -3.0e38f) ? expf(s[k] - rmax) : 0.f; Psh[lane * 16 + k] = __float2half(p); lnew += p; }
                for (int d = 0; d < hd; ++d) Osh[lane * hd + d] *= corr;
                m[lane] = rmax; l[lane] = lnew;
            } else for (int k = 0; k < 16; ++k) Psh[lane * 16 + k] = __float2half(0.f);
        }
        __syncwarp();
        for (int n = 0; n < HT; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> of;
            wmma::load_matrix_sync(of, Osh + n * 16, hd, wmma::mem_row_major);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
            wmma::load_matrix_sync(af, Psh, 16);
            wmma::load_matrix_sync(bf, Vsh + n * 16, hd);
            wmma::mma_sync(of, af, bf, of);
            wmma::store_matrix_sync(Osh + n * 16, of, hd, wmma::mem_row_major);
        }
        __syncwarp();
    }
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd; if (q0 + r < P) {
        float g = g_all[(size_t)(q0 + r) * att + h * hd + d];
        attn[(size_t)(q0 + r) * att + h * hd + d] = (Osh[i] / l[r]) * (1.f / (1.f + expf(-g))); } }
}

// ---- VARIANT A: 32-key tile (KT=32). Two 16x16 QK mmas + two P@V passes per
//   softmax epilogue => halves the number of O-rescale passes (each is hd=256
//   serial mults per active lane — the dominant serial cost at hd=256).
__global__ void k_fattn_wmma_kt32(const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc, float *__restrict__ attn,
    int nq, int nkv, int hd, int kvd, int pos_start, int P) {
    int qt = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    int rep = nq / nkv, kvh = h / rep, kv_off = kvh * hd, att = nq * hd, HT = hd / 16, q0 = qt * 16;
    extern __shared__ char smem[];
    half *Qsh = (half *) smem;                       // [16*hd]
    half *Ksh = Qsh + 16 * hd, *Vsh = Ksh + 32 * hd; // K [32*hd], V [32*hd]
    half *Psh = Vsh + 32 * hd;                        // [16*32]
    float *Osh = (float *) (Psh + 16 * 32);          // [16*hd]
    float *Ssh = Osh + 16 * hd;                       // [16*32]
    __shared__ float m[16], l[16];
    int qmax = min(16, P - q0);
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd;
        Qsh[i] = (q0 + r < P) ? __float2half(q_all[(size_t)(q0 + r) * att + h * hd + d]) : __float2half(0.f);
        Osh[i] = 0.f; }
    if (lane < 16) { m[lane] = -3.402823466e38f; l[lane] = 0.f; }
    __syncwarp();
    float scale = rsqrtf((float) hd);
    int len_max = pos_start + q0 + qmax;
    for (int k0 = 0; k0 < len_max; k0 += 32) {
        int kn = min(32, len_max - k0);
        for (int i = lane; i < 32 * hd; i += 32) { int r = i / hd, d = i % hd;
            Ksh[i] = (r < kn) ? __float2half(Kc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f);
            Vsh[i] = (r < kn) ? __float2half(Vc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f); }
        __syncwarp();
        // QK^T for the two 16-key sub-tiles => Ssh[16][32]
        for (int sub = 0; sub < 2; ++sub) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
            wmma::fill_fragment(cf, 0.f);
            for (int kt = 0; kt < HT; ++kt) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
                wmma::load_matrix_sync(af, Qsh + kt * 16, hd);
                wmma::load_matrix_sync(bf, Ksh + sub * 16 * hd + kt * 16, hd);
                wmma::mma_sync(cf, af, bf, cf);
            }
            wmma::store_matrix_sync(Ssh + sub * 16, cf, 32, wmma::mem_row_major);
        }
        __syncwarp();
        if (lane < 16) {
            if (lane < qmax) {
                int qpos = pos_start + q0 + lane; float rmax = m[lane]; float s[32];
                for (int k = 0; k < 32; ++k) { float sv = (k < kn && (k0 + k) <= qpos) ? Ssh[lane * 32 + k] * scale : -3.402823466e38f; s[k] = sv; rmax = fmaxf(rmax, sv); }
                float corr = expf(m[lane] - rmax), lnew = l[lane] * corr;
                for (int k = 0; k < 32; ++k) { float p = (s[k] > -3.0e38f) ? expf(s[k] - rmax) : 0.f; Psh[lane * 32 + k] = __float2half(p); lnew += p; }
                for (int d = 0; d < hd; ++d) Osh[lane * hd + d] *= corr;
                m[lane] = rmax; l[lane] = lnew;
            } else for (int k = 0; k < 32; ++k) Psh[lane * 32 + k] = __float2half(0.f);
        }
        __syncwarp();
        // O += P[16x32] @ V[32xhd]  (two 16-contraction mmas accumulated)
        for (int n = 0; n < HT; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> of;
            wmma::load_matrix_sync(of, Osh + n * 16, hd, wmma::mem_row_major);
            for (int sub = 0; sub < 2; ++sub) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
                wmma::load_matrix_sync(af, Psh + sub * 16, 32);
                wmma::load_matrix_sync(bf, Vsh + sub * 16 * hd + n * 16, hd);
                wmma::mma_sync(of, af, bf, of);
            }
            wmma::store_matrix_sync(Osh + n * 16, of, hd, wmma::mem_row_major);
        }
        __syncwarp();
    }
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd; if (q0 + r < P) {
        float g = g_all[(size_t)(q0 + r) * att + h * hd + d];
        attn[(size_t)(q0 + r) * att + h * hd + d] = (Osh[i] / l[r]) * (1.f / (1.f + expf(-g))); } }
}

// ---- VARIANT B: fp16 O accumulator. Halves Osh (16KB->8KB at hd=256) to ease
//   the shared-memory occupancy ceiling. Precision: O kept fp16 in shared,
//   rescaled each key tile. Checked vs fp32 oracle (rel err must stay <1e-3).
__global__ void k_fattn_wmma_o16(const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc, float *__restrict__ attn,
    int nq, int nkv, int hd, int kvd, int pos_start, int P) {
    int qt = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    int rep = nq / nkv, kvh = h / rep, kv_off = kvh * hd, att = nq * hd, HT = hd / 16, q0 = qt * 16;
    extern __shared__ char smem[];
    half *Qsh = (half *) smem;
    half *Ksh = Qsh + 16 * hd, *Vsh = Ksh + 16 * hd, *Psh = Vsh + 16 * hd;
    half *Osh = Psh + 16 * 16;                        // fp16 O [16*hd]
    float *Ssh = (float *) (Osh + 16 * hd);           // [16*16]
    __shared__ float m[16], l[16];
    int qmax = min(16, P - q0);
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd;
        Qsh[i] = (q0 + r < P) ? __float2half(q_all[(size_t)(q0 + r) * att + h * hd + d]) : __float2half(0.f);
        Osh[i] = __float2half(0.f); }
    if (lane < 16) { m[lane] = -3.402823466e38f; l[lane] = 0.f; }
    __syncwarp();
    float scale = rsqrtf((float) hd);
    int len_max = pos_start + q0 + qmax;
    for (int k0 = 0; k0 < len_max; k0 += 16) {
        int kn = min(16, len_max - k0);
        for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd;
            Ksh[i] = (r < kn) ? __float2half(Kc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f);
            Vsh[i] = (r < kn) ? __float2half(Vc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f); }
        __syncwarp();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
        wmma::fill_fragment(cf, 0.f);
        for (int kt = 0; kt < HT; ++kt) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
            wmma::load_matrix_sync(af, Qsh + kt * 16, hd);
            wmma::load_matrix_sync(bf, Ksh + kt * 16, hd);
            wmma::mma_sync(cf, af, bf, cf);
        }
        wmma::store_matrix_sync(Ssh, cf, 16, wmma::mem_row_major);
        __syncwarp();
        if (lane < 16) {
            if (lane < qmax) {
                int qpos = pos_start + q0 + lane; float rmax = m[lane]; float s[16];
                for (int k = 0; k < 16; ++k) { float sv = (k < kn && (k0 + k) <= qpos) ? Ssh[lane * 16 + k] * scale : -3.402823466e38f; s[k] = sv; rmax = fmaxf(rmax, sv); }
                float corr = expf(m[lane] - rmax), lnew = l[lane] * corr;
                for (int k = 0; k < 16; ++k) { float p = (s[k] > -3.0e38f) ? expf(s[k] - rmax) : 0.f; Psh[lane * 16 + k] = __float2half(p); lnew += p; }
                for (int d = 0; d < hd; ++d) Osh[lane * hd + d] = __float2half(__half2float(Osh[lane * hd + d]) * corr);
                m[lane] = rmax; l[lane] = lnew;
            } else for (int k = 0; k < 16; ++k) Psh[lane * 16 + k] = __float2half(0.f);
        }
        __syncwarp();
        for (int n = 0; n < HT; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> of;   // fp16 accum
            wmma::load_matrix_sync(of, Osh + n * 16, hd, wmma::mem_row_major);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
            wmma::load_matrix_sync(af, Psh, 16);
            wmma::load_matrix_sync(bf, Vsh + n * 16, hd);
            wmma::mma_sync(of, af, bf, of);
            wmma::store_matrix_sync(Osh + n * 16, of, hd, wmma::mem_row_major);
        }
        __syncwarp();
    }
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd; if (q0 + r < P) {
        float g = g_all[(size_t)(q0 + r) * att + h * hd + d];
        attn[(size_t)(q0 + r) * att + h * hd + d] = (__half2float(Osh[i]) / l[r]) * (1.f / (1.f + expf(-g))); } }
}

static float* dup(const vector<float>&h){float*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}

int main(int argc, char** argv){
    int hd=256,nq=16,nkv=2,P=256;
    if(argc>=5){hd=atoi(argv[1]);nq=atoi(argv[2]);nkv=atoi(argv[3]);P=atoi(argv[4]);}
    int kvd=nkv*hd,att=nq*hd;
    printf("== dims hd=%d nq=%d nkv=%d P=%d  (rep=%d, kvd=%d, att=%d) ==\n",hd,nq,nkv,P,nq/nkv,kvd,att);
    // shared-mem sizes
    size_t shm_w  = (size_t)(48*hd+256)*2 + (size_t)(16*hd+256)*4;
    size_t shm_kt = (size_t)(16*hd + 2*32*hd + 16*32)*2 + (size_t)(16*hd + 16*32)*4;
    size_t shm_o16= (size_t)(48*hd + 256 + 16*hd)*2 + (size_t)(16*16)*4;
    int TK = (hd<=128)?32:16, TQW=32;
    size_t shm_t = (size_t)2*TK*hd*4;
    printf("shared bytes: wmma=%zu  kt32=%zu  o16=%zu  tiled=%zu\n",shm_w,shm_kt,shm_o16,shm_t);
    // opt-in for >48KB dynamic shared (Ada allows up to ~99KB)
    cudaFuncSetAttribute(k_fattn_wmma,      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_w);
    cudaFuncSetAttribute(k_fattn_wmma_kt32, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_kt);
    cudaFuncSetAttribute(k_fattn_wmma_o16,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_o16);

    std::mt19937 rng(3);std::normal_distribution<float> nd(0,1);
    int positions[]={0,4096,12288,25344};
    struct Run{const char*name;int kind;size_t shm;};
    Run runs[]={{"tiled",0,shm_t},{"wmma ",1,shm_w},{"kt32 ",2,shm_kt},{"o16  ",3,shm_o16}};
    for(int pi=0;pi<4;++pi){
        int ps=positions[pi];int len=ps+P;
        auto rnd=[&](size_t n){vector<float>x(n);for(auto&e:x)e=nd(rng)*0.3f;return x;};
        vector<float> qa=rnd((size_t)P*att),ga=rnd((size_t)P*att),K=rnd((size_t)len*kvd),V=rnd((size_t)len*kvd);
        float*dqa=dup(qa),*dga=dup(ga),*dK=dup(K),*dV=dup(V),*dref,*dw;
        CK(cudaMalloc(&dref,(size_t)P*att*4));CK(cudaMalloc(&dw,(size_t)P*att*4));
        k_fattn_attend_chunk<<<(size_t)nq*P,hd>>>(dqa,dga,dK,dV,dref,nq,nkv,hd,kvd,ps);
        CK(cudaDeviceSynchronize());
        vector<float> a(P*att);CK(cudaMemcpy(a.data(),dref,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        dim3 gw((P+15)/16,nq), gt((P+TQW-1)/TQW,nq);
        printf("pos=%5d len=%5d:\n",ps,len);
        for(auto&r:runs){
            auto launch=[&](float*o){
                if(r.kind==0) k_fattn_attend_tile<<<gt,TQW*32,r.shm>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P,TK);
                else if(r.kind==1) k_fattn_wmma<<<gw,32,r.shm>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P);
                else if(r.kind==2) k_fattn_wmma_kt32<<<gw,32,r.shm>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P);
                else k_fattn_wmma_o16<<<gw,32,r.shm>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P);
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
