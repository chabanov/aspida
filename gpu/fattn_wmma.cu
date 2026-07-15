// Tensor-core (wmma) FlashAttention for aspida prefill, referencing the online-
// softmax tiling of llama.cpp fattn-mma-f16.cuh, implemented on the wmma pattern
// aspida already ships in k_q8_wmma. One warp = one 16-query tile x head; Q@K^T
// and P@V via 16x16x16 wmma (fp16 in, fp32 accum), softmax in shared, O rescaled
// between key tiles. Validated vs k_fattn_attend_chunk (fp32 oracle).
//   nvcc -O3 -arch=native fattn_wmma.cu -o fw && ./fw
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

// hd must be a multiple of 16. QT=KT=16.
__global__ void k_fattn_wmma(const float*__restrict__ q_all,const float*__restrict__ g_all,
    const float*__restrict__ Kc,const float*__restrict__ Vc,float*__restrict__ attn,
    int nq,int nkv,int hd,int kvd,int pos_start,int P){
    int qt=blockIdx.x, h=blockIdx.y, lane=threadIdx.x;
    int rep=nq/nkv, kvh=h/rep, kv_off=kvh*hd, att=nq*hd, HT=hd/16, q0=qt*16;
    extern __shared__ char smem[];
    half *Qsh=(half*)smem;                 // [16*hd]
    half *Ksh=Qsh+16*hd, *Vsh=Ksh+16*hd, *Psh=Vsh+16*hd;   // [16*hd],[16*hd],[16*16]
    float *Osh=(float*)(Psh+16*16);        // [16*hd]
    float *Ssh=Osh+16*hd;                  // [16*16]
    __shared__ float m[16],l[16];
    int qmax=min(16,P-q0);
    for(int i=lane;i<16*hd;i+=32){int r=i/hd,d=i%hd;
        Qsh[i]=(r<qmax)?__float2half(q_all[(size_t)(q0+r)*att+h*hd+d]):__float2half(0.f); Osh[i]=0.f;}
    if(lane<16){m[lane]=-3.402823466e38f;l[lane]=0.f;}
    __syncwarp();
    float scale=rsqrtf((float)hd);
    int len_max=pos_start+q0+qmax;
    for(int k0=0;k0<len_max;k0+=16){
        int kn=min(16,len_max-k0);
        for(int i=lane;i<16*hd;i+=32){int r=i/hd,d=i%hd;
            float kv=(r<kn)?Kc[(size_t)(k0+r)*kvd+kv_off+d]:0.f; Ksh[i]=__float2half(kv);
            float vv=(r<kn)?Vc[(size_t)(k0+r)*kvd+kv_off+d]:0.f; Vsh[i]=__float2half(vv);}
        __syncwarp();
        wmma::fragment<wmma::accumulator,16,16,16,float> cf; wmma::fill_fragment(cf,0.f);
        for(int kt=0;kt<HT;++kt){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
            wmma::load_matrix_sync(af,Qsh+kt*16,hd);
            wmma::load_matrix_sync(bf,Ksh+kt*16,hd);   // col_major => K^T
            wmma::mma_sync(cf,af,bf,cf);
        }
        wmma::store_matrix_sync(Ssh,cf,16,wmma::mem_row_major);
        __syncwarp();
        if(lane<16){
            if(lane<qmax){
                int qpos=pos_start+q0+lane; float rmax=m[lane]; float s[16];
                for(int k=0;k<16;++k){float sv=(k<kn&&(k0+k)<=qpos)?Ssh[lane*16+k]*scale:-3.402823466e38f;s[k]=sv;rmax=fmaxf(rmax,sv);}
                float corr=expf(m[lane]-rmax),lnew=l[lane]*corr;
                for(int k=0;k<16;++k){float p=(s[k]>-3.0e38f)?expf(s[k]-rmax):0.f;Psh[lane*16+k]=__float2half(p);lnew+=p;}
                for(int d=0;d<hd;++d)Osh[lane*hd+d]*=corr;
                m[lane]=rmax;l[lane]=lnew;
            } else for(int k=0;k<16;++k)Psh[lane*16+k]=__float2half(0.f);
        }
        __syncwarp();
        for(int n=0;n<HT;++n){
            wmma::fragment<wmma::accumulator,16,16,16,float> of;
            wmma::load_matrix_sync(of,Osh+n*16,hd,wmma::mem_row_major);
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(af,Psh,16);
            wmma::load_matrix_sync(bf,Vsh+n*16,hd);
            wmma::mma_sync(of,af,bf,of);
            wmma::store_matrix_sync(Osh+n*16,of,hd,wmma::mem_row_major);
        }
        __syncwarp();
    }
    for(int i=lane;i<16*hd;i+=32){int r=i/hd,d=i%hd; if(q0+r<P){
        float g=g_all[(size_t)(q0+r)*att+h*hd+d];
        attn[(size_t)(q0+r)*att+h*hd+d]=(Osh[i]/l[r])*(1.f/(1.f+expf(-g)));}}
}
static float* dup(const vector<float>&h){float*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}
int main(){
    int hd=128,nq=32,nkv=8,P=256; int kvd=nkv*hd,att=nq*hd;
    std::mt19937 rng(3);std::normal_distribution<float> nd(0,1);
    int positions[]={0,4096,12288,25344};
    for(int pi=0;pi<4;++pi){
        int ps=positions[pi];int len=ps+P;
        auto rnd=[&](size_t n){vector<float>x(n);for(auto&e:x)e=nd(rng)*0.3f;return x;};
        vector<float> qa=rnd((size_t)P*att),ga=rnd((size_t)P*att),K=rnd((size_t)len*kvd),V=rnd((size_t)len*kvd);
        float*dqa=dup(qa),*dga=dup(ga),*dK=dup(K),*dV=dup(V),*dref,*dw;
        CK(cudaMalloc(&dref,(size_t)P*att*4));CK(cudaMalloc(&dw,(size_t)P*att*4));
        auto run_ref=[&](float*o){k_fattn_attend_chunk<<<(size_t)nq*P,hd>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps);};
        size_t shm=(size_t)(16*hd*2*3 + 16*16*2)/*half bytes*/ + (size_t)(16*hd+16*16)*4;
        dim3 grid((P+15)/16,nq);
        auto run_w=[&](float*o){k_fattn_wmma<<<grid,32,shm>>>(dqa,dga,dK,dV,o,nq,nkv,hd,kvd,ps,P);};
        run_ref(dref);run_w(dw);CK(cudaDeviceSynchronize());
        cudaError_t e=cudaGetLastError(); if(e){printf("launch err: %s (shm=%zu)\n",cudaGetErrorString(e),shm);return 1;}
        vector<float> a(P*att),b(P*att);CK(cudaMemcpy(a.data(),dref,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(),dw,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        float mx=0,denom=0;for(size_t i=0;i<a.size();++i){mx=std::max(mx,fabsf(a[i]-b[i]));denom=std::max(denom,fabsf(a[i]));}
        for(int i=0;i<3;++i)run_w(dw);CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);int N=30;
        cudaEventRecord(e0);for(int i=0;i<N;++i)run_w(dw);cudaEventRecord(e1);CK(cudaEventSynchronize(e1));
        float t=0;cudaEventElapsedTime(&t,e0,e1);
        printf("pos=%5d len=%5d  wmma=%.3f ms/chunk  max|err|=%.2e (rel~%.1e)\n",ps,len,t/N,mx,mx/(denom+1e-9f));
        cudaFree(dqa);cudaFree(dga);cudaFree(dK);cudaFree(dV);cudaFree(dref);cudaFree(dw);
    }
    return 0;
}
