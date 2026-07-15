// wmma FlashAttention, SPLIT-K (flash-decoding): split the key range into S slices,
// one block per (query-tile, head, slice) -> S x more blocks -> fills the GPU (v1
// launched only P/16*nq=512 blocks/layer, ~3.6/SM, latency-starved). Each block does
// online softmax over its slice -> partial (O,m,l); combine kernel merges per query.
// Validated vs k_fattn_attend_chunk.  nvcc -O3 -arch=native fattn_sk.cu -o fsk
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
namespace w = nvcuda::wmma;

// ---- bit-exact fp32 oracle (aspida k_fattn_attend_chunk) ----
__global__ void k_oracle(const float*__restrict__ q_all,const float*__restrict__ g_all,
    const float*__restrict__ Kc,const float*__restrict__ Vc,float*__restrict__ attn,
    int nq,int nkv,int hd,int kvd,int pos_start){
    int t=blockIdx.x/nq,h=blockIdx.x%nq,d=threadIdx.x;
    int len=pos_start+t+1,rep=nq/nkv,kvh=h/rep,q_off=h*hd,kv_off=kvh*hd,att=nq*hd;
    const float*q=q_all+(size_t)t*att+q_off; float scale=rsqrtf((float)hd);
    __shared__ float red[256],qsh[256]; qsh[d]=q[d]; __syncthreads();
    float m=-3.402823466e38f,l=0.f,acc=0.f;
    for(int s=0;s<len;++s){const float*k=Kc+(size_t)s*kvd+kv_off; float part=qsh[d]*k[d];
        red[d]=part; __syncthreads();
        for(int o=blockDim.x/2;o>0;o>>=1){if(d<o)red[d]+=red[d+o];__syncthreads();}
        float dot=red[0]*scale; __syncthreads();
        float mn=fmaxf(m,dot),corr=expf(m-mn),p=expf(dot-mn);
        l=l*corr+p; acc=acc*corr+p*Vc[(size_t)s*kvd+kv_off+d]; m=mn;}
    float g=g_all[(size_t)t*att+q_off+d];
    attn[(size_t)t*att+q_off+d]=(acc/l)*(1.f/(1.f+expf(-g)));
}

// ---- split-K attend: partial (O,m,l) per (qtile, head, slice) ----
__global__ void k_splitk(const float*__restrict__ q_all,const float*__restrict__ Kc,
    const float*__restrict__ Vc,float*__restrict__ Opart,float*__restrict__ mpart,float*__restrict__ lpart,
    int nq,int nkv,int hd,int kvd,int pos_start,int P,int S,int ntiles){
    int qt=blockIdx.x,h=blockIdx.y,sl=blockIdx.z,lane=threadIdx.x;
    int rep=nq/nkv,kvh=h/rep,kv_off=kvh*hd,att=nq*hd,HT=hd/16,q0=qt*16;
    extern __shared__ char smem[];
    half *Qsh=(half*)smem,*Ksh=Qsh+16*hd,*Vsh=Ksh+16*hd,*Psh=Vsh+16*hd;
    float *Osh=(float*)(Psh+16*16),*Ssh=Osh+16*hd;
    __shared__ float m[16],l[16];
    int qmax=min(16,P-q0);
    for(int i=lane;i<16*hd;i+=32){int r=i/hd,d=i%hd;
        Qsh[i]=(q0+r<P)?__float2half(q_all[(size_t)(q0+r)*att+h*hd+d]):__float2half(0.f);Osh[i]=0.f;}
    if(lane<16){m[lane]=-3.402823466e38f;l[lane]=0.f;}
    __syncwarp();
    float scale=rsqrtf((float)hd);
    int total=pos_start+q0+qmax, Wd=(pos_start+P+S-1)/S, klo=sl*Wd, khi=min((sl+1)*Wd,total);
    for(int k0=(klo/16)*16;k0<khi;k0+=16){
        int kn=min(16,khi-k0); if(kn<=0)break;
        for(int i=lane;i<16*hd;i+=32){int r=i/hd,d=i%hd; bool ok=(k0+r>=klo&&r<kn);
            Ksh[i]=ok?__float2half(Kc[(size_t)(k0+r)*kvd+kv_off+d]):__float2half(0.f);
            Vsh[i]=ok?__float2half(Vc[(size_t)(k0+r)*kvd+kv_off+d]):__float2half(0.f);}
        __syncwarp();
        w::fragment<w::accumulator,16,16,16,float> cf; w::fill_fragment(cf,0.f);
        for(int kt=0;kt<HT;++kt){w::fragment<w::matrix_a,16,16,16,half,w::row_major> af;
            w::fragment<w::matrix_b,16,16,16,half,w::col_major> bf;
            w::load_matrix_sync(af,Qsh+kt*16,hd);w::load_matrix_sync(bf,Ksh+kt*16,hd);w::mma_sync(cf,af,bf,cf);}
        w::store_matrix_sync(Ssh,cf,16,w::mem_row_major); __syncwarp();
        if(lane<16&&lane<qmax){
            int qpos=pos_start+q0+lane; float rmax=m[lane]; float s[16];
            for(int k=0;k<16;++k){int kp=k0+k;float sv=(k<kn&&kp>=klo&&kp<=qpos)?Ssh[lane*16+k]*scale:-3.402823466e38f;s[k]=sv;rmax=fmaxf(rmax,sv);}
            float corr=expf(m[lane]-rmax),lnew=l[lane]*corr;
            for(int k=0;k<16;++k){float p=(s[k]>-3.0e38f)?expf(s[k]-rmax):0.f;Psh[lane*16+k]=__float2half(p);lnew+=p;}
            for(int d=0;d<hd;++d)Osh[lane*hd+d]*=corr; m[lane]=rmax;l[lane]=lnew;
        } else if(lane<16) for(int k=0;k<16;++k)Psh[lane*16+k]=__float2half(0.f);
        __syncwarp();
        for(int n=0;n<HT;++n){w::fragment<w::accumulator,16,16,16,float> of;
            w::load_matrix_sync(of,Osh+n*16,hd,w::mem_row_major);
            w::fragment<w::matrix_a,16,16,16,half,w::row_major> af;w::fragment<w::matrix_b,16,16,16,half,w::row_major> bf;
            w::load_matrix_sync(af,Psh,16);w::load_matrix_sync(bf,Vsh+n*16,hd);
            w::mma_sync(of,af,bf,of);w::store_matrix_sync(Osh+n*16,of,hd,w::mem_row_major);}
        __syncwarp();
    }
    size_t base=(((size_t)sl*ntiles+qt)*nq+h);
    for(int i=lane;i<16*hd;i+=32)Opart[base*16*hd+i]=Osh[i];
    if(lane<16){mpart[base*16+lane]=m[lane];lpart[base*16+lane]=l[lane];}
}
__global__ void k_combine(const float*__restrict__ Opart,const float*__restrict__ mpart,
    const float*__restrict__ lpart,const float*__restrict__ g_all,float*__restrict__ attn,
    int nq,int hd,int P,int S,int ntiles){
    int qt=blockIdx.x,h=blockIdx.y,d=threadIdx.x; int att=nq*hd,q0=qt*16;
    for(int r=0;r<16;++r){ if(q0+r>=P)break;
        float M=-3.402823466e38f;
        for(int s=0;s<S;++s)M=fmaxf(M,mpart[(((size_t)s*ntiles+qt)*nq+h)*16+r]);
        float acc=0.f,L=0.f;
        for(int s=0;s<S;++s){size_t b=(((size_t)s*ntiles+qt)*nq+h);float ms=mpart[b*16+r];
            if(ms<-3.0e38f)continue; float wt=expf(ms-M); acc+=wt*Opart[b*16*hd+r*hd+d]; L+=wt*lpart[b*16+r];}
        float g=g_all[(size_t)(q0+r)*att+h*hd+d];
        attn[(size_t)(q0+r)*att+h*hd+d]=(acc/L)*(1.f/(1.f+expf(-g)));
    }
}
static float* dup(const vector<float>&h){float*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}
int main(int argc,char**argv){
    int hd=128,nq=32,nkv=8,P=256; int kvd=nkv*hd,att=nq*hd,ntiles=(P+15)/16;
    int S=(argc>1?atoi(argv[1]):8);
    size_t shm=(size_t)(48*hd+256)*2+(size_t)(16*hd+256)*4;
    std::mt19937 rng(3);std::normal_distribution<float> nd(0,1);
    int positions[]={0,4096,12288,25344};
    float*Op,*mp,*lp; CK(cudaMalloc(&Op,(size_t)S*ntiles*nq*16*hd*4));
    CK(cudaMalloc(&mp,(size_t)S*ntiles*nq*16*4));CK(cudaMalloc(&lp,(size_t)S*ntiles*nq*16*4));
    for(int pi=0;pi<4;++pi){
        int ps=positions[pi];int len=ps+P;
        auto rnd=[&](size_t n){vector<float>x(n);for(auto&e:x)e=nd(rng)*0.3f;return x;};
        vector<float> qa=rnd((size_t)P*att),ga=rnd((size_t)P*att),K=rnd((size_t)len*kvd),V=rnd((size_t)len*kvd);
        float*dqa=dup(qa),*dga=dup(ga),*dK=dup(K),*dV=dup(V),*dref,*dw;
        CK(cudaMalloc(&dref,(size_t)P*att*4));CK(cudaMalloc(&dw,(size_t)P*att*4));
        k_oracle<<<(size_t)nq*P,hd>>>(dqa,dga,dK,dV,dref,nq,nkv,hd,kvd,ps);
        dim3 gA(ntiles,nq,S),gC(ntiles,nq);
        auto run=[&](){k_splitk<<<gA,32,shm>>>(dqa,dK,dV,Op,mp,lp,nq,nkv,hd,kvd,ps,P,S,ntiles);
                       k_combine<<<gC,hd>>>(Op,mp,lp,dga,dw,nq,hd,P,S,ntiles);};
        run();CK(cudaDeviceSynchronize());cudaError_t e=cudaGetLastError();if(e){printf("err %s\n",cudaGetErrorString(e));return 1;}
        vector<float> a(P*att),b(P*att);CK(cudaMemcpy(a.data(),dref,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(),dw,(size_t)P*att*4,cudaMemcpyDeviceToHost));
        float mx=0,dn=0;for(size_t i=0;i<a.size();++i){mx=std::max(mx,fabsf(a[i]-b[i]));dn=std::max(dn,fabsf(a[i]));}
        for(int i=0;i<3;++i)run();CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);int N=30;
        cudaEventRecord(e0);for(int i=0;i<N;++i)run();cudaEventRecord(e1);CK(cudaEventSynchronize(e1));
        float t=0;cudaEventElapsedTime(&t,e0,e1);
        printf("pos=%5d len=%5d  splitK(S=%d)=%.3f ms  max|err|=%.2e rel~%.1e\n",ps,len,S,t/N,mx,mx/(dn+1e-9f));
        cudaFree(dqa);cudaFree(dga);cudaFree(dK);cudaFree(dV);cudaFree(dref);cudaFree(dw);
    }
    return 0;
}
