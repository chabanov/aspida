// bench_densedecode.cu — decode-regime (B<=16) dense Q8_0 projection bench.
// Question: at real Ornith-35B attention-projection shapes and B=8 (8 lanes),
// which beats the current scalar k_q8_0_wb — a B-templated acc[BT] variant
// (restores occupancy the acc[MAXB=32] array destroys) or the tensor-core
// k_q8_wmma?  All on H200. Correctness = NMSE vs an fp64 CPU reference on the
// SAME synthetic Q8_0 weights.
//
// build: nvcc -O3 --fmad=false -arch=native gpu/bench_densedecode.cu -o /tmp/bdd
// run:   flock /tmp/aspida_bench.lock -c /tmp/bdd
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define MAXB 32
#define WMM_TM 16
#define WMM_TN 16
#define WMM_TK 32

__device__ __forceinline__ float f16(const uint8_t*p){__half h;*reinterpret_cast<uint16_t*>(&h)=(uint16_t)p[0]|((uint16_t)p[1]<<8);return __half2float(h);}
__device__ __forceinline__ float warp_reduce(float v){for(int o=16;o;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);return v;}

// ---- current production kernel: acc[MAXB], warp-per-output-row ----
__global__ void k_q8_0_wb(const uint8_t*__restrict__ w,const float*__restrict__ x,
                          float*__restrict__ y,int in,int out,int B){
    int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31;
    if(row>=out)return;
    int nb=in/32; size_t bpr=(size_t)nb*34;
    const uint8_t*r=w+(size_t)row*bpr; float acc[MAXB];
    for(int b=0;b<B;++b)acc[b]=0.f;
    for(int blk=0;blk<nb;++blk){
        const uint8_t*bl=r+(size_t)blk*34; float d=f16(bl);
        const int8_t*qs=(const int8_t*)(bl+2);
        float wv=d*(float)qs[lane]; int i=blk*32+lane;
        for(int b=0;b<B;++b)acc[b]+=wv*x[(size_t)b*in+i];
    }
    for(int b=0;b<B;++b){float a=warp_reduce(acc[b]);if(lane==0)y[(size_t)b*out+row]=a;}
}

// ---- proposed fix: B templated -> acc[BT] uses exactly BT regs ----
template<int BT>
__global__ void k_q8_0_wb_T(const uint8_t*__restrict__ w,const float*__restrict__ x,
                            float*__restrict__ y,int in,int out){
    int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31;
    if(row>=out)return;
    int nb=in/32; size_t bpr=(size_t)nb*34;
    const uint8_t*r=w+(size_t)row*bpr; float acc[BT];
    #pragma unroll
    for(int b=0;b<BT;++b)acc[b]=0.f;
    for(int blk=0;blk<nb;++blk){
        const uint8_t*bl=r+(size_t)blk*34; float d=f16(bl);
        const int8_t*qs=(const int8_t*)(bl+2);
        float wv=d*(float)qs[lane]; int i=blk*32+lane;
        #pragma unroll
        for(int b=0;b<BT;++b)acc[b]+=wv*x[(size_t)b*in+i];
    }
    #pragma unroll
    for(int b=0;b<BT;++b){float a=warp_reduce(acc[b]);if(lane==0)y[(size_t)b*out+row]=a;}
}

// ---- tensor-core baseline (verbatim k_q8_wmma, fp16 dequant) ----
__global__ void k_q8_wmma(const uint8_t*__restrict__ w,const float*__restrict__ x,
                          float*__restrict__ y,int in,int out,int B){
    int n0=blockIdx.x*WMM_TN,m0=blockIdx.y*WMM_TM,lane=threadIdx.x&31;
    int nb=in/32; size_t bpr=(size_t)nb*34;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> cf;
    nvcuda::wmma::fill_fragment(cf,0.0f);
    __shared__ half As[WMM_TM*WMM_TK];
    __shared__ half Bs[WMM_TK*WMM_TN];
    for(int blk=0;blk<nb;++blk){
        for(int n=0;n<WMM_TN;++n){int gn=n0+n;
            if(gn<out){const uint8_t*bl=w+(size_t)gn*bpr+(size_t)blk*34;float d=f16(bl);
                const int8_t*qs=(const int8_t*)(bl+2);Bs[(size_t)lane*WMM_TN+n]=__float2half(d*(float)qs[lane]);}
            else Bs[(size_t)lane*WMM_TN+n]=__float2half(0.f);}
        for(int m=0;m<WMM_TM;++m){int gm=m0+m;int i=blk*32+lane;
            As[m*WMM_TK+lane]=(gm<B)?__float2half(x[(size_t)gm*in+i]):__float2half(0.f);}
        __syncwarp();
        for(int k16=0;k16<2;++k16){
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af,As+k16*16,WMM_TK);
            nvcuda::wmma::load_matrix_sync(bf,Bs+(size_t)k16*16*WMM_TN,WMM_TN);
            nvcuda::wmma::mma_sync(cf,af,bf,cf);
        }
        __syncwarp();
    }
    __shared__ float Cs[WMM_TM*WMM_TN];
    nvcuda::wmma::store_matrix_sync(Cs,cf,WMM_TN,nvcuda::wmma::mem_row_major);
    for(int idx=lane;idx<WMM_TM*WMM_TN;idx+=32){int m=idx/WMM_TN,n=idx%WMM_TN;
        int gm=m0+m,gn=n0+n; if(gm<B&&gn<out)y[(size_t)gm*out+gn]=Cs[idx];}
}

struct Shape{const char*name;int in,out;};

int main(){
    std::mt19937 rng(1234); std::uniform_real_distribution<float> U(-1.f,1.f);
    Shape shapes[]={{"q_proj 2048->4096",2048,4096},{"o_proj 4096->2048",4096,2048},
                    {"kv_proj 2048->512",2048,512}};
    const int B=8, IT=200;
    size_t fb,tb; cudaMemGetInfo(&fb,&tb); printf("H200 VRAM free=%zuMB, B=%d\n",fb/1048576,B);
    cudaStream_t st; cudaStreamCreate(&st);
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for(Shape S:shapes){
        int in=S.in,out=S.out,nb=in/32; size_t bpr=(size_t)nb*34;
        // synthetic Q8_0 weights
        std::vector<uint8_t> hw((size_t)out*bpr);
        for(int r=0;r<out;++r)for(int blk=0;blk<nb;++blk){uint8_t*bl=hw.data()+(size_t)r*bpr+(size_t)blk*34;
            half d=__float2half(0.01f+0.001f*(r%7)); *reinterpret_cast<uint16_t*>(bl)=*reinterpret_cast<uint16_t*>(&d);
            for(int j=0;j<32;++j)((int8_t*)(bl+2))[j]=(int8_t)((rng()%255)-127);}
        std::vector<float> hx((size_t)B*in); for(auto&v:hx)v=U(rng);
        uint8_t*dw; float*dx,*dy; CK(cudaMalloc(&dw,hw.size())); CK(cudaMalloc(&dx,hx.size()*4));
        CK(cudaMalloc(&dy,(size_t)B*out*4));
        CK(cudaMemcpy(dw,hw.data(),hw.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
        // fp64 CPU reference
        std::vector<double> ref((size_t)B*out,0.0);
        for(int r=0;r<out;++r)for(int blk=0;blk<nb;++blk){const uint8_t*bl=hw.data()+(size_t)r*bpr+(size_t)blk*34;
            __half hd;*reinterpret_cast<uint16_t*>(&hd)=*reinterpret_cast<const uint16_t*>(bl);double d=__half2float(hd);
            const int8_t*qs=(const int8_t*)(bl+2);
            for(int b=0;b<B;++b)for(int j=0;j<32;++j)ref[(size_t)b*out+r]+=d*qs[j]*(double)hx[(size_t)b*in+blk*32+j];}
        std::vector<float> hy((size_t)B*out);
        auto nmse=[&](){CK(cudaMemcpy(hy.data(),dy,(size_t)B*out*4,cudaMemcpyDeviceToHost));
            double se=0,sr=0;for(size_t i=0;i<hy.size();++i){double e=hy[i]-ref[i];se+=e*e;sr+=ref[i]*ref[i];}return se/(sr+1e-12);};
        int TPB=256,WPB=TPB/32,blocks=(out+WPB-1)/WPB;
        auto scalar=[&](){k_q8_0_wb<<<blocks,TPB,0,st>>>(dw,dx,dy,in,out,B);};
        auto tmpl=[&](){k_q8_0_wb_T<8><<<blocks,TPB,0,st>>>(dw,dx,dy,in,out);};
        auto wmma=[&](){dim3 g((out+15)/16,(B+15)/16);k_q8_wmma<<<g,32,0,st>>>(dw,dx,dy,in,out,B);};
        auto timeit=[&](auto fn){for(int i=0;i<10;++i)fn();cudaStreamSynchronize(st);
            float best=1e30f;for(int r=0;r<3;++r){cudaEventRecord(e0,st);for(int i=0;i<IT;++i)fn();cudaEventRecord(e1,st);
            cudaEventSynchronize(e1);float ms;cudaEventElapsedTime(&ms,e0,e1);ms/=IT;if(ms<best)best=ms;}return best;};
        float ts=timeit(scalar); scalar(); cudaStreamSynchronize(st); double ns=nmse();
        float tt=timeit(tmpl);   tmpl();   cudaStreamSynchronize(st); double nt=nmse();
        float tw=timeit(wmma);   wmma();   cudaStreamSynchronize(st); double nw=nmse();
        printf("\n== %s  (B=%d) ==\n",S.name,B);
        printf("  scalar acc[32]  : %8.4f ms   NMSE=%.2e   (baseline)\n",ts,ns);
        printf("  templ  acc[8]   : %8.4f ms   NMSE=%.2e   speedup %.2fx\n",tt,nt,ts/tt);
        printf("  wmma  (fp16 TC) : %8.4f ms   NMSE=%.2e   speedup %.2fx\n",tw,nw,ts/tw);
        cudaFree(dw);cudaFree(dx);cudaFree(dy);
    }
    return 0;
}
