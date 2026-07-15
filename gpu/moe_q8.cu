// Real-weights MoE test: Q8_0 expert weights. Compares per-token Q8 GEMV (aspida
// today) vs grouped [dequant Q8->fp16 for active experts + cuBLAS batched GEMM]
// (the ggml large-batch path). Includes the dequant cost so the speedup is honest.
//   nvcc -O3 -arch=native moe_q8.cu -lcublas -o mq && ./mq
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
#define CB(x) do{cublasStatus_t s=(x); if(s){printf("cuBLAS %d @ %d\n",(int)s,__LINE__);exit(1);} }while(0)
using std::vector;
// Q8_0: per 32-value block = fp16 scale (2B) + 32 int8 (32B) = 34B.
static const int QB=32, QBPB=34;

// per-token: warp per (pair,row), dot X(fp16,dim) . dequant(Wq8 row) ; current aspida structure
__global__ void k_pertoken_q8(const half*__restrict__ X,const uint8_t*__restrict__ Wq,
    const int*__restrict__ ptok,const int*__restrict__ pexp,float*__restrict__ Y,int npair,int dim,int intermed){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(wid>=npair*intermed)return;
    int p=wid/intermed,r=wid%intermed; const half*x=X+(size_t)ptok[p]*dim;
    size_t rowbytes=(size_t)(dim/QB)*QBPB; const uint8_t*wr=Wq+(((size_t)pexp[p]*intermed+r)*rowbytes);
    float acc=0.f;
    for(int b=0;b<dim/QB;++b){const uint8_t*bl=wr+(size_t)b*QBPB; __half sc; *(uint16_t*)&sc=*(const uint16_t*)bl; float s=__half2float(sc);
        const int8_t*q=(const int8_t*)(bl+2);
        for(int i=lane;i<QB;i+=32)acc+=__half2float(x[b*QB+i])*(s*(float)q[i]); }
    for(int o=16;o>0;o>>=1)acc+=__shfl_xor_sync(0xffffffffu,acc,o); if(lane==0)Y[(size_t)p*intermed+r]=acc;
}
// dequant one expert's Q8 weight [intermed,dim] -> fp16 Wf[intermed,dim]
__global__ void k_deq_expert(const uint8_t*__restrict__ Wq,half*__restrict__ Wf,int e,int intermed,int dim){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t tot=(size_t)intermed*dim; if(idx>=tot)return;
    int r=idx/dim, c=idx%dim; size_t rowbytes=(size_t)(dim/QB)*QBPB;
    const uint8_t*bl=Wq+((size_t)e*intermed+r)*rowbytes+(size_t)(c/QB)*QBPB;
    __half sc; *(uint16_t*)&sc=*(const uint16_t*)bl; const int8_t*q=(const int8_t*)(bl+2);
    Wf[idx]=__float2half(__half2float(sc)*(float)q[c%QB]);
}
__global__ void k_gather(const half*__restrict__ X,const int*__restrict__ gtok,half*__restrict__ Xg,int npair,int dim){
    int p=blockIdx.x,d=threadIdx.x+blockIdx.y*blockDim.x; if(d>=dim)return; Xg[(size_t)p*dim+d]=X[(size_t)gtok[p]*dim+d];
}
static half* duph(const vector<half>&h){half*p;CK(cudaMalloc(&p,h.size()*2));CK(cudaMemcpy(p,h.data(),h.size()*2,cudaMemcpyHostToDevice));return p;}
static int* dupi(const vector<int>&h){int*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}
int main(){
    int P=256,dim=2048,intermed=768,n_exp=64,top_k=8;
    std::mt19937 rng(5);std::normal_distribution<float> nd(0,1);std::uniform_int_distribution<int> ue(0,n_exp-1);
    vector<half> X((size_t)P*dim);for(auto&e:X)e=__float2half(nd(rng)*0.1f);
    // build Q8_0 weights (quantize random fp32)
    size_t rowbytes=(size_t)(dim/QB)*QBPB; vector<uint8_t> Wq((size_t)n_exp*intermed*rowbytes);
    for(size_t e=0;e<(size_t)n_exp;++e)for(int r=0;r<intermed;++r){uint8_t*row=Wq.data()+((e*intermed+r)*rowbytes);
        for(int b=0;b<dim/QB;++b){float v[32],amax=0; for(int i=0;i<QB;++i){v[i]=nd(rng)*0.05f;amax=std::max(amax,fabsf(v[i]));}
            float s=amax/127.f; __half sc=__float2half(s); uint8_t*bl=row+(size_t)b*QBPB; *(uint16_t*)bl=*(uint16_t*)&sc;
            int8_t*q=(int8_t*)(bl+2); for(int i=0;i<QB;++i)q[i]=(int8_t)lroundf(s>0?v[i]/s:0.f);}}
    vector<int> ptok,pexp; vector<vector<int>> byexp(n_exp);
    for(int t=0;t<P;++t){vector<int> es; while((int)es.size()<top_k){int e=ue(rng);if(std::find(es.begin(),es.end(),e)==es.end())es.push_back(e);}
        for(int e:es){ptok.push_back(t);pexp.push_back(e);}}
    int npair=ptok.size();
    vector<int> grp_tok,grp_off,grp_exp; grp_off.push_back(0); vector<int> pair_gidx(npair);
    for(int p=0;p<npair;++p)byexp[pexp[p]].push_back(p);
    for(int e=0;e<n_exp;++e){if(byexp[e].empty())continue; grp_exp.push_back(e);
        for(int p:byexp[e]){pair_gidx[p]=grp_tok.size();grp_tok.push_back(ptok[p]);} grp_off.push_back(grp_tok.size());}
    int ngrp=grp_exp.size();
    half*dX=duph(X); uint8_t*dWq;CK(cudaMalloc(&dWq,Wq.size()));CK(cudaMemcpy(dWq,Wq.data(),Wq.size(),cudaMemcpyHostToDevice));
    int*dptok=dupi(ptok),*dpexp=dupi(pexp),*dgtok=dupi(grp_tok);
    float*dYa,*dYc; CK(cudaMalloc(&dYa,(size_t)npair*intermed*4));CK(cudaMalloc(&dYc,(size_t)npair*intermed*4));
    half*dXg,*dWf; CK(cudaMalloc(&dXg,(size_t)npair*dim*2)); CK(cudaMalloc(&dWf,(size_t)intermed*dim*2));
    cublasHandle_t hdl;CB(cublasCreate(&hdl));CB(cublasSetMathMode(hdl,CUBLAS_TENSOR_OP_MATH));float al=1,be=0;
    auto rA=[&](){k_pertoken_q8<<<((size_t)npair*intermed*32+255)/256,256>>>(dX,dWq,dptok,dpexp,dYa,npair,dim,intermed);};
    auto rC=[&](){ dim3 gg(npair,(dim+255)/256); k_gather<<<gg,256>>>(dX,dgtok,dXg,npair,dim);
        for(int g=0;g<ngrp;++g){int off=grp_off[g],ne=grp_off[g+1]-off; if(ne<=0)continue;
            k_deq_expert<<<((size_t)intermed*dim+255)/256,256>>>(dWq,dWf,grp_exp[g],intermed,dim);
            CB(cublasGemmEx(hdl,CUBLAS_OP_T,CUBLAS_OP_N,intermed,ne,dim,&al,dWf,CUDA_R_16F,dim,
                dXg+(size_t)off*dim,CUDA_R_16F,dim,&be,dYc+(size_t)off*intermed,CUDA_R_32F,intermed,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)); } };
    rA();rC();CK(cudaDeviceSynchronize());CK(cudaGetLastError());
    vector<float> Ya((size_t)npair*intermed),Yc((size_t)npair*intermed);
    CK(cudaMemcpy(Ya.data(),dYa,(size_t)npair*intermed*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Yc.data(),dYc,(size_t)npair*intermed*4,cudaMemcpyDeviceToHost));
    float mx=0,dn=0;for(int p=0;p<npair;++p){int gp=pair_gidx[p];for(int r=0;r<intermed;++r){
        float a=Ya[(size_t)p*intermed+r],c=Yc[(size_t)gp*intermed+r];mx=std::max(mx,fabsf(a-c));dn=std::max(dn,fabsf(a));}}
    auto tm=[&](auto fn){for(int i=0;i<5;++i)fn();CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);int N=50;
        cudaEventRecord(e0);for(int i=0;i<N;++i)fn();cudaEventRecord(e1);CK(cudaEventSynchronize(e1));
        float t=0;cudaEventElapsedTime(&t,e0,e1);return t/N;};
    float ta=tm(rA),tc=tm(rC);
    printf("MoE gate/up Q8 real-weights: P=%d dim=%d intermed=%d n_exp=%d top_k=%d npair=%d ngrp=%d\n",P,dim,intermed,n_exp,top_k,npair,ngrp);
    printf("  per-token Q8 GEMV (aspida today) : %.3f ms\n  grouped dequant+cuBLAS (ggml way): %.3f ms  speedup=%.2fx  rel~%.1e\n",
        ta,tc,ta/tc,mx/(dn+1e-9f));
    return 0;
}
