// MoE gate/up projection: per-token GEMV (current aspida) vs grouped wmma vs
// grouped cuBLAS (the ggml large-batch path: gather tokens per expert, batched
// tensor-core GEMM). fp16 weights. Correctness vs per-token + timing.
//   nvcc -O3 -arch=native moe_cublas.cu -lcublas -o mc && ./mc
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

// per-token: one warp per (pair,row) scalar dot (current structure)
__global__ void k_pertoken(const half*__restrict__ X,const half*__restrict__ W,
    const int*__restrict__ ptok,const int*__restrict__ pexp,float*__restrict__ Y,int npair,int dim,int intermed){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(wid>=npair*intermed)return;
    int p=wid/intermed,r=wid%intermed; const half*x=X+(size_t)ptok[p]*dim; const half*wr=W+((size_t)pexp[p]*intermed+r)*dim;
    float acc=0.f; for(int i=lane;i<dim;i+=32)acc+=__half2float(x[i])*__half2float(wr[i]);
    for(int o=16;o>0;o>>=1)acc+=__shfl_xor_sync(0xffffffffu,acc,o); if(lane==0)Y[(size_t)p*intermed+r]=acc;
}
// gather Xg[pair,dim] = X[grp_tok[pair]]
__global__ void k_gather(const half*__restrict__ X,const int*__restrict__ gtok,half*__restrict__ Xg,int npair,int dim){
    int p=blockIdx.x,d=threadIdx.x+blockIdx.y*blockDim.x; if(d>=dim)return; Xg[(size_t)p*dim+d]=X[(size_t)gtok[p]*dim+d];
}
static half* duph(const vector<half>&h){half*p;CK(cudaMalloc(&p,h.size()*2));CK(cudaMemcpy(p,h.data(),h.size()*2,cudaMemcpyHostToDevice));return p;}
static int* dupi(const vector<int>&h){int*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}
int main(){
    int P=256,dim=2048,intermed=768,n_exp=64,top_k=8;
    std::mt19937 rng(5);std::normal_distribution<float> nd(0,1);std::uniform_int_distribution<int> ue(0,n_exp-1);
    vector<half> X((size_t)P*dim),W((size_t)n_exp*intermed*dim);
    for(auto&e:X)e=__float2half(nd(rng)*0.1f); for(auto&e:W)e=__float2half(nd(rng)*0.05f);
    vector<int> ptok,pexp; vector<vector<int>> byexp(n_exp);
    for(int t=0;t<P;++t){vector<int> es; while((int)es.size()<top_k){int e=ue(rng);if(std::find(es.begin(),es.end(),e)==es.end())es.push_back(e);}
        for(int e:es){ptok.push_back(t);pexp.push_back(e);}}
    int npair=ptok.size();
    // grouped: pairs sorted by expert
    vector<int> grp_tok,grp_off,grp_exp; grp_off.push_back(0); vector<int> pair_gidx(npair);
    for(int p=0;p<npair;++p)byexp[pexp[p]].push_back(p);
    for(int e=0;e<n_exp;++e){ if(byexp[e].empty())continue; grp_exp.push_back(e);
        for(int p:byexp[e]){pair_gidx[p]=grp_tok.size(); grp_tok.push_back(ptok[p]);} grp_off.push_back(grp_tok.size()); }
    int ngrp=grp_exp.size();
    half*dX=duph(X),*dW=duph(W); int*dptok=dupi(ptok),*dpexp=dupi(pexp),*dgtok=dupi(grp_tok);
    float*dYa,*dYc; CK(cudaMalloc(&dYa,(size_t)npair*intermed*4));CK(cudaMalloc(&dYc,(size_t)npair*intermed*4));
    half*dXg; CK(cudaMalloc(&dXg,(size_t)npair*dim*2));
    cublasHandle_t hdl; CB(cublasCreate(&hdl)); CB(cublasSetMathMode(hdl,CUBLAS_TENSOR_OP_MATH));
    float al=1.f,be=0.f;
    // (A) per-token
    auto rA=[&](){k_pertoken<<<((size_t)npair*intermed*32+255)/256,256>>>(dX,dW,dptok,dpexp,dYa,npair,dim,intermed);};
    // (C) grouped cuBLAS: gather, then per-expert GEMM  Yc[n_e,intermed]=Xg_e[n_e,dim]@W_e[intermed,dim]^T
    auto rC=[&](){
        dim3 gg(npair,(dim+255)/256); k_gather<<<gg,256>>>(dX,dgtok,dXg,npair,dim);
        for(int g=0;g<ngrp;++g){int off=grp_off[g],ne=grp_off[g+1]-off; if(ne<=0)continue;
            const half*Xge=dXg+(size_t)off*dim; const half*We=dW+(size_t)grp_exp[g]*intermed*dim; float*Yce=dYc+(size_t)off*intermed;
            CB(cublasGemmEx(hdl,CUBLAS_OP_T,CUBLAS_OP_N,intermed,ne,dim,&al,
                We,CUDA_R_16F,dim, Xge,CUDA_R_16F,dim, &be, Yce,CUDA_R_32F,intermed,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)); }
    };
    rA();rC();CK(cudaDeviceSynchronize());CK(cudaGetLastError());
    vector<float> Ya((size_t)npair*intermed),Yc((size_t)npair*intermed);
    CK(cudaMemcpy(Ya.data(),dYa,(size_t)npair*intermed*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Yc.data(),dYc,(size_t)npair*intermed*4,cudaMemcpyDeviceToHost));
    float mx=0,dn=0; for(int p=0;p<npair;++p){int gp=pair_gidx[p]; for(int r=0;r<intermed;++r){
        float a=Ya[(size_t)p*intermed+r],c=Yc[(size_t)gp*intermed+r]; mx=std::max(mx,fabsf(a-c)); dn=std::max(dn,fabsf(a)); }}
    auto tm=[&](auto fn){for(int i=0;i<5;++i)fn();CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);int N=50;
        cudaEventRecord(e0);for(int i=0;i<N;++i)fn();cudaEventRecord(e1);CK(cudaEventSynchronize(e1));
        float t=0;cudaEventElapsedTime(&t,e0,e1);return t/N;};
    float ta=tm(rA),tc=tm(rC);
    printf("MoE gate/up: P=%d dim=%d intermed=%d n_exp=%d top_k=%d npair=%d ngrp=%d\n",P,dim,intermed,n_exp,top_k,npair,ngrp);
    printf("  per-token GEMV (current): %.3f ms\n  grouped cuBLAS (ggml way): %.3f ms  speedup=%.2fx  max|err|=%.2e rel~%.1e\n",
        ta,tc,ta/tc,mx,mx/(dn+1e-9f));
    return 0;
}
