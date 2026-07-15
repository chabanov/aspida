// Validate the MoE parity lever: aspida prefill runs experts as per-(token,expert,row)
// warp-scalar GEMV (k_moe_gu_p) — expert weights re-read per token, no tensor cores.
// Ollama groups tokens by expert -> batched tensor-core GEMM. This microbench compares
// the gate/up projection both ways (fp16) for correctness + speed at MoE dims.
//   nvcc -O3 -arch=native moe_group.cu -o mg && ./mg
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s\n",cudaGetErrorString(e));exit(1);} }while(0)
using std::vector; namespace w = nvcuda::wmma;

// Y[pair, r] = dot(X[token_of_pair], W[expert_of_pair][r]);  W row-major [E][intermed][dim]
// ---- (A) current structure: one warp per (pair, row) scalar dot over dim ----
__global__ void k_pertoken(const half*__restrict__ X,const half*__restrict__ W,
    const int*__restrict__ ptok,const int*__restrict__ pexp,float*__restrict__ Y,
    int npair,int dim,int intermed){
    int wid=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
    if(wid>=npair*intermed)return;
    int p=wid/intermed, r=wid%intermed; int tok=ptok[p], e=pexp[p];
    const half*x=X+(size_t)tok*dim; const half*wr=W+((size_t)e*intermed+r)*dim;
    float acc=0.f; for(int i=lane;i<dim;i+=32)acc+=__half2float(x[i])*__half2float(wr[i]);
    for(int o=16;o>0;o>>=1)acc+=__shfl_xor_sync(0xffffffffu,acc,o);
    if(lane==0)Y[(size_t)p*intermed+r]=acc;
}

// ---- (B) grouped tensor-core: per expert, batched GEMM over its tokens ----
// For expert e with n_e tokens (gathered rows Xg[n_e][dim]), Y_e[n_e][intermed] =
// Xg @ W[e]^T. wmma 16x16x16: M=token tile, N=intermed tile, K=dim. One warp per
// (token-tile, intermed-tile) of one expert.
__global__ void k_grouped(const half*__restrict__ X,const half*__restrict__ W,
    const int*__restrict__ grp_tok,const int*__restrict__ grp_off,const int*__restrict__ grp_exp,
    const int*__restrict__ tile_grp,float*__restrict__ Ypair,const int*__restrict__ tile_pairbase,
    int dim,int intermed){
    int g=tile_grp[blockIdx.z];                    // which expert-group this tile block serves
    int e=grp_exp[g]; int off=grp_off[g], n_e=grp_off[g+1]-off;
    int mt=blockIdx.x, nt=blockIdx.y;              // token-tile, intermed-tile (16 each)
    int m0=mt*16, n0=nt*16; if(m0>=n_e)return;
    int lane=threadIdx.x&31;
    __shared__ half As[16*16], Bs[16*16];
    w::fragment<w::accumulator,16,16,16,float> cf; w::fill_fragment(cf,0.f);
    for(int k0=0;k0<dim;k0+=16){
        for(int i=lane;i<16*16;i+=32){int mr=i/16,kk=i%16;
            int tok=(m0+mr<n_e)?grp_tok[off+m0+mr]:0;
            As[i]=(m0+mr<n_e)?X[(size_t)tok*dim+k0+kk]:__float2half(0.f);
            int rr=n0+(i/16); // reuse i for B: B[k][n] = W[e][n0+col][k0+row]
        }
        for(int i=lane;i<16*16;i+=32){int kk=i/16,nn=i%16; // Bs[k][n]
            Bs[i]=(n0+nn<intermed)?W[((size_t)e*intermed+(n0+nn))*dim+k0+kk]:__float2half(0.f);}
        __syncwarp();
        w::fragment<w::matrix_a,16,16,16,half,w::row_major> af;
        w::fragment<w::matrix_b,16,16,16,half,w::row_major> bf;
        w::load_matrix_sync(af,As,16); w::load_matrix_sync(bf,Bs,16); w::mma_sync(cf,af,bf,cf);
        __syncwarp();
    }
    __shared__ float Cs[16*16]; w::store_matrix_sync(Cs,cf,16,w::mem_row_major);
    int pbase=tile_pairbase[g];
    for(int i=lane;i<16*16;i+=32){int mr=i/16,nn=i%16; if(m0+mr<n_e&&n0+nn<intermed)
        Ypair[(size_t)(pbase+m0+mr)*intermed+n0+nn]=Cs[i];}
}
static half* duph(const vector<half>&h){half*p;CK(cudaMalloc(&p,h.size()*2));CK(cudaMemcpy(p,h.data(),h.size()*2,cudaMemcpyHostToDevice));return p;}
static int* dupi(const vector<int>&h){int*p;CK(cudaMalloc(&p,h.size()*4));CK(cudaMemcpy(p,h.data(),h.size()*4,cudaMemcpyHostToDevice));return p;}
int main(){
    int P=256,dim=2048,intermed=768,n_exp=64,top_k=8;
    std::mt19937 rng(5);std::normal_distribution<float> nd(0,1);std::uniform_int_distribution<int> ue(0,n_exp-1);
    vector<half> X((size_t)P*dim),W((size_t)n_exp*intermed*dim);
    for(auto&e:X)e=__float2half(nd(rng)*0.1f); for(auto&e:W)e=__float2half(nd(rng)*0.05f);
    // routing: each token -> top_k distinct experts. pairs in TOKEN order (per-token layout).
    vector<int> ptok,pexp; vector<vector<int>> exp_tokens(n_exp);
    for(int t=0;t<P;++t){vector<int> es; while((int)es.size()<top_k){int e=ue(rng);if(std::find(es.begin(),es.end(),e)==es.end())es.push_back(e);}
        for(int e:es){ptok.push_back(t);pexp.push_back(e);}}
    int npair=ptok.size();
    // grouped layout: pairs sorted by expert. grp_off, grp_tok, grp_exp, pairbase.
    vector<int> grp_tok,grp_off,grp_exp,tile_pairbase; grp_off.push_back(0);
    // build per-expert token lists preserving a mapping pair->grouped index for compare
    vector<int> pair_grouped_index(npair);
    // first: for each (token,expert) pair index, we need its grouped position.
    // group pairs by expert:
    vector<vector<int>> byexp(n_exp); for(int p=0;p<npair;++p)byexp[pexp[p]].push_back(p);
    int ng=0; vector<int> tile_grp;
    for(int e=0;e<n_exp;++e){ if(byexp[e].empty())continue;
        int base=grp_tok.size(); tile_pairbase.push_back(base); grp_exp.push_back(e);
        for(int p:byexp[e]){grp_tok.push_back(ptok[p]); pair_grouped_index[p]=base+(grp_tok.size()-1-base);}
        grp_off.push_back(grp_tok.size());
        int nt_tiles=(byexp[e].size()+15)/16;
        for(int mt=0;mt<nt_tiles;++mt)tile_grp.push_back(ng);
        ng++;
    }
    int ntiles_intermed=(intermed+15)/16;
    // device
    half*dX=duph(X),*dW=duph(W); int*dptok=dupi(ptok),*dpexp=dupi(pexp);
    int*dgtok=dupi(grp_tok),*dgoff=dupi(grp_off),*dgexp=dupi(grp_exp),*dtileg=dupi(tile_grp),*dpbase=dupi(tile_pairbase);
    float*dYa,*dYb; CK(cudaMalloc(&dYa,(size_t)npair*intermed*4));CK(cudaMalloc(&dYb,(size_t)npair*intermed*4));
    // (A) per-token
    int totalw=npair*intermed; auto rA=[&](){k_pertoken<<<(totalw*32+255)/256,256>>>(dX,dW,dptok,dpexp,dYa,npair,dim,intermed);};
    // (B) grouped: grid (max_token_tiles, intermed_tiles, num tile-groups)
    int max_mt=0; for(int e=0;e<n_exp;++e)if(!byexp[e].empty())max_mt=std::max(max_mt,(int)((byexp[e].size()+15)/16));
    dim3 gB(max_mt,ntiles_intermed,(unsigned)tile_grp.size());
    auto rB=[&](){k_grouped<<<gB,32>>>(dX,dW,dgtok,dgoff,dgexp,dtileg,dYb,dpbase,dim,intermed);};
    rA();rB();CK(cudaDeviceSynchronize());CK(cudaGetLastError());
    // compare: Yb is in grouped-pair order; map pair p -> grouped index
    vector<float> Ya((size_t)npair*intermed),Yb((size_t)npair*intermed);
    CK(cudaMemcpy(Ya.data(),dYa,(size_t)npair*intermed*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Yb.data(),dYb,(size_t)npair*intermed*4,cudaMemcpyDeviceToHost));
    float mx=0,dn=0; for(int p=0;p<npair;++p){int gp=pair_grouped_index[p];
        for(int r=0;r<intermed;++r){float a=Ya[(size_t)p*intermed+r],b=Yb[(size_t)gp*intermed+r];mx=std::max(mx,fabsf(a-b));dn=std::max(dn,fabsf(a));}}
    auto tm=[&](auto fn){for(int i=0;i<5;++i)fn();CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);int N=50;
        cudaEventRecord(e0);for(int i=0;i<N;++i)fn();cudaEventRecord(e1);CK(cudaEventSynchronize(e1));
        float t=0;cudaEventElapsedTime(&t,e0,e1);return t/N;};
    float ta=tm(rA),tb=tm(rB);
    printf("MoE gate/up proj: P=%d dim=%d intermed=%d n_exp=%d top_k=%d npair=%d\n",P,dim,intermed,n_exp,top_k,npair);
    printf("  per-token GEMV (current): %.3f ms\n  grouped wmma GEMM       : %.3f ms   speedup=%.2fx   max|err|=%.2e rel~%.1e\n",
        ta,tb,ta/tb,mx,mx/(dn+1e-9f));
    return 0;
}
