// student_kernels.cuh — shared CUDA kernels for the resident Student
// (each individually grad-checked in steps 5a/5d–5h). Included by
// student_resident.cu (self-test) and the Ada-driven session shim (Stage C2).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// ---- tensor-core (WMMA) FP16 forward GEMM: C[M,N]=A[M,K]·B[K,N], FP32 accumulate.
// Caller guarantees M,K,N multiples of 16 (else use the FP32 path). 6.6x on L40S.
__global__ void k_f2h(const float* in, __half* out, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=__float2half(in[i]); }
__global__ void k_mm_wmma(const __half*A,const __half*B,float*C,int M,int K,int N){
  using namespace nvcuda;
  int warpM=(blockIdx.x*blockDim.x+threadIdx.x)/warpSize;
  int warpN= blockIdx.y*blockDim.y+threadIdx.y;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.0f);
  for(int k=0;k<K;k+=16){
    int aRow=warpM*16, bCol=warpN*16;
    if(aRow<M && bCol<N){
      wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
      wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b;
      wmma::load_matrix_sync(a, A+aRow*K+k, K);
      wmma::load_matrix_sync(b, B+k*N+bCol, N);
      wmma::mma_sync(acc,a,b,acc);
    }
  }
  int cRow=warpM*16, cCol=warpN*16;
  if(cRow<M && cCol<N) wmma::store_matrix_sync(C+cRow*N+cCol, acc, N, wmma::mem_row_major);
}

__global__ void k_emb_fwd(const float*E,const int*Id,float*O,int s,int d){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=s*d) return; int r=t/d,c=t%d; O[t]=E[Id[r]*d+c]; }
__global__ void k_emb_bwd(const float*dO,const int*Id,float*dE,int s,int d){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=s*d) return; int r=t/d,c=t%d; atomicAdd(&dE[Id[r]*d+c],dO[t]); }
__global__ void k_rms_fwd(const float*x,const float*g,float*y,float*ri,int n,int d,float eps){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n) return;
  float ms=0; for(int k=0;k<d;k++){float v=x[r*d+k];ms+=v*v;} ms/=d; float q=rsqrtf(ms+eps); ri[r]=q;
  for(int k=0;k<d;k++) y[r*d+k]=x[r*d+k]*q*g[k]; }
__global__ void k_rms_bwd(const float*x,const float*g,const float*dy,const float*ri,
                                 float*dx,float*dg,int n,int d){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n) return; float q=ri[r],Sd=0;
  for(int j=0;j<d;j++) Sd+=dy[r*d+j]*g[j]*x[r*d+j]; float c=q*q*q/d*Sd;
  for(int i=0;i<d;i++){ dx[r*d+i]=q*g[i]*dy[r*d+i]-c*x[r*d+i]; atomicAdd(&dg[i],dy[r*d+i]*x[r*d+i]*q);} }
__global__ void k_mm(const float*A,const float*B,float*C,int M,int K,int N){
  int c=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&c<N){float s=0;for(int k=0;k<K;k++)s+=A[r*K+k]*B[k*N+c];C[r*N+c]=s;} }
// shared-memory tiled GEMM (same 16x16 launch as k_mm); validated 1.3x faster, exact.
#define KTILE 16
__global__ void k_mm_tiled(const float*A,const float*B,float*C,int M,int K,int N){
  __shared__ float As[KTILE][KTILE], Bs[KTILE][KTILE];
  int row=blockIdx.y*KTILE+threadIdx.y, col=blockIdx.x*KTILE+threadIdx.x; float acc=0;
  for(int t=0;t<(K+KTILE-1)/KTILE;t++){
    int ak=t*KTILE+threadIdx.x, bk=t*KTILE+threadIdx.y;
    As[threadIdx.y][threadIdx.x]=(row<M&&ak<K)?A[row*K+ak]:0.0f;
    Bs[threadIdx.y][threadIdx.x]=(bk<K&&col<N)?B[bk*N+col]:0.0f;
    __syncthreads();
    #pragma unroll
    for(int k=0;k<KTILE;k++) acc+=As[threadIdx.y][k]*Bs[k][threadIdx.x];
    __syncthreads();
  }
  if(row<M&&col<N) C[row*N+col]=acc; }
__global__ void k_mm_ABt(const float*dC,const float*B,float*dA,int M,int K,int N){
  int k=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&k<K){float s=0;for(int n=0;n<N;n++)s+=dC[r*N+n]*B[k*N+n];dA[r*K+k]=s;} }
__global__ void k_mm_AtB(const float*A,const float*dC,float*dB,int M,int K,int N){
  int n=blockIdx.x*blockDim.x+threadIdx.x,k=blockIdx.y*blockDim.y+threadIdx.y;
  if(k<K&&n<N){float s=0;for(int m=0;m<M;m++)s+=A[m*K+k]*dC[m*N+n];dB[k*N+n]=s;} }
__global__ void k_add(const float*a,const float*b,float*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]+b[i]; }
__global__ void k_sgd(float*w,const float*g,int n,float lr){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) w[i]-=lr*g[i]; }
__global__ void k_scale(float*a,int n,float f){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]*=f; }
__global__ void k_adamw(float*W,const float*G,float*Mm,float*Vv,int n,
                        float lr,float b1,float b2,float eps,float wd,float bc1,float bc2){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float g=G[i];
  float m=b1*Mm[i]+(1.f-b1)*g, v=b2*Vv[i]+(1.f-b2)*g*g; Mm[i]=m; Vv[i]=v;
  W[i]-=lr*((m/bc1)/(sqrtf(v/bc2)+eps)+wd*W[i]); }
__global__ void k_rope_fwd(const float*x,float*y,int s,int d,float base){
  int hp=d/2,t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=s*hp)return; int p=t/hp,i=t%hp;
  float inv=powf(base,-2.0f*i/d),th=p*inv,c=cosf(th),sn=sinf(th); int o=p*d+2*i;
  float x0=x[o],x1=x[o+1]; y[o]=x0*c-x1*sn; y[o+1]=x0*sn+x1*c; }
__global__ void k_rope_bwd(const float*dy,float*dx,int s,int d,float base){
  int hp=d/2,t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=s*hp)return; int p=t/hp,i=t%hp;
  float inv=powf(base,-2.0f*i/d),th=p*inv,c=cosf(th),sn=sinf(th); int o=p*d+2*i;
  float d0=dy[o],d1=dy[o+1]; dx[o]=d0*c+d1*sn; dx[o+1]=-d0*sn+d1*c; }
__global__ void k_attn_fwd(const float*Q,const float*K,const float*Vv,float*P,float*O,int s,int d,float sc){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=s)return; float mx=-1e30f;
  for(int j=0;j<=i;j++){float v=0;for(int e=0;e<d;e++)v+=Q[i*d+e]*K[j*d+e]; v*=sc; P[i*s+j]=v; if(v>mx)mx=v;}
  float sum=0; for(int j=0;j<=i;j++){float ex=expf(P[i*s+j]-mx);P[i*s+j]=ex;sum+=ex;}
  for(int j=0;j<=i;j++)P[i*s+j]/=sum; for(int j=i+1;j<s;j++)P[i*s+j]=0;
  for(int e=0;e<d;e++){float o=0;for(int j=0;j<=i;j++)o+=P[i*s+j]*Vv[j*d+e];O[i*d+e]=o;} }
// caller must size dpj[]/dq[] for the compile-time S,D (passed via SMAXS/SMAXD)
template<int SMAXS,int SMAXD>
__global__ void k_attn_bwd(const float*Q,const float*K,const float*Vv,const float*P,const float*dO,
                           float*dQ,float*dK,float*dV,int s,int d,float sc){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=s)return; float dpj[SMAXS],ssum=0;
  for(int j=0;j<=i;j++){float dp=0;for(int e=0;e<d;e++)dp+=dO[i*d+e]*Vv[j*d+e]; dpj[j]=dp; ssum+=P[i*s+j]*dp;}
  float dq[SMAXD]; for(int e=0;e<d;e++)dq[e]=0;
  for(int j=0;j<=i;j++){ float ds=P[i*s+j]*(dpj[j]-ssum);
    for(int e=0;e<d;e++){ dq[e]+=sc*ds*K[j*d+e]; atomicAdd(&dK[j*d+e],sc*ds*Q[i*d+e]); atomicAdd(&dV[j*d+e],P[i*s+j]*dO[i*d+e]); } }
  for(int e=0;e<d;e++) dQ[i*d+e]=dq[e]; }
// ---- multi-head variants (H heads × dh; head hd uses column slice [hd*dh:(hd+1)*dh]) ----
__global__ void k_rope_mh_fwd(const float*x,float*y,int s,int h,int dh,float base){
  int hp=dh/2,t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=s*h*hp)return;
  int pr=t%hp,rest=t/hp,hd=rest%h,p=rest/h,D=h*dh,o=p*D+hd*dh+2*pr;
  float inv=powf(base,-2.0f*pr/dh),th=p*inv,c=cosf(th),sn=sinf(th);
  float x0=x[o],x1=x[o+1]; y[o]=x0*c-x1*sn; y[o+1]=x0*sn+x1*c; }
__global__ void k_rope_mh_bwd(const float*dy,float*dx,int s,int h,int dh,float base){
  int hp=dh/2,t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=s*h*hp)return;
  int pr=t%hp,rest=t/hp,hd=rest%h,p=rest/h,D=h*dh,o=p*D+hd*dh+2*pr;
  float inv=powf(base,-2.0f*pr/dh),th=p*inv,c=cosf(th),sn=sinf(th);
  float d0=dy[o],d1=dy[o+1]; dx[o]=d0*c+d1*sn; dx[o+1]=-d0*sn+d1*c; }
__global__ void k_mha_fwd(const float*Q,const float*K,const float*Vv,float*P,float*O,
                          int s,int h,int dh,float sc){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=h*s)return; int hd=t/s,i=t%s,D=h*dh,b=hd*dh;
  float mx=-1e30f;
  for(int j=0;j<=i;j++){float v=0;for(int e=0;e<dh;e++)v+=Q[i*D+b+e]*K[j*D+b+e]; v*=sc; P[(hd*s+i)*s+j]=v; if(v>mx)mx=v;}
  float sum=0; for(int j=0;j<=i;j++){float ex=expf(P[(hd*s+i)*s+j]-mx);P[(hd*s+i)*s+j]=ex;sum+=ex;}
  for(int j=0;j<=i;j++)P[(hd*s+i)*s+j]/=sum;
  for(int e=0;e<dh;e++){float o=0;for(int j=0;j<=i;j++)o+=P[(hd*s+i)*s+j]*Vv[j*D+b+e];O[i*D+b+e]=o;} }
// Runtime-sized (no per-thread length-S array): two passes recompute dp so S is
// a runtime parameter. dq is per-head (head_dim), bounded by MAX_DH.
static const int MAX_DH = 256;
__global__ void k_mha_bwd(const float*Q,const float*K,const float*Vv,const float*P,const float*dO,
                          float*dQ,float*dK,float*dV,int s,int h,int dh,float sc){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=h*s)return; int hd=t/s,i=t%s,D=h*dh,b=hd*dh;
  float ssum=0;
  for(int j=0;j<=i;j++){float dp=0;for(int e=0;e<dh;e++)dp+=dO[i*D+b+e]*Vv[j*D+b+e]; ssum+=P[(hd*s+i)*s+j]*dp;}
  float dq[MAX_DH]; for(int e=0;e<dh;e++)dq[e]=0;
  for(int j=0;j<=i;j++){ float dp=0;for(int e=0;e<dh;e++)dp+=dO[i*D+b+e]*Vv[j*D+b+e];
    float ds=P[(hd*s+i)*s+j]*(dp-ssum);
    for(int e=0;e<dh;e++){ dq[e]+=sc*ds*K[j*D+b+e]; atomicAdd(&dK[j*D+b+e],sc*ds*Q[i*D+b+e]); atomicAdd(&dV[j*D+b+e],P[(hd*s+i)*s+j]*dO[i*D+b+e]); } }
  for(int e=0;e<dh;e++) dQ[i*D+b+e]=dq[e]; }
__global__ void k_swiglu_fwd(const float*a,const float*b,float*h,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float s=1.0f/(1.0f+expf(-a[i])); h[i]=(a[i]*s)*b[i]; }
__global__ void k_swiglu_bwd(const float*a,const float*b,const float*dh,float*da,float*db,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float s=1.0f/(1.0f+expf(-a[i]));
  float silu=a[i]*s; da[i]=dh[i]*b[i]*(s*(1.0f+a[i]*(1.0f-s))); db[i]=dh[i]*silu; }
// soft-target (distillation) cross-entropy: target is a teacher distribution Q
// (per-row, sums to 1) instead of a hard class. loss = -sum_k Q*log softmax(Z);
// gradient dZ = softmax(Z) - Q. Same backward chain as hard CE.
__global__ void k_ce_soft(const float*Z,const float*Q,float*loss,float*dZ,int s,int v){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=s)return; float mx=-1e30f;
  for(int k=0;k<v;k++) if(Z[r*v+k]>mx)mx=Z[r*v+k];
  float sum=0; for(int k=0;k<v;k++) sum+=expf(Z[r*v+k]-mx); float lse=mx+logf(sum);
  float l=0; for(int k=0;k<v;k++){ float p=expf(Z[r*v+k]-lse); float q=Q[r*v+k];
    l += -q*(Z[r*v+k]-lse); dZ[r*v+k]=p-q; }
  loss[r]=l;
}
__global__ void k_ce(const float*Z,const int*Y,float*loss,float*dZ,int s,int v){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=s)return; float mx=-1e30f;
  for(int k=0;k<v;k++) if(Z[r*v+k]>mx)mx=Z[r*v+k]; float sum=0; for(int k=0;k<v;k++)sum+=expf(Z[r*v+k]-mx);
  float lse=mx+logf(sum); loss[r]=lse-Z[r*v+Y[r]];
  for(int k=0;k<v;k++) dZ[r*v+k]=expf(Z[r*v+k]-lse)-(k==Y[r]?1.0f:0.0f); }
