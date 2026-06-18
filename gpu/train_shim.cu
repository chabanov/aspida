// train_shim.cu — C-ABI shared library exposing the GPU training kernels for
// dlopen from Ada (Train_GPU), mirroring the inference shim (libaspidagpu.so).
// Each entry takes HOST pointers and manages device memory internally
// (correctness-first; activations-resident optimization comes later).
//
//   build .so:   nvcc -O3 -arch=native --shared -Xcompiler -fPIC \
//                  gpu/train_shim.cu -o libaspidatrain.so
//   self-test:   nvcc -O3 -arch=native -DSHIM_TEST gpu/train_shim.cu -o shimtest && ./shimtest

#include <cuda_runtime.h>
#include <cstdlib>

// ---- kernels (validated in train_kernels.cu / train_kernels2.cu) ----------
__global__ void k_mm_fwd(const float*A,const float*B,float*C,int M,int K,int N){
  int c=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&c<N){float s=0;for(int k=0;k<K;k++)s+=A[r*K+k]*B[k*N+c];C[r*N+c]=s;}
}
__global__ void k_mm_dA(const float*dC,const float*B,float*dA,int M,int K,int N){
  int k=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&k<K){float s=0;for(int n=0;n<N;n++)s+=dC[r*N+n]*B[k*N+n];dA[r*K+k]=s;}
}
__global__ void k_mm_dB(const float*A,const float*dC,float*dB,int M,int K,int N){
  int n=blockIdx.x*blockDim.x+threadIdx.x, k=blockIdx.y*blockDim.y+threadIdx.y;
  if(k<K&&n<N){float s=0;for(int m=0;m<M;m++)s+=A[m*K+k]*dC[m*N+n];dB[k*N+n]=s;}
}
__global__ void k_softmax_bwd(const float*P,const float*dP,float*dS,int R,int N){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=R)return;
  float dot=0;for(int n=0;n<N;n++)dot+=P[r*N+n]*dP[r*N+n];
  for(int n=0;n<N;n++)dS[r*N+n]=P[r*N+n]*(dP[r*N+n]-dot);
}
__global__ void k_rmsnorm_bwd(const float*X,const float*G,const float*DY,float*DX,float*DG,int R,int D,float eps){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=R)return;
  float ms=0;for(int j=0;j<D;j++){float v=X[r*D+j];ms+=v*v;}
  float ri=1.f/sqrtf(ms/D+eps),ri3=ri*ri*ri,c=0;
  for(int j=0;j<D;j++)c+=(DY[r*D+j]*G[j])*X[r*D+j];
  for(int j=0;j<D;j++){float dn=DY[r*D+j]*G[j];
    DX[r*D+j]=ri*dn-(ri3/D)*X[r*D+j]*c;
    atomicAdd(&DG[j],DY[r*D+j]*(X[r*D+j]*ri));}
}

// ---- host-pointer C-ABI wrappers ------------------------------------------
static float* up(const float*h,size_t n){float*d;cudaMalloc(&d,n*4);if(h)cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice);else cudaMemset(d,0,n*4);return d;}
static void   down(float*h,float*d,size_t n){cudaMemcpy(h,d,n*4,cudaMemcpyDeviceToHost);cudaFree(d);}

extern "C" {

void aspida_mm_fwd(const float*A,const float*B,float*C,int M,int K,int N){
  float*a=up(A,(size_t)M*K),*b=up(B,(size_t)K*N),*c=up(0,(size_t)M*N);
  dim3 bl(16,16),g((N+15)/16,(M+15)/16); k_mm_fwd<<<g,bl>>>(a,b,c,M,K,N);
  cudaFree(a);cudaFree(b);down(C,c,(size_t)M*N);
}
void aspida_mm_dA(const float*dC,const float*B,float*dA,int M,int K,int N){
  float*dc=up(dC,(size_t)M*N),*b=up(B,(size_t)K*N),*da=up(0,(size_t)M*K);
  dim3 bl(16,16),g((K+15)/16,(M+15)/16); k_mm_dA<<<g,bl>>>(dc,b,da,M,K,N);
  cudaFree(dc);cudaFree(b);down(dA,da,(size_t)M*K);
}
void aspida_mm_dB(const float*A,const float*dC,float*dB,int M,int K,int N){
  float*a=up(A,(size_t)M*K),*dc=up(dC,(size_t)M*N),*db=up(0,(size_t)K*N);
  dim3 bl(16,16),g((N+15)/16,(K+15)/16); k_mm_dB<<<g,bl>>>(a,dc,db,M,K,N);
  cudaFree(a);cudaFree(dc);down(dB,db,(size_t)K*N);
}
void aspida_softmax_bwd(const float*P,const float*dP,float*dS,int R,int N){
  float*p=up(P,(size_t)R*N),*dp=up(dP,(size_t)R*N),*ds=up(0,(size_t)R*N);
  k_softmax_bwd<<<(R+127)/128,128>>>(p,dp,ds,R,N);
  cudaFree(p);cudaFree(dp);down(dS,ds,(size_t)R*N);
}
void aspida_rmsnorm_bwd(const float*X,const float*G,const float*DY,float*DX,float*DG,int R,int D,float eps){
  float*x=up(X,(size_t)R*D),*g=up(G,(size_t)D),*dy=up(DY,(size_t)R*D),
        *dx=up(0,(size_t)R*D),*dg=up(0,(size_t)D);
  k_rmsnorm_bwd<<<(R+127)/128,128>>>(x,g,dy,dx,dg,R,D,eps);
  cudaFree(x);cudaFree(g);cudaFree(dy);down(DX,dx,(size_t)R*D);down(DG,dg,(size_t)D);
}

} // extern C

#ifdef SHIM_TEST
#include <cstdio>
#include <cmath>
static double er(const float*a,const float*b,int n){double md=0,mb=0;for(int i=0;i<n;i++){double d=fabs((double)a[i]-b[i]);if(d>md)md=d;double bb=fabs((double)b[i]);if(bb>mb)mb=bb;}return md/(mb+1e-12);}
int main(){
  srand(3); int M=64,K=48,N=32; size_t sa=M*K,sb=K*N,sc=M*N;
  float*A=(float*)malloc(sa*4),*B=(float*)malloc(sb*4),*C=(float*)malloc(sc*4),*R=(float*)malloc(sc*4);
  for(size_t i=0;i<sa;i++)A[i]=rand()/(float)RAND_MAX-0.5f;
  for(size_t i=0;i<sb;i++)B[i]=rand()/(float)RAND_MAX-0.5f;
  aspida_mm_fwd(A,B,C,M,K,N);
  for(int r=0;r<M;r++)for(int c=0;c<N;c++){double s=0;for(int k=0;k<K;k++)s+=(double)A[r*K+k]*B[k*N+c];R[r*N+c]=(float)s;}
  printf("shim aspida_mm_fwd err=%.2e %s\n",er(C,R,sc),er(C,R,sc)<2e-3?"OK":"FAIL");
  return er(C,R,sc)<2e-3?0:1;
}
#endif
