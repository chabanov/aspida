// test_gemm_wmma.cu — P1: tensor-core (WMMA) FP16 GEMM. The production perf lever:
// FP16 inputs, FP32 accumulate on the tensor cores. Validated against the FP32
// reference within FP16 tolerance (NOT bit-exact — FP16 input rounding) and timed
// vs the naive/tiled FP32 path. This is the kernel the resident Student adopts
// (FP16 weights/activations + FP32 master) for real-scale training.
//
//   nvcc -O3 -arch=native test_gemm_wmma.cu -o test_gemm_wmma && ./test_gemm_wmma

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

__global__ void k_f2h(const float* in, half* out, long n){
  long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=__float2half(in[i]);
}
__global__ void k_mm_fp32(const float*A,const float*B,float*C,int M,int K,int N){
  int c=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&c<N){ float s=0; for(int k=0;k<K;k++) s+=A[r*K+k]*B[k*N+c]; C[r*N+c]=s; }
}

static const int WM=16, WN=16, WK=16;
__global__ void k_wmma(const half*A,const half*B,float*C,int M,int K,int N){
  int warpM=(blockIdx.x*blockDim.x+threadIdx.x)/warpSize;
  int warpN= blockIdx.y*blockDim.y+threadIdx.y;
  wmma::fragment<wmma::accumulator,WM,WN,WK,float> acc;
  wmma::fill_fragment(acc,0.0f);
  for(int k=0;k<K;k+=WK){
    int aRow=warpM*WM, aCol=k, bRow=k, bCol=warpN*WN;
    if(aRow<M && bCol<N && aCol<K && bRow<K){
      wmma::fragment<wmma::matrix_a,WM,WN,WK,half,wmma::row_major> a;
      wmma::fragment<wmma::matrix_b,WM,WN,WK,half,wmma::row_major> b;
      wmma::load_matrix_sync(a, A+aRow*K+aCol, K);
      wmma::load_matrix_sync(b, B+bRow*N+bCol, N);
      wmma::mma_sync(acc,a,b,acc);
    }
  }
  int cRow=warpM*WM, cCol=warpN*WN;
  if(cRow<M && cCol<N) wmma::store_matrix_sync(C+cRow*N+cCol, acc, N, wmma::mem_row_major);
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  const int M=1024,K=1024,N=1024;
  printf("device: %s (sm_%d%d) | WMMA FP16 GEMM %dx%dx%d\n",p.name,p.major,p.minor,M,K,N);
  float *hA=(float*)malloc((size_t)M*K*4),*hB=(float*)malloc((size_t)K*N*4);
  srand(1); for(int i=0;i<M*K;i++)hA[i]=(rand()/(float)RAND_MAX-0.5f);
  for(int i=0;i<K*N;i++)hB[i]=(rand()/(float)RAND_MAX-0.5f);
  float *dAf,*dBf,*dCref,*dCw; half *dAh,*dBh;
  CK(cudaMalloc(&dAf,(size_t)M*K*4));CK(cudaMalloc(&dBf,(size_t)K*N*4));
  CK(cudaMalloc(&dCref,(size_t)M*N*4));CK(cudaMalloc(&dCw,(size_t)M*N*4));
  CK(cudaMalloc(&dAh,(size_t)M*K*2));CK(cudaMalloc(&dBh,(size_t)K*N*2));
  CK(cudaMemcpy(dAf,hA,(size_t)M*K*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dBf,hB,(size_t)K*N*4,cudaMemcpyHostToDevice));
  k_f2h<<<(M*K+255)/256,256>>>(dAf,dAh,(long)M*K);
  k_f2h<<<(K*N+255)/256,256>>>(dBf,dBh,(long)K*N);

  dim3 tb32(16,16), g32((N+15)/16,(M+15)/16);
  dim3 tbw(128,4);
  dim3 gw( (M + (WM*tbw.x/32) - 1) / (WM*tbw.x/32), (N + (WN*tbw.y) - 1) / (WN*tbw.y) );
  cudaEvent_t a,b; CK(cudaEventCreate(&a));CK(cudaEventCreate(&b)); const int IT=50;

  k_mm_fp32<<<g32,tb32>>>(dAf,dBf,dCref,M,K,N); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a)); for(int i=0;i<IT;i++) k_mm_fp32<<<g32,tb32>>>(dAf,dBf,dCref,M,K,N);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float t1; CK(cudaEventElapsedTime(&t1,a,b));

  k_wmma<<<gw,tbw>>>(dAh,dBh,dCw,M,K,N); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a)); for(int i=0;i<IT;i++) k_wmma<<<gw,tbw>>>(dAh,dBh,dCw,M,K,N);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float t2; CK(cudaEventElapsedTime(&t2,a,b));

  float *cr=(float*)malloc((size_t)M*N*4),*cw=(float*)malloc((size_t)M*N*4);
  CK(cudaMemcpy(cr,dCref,(size_t)M*N*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(cw,dCw,(size_t)M*N*4,cudaMemcpyDeviceToHost));
  //  Frobenius relative error — the right GEMM accuracy metric (element-wise rel
  //  is meaningless on near-zero entries of C).
  double num=0, den=0; for(int i=0;i<M*N;i++){ double d=(double)cr[i]-cw[i]; num+=d*d; den+=(double)cr[i]*cr[i]; }
  double frob=sqrt(num/den);
  double fl=2.0*M*K*(double)N*IT;
  printf("  FP32 naive : %.2f ms  %.1f GFLOP/s\n", t1/IT, fl/(t1/1e3)/1e9);
  printf("  WMMA FP16  : %.2f ms  %.1f GFLOP/s  (%.1fx)\n", t2/IT, fl/(t2/1e3)/1e9, t1/t2);
  printf("  Frobenius rel err vs FP32 = %.3e (FP16 input rounding)\n", frob);
  bool ok = frob<1e-2 && t2<t1;
  printf("RESULT: %s (tensor-core GEMM correct within FP16 tol & faster)\n", ok?"PASS":"FAIL");
  return ok?0:1;
}
