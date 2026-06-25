// test_gemm_tiled.cu — Stream B perf: a shared-memory TILED FP32 GEMM to replace
// the naive k_mm in the resident Student. Validated for correctness against the
// naive kernel (max rel diff within FP32 reassociation tolerance) and timed for
// speedup. Honest scope: this is the tiled-GEMM lever; FP16/tensor-core (WMMA)
// is a further step (it changes numerics — not bit-parity — and needs FP16 I/O).
//
//   nvcc -O3 -arch=native test_gemm_tiled.cu -o test_gemm_tiled && ./test_gemm_tiled

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

#define TS 16
__global__ void k_mm(const float*A,const float*B,float*C,int M,int K,int N){
  int c=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&c<N){float s=0;for(int k=0;k<K;k++)s+=A[r*K+k]*B[k*N+c];C[r*N+c]=s;}
}
__global__ void k_mm_tiled(const float*A,const float*B,float*C,int M,int K,int N){
  __shared__ float As[TS][TS], Bs[TS][TS];
  int row=blockIdx.y*TS+threadIdx.y, col=blockIdx.x*TS+threadIdx.x;
  float acc=0;
  for(int t=0;t<(K+TS-1)/TS;t++){
    int ak=t*TS+threadIdx.x, bk=t*TS+threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row<M && ak<K) ? A[row*K+ak] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (bk<K && col<N) ? B[bk*N+col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for(int k=0;k<TS;k++) acc += As[threadIdx.y][k]*Bs[k][threadIdx.x];
    __syncthreads();
  }
  if(row<M && col<N) C[row*N+col]=acc;
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  const int M=1024,K=1024,N=1024;
  printf("device: %s | GEMM %dx%dx%d  naive vs tiled\n",p.name,M,K,N);
  float *hA=(float*)malloc((size_t)M*K*4),*hB=(float*)malloc((size_t)K*N*4);
  srand(1); for(int i=0;i<M*K;i++)hA[i]=rand()/(float)RAND_MAX-0.5f;
  for(int i=0;i<K*N;i++)hB[i]=rand()/(float)RAND_MAX-0.5f;
  float *dA,*dB,*dC1,*dC2; CK(cudaMalloc(&dA,(size_t)M*K*4));CK(cudaMalloc(&dB,(size_t)K*N*4));
  CK(cudaMalloc(&dC1,(size_t)M*N*4));CK(cudaMalloc(&dC2,(size_t)M*N*4));
  CK(cudaMemcpy(dA,hA,(size_t)M*K*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,hB,(size_t)K*N*4,cudaMemcpyHostToDevice));
  dim3 tb(TS,TS), g((N+TS-1)/TS,(M+TS-1)/TS);
  cudaEvent_t a,b; CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
  const int IT=30;
  // naive
  k_mm<<<g,tb>>>(dA,dB,dC1,M,K,N); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a)); for(int i=0;i<IT;i++) k_mm<<<g,tb>>>(dA,dB,dC1,M,K,N);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float t1; CK(cudaEventElapsedTime(&t1,a,b));
  // tiled
  k_mm_tiled<<<g,tb>>>(dA,dB,dC2,M,K,N); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a)); for(int i=0;i<IT;i++) k_mm_tiled<<<g,tb>>>(dA,dB,dC2,M,K,N);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float t2; CK(cudaEventElapsedTime(&t2,a,b));
  // correctness
  float *c1=(float*)malloc((size_t)M*N*4),*c2=(float*)malloc((size_t)M*N*4);
  CK(cudaMemcpy(c1,dC1,(size_t)M*N*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(c2,dC2,(size_t)M*N*4,cudaMemcpyDeviceToHost));
  float maxrel=0; for(int i=0;i<M*N;i++){ float ad=fabsf(c1[i]-c2[i]); float rel=ad/(fabsf(c1[i])+fabsf(c2[i])+1e-6f); if(rel>maxrel)maxrel=rel; }
  double fl=2.0*M*K*(double)N*IT;
  printf("  naive: %.2f ms  %.1f GFLOP/s\n", t1/IT, fl/(t1/1e3)/1e9);
  printf("  tiled: %.2f ms  %.1f GFLOP/s  (%.2fx)\n", t2/IT, fl/(t2/1e3)/1e9, t1/t2);
  printf("  max rel diff (tiled vs naive) = %.3e\n", maxrel);
  bool ok = maxrel<1e-3f && t2<t1;
  printf("RESULT: %s (tiled GEMM correct & faster)\n", ok?"PASS":"FAIL");
  return ok?0:1;
}
