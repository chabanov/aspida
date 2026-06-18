// train_kernels.cu — GPU kernels for the from-scratch training engine.
//
// The dominant training op is the dense linear layer; here are its three
// matmuls in FP32 (row-major), which mirror Train.Linear_NB_Forward/Backward:
//   forward:  Y[M,N] = X[M,K] . W[K,N]        (X=A, W=B, Y=C)
//   d-input:  dX[M,K] = dY[M,N] . W[K,N]^T     (dA = dC . B^T)
//   d-weight: dW[K,N] = X[M,K]^T . dY[M,N]     (dB = A^T . dC)
//
// This file is self-validating: built standalone it checks each kernel against
// a CPU reference on random data across several shapes.
//   nvcc -O3 -arch=native train_kernels.cu -o train_kernels && ./train_kernels

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---- kernels (one thread per output element; correctness-first) ----------
__global__ void mm_fwd(const float* A, const float* B, float* C,
                       int M, int K, int N) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;   // 0..N
  int r = blockIdx.y * blockDim.y + threadIdx.y;   // 0..M
  if (r < M && c < N) {
    float s = 0.f;
    for (int k = 0; k < K; ++k) s += A[r * K + k] * B[k * N + c];
    C[r * N + c] = s;
  }
}

// dA[M,K] = dC[M,N] . B[K,N]^T   ->  dA[r,k] = sum_n dC[r,n] * B[k,n]
__global__ void mm_dA(const float* dC, const float* B, float* dA,
                      int M, int K, int N) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;   // 0..K
  int r = blockIdx.y * blockDim.y + threadIdx.y;   // 0..M
  if (r < M && k < K) {
    float s = 0.f;
    for (int n = 0; n < N; ++n) s += dC[r * N + n] * B[k * N + n];
    dA[r * K + k] = s;
  }
}

// dB[K,N] = A[M,K]^T . dC[M,N]   ->  dB[k,n] = sum_m A[m,k] * dC[m,n]
__global__ void mm_dB(const float* A, const float* dC, float* dB,
                      int M, int K, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;   // 0..N
  int k = blockIdx.y * blockDim.y + threadIdx.y;   // 0..K
  if (k < K && n < N) {
    float s = 0.f;
    for (int m = 0; m < M; ++m) s += A[m * K + k] * dC[m * N + n];
    dB[k * N + n] = s;
  }
}

// ---- CPU references -------------------------------------------------------
static void cpu_fwd(const float*A,const float*B,float*C,int M,int K,int N){
  for(int r=0;r<M;r++)for(int c=0;c<N;c++){double s=0;for(int k=0;k<K;k++)s+=(double)A[r*K+k]*B[k*N+c];C[r*N+c]=(float)s;}
}
static void cpu_dA(const float*dC,const float*B,float*dA,int M,int K,int N){
  for(int r=0;r<M;r++)for(int k=0;k<K;k++){double s=0;for(int n=0;n<N;n++)s+=(double)dC[r*N+n]*B[k*N+n];dA[r*K+k]=(float)s;}
}
static void cpu_dB(const float*A,const float*dC,float*dB,int M,int K,int N){
  for(int k=0;k<K;k++)for(int n=0;n<N;n++){double s=0;for(int m=0;m<M;m++)s+=(double)A[m*K+k]*dC[m*N+n];dB[k*N+n]=(float)s;}
}

// scale-normalized error: max|gpu-cpu| / max|cpu| (robust for FP32 matmul,
// where per-element relative error is meaningless near zero).
static double max_rel(const float*a,const float*b,int n){
  double md=0, mb=0;
  for(int i=0;i<n;i++){ double d=fabs((double)a[i]-b[i]); if(d>md)md=d;
                        double bb=fabs((double)b[i]); if(bb>mb)mb=bb; }
  return md/(mb+1e-12);
}
static void fill(float*p,int n){ for(int i=0;i<n;i++) p[i]=(float)((rand()/(double)RAND_MAX)*2.0-1.0); }

static dim3 grid2(int xs,int ys,dim3 b){ return dim3((xs+b.x-1)/b.x,(ys+b.y-1)/b.y); }

static int check_shape(int M,int K,int N){
  size_t szA=M*K, szB=K*N, szC=M*N;
  float *hA=(float*)malloc(szA*4),*hB=(float*)malloc(szB*4),*hdC=(float*)malloc(szC*4);
  float *hC=(float*)malloc(szC*4),*hdA=(float*)malloc(szA*4),*hdB=(float*)malloc(szB*4);
  float *rC=(float*)malloc(szC*4),*rdA=(float*)malloc(szA*4),*rdB=(float*)malloc(szB*4);
  fill(hA,szA); fill(hB,szB); fill(hdC,szC);
  float *dA_,*dB_,*ddC,*dC_,*ddA,*ddB;
  cudaMalloc(&dA_,szA*4);cudaMalloc(&dB_,szB*4);cudaMalloc(&ddC,szC*4);
  cudaMalloc(&dC_,szC*4);cudaMalloc(&ddA,szA*4);cudaMalloc(&ddB,szB*4);
  cudaMemcpy(dA_,hA,szA*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dB_,hB,szB*4,cudaMemcpyHostToDevice);
  cudaMemcpy(ddC,hdC,szC*4,cudaMemcpyHostToDevice);
  dim3 b(16,16);
  mm_fwd<<<grid2(N,M,b),b>>>(dA_,dB_,dC_,M,K,N);
  mm_dA <<<grid2(K,M,b),b>>>(ddC,dB_,ddA,M,K,N);
  mm_dB <<<grid2(N,K,b),b>>>(dA_,ddC,ddB,M,K,N);
  cudaDeviceSynchronize();
  cudaMemcpy(hC,dC_,szC*4,cudaMemcpyDeviceToHost);
  cudaMemcpy(hdA,ddA,szA*4,cudaMemcpyDeviceToHost);
  cudaMemcpy(hdB,ddB,szB*4,cudaMemcpyDeviceToHost);
  cpu_fwd(hA,hB,rC,M,K,N); cpu_dA(hdC,hB,rdA,M,K,N); cpu_dB(hA,hdC,rdB,M,K,N);
  double e1=max_rel(hC,rC,szC), e2=max_rel(hdA,rdA,szA), e3=max_rel(hdB,rdB,szB);
  int ok = (e1<2e-3 && e2<2e-3 && e3<2e-3);
  printf("  [%3dx%3dx%3d] fwd=%.2e dA=%.2e dB=%.2e  %s\n",M,K,N,e1,e2,e3, ok?"OK":"FAIL");
  cudaFree(dA_);cudaFree(dB_);cudaFree(ddC);cudaFree(dC_);cudaFree(ddA);cudaFree(ddB);
  free(hA);free(hB);free(hdC);free(hC);free(hdA);free(hdB);free(rC);free(rdA);free(rdB);
  return ok;
}

int main(){
  srand(1234);
  printf("=== Aspida GPU training kernels: matmul fwd/dA/dB vs CPU ===\n");
  int all=1;
  all &= check_shape(64,48,32);
  all &= check_shape(128,256,64);
  all &= check_shape(32,32,128);
  all &= check_shape(17,33,49);   // non-multiples of 16
  printf(all ? "\nRESULT: PASS (GPU == CPU)\n" : "\nRESULT: FAIL\n");
  return all?0:1;
}
