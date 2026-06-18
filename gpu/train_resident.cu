// train_resident.cu — a complete GPU-resident training loop: weights, AdamW
// moments and the dataset all live on the device across steps; only a scalar
// loss is copied back. Demonstrates the resident pattern (no per-step host
// round-trips) and throughput at a real layer size. Fits W to recover a
// target linear map T = X·W*  via MSE + AdamW — loss must collapse.
//   nvcc -O3 -arch=native train_resident.cu -o train_resident && ./train_resident

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void k_mm(const float*A,const float*B,float*C,int M,int K,int N){
  int c=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&c<N){float s=0;for(int k=0;k<K;k++)s+=A[r*K+k]*B[k*N+c];C[r*N+c]=s;}
}
// dW[K,N] = X[M,K]^T . dY[M,N]
__global__ void k_dW(const float*X,const float*dY,float*dW,int M,int K,int N){
  int n=blockIdx.x*blockDim.x+threadIdx.x, k=blockIdx.y*blockDim.y+threadIdx.y;
  if(k<K&&n<N){float s=0;for(int m=0;m<M;m++)s+=X[m*K+k]*dY[m*N+n];dW[k*N+n]=s;}
}
// dY = (Y - T); also atomic-accumulate squared error into loss[0]
__global__ void k_resid(const float*Y,const float*T,float*dY,float*loss,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  float d=Y[i]-T[i]; dY[i]=d; atomicAdd(loss,d*d);
}
__global__ void k_adam(float*W,const float*dW,float*m,float*v,int n,
                       float lr,float b1,float b2,float eps,float bc1,float bc2){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  float g=dW[i];
  m[i]=b1*m[i]+(1.f-b1)*g; v[i]=b2*v[i]+(1.f-b2)*g*g;
  W[i]-=lr*((m[i]/bc1)/(sqrtf(v[i]/bc2)+eps));
}

static void fill(float*p,size_t n){for(size_t i=0;i<n;i++)p[i]=(rand()/(float)RAND_MAX)*2-1;}

// resident training of W[K,N] to fit T=X·Wtrue, M rows. Returns final loss.
static double fit(int M,int K,int N,int steps,float lr,int verbose){
  size_t sX=(size_t)M*K, sW=(size_t)K*N, sY=(size_t)M*N;
  float *hX=(float*)malloc(sX*4),*hWt=(float*)malloc(sW*4);
  fill(hX,sX); fill(hWt,sW);
  // device-resident buffers (allocated ONCE, persist across all steps)
  float *X,*Wt,*T,*W,*Y,*dY,*dW,*mW,*vW,*dloss;
  cudaMalloc(&X,sX*4);cudaMalloc(&Wt,sW*4);cudaMalloc(&T,sY*4);
  cudaMalloc(&W,sW*4);cudaMalloc(&Y,sY*4);cudaMalloc(&dY,sY*4);cudaMalloc(&dW,sW*4);
  cudaMalloc(&mW,sW*4);cudaMalloc(&vW,sW*4);cudaMalloc(&dloss,4);
  cudaMemcpy(X,hX,sX*4,cudaMemcpyHostToDevice);
  cudaMemcpy(Wt,hWt,sW*4,cudaMemcpyHostToDevice);
  cudaMemset(W,0,sW*4);cudaMemset(mW,0,sW*4);cudaMemset(vW,0,sW*4);
  dim3 b2(16,16);
  // T = X·Wtrue (target), once
  k_mm<<<dim3((N+15)/16,(M+15)/16),b2>>>(X,Wt,T,M,K,N);
  // tiny random init for W
  { float*hW=(float*)malloc(sW*4); for(size_t i=0;i<sW;i++)hW[i]=(rand()/(float)RAND_MAX-0.5f)*0.02f;
    cudaMemcpy(W,hW,sW*4,cudaMemcpyHostToDevice); free(hW); }
  cudaDeviceSynchronize();

  cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1); cudaEventRecord(e0);
  float L0=0,Lf=0; float invM=1.0f/M;
  for(int s=1;s<=steps;s++){
    cudaMemset(dloss,0,4);
    k_mm<<<dim3((N+15)/16,(M+15)/16),b2>>>(X,W,Y,M,K,N);          // Y = X·W
    k_resid<<<(sY+255)/256,256>>>(Y,T,dY,dloss,sY);              // dY=Y-T, loss
    k_dW<<<dim3((N+15)/16,(K+15)/16),b2>>>(X,dY,dW,M,K,N);       // dW = X^T·dY
    float bc1=1-powf(0.9f,s), bc2=1-powf(0.999f,s);
    k_adam<<<(sW+255)/256,256>>>(W,dW,mW,vW,sW,lr*invM,0.9f,0.999f,1e-8f,bc1,bc2);
    if(s==1||s%(steps/5)==0){ float l; cudaMemcpy(&l,dloss,4,cudaMemcpyDeviceToHost);
      double mse=l/(double)sY; if(s==1)L0=mse; Lf=mse;
      if(verbose) printf("    step %4d  MSE=%.3e\n",s,mse); }
  }
  cudaEventRecord(e1);cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms,e0,e1);
  double gflop = 2.0*(2.0*(double)M*K*N) * steps / 1e9;   // fwd + dW matmuls
  printf("  [M=%d K=%d N=%d x%d steps]  %.1f ms  (%.0f GFLOP/s)  MSE %.2e -> %.2e\n",
         M,K,N,steps,ms,gflop/(ms/1000.0),L0,Lf);
  // final loss
  float lf; cudaMemcpy(&lf,dloss,4,cudaMemcpyDeviceToHost);
  double mse=lf/(double)sY;
  cudaFree(X);cudaFree(Wt);cudaFree(T);cudaFree(W);cudaFree(Y);cudaFree(dY);
  cudaFree(dW);cudaFree(mW);cudaFree(vW);cudaFree(dloss);free(hX);free(hWt);
  return mse;
}

int main(){
  srand(11);
  printf("=== Aspida GPU-resident AdamW training loop ===\n");
  printf(" small (correctness — must converge):\n");
  double s = fit(64,128,96,800,1.0f,1);
  printf(" real layer sizes (throughput, weights/data resident):\n");
  fit(256,1024,1024,200,1.0f,0);
  fit(512,2048,2048,100,1.0f,0);
  int ok = (s < 1e-4);
  printf(ok ? "\nRESULT: PASS (GPU-resident training converges: MSE %.2e)\n"
            : "\nRESULT: FAIL (MSE %.2e)\n", s);
  return ok?0:1;
}
