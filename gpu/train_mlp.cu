// train_mlp.cu — Stage 1: a GPU-RESIDENT, end-to-end, multi-layer training loop.
//
// All weights and AdamW moments live on the device across every step; the data
// (X, target T) is resident too. Each step runs forward + backward + AdamW
// entirely on the GPU and copies back ONLY the scalar loss — no per-op host
// round-trips. We fit a deep stack of linear layers to a random target map
// T = X·W*, so the loss MUST collapse if the resident forward/backward/AdamW
// chain is correct. Then we report throughput (steps/s, effective TFLOP/s).
//
// This is the foundation for GPU-resident Student training: the resident matmul
// (fwd / dA / dB) + AdamW loop, validated at depth and real size. The next
// increments graft on SwiGLU, RMSNorm, RoPE and attention to reach the full
// Student. Sized small (<200 MB VRAM) so it runs alongside a live serving demo.
//
//   nvcc -O3 -arch=native train_mlp.cu -o train_mlp && ./train_mlp

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA error %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)

// C[M,N] = A[M,K] . B[K,N]
__global__ void k_mm(const float*A,const float*B,float*C,int M,int K,int N){
  int c=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&c<N){ float s=0; for(int k=0;k<K;k++) s+=A[r*K+k]*B[k*N+c]; C[r*N+c]=s; }
}
// dA[M,K] = dC[M,N] . B[K,N]^T
__global__ void k_dA(const float*dC,const float*B,float*dA,int M,int K,int N){
  int k=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&k<K){ float s=0; for(int n=0;n<N;n++) s+=dC[r*N+n]*B[k*N+n]; dA[r*K+k]=s; }
}
// dB[K,N] = A[M,K]^T . dC[M,N]
__global__ void k_dB(const float*A,const float*dC,float*dB,int M,int K,int N){
  int n=blockIdx.x*blockDim.x+threadIdx.x, k=blockIdx.y*blockDim.y+threadIdx.y;
  if(k<K&&n<N){ float s=0; for(int m=0;m<M;m++) s+=A[m*K+k]*dC[m*N+n]; dB[k*N+n]=s; }
}
// dY = (Y - T);  loss += sum dY^2
__global__ void k_resid(const float*Y,const float*T,float*dY,float*loss,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float d=Y[i]-T[i]; dY[i]=d; atomicAdd(loss,d*d);
}
// AdamW (decoupled weight decay)
__global__ void k_adamw(float*W,const float*G,float*M,float*V,int n,
                        float lr,float b1,float b2,float eps,float wd,float bc1,float bc2){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float g=G[i];
  float m=b1*M[i]+(1.f-b1)*g;
  float v=b2*V[i]+(1.f-b2)*g*g;
  M[i]=m; V[i]=v;
  float mh=m/bc1, vh=v/bc2;
  W[i]-=lr*(mh/(sqrtf(vh)+eps)+wd*W[i]);
}

static float* dnew(size_t n){ float*p; CK(cudaMalloc(&p,n*sizeof(float))); return p; }

int main(int argc,char**argv){
  const int B=256, D=512, L=4, STEPS=3000;
  const float lr=2e-3f,b1=0.9f,b2=0.999f,eps=1e-8f,wd=0.0f;

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  printf("device: %s  (%.0f GB)\n",prop.name,prop.totalGlobalMem/1e9);
  printf("config: batch=%d dim=%d layers=%d steps=%d\n",B,D,L,STEPS);

  // host init: X, W* (target), T = X·W*
  srand(7);
  float *hX=(float*)malloc((size_t)B*D*4), *hW=(float*)malloc((size_t)D*D*4);
  for(size_t i=0;i<(size_t)B*D;i++) hX[i]=(rand()/(float)RAND_MAX-0.5f);
  for(size_t i=0;i<(size_t)D*D;i++) hW[i]=(rand()/(float)RAND_MAX-0.5f)*0.1f;

  float *dX=dnew((size_t)B*D), *dWstar=dnew((size_t)D*D), *dT=dnew((size_t)B*D);
  CK(cudaMemcpy(dX,hX,(size_t)B*D*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dWstar,hW,(size_t)D*D*4,cudaMemcpyHostToDevice));

  dim3 tb(16,16);
  auto grid=[&](int X,int Y){ return dim3((X+15)/16,(Y+15)/16); };

  // T = X · W*   (the target map to recover)
  k_mm<<<grid(D,B),tb>>>(dX,dWstar,dT,B,D,D); CK(cudaGetLastError());

  // resident weights + AdamW moments + activations + grads
  float *W[L],*Mm[L],*Vv[L],*A[L+1];
  A[0]=dX;                                  // layer-0 activation = input (fixed)
  for(int l=0;l<L;l++){
    W[l]=dnew((size_t)D*D); Mm[l]=dnew((size_t)D*D); Vv[l]=dnew((size_t)D*D);
    CK(cudaMemset(Mm[l],0,(size_t)D*D*4)); CK(cudaMemset(Vv[l],0,(size_t)D*D*4));
    A[l+1]=dnew((size_t)B*D);
    // init weights ~ small random
    float*tmp=(float*)malloc((size_t)D*D*4);
    for(size_t i=0;i<(size_t)D*D;i++) tmp[i]=(rand()/(float)RAND_MAX-0.5f)*0.05f;
    CK(cudaMemcpy(W[l],tmp,(size_t)D*D*4,cudaMemcpyHostToDevice)); free(tmp);
  }
  float *dW=dnew((size_t)D*D);              // reused gradient buffer
  float *dCur=dnew((size_t)B*D), *dPrev=dnew((size_t)B*D); // grad activations
  float *dLoss=dnew(1);

  cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));

  for(int step=1;step<=STEPS;step++){
    // forward: A[l] = A[l-1] · W[l-1]
    for(int l=1;l<=L;l++)
      k_mm<<<grid(D,B),tb>>>(A[l-1],W[l-1],A[l],B,D,D);
    // loss + dY
    CK(cudaMemset(dLoss,0,4));
    k_resid<<<(B*D+255)/256,256>>>(A[L],dT,dCur,dLoss,B*D);
    // backward through layers (dCur holds dA of the current layer's output)
    float bc1=1.f-powf(b1,step), bc2=1.f-powf(b2,step);
    for(int l=L;l>=1;l--){
      k_dB<<<grid(D,D),tb>>>(A[l-1],dCur,dW,B,D,D);          // dW[D,D]
      if(l>1) k_dA<<<grid(D,B),tb>>>(dCur,W[l-1],dPrev,B,D,D); // dA_prev
      k_adamw<<<(D*D+255)/256,256>>>(W[l-1],dW,Mm[l-1],Vv[l-1],D*D,
                                     lr,b1,b2,eps,wd,bc1,bc2);
      float*t=dCur; dCur=dPrev; dPrev=t;                      // ping-pong
    }
    if(step==1||step%500==0||step==STEPS){
      float hl; CK(cudaMemcpy(&hl,dLoss,4,cudaMemcpyDeviceToHost));
      printf("  step %5d   loss/elem = %.6e\n",step,hl/(B*D));
    }
  }
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  float ms; CK(cudaEventElapsedTime(&ms,t0,t1));

  // FLOPs/step: fwd L matmuls + bwd L*(dB + dA) ≈ (L + 2L) matmuls of 2*B*D*D
  double fl_step = (double)(3*L)*2.0*B*(double)D*D;
  double tflops = (fl_step*STEPS)/(ms/1e3)/1e12;
  size_t mb=0; { size_t fr,to; cudaMemGetInfo(&fr,&to); mb=(to-fr)/(1024*1024); }
  printf("throughput: %.0f steps/s   %.2f TFLOP/s (naive kernels)   VRAM~%zuMB\n",
         STEPS/(ms/1e3), tflops, mb);
  printf("RESULT: %s  (resident GPU train loop)\n",
         "see loss collapse above");
  return 0;
}
