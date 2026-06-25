// train_resident_shim.cu — Step 5a: a RESIDENT training SESSION with a C ABI.
//
// Weights, AdamW moments, activations and the dataset all stay on the device
// across every step; only the scalar loss crosses the bus per step. This is the
// callable substrate the Ada `Student` will drive in Step 5b (via dlopen, like
// LLM_GPU), with NO per-op host round-trips (unlike the old train_shim.cu).
//
// C ABI:
//   void*  art_create(L,B,D,lr)        -- resident L-layer (DxD) linear stack
//   void   art_set_data(h, X[B*D], T[B*D])   -- upload once (resident)
//   float  art_step(h)                 -- fwd+bwd+AdamW on device; returns loss/elem
//   void   art_get_loss(h, out)        -- last loss/elem
//   void   art_free(h)
//
// Self-test fits a deep linear stack to T = X·W* (loss must collapse).
//   test: nvcc -O3 -arch=native train_resident_shim.cu -o art_test && ./art_test
//   lib : nvcc -O3 -arch=native -shared -Xcompiler -fPIC -DART_NO_MAIN \
//            train_resident_shim.cu -o libaspidatrain.so

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA error %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort();} }while(0)

__global__ void k_mm(const float*A,const float*B,float*C,int M,int K,int N){
  int c=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&c<N){ float s=0; for(int k=0;k<K;k++) s+=A[r*K+k]*B[k*N+c]; C[r*N+c]=s; }
}
__global__ void k_dA(const float*dC,const float*B,float*dA,int M,int K,int N){
  int k=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<M&&k<K){ float s=0; for(int n=0;n<N;n++) s+=dC[r*N+n]*B[k*N+n]; dA[r*K+k]=s; }
}
__global__ void k_dB(const float*A,const float*dC,float*dB,int M,int K,int N){
  int n=blockIdx.x*blockDim.x+threadIdx.x, k=blockIdx.y*blockDim.y+threadIdx.y;
  if(k<K&&n<N){ float s=0; for(int m=0;m<M;m++) s+=A[m*K+k]*dC[m*N+n]; dB[k*N+n]=s; }
}
__global__ void k_resid(const float*Y,const float*T,float*dY,float*loss,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float d=Y[i]-T[i]; dY[i]=d; atomicAdd(loss,d*d);
}
__global__ void k_adamw(float*W,const float*G,float*M,float*V,int n,
                        float lr,float b1,float b2,float eps,float wd,float bc1,float bc2){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float g=G[i];
  float m=b1*M[i]+(1.f-b1)*g, v=b2*V[i]+(1.f-b2)*g*g;
  M[i]=m; V[i]=v;
  W[i]-=lr*((m/bc1)/(sqrtf(v/bc2)+eps)+wd*W[i]);
}

struct Session {
  int L,B,D; float lr,b1,b2,eps,wd; long step; float last_loss;
  float **W,**M,**V,**A; float *dW,*dCur,*dPrev,*dLoss,*dT;
};

static float* dz(size_t n){ float*p; CK(cudaMalloc(&p,n*4)); CK(cudaMemset(p,0,n*4)); return p; }

extern "C" void* art_create(int L,int B,int D,float lr){
  Session* s=new Session();
  s->L=L; s->B=B; s->D=D; s->lr=lr; s->b1=0.9f; s->b2=0.999f; s->eps=1e-8f;
  s->wd=0.f; s->step=0; s->last_loss=0;
  s->W=new float*[L]; s->M=new float*[L]; s->V=new float*[L]; s->A=new float*[L+1];
  for(int i=0;i<=L;i++) s->A[i]=dz((size_t)B*D);
  float* tmp=(float*)malloc((size_t)D*D*4);
  for(int l=0;l<L;l++){
    s->W[l]=dz((size_t)D*D); s->M[l]=dz((size_t)D*D); s->V[l]=dz((size_t)D*D);
    for(size_t i=0;i<(size_t)D*D;i++) tmp[i]=(rand()/(float)RAND_MAX-0.5f)*0.05f;
    CK(cudaMemcpy(s->W[l],tmp,(size_t)D*D*4,cudaMemcpyHostToDevice));
  }
  free(tmp);
  s->dW=dz((size_t)D*D); s->dCur=dz((size_t)B*D); s->dPrev=dz((size_t)B*D);
  s->dLoss=dz(1); s->dT=dz((size_t)B*D);
  return s;
}

extern "C" void art_set_data(void* h,const float* X,const float* T){
  Session* s=(Session*)h;
  CK(cudaMemcpy(s->A[0],X,(size_t)s->B*s->D*4,cudaMemcpyHostToDevice)); // resident input
  CK(cudaMemcpy(s->dT, T,(size_t)s->B*s->D*4,cudaMemcpyHostToDevice)); // resident target
}

extern "C" float art_step(void* h){
  Session* s=(Session*)h; int B=s->B,D=s->D,L=s->L;
  dim3 tb(16,16);
  auto g=[&](int X,int Y){ return dim3((X+15)/16,(Y+15)/16); };
  for(int l=1;l<=L;l++) k_mm<<<g(D,B),tb>>>(s->A[l-1],s->W[l-1],s->A[l],B,D,D);
  CK(cudaMemset(s->dLoss,0,4));
  k_resid<<<(B*D+255)/256,256>>>(s->A[L],s->dT,s->dCur,s->dLoss,B*D);
  s->step++;
  float bc1=1.f-powf(s->b1,(float)s->step), bc2=1.f-powf(s->b2,(float)s->step);
  for(int l=L;l>=1;l--){
    k_dB<<<g(D,D),tb>>>(s->A[l-1],s->dCur,s->dW,B,D,D);
    if(l>1) k_dA<<<g(D,B),tb>>>(s->dCur,s->W[l-1],s->dPrev,B,D,D);
    k_adamw<<<(D*D+255)/256,256>>>(s->W[l-1],s->dW,s->M[l-1],s->V[l-1],D*D,
                                   s->lr,s->b1,s->b2,s->eps,s->wd,bc1,bc2);
    float*t=s->dCur; s->dCur=s->dPrev; s->dPrev=t;
  }
  float hl; CK(cudaMemcpy(&hl,s->dLoss,4,cudaMemcpyDeviceToHost));
  s->last_loss=hl/(B*D);
  return s->last_loss;
}

extern "C" void art_get_loss(void* h,float* out){ *out=((Session*)h)->last_loss; }

// ---- grad-check support (Step 5c): forward-only loss, analytic gradient
// (NO AdamW update), and single-weight read/perturb. The reduction is
// E = 0.5*sum((Y-T)^2); backward propagates dE/dY = (Y-T), so art_grad_at
// returns dE/dW[layer][idx] — directly comparable to a finite difference of E.
extern "C" float art_loss_only(void* h){
  Session* s=(Session*)h; int B=s->B,D=s->D,L=s->L; dim3 tb(16,16);
  auto g=[&](int X,int Y){ return dim3((X+15)/16,(Y+15)/16); };
  for(int l=1;l<=L;l++) k_mm<<<g(D,B),tb>>>(s->A[l-1],s->W[l-1],s->A[l],B,D,D);
  CK(cudaMemset(s->dLoss,0,4));
  k_resid<<<(B*D+255)/256,256>>>(s->A[L],s->dT,s->dCur,s->dLoss,B*D);
  float hl; CK(cudaMemcpy(&hl,s->dLoss,4,cudaMemcpyDeviceToHost));
  return 0.5f*hl;
}
extern "C" float art_grad_at(void* h,int layer,int idx){
  Session* s=(Session*)h; int B=s->B,D=s->D,L=s->L; dim3 tb(16,16);
  auto g=[&](int X,int Y){ return dim3((X+15)/16,(Y+15)/16); };
  for(int l=1;l<=L;l++) k_mm<<<g(D,B),tb>>>(s->A[l-1],s->W[l-1],s->A[l],B,D,D);
  CK(cudaMemset(s->dLoss,0,4));
  k_resid<<<(B*D+255)/256,256>>>(s->A[L],s->dT,s->dCur,s->dLoss,B*D);
  float gout=0;
  for(int l=L;l>=1;l--){
    k_dB<<<g(D,D),tb>>>(s->A[l-1],s->dCur,s->dW,B,D,D);
    if(l-1==layer) CK(cudaMemcpy(&gout,s->dW+idx,4,cudaMemcpyDeviceToHost));
    if(l>1) k_dA<<<g(D,B),tb>>>(s->dCur,s->W[l-1],s->dPrev,B,D,D);
    float*t=s->dCur; s->dCur=s->dPrev; s->dPrev=t;
  }
  return gout;   // analytic dE/dW[layer][idx], no weight update
}
extern "C" float art_w_get(void* h,int layer,int idx){
  float v; CK(cudaMemcpy(&v,((Session*)h)->W[layer]+idx,4,cudaMemcpyDeviceToHost)); return v;
}
extern "C" void art_w_set(void* h,int layer,int idx,float v){
  CK(cudaMemcpy(((Session*)h)->W[layer]+idx,&v,4,cudaMemcpyHostToDevice));
}

extern "C" void art_free(void* h){
  Session* s=(Session*)h;
  for(int i=0;i<=s->L;i++) cudaFree(s->A[i]);
  for(int l=0;l<s->L;l++){ cudaFree(s->W[l]); cudaFree(s->M[l]); cudaFree(s->V[l]); }
  cudaFree(s->dW); cudaFree(s->dCur); cudaFree(s->dPrev); cudaFree(s->dLoss); cudaFree(s->dT);
  delete[] s->W; delete[] s->M; delete[] s->V; delete[] s->A; delete s;
}

#ifndef ART_NO_MAIN
int main(){
  const int L=4,B=256,D=512,STEPS=3000;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s  | resident session L=%d B=%d D=%d\n",p.name,L,B,D);
  srand(7);
  float *X=(float*)malloc((size_t)B*D*4), *Wt=(float*)malloc((size_t)D*D*4),
        *T=(float*)malloc((size_t)B*D*4);
  for(size_t i=0;i<(size_t)B*D;i++) X[i]=rand()/(float)RAND_MAX-0.5f;
  for(size_t i=0;i<(size_t)D*D;i++) Wt[i]=(rand()/(float)RAND_MAX-0.5f)*0.1f;
  for(int r=0;r<B;r++) for(int c=0;c<D;c++){ float s=0;
    for(int k=0;k<D;k++) s+=X[r*D+k]*Wt[k*D+c]; T[r*D+c]=s; }            // T = X·W*
  void* h=art_create(L,B,D,2e-3f);
  art_set_data(h,X,T);                                                   // upload ONCE
  cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  float loss=0;
  for(int s=1;s<=STEPS;s++){ loss=art_step(h);
    if(s==1||s%500==0||s==STEPS) printf("  step %5d  loss/elem=%.6e\n",s,loss); }
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  float ms; CK(cudaEventElapsedTime(&ms,t0,t1));
  double fl=(double)(3*L)*2.0*B*(double)D*D*STEPS;
  printf("throughput: %.0f steps/s  %.2f TFLOP/s (resident session)\n",
         STEPS/(ms/1e3), fl/(ms/1e3)/1e12);
  printf("RESULT: %s\n", loss<1e-3 ? "PASS (resident session trains; loss collapsed)"
                                    : "FAIL (loss did not collapse)");
  art_free(h);
  return 0;
}
#endif
