// test_swiglu_gpu.cu — Step 5e: SwiGLU activation op toward a GPU-resident
// Student. The MLP matmuls (W_gate/W_up/W_down) are the resident matmul; the new
// op is the elementwise gated activation h = SiLU(a) * b, SiLU(a)=a*sigmoid(a).
// Forward + backward CUDA kernels, grad-checked against a finite difference of
// E = 0.5*sum((h-t)^2) for BOTH gate pre-activation da and up-projection db.
//
//   s=sigmoid(a); silu=a*s; silu'=s*(1+a*(1-s))
//   da_i = dh_i * b_i * silu'(a_i);   db_i = dh_i * silu(a_i)
//
//   nvcc -O3 -arch=native test_swiglu_gpu.cu -o test_swiglu_gpu && ./test_swiglu_gpu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

static const int N=64;

__device__ __forceinline__ float sig(float a){ return 1.0f/(1.0f+expf(-a)); }

__global__ void k_swiglu_fwd(const float*a,const float*b,float*h,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  h[i]=(a[i]*sig(a[i]))*b[i];
}
__global__ void k_swiglu_bwd(const float*a,const float*b,const float*dh,
                             float*da,float*db,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float s=sig(a[i]); float silu=a[i]*s; float dsilu=s*(1.0f+a[i]*(1.0f-s));
  da[i]=dh[i]*b[i]*dsilu;
  db[i]=dh[i]*silu;
}

static float *dA,*dB,*dH,*dDa,*dDb,*dDh;
static float hA[N],hB[N],hT[N],hH[N];

static float fwd_loss(const float*a,const float*b){
  CK(cudaMemcpy(dA,a,sizeof(hA),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,b,sizeof(hB),cudaMemcpyHostToDevice));
  k_swiglu_fwd<<<(N+63)/64,64>>>(dA,dB,dH,N);
  CK(cudaMemcpy(hH,dH,sizeof(hH),cudaMemcpyDeviceToHost));
  float e=0; for(int i=0;i<N;i++){ float d=hH[i]-hT[i]; e+=d*d; } return 0.5f*e;
}
static void analytic(float*da,float*db){
  fwd_loss(hA,hB);
  float hDh[N]; for(int i=0;i<N;i++) hDh[i]=hH[i]-hT[i];
  CK(cudaMemcpy(dDh,hDh,sizeof(hDh),cudaMemcpyHostToDevice));
  k_swiglu_bwd<<<(N+63)/64,64>>>(dA,dB,dDh,dDa,dDb,N);
  CK(cudaMemcpy(da,dDa,sizeof(hA),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(db,dDb,sizeof(hB),cudaMemcpyDeviceToHost));
}

static long seed=11;
static float rnd(){ seed=(seed*1103515245+12345)&0x7fffffff; return (float)seed/2147483648.0f-0.5f; }

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s | SwiGLU grad-check N=%d\n",p.name,N);
  for(int i=0;i<N;i++){ hA[i]=rnd()*4.0f; hB[i]=rnd(); hT[i]=rnd(); }  // a in [-2,2]
  CK(cudaMalloc(&dA,sizeof(hA))); CK(cudaMalloc(&dB,sizeof(hB)));
  CK(cudaMalloc(&dH,sizeof(hH))); CK(cudaMalloc(&dDa,sizeof(hA)));
  CK(cudaMalloc(&dDb,sizeof(hB))); CK(cudaMalloc(&dDh,sizeof(hH)));

  float da[N], db[N]; analytic(da,db);
  const float eps=1e-3f; int bad=0; float maxrel=0;
  auto check=[&](const char*tag,float*vec,float ana,int idx){
    float w0=vec[idx];
    vec[idx]=w0+eps; float Lp=fwd_loss(hA,hB);
    vec[idx]=w0-eps; float Lm=fwd_loss(hA,hB);
    vec[idx]=w0;
    float fd=(Lp-Lm)/(2*eps), ad=fabsf(ana-fd);
    float rel=ad/(fabsf(ana)+fabsf(fd)+1e-9f);
    bool ok = rel<2e-2f || ad<5e-5f; if(!ok) bad++;
    if(ad>=5e-5f && rel>maxrel) maxrel=rel;   // only meaningful rel (skip near-zero)
    printf("  %s[%d] analytic=% .6e fd=% .6e rel=%.3e %s\n",tag,idx,ana,fd,rel,ok?"ok":"BAD");
  };
  printf(" da (gate pre-activation grad):\n");
  for(int t=0;t<4;t++){ int i=(t*13+2)%N; check("da",hA,da[i],i); }
  printf(" db (up-projection grad):\n");
  for(int t=0;t<4;t++){ int i=(t*17+5)%N; check("db",hB,db[i],i); }

  printf("RESULT: %s (SwiGLU fwd+bwd grad-checked; max rel=%.3e)\n",
         bad==0?"PASS":"FAIL", maxrel);
  return bad==0?0:1;
}
