// test_celoss_soft.cu — P3: soft-target (distillation) cross-entropy, grad-checked.
// Target is a teacher distribution Q (per row, sums to 1); loss = -sum Q*log p,
// gradient dZ = softmax(Z) - Q. Validated vs a finite difference of the loss.
//
//   nvcc -O3 -arch=native test_celoss_soft.cu -o test_celoss_soft && ./test_celoss_soft

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

static const int S=4, V=8;
__global__ void k_ce_soft(const float*Z,const float*Q,float*loss,float*dZ,int s,int v){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=s)return; float mx=-1e30f;
  for(int k=0;k<v;k++) if(Z[r*v+k]>mx)mx=Z[r*v+k];
  float sum=0; for(int k=0;k<v;k++) sum+=expf(Z[r*v+k]-mx); float lse=mx+logf(sum);
  float l=0; for(int k=0;k<v;k++){ float p=expf(Z[r*v+k]-lse); float q=Q[r*v+k];
    l += -q*(Z[r*v+k]-lse); dZ[r*v+k]=p-q; }
  loss[r]=l;
}
static float *dZ,*dQ,*dL,*ddZ; static float hZ[S*V],hQ[S*V];
static double loss_of(){
  CK(cudaMemcpy(dZ,hZ,sizeof(hZ),cudaMemcpyHostToDevice));
  k_ce_soft<<<(S+31)/32,32>>>(dZ,dQ,dL,ddZ,S,V);
  float hl[S]; CK(cudaMemcpy(hl,dL,sizeof(hl),cudaMemcpyDeviceToHost));
  double e=0; for(int i=0;i<S;i++)e+=hl[i]; return e;
}
static long sd=7; static float rnd(){ sd=(sd*1103515245+12345)&0x7fffffff; return (float)sd/2147483648.0f; }
int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s | soft-target CE grad-check S=%d V=%d\n",p.name,S,V);
  for(int i=0;i<S*V;i++) hZ[i]=rnd()-0.5f;
  for(int r=0;r<S;r++){ float s=0; for(int k=0;k<V;k++){ hQ[r*V+k]=rnd()+0.05f; s+=hQ[r*V+k]; }
    for(int k=0;k<V;k++) hQ[r*V+k]/=s; }                      // teacher dist per row
  CK(cudaMalloc(&dZ,sizeof(hZ)));CK(cudaMalloc(&dQ,sizeof(hQ)));CK(cudaMalloc(&dL,S*4));CK(cudaMalloc(&ddZ,sizeof(hZ)));
  CK(cudaMemcpy(dQ,hQ,sizeof(hQ),cudaMemcpyHostToDevice));
  loss_of(); float gz[S*V]; CK(cudaMemcpy(gz,ddZ,sizeof(gz),cudaMemcpyDeviceToHost));
  const float eps=1e-3f; int bad=0; float maxrel=0;
  for(int t=0;t<6;t++){ int i=(t*11+2)%(S*V); float w0=hZ[i];
    hZ[i]=w0+eps; double Lp=loss_of(); hZ[i]=w0-eps; double Lm=loss_of(); hZ[i]=w0;
    float fd=(float)((Lp-Lm)/(2*eps)), ad=fabsf(gz[i]-fd), rel=ad/(fabsf(gz[i])+fabsf(fd)+1e-9f);
    bool ok=rel<2e-2f||ad<2e-4f; if(!ok)bad++; if(rel>maxrel)maxrel=rel;
    printf("  dZ[%d] analytic=% .6e fd=% .6e rel=%.3e %s\n",i,gz[i],fd,rel,ok?"ok":"BAD"); }
  printf("RESULT: %s (soft-target CE grad-checked; max rel=%.3e)\n", bad==0?"PASS":"FAIL", maxrel);
  return bad==0?0:1;
}
