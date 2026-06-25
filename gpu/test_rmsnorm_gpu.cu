// test_rmsnorm_gpu.cu — Step 5d: the first transformer op grafted toward a
// GPU-resident Student. RMSNorm forward + backward as CUDA kernels, grad-checked
// against a finite difference of E = 0.5*sum((y-t)^2) for BOTH the input dx and
// the gain dg. Same discipline as 5c (combined rel/abs tolerance).
//
//   y_i = g_i * x_i * r,   r = 1/sqrt(mean_k x_k^2 + eps)
//   dg_i = sum_rows dy_i * x_i * r
//   dx_i = r*g_i*dy_i - (r^3 * x_i / D) * sum_j (dy_j * g_j * x_j)
//
//   nvcc -O3 -arch=native test_rmsnorm_gpu.cu -o test_rmsnorm_gpu && ./test_rmsnorm_gpu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

static const int B=4, D=16;
static const float EPS=1e-5f;

__global__ void k_rms_fwd(const float*x,const float*g,float*y,float*rinv,int b,int d,float eps){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=b) return;
  float ms=0; for(int k=0;k<d;k++){ float v=x[r*d+k]; ms+=v*v; } ms/=d;
  float ri=rsqrtf(ms+eps); rinv[r]=ri;
  for(int k=0;k<d;k++) y[r*d+k]=x[r*d+k]*ri*g[k];
}
__global__ void k_rms_bwd(const float*x,const float*g,const float*dy,const float*rinv,
                          float*dx,float*dg,int b,int d){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=b) return;
  float ri=rinv[r], S=0;
  for(int j=0;j<d;j++) S+=dy[r*d+j]*g[j]*x[r*d+j];
  float c=ri*ri*ri/d*S;
  for(int i=0;i<d;i++){
    dx[r*d+i]=ri*g[i]*dy[r*d+i]-c*x[r*d+i];
    atomicAdd(&dg[i], dy[r*d+i]*x[r*d+i]*ri);
  }
}

static float *dX,*dG,*dY,*dRi,*dDx,*dDg,*dDy;
static float hX[B*D],hG[D],hT[B*D],hY[B*D];

static float fwd_loss(const float*x,const float*g){
  CK(cudaMemcpy(dX,x,sizeof(hX),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dG,g,sizeof(hG),cudaMemcpyHostToDevice));
  k_rms_fwd<<<(B+31)/32,32>>>(dX,dG,dY,dRi,B,D,EPS);
  CK(cudaMemcpy(hY,dY,sizeof(hY),cudaMemcpyDeviceToHost));
  float e=0; for(int i=0;i<B*D;i++){ float d=hY[i]-hT[i]; e+=d*d; } return 0.5f*e;
}
static void analytic(float*dx,float*dg){
  fwd_loss(hX,hG);                              // fills hY, dRi
  float hDy[B*D]; for(int i=0;i<B*D;i++) hDy[i]=hY[i]-hT[i];
  CK(cudaMemcpy(dDy,hDy,sizeof(hDy),cudaMemcpyHostToDevice));
  CK(cudaMemset(dDg,0,sizeof(hG)));
  k_rms_bwd<<<(B+31)/32,32>>>(dX,dG,dDy,dRi,dDx,dDg,B,D);
  CK(cudaMemcpy(dx,dDx,sizeof(hX),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(dg,dDg,sizeof(hG),cudaMemcpyDeviceToHost));
}

static long seed=7;
static float rnd(){ seed=(seed*1103515245+12345)&0x7fffffff; return (float)seed/2147483648.0f-0.5f; }

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s | RMSNorm grad-check B=%d D=%d\n",p.name,B,D);
  for(int i=0;i<B*D;i++) hX[i]=rnd();
  for(int i=0;i<D;i++)   hG[i]=1.0f+rnd()*0.1f;     // gain ~1
  for(int i=0;i<B*D;i++) hT[i]=rnd();
  CK(cudaMalloc(&dX,sizeof(hX))); CK(cudaMalloc(&dG,sizeof(hG)));
  CK(cudaMalloc(&dY,sizeof(hX))); CK(cudaMalloc(&dRi,B*sizeof(float)));
  CK(cudaMalloc(&dDx,sizeof(hX))); CK(cudaMalloc(&dDg,sizeof(hG)));
  CK(cudaMalloc(&dDy,sizeof(hX)));

  float dx[B*D], dg[D]; analytic(dx,dg);
  const float eps=1e-3f; int bad=0; float maxrel=0;
  auto check=[&](const char*tag,float*vec,float ana,int idx){
    float w0=vec[idx];
    vec[idx]=w0+eps; float Lp=fwd_loss(hX,hG);
    vec[idx]=w0-eps; float Lm=fwd_loss(hX,hG);
    vec[idx]=w0;
    float fd=(Lp-Lm)/(2*eps), ad=fabsf(ana-fd);
    float rel=ad/(fabsf(ana)+fabsf(fd)+1e-9f);
    bool ok = rel<2e-2f || ad<5e-5f; if(!ok) bad++;
    if(rel>maxrel) maxrel=rel;
    printf("  %s[%d] analytic=% .6e fd=% .6e rel=%.3e %s\n",tag,idx,ana,fd,rel,ok?"ok":"BAD");
  };
  printf(" dx (input grad):\n");
  for(int t=0;t<4;t++){ int i=(t*23+3)%(B*D); check("dx",hX,dx[i],i); }
  printf(" dg (gain grad):\n");
  for(int t=0;t<4;t++){ int i=(t*5+1)%D; check("dg",hG,dg[i],i); }

  printf("RESULT: %s (RMSNorm fwd+bwd grad-checked; max rel=%.3e)\n",
         bad==0?"PASS":"FAIL", maxrel);
  return bad==0?0:1;
}
