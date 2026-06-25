// test_rope_gpu.cu — Step 5f: RoPE op toward a GPU-resident Student. Rotary
// position embedding rotates each dim-pair (2i,2i+1) of a [S,D] tensor by
// theta = pos * base^(-2i/D). It has no learnable params; the backward is the
// rotation by -theta (the rotation is orthogonal, so R^T = R(-theta)). Forward +
// backward CUDA kernels, grad-checked vs a finite difference of E=0.5*sum((y-t)^2).
//
//   y0 = x0*c - x1*s ; y1 = x0*s + x1*c           (c=cos th, s=sin th)
//   dx0 = dy0*c + dy1*s ; dx1 = -dy0*s + dy1*c
//
//   nvcc -O3 -arch=native test_rope_gpu.cu -o test_rope_gpu && ./test_rope_gpu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

static const int S=4, D=16;            // S positions, head_dim D (even)
static const float BASE=10000.0f;

__global__ void k_rope_fwd(const float*x,float*y,int s,int d,float base){
  int hp=d/2, tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=s*hp) return;
  int p=tid/hp, i=tid%hp;
  float inv=powf(base,-2.0f*i/d), th=p*inv, c=cosf(th), sn=sinf(th);
  int o=p*d+2*i; float x0=x[o], x1=x[o+1];
  y[o]=x0*c-x1*sn; y[o+1]=x0*sn+x1*c;
}
__global__ void k_rope_bwd(const float*dy,float*dx,int s,int d,float base){
  int hp=d/2, tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=s*hp) return;
  int p=tid/hp, i=tid%hp;
  float inv=powf(base,-2.0f*i/d), th=p*inv, c=cosf(th), sn=sinf(th);
  int o=p*d+2*i; float d0=dy[o], d1=dy[o+1];
  dx[o]=d0*c+d1*sn; dx[o+1]=-d0*sn+d1*c;     // rotate by -theta
}

static float *dX,*dY,*dDx,*dDy;
static float hX[S*D],hT[S*D],hY[S*D];

static float fwd_loss(const float*x){
  CK(cudaMemcpy(dX,x,sizeof(hX),cudaMemcpyHostToDevice));
  k_rope_fwd<<<(S*(D/2)+63)/64,64>>>(dX,dY,S,D,BASE);
  CK(cudaMemcpy(hY,dY,sizeof(hY),cudaMemcpyDeviceToHost));
  float e=0; for(int i=0;i<S*D;i++){ float dd=hY[i]-hT[i]; e+=dd*dd; } return 0.5f*e;
}
static void analytic(float*dx){
  fwd_loss(hX);
  float hDy[S*D]; for(int i=0;i<S*D;i++) hDy[i]=hY[i]-hT[i];
  CK(cudaMemcpy(dDy,hDy,sizeof(hDy),cudaMemcpyHostToDevice));
  k_rope_bwd<<<(S*(D/2)+63)/64,64>>>(dDy,dDx,S,D,BASE);
  CK(cudaMemcpy(dx,dDx,sizeof(hX),cudaMemcpyDeviceToHost));
}

static long seed=29;
static float rnd(){ seed=(seed*1103515245+12345)&0x7fffffff; return (float)seed/2147483648.0f-0.5f; }

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s | RoPE grad-check S=%d D=%d\n",p.name,S,D);
  for(int i=0;i<S*D;i++){ hX[i]=rnd(); hT[i]=rnd(); }
  CK(cudaMalloc(&dX,sizeof(hX))); CK(cudaMalloc(&dY,sizeof(hY)));
  CK(cudaMalloc(&dDx,sizeof(hX))); CK(cudaMalloc(&dDy,sizeof(hY)));

  float dx[S*D]; analytic(dx);
  const float eps=1e-3f; int bad=0; float maxrel=0;
  printf(" dx (input grad):\n");
  for(int t=0;t<6;t++){
    int idx=(t*19+3)%(S*D); float w0=hX[idx];
    hX[idx]=w0+eps; float Lp=fwd_loss(hX);
    hX[idx]=w0-eps; float Lm=fwd_loss(hX);
    hX[idx]=w0;
    float fd=(Lp-Lm)/(2*eps), ad=fabsf(dx[idx]-fd);
    float rel=ad/(fabsf(dx[idx])+fabsf(fd)+1e-9f);
    bool ok = rel<2e-2f || ad<5e-5f; if(!ok) bad++;
    if(ad>=5e-5f && rel>maxrel) maxrel=rel;
    printf("  dx[%d] analytic=% .6e fd=% .6e rel=%.3e %s\n",idx,dx[idx],fd,rel,ok?"ok":"BAD");
  }
  printf("RESULT: %s (RoPE fwd+bwd grad-checked; max rel=%.3e)\n",
         bad==0?"PASS":"FAIL", maxrel);
  return bad==0?0:1;
}
