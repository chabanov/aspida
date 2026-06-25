// test_attention_mh_gpu.cu — Stream B production: MULTI-HEAD causal attention
// (generalises the single-head 5g op). Q/K/V are [S, H*dh]; head h works on the
// column slice [h*dh : (h+1)*dh], softmax per head, scale = 1/sqrt(dh). Forward +
// backward CUDA kernels, grad-checked for dQ/dK/dV vs a finite difference of
// E=0.5*sum((out-t)^2). This is the attention the resident Student upgrades to.
//
//   nvcc -O3 -arch=native test_attention_mh_gpu.cu -o test_attention_mh_gpu && ./test_attention_mh_gpu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

static const int S=4, H=2, DH=8, D=H*DH;       // 2 heads x 8 = dim 16
static const float SCALE=0.35355339f;          // 1/sqrt(8)

__global__ void k_mha_fwd(const float*Q,const float*K,const float*Vv,float*P,float*O,
                          int s,int h,int dh,float sc){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=h*s) return; int hd=t/s, i=t%s;
  int D=h*dh, b=hd*dh; float mx=-1e30f;
  for(int j=0;j<=i;j++){ float v=0; for(int e=0;e<dh;e++) v+=Q[i*D+b+e]*K[j*D+b+e];
    v*=sc; P[(hd*s+i)*s+j]=v; if(v>mx)mx=v; }
  float sum=0; for(int j=0;j<=i;j++){ float ex=expf(P[(hd*s+i)*s+j]-mx); P[(hd*s+i)*s+j]=ex; sum+=ex; }
  for(int j=0;j<=i;j++) P[(hd*s+i)*s+j]/=sum;
  for(int e=0;e<dh;e++){ float o=0; for(int j=0;j<=i;j++) o+=P[(hd*s+i)*s+j]*Vv[j*D+b+e]; O[i*D+b+e]=o; }
}
__global__ void k_mha_bwd(const float*Q,const float*K,const float*Vv,const float*P,const float*dO,
                          float*dQ,float*dK,float*dV,int s,int h,int dh,float sc){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=h*s) return; int hd=t/s, i=t%s;
  int D=h*dh, b=hd*dh; float dpj[S], ssum=0;
  for(int j=0;j<=i;j++){ float dp=0; for(int e=0;e<dh;e++) dp+=dO[i*D+b+e]*Vv[j*D+b+e];
    dpj[j]=dp; ssum+=P[(hd*s+i)*s+j]*dp; }
  float dq[DH]; for(int e=0;e<dh;e++) dq[e]=0;
  for(int j=0;j<=i;j++){ float ds=P[(hd*s+i)*s+j]*(dpj[j]-ssum);
    for(int e=0;e<dh;e++){ dq[e]+=sc*ds*K[j*D+b+e]; atomicAdd(&dK[j*D+b+e],sc*ds*Q[i*D+b+e]); atomicAdd(&dV[j*D+b+e],P[(hd*s+i)*s+j]*dO[i*D+b+e]); } }
  for(int e=0;e<dh;e++) dQ[i*D+b+e]=dq[e];
}

static float *dQ_,*dK_,*dV_,*dP,*dOut,*ddQ,*ddK,*ddV,*ddOut;
static float hQ[S*D],hK[S*D],hV[S*D],hT[S*D],hO[S*D];

static double fwd_loss(){
  CK(cudaMemcpy(dQ_,hQ,sizeof(hQ),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK_,hK,sizeof(hK),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV_,hV,sizeof(hV),cudaMemcpyHostToDevice));
  k_mha_fwd<<<(H*S+31)/32,32>>>(dQ_,dK_,dV_,dP,dOut,S,H,DH,SCALE);
  CK(cudaMemcpy(hO,dOut,sizeof(hO),cudaMemcpyDeviceToHost));
  double e=0; for(int i=0;i<S*D;i++){ double dd=(double)hO[i]-hT[i]; e+=dd*dd; } return 0.5*e;
}
static void analytic(float*gq,float*gk,float*gv){
  fwd_loss();
  float hdO[S*D]; for(int i=0;i<S*D;i++) hdO[i]=hO[i]-hT[i];
  CK(cudaMemcpy(ddOut,hdO,sizeof(hdO),cudaMemcpyHostToDevice));
  CK(cudaMemset(ddQ,0,sizeof(hQ)));CK(cudaMemset(ddK,0,sizeof(hK)));CK(cudaMemset(ddV,0,sizeof(hV)));
  k_mha_bwd<<<(H*S+31)/32,32>>>(dQ_,dK_,dV_,dP,ddOut,ddQ,ddK,ddV,S,H,DH,SCALE);
  CK(cudaMemcpy(gq,ddQ,sizeof(hQ),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gk,ddK,sizeof(hK),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gv,ddV,sizeof(hV),cudaMemcpyDeviceToHost));
}
static long seed=404;
static float rnd(){ seed=(seed*1103515245+12345)&0x7fffffff; return (float)seed/2147483648.0f-0.5f; }

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s | multi-head attention grad-check S=%d H=%d dh=%d (D=%d)\n",p.name,S,H,DH,D);
  for(int i=0;i<S*D;i++){ hQ[i]=rnd(); hK[i]=rnd(); hV[i]=rnd(); hT[i]=rnd(); }
  CK(cudaMalloc(&dQ_,sizeof(hQ)));CK(cudaMalloc(&dK_,sizeof(hK)));CK(cudaMalloc(&dV_,sizeof(hV)));
  CK(cudaMalloc(&dP,H*S*S*4));CK(cudaMalloc(&dOut,sizeof(hO)));
  CK(cudaMalloc(&ddQ,sizeof(hQ)));CK(cudaMalloc(&ddK,sizeof(hK)));CK(cudaMalloc(&ddV,sizeof(hV)));CK(cudaMalloc(&ddOut,sizeof(hO)));

  float gq[S*D],gk[S*D],gv[S*D]; analytic(gq,gk,gv);
  const float eps=5e-3f; int bad=0; float maxrel=0;
  auto check=[&](const char*tag,float*vec,float*ana,int idx){
    float w0=vec[idx]; vec[idx]=w0+eps; double Lp=fwd_loss(); vec[idx]=w0-eps; double Lm=fwd_loss(); vec[idx]=w0;
    float fd=(float)((Lp-Lm)/(2*eps)),ad=fabsf(ana[idx]-fd),rel=ad/(fabsf(ana[idx])+fabsf(fd)+1e-9f);
    bool ok=rel<2e-2f||ad<2e-4f; if(!ok)bad++; if(rel>maxrel)maxrel=rel;
    printf("  %s[%d](head %d) analytic=% .6e fd=% .6e rel=%.3e %s\n",tag,idx,(idx/DH)%H,ana[idx],fd,rel,ok?"ok":"BAD"); };
  printf(" dQ/dK/dV across both heads:\n");
  check("dQ",hQ,gq,2); check("dQ",hQ,gq,1*DH+3);     // head0, head1
  check("dK",hK,gk,5); check("dK",hK,gk,1*DH+6);
  check("dV",hV,gv,0); check("dV",hV,gv,1*DH+1);

  printf("RESULT: %s (multi-head attention fwd+bwd grad-checked; max rel=%.3e)\n",
         bad==0?"PASS":"FAIL",maxrel);
  return bad==0?0:1;
}
