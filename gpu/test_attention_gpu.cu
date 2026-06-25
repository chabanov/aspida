// test_attention_gpu.cu — Step 5g: causal scaled-dot-product attention toward a
// GPU-resident Student — the hardest op (backward flows through the softmax
// jacobian). Single head, S positions, head_dim D, causal mask.
//
//   score[i][j] = (Q[i]·K[j])/sqrt(D)   (j<=i)
//   p[i] = softmax_j score[i]
//   out[i] = sum_j p[i][j] V[j]
// backward (dout):
//   dp[i][j]   = dout[i]·V[j]
//   dscore[i][j] = p[i][j]*(dp[i][j] - sum_k p[i][k] dp[i][k])   (softmax jac)
//   dQ[i] = scale * sum_j dscore[i][j] K[j]
//   dK[j] = scale * sum_i dscore[i][j] Q[i]
//   dV[j] = sum_i p[i][j] dout[i]
// grad-checked vs a finite difference of E=0.5*sum((out-t)^2) for dQ, dK, dV.
//
//   nvcc -O3 -arch=native test_attention_gpu.cu -o test_attention_gpu && ./test_attention_gpu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

static const int S=4, D=8;
static const float SCALE=0.35355339f;       // 1/sqrt(8)

__global__ void k_attn_fwd(const float*Q,const float*K,const float*V,
                           float*P,float*Out,int s,int d,float scale){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=s) return;
  float mx=-1e30f;
  for(int j=0;j<=i;j++){ float sc=0; for(int e=0;e<d;e++) sc+=Q[i*d+e]*K[j*d+e];
    sc*=scale; P[i*s+j]=sc; if(sc>mx) mx=sc; }
  float sum=0; for(int j=0;j<=i;j++){ float ex=expf(P[i*s+j]-mx); P[i*s+j]=ex; sum+=ex; }
  for(int j=0;j<=i;j++) P[i*s+j]/=sum;
  for(int j=i+1;j<s;j++) P[i*s+j]=0.0f;
  for(int e=0;e<d;e++){ float o=0; for(int j=0;j<=i;j++) o+=P[i*s+j]*V[j*d+e]; Out[i*d+e]=o; }
}
__global__ void k_attn_bwd(const float*Q,const float*K,const float*V,const float*P,
                           const float*dOut,float*dQ,float*dK,float*dV,int s,int d,float scale){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=s) return;
  float dpj[S]; float ssum=0;
  for(int j=0;j<=i;j++){ float dp=0; for(int e=0;e<d;e++) dp+=dOut[i*d+e]*V[j*d+e];
    dpj[j]=dp; ssum+=P[i*s+j]*dp; }
  float dq[D]; for(int e=0;e<d;e++) dq[e]=0;
  for(int j=0;j<=i;j++){
    float dsc=P[i*s+j]*(dpj[j]-ssum);
    for(int e=0;e<d;e++){
      dq[e]+=scale*dsc*K[j*d+e];
      atomicAdd(&dK[j*d+e], scale*dsc*Q[i*d+e]);
      atomicAdd(&dV[j*d+e], P[i*s+j]*dOut[i*d+e]);
    }
  }
  for(int e=0;e<d;e++) dQ[i*d+e]=dq[e];
}

static float *dQ_,*dK_,*dV_,*dP,*dOut,*ddQ,*ddK,*ddV,*ddOut;
static float hQ[S*D],hK[S*D],hV[S*D],hT[S*D],hO[S*D];

static float fwd_loss(){
  CK(cudaMemcpy(dQ_,hQ,sizeof(hQ),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK_,hK,sizeof(hK),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV_,hV,sizeof(hV),cudaMemcpyHostToDevice));
  k_attn_fwd<<<(S+31)/32,32>>>(dQ_,dK_,dV_,dP,dOut,S,D,SCALE);
  CK(cudaMemcpy(hO,dOut,sizeof(hO),cudaMemcpyDeviceToHost));
  double e=0; for(int i=0;i<S*D;i++){ double dd=(double)hO[i]-hT[i]; e+=dd*dd; }
  return (float)(0.5*e);   // double-accumulate to cut summation roundoff
}
static void analytic(float*gq,float*gk,float*gv){
  fwd_loss();
  float hdO[S*D]; for(int i=0;i<S*D;i++) hdO[i]=hO[i]-hT[i];
  CK(cudaMemcpy(ddOut,hdO,sizeof(hdO),cudaMemcpyHostToDevice));
  CK(cudaMemset(ddQ,0,sizeof(hQ))); CK(cudaMemset(ddK,0,sizeof(hK))); CK(cudaMemset(ddV,0,sizeof(hV)));
  k_attn_bwd<<<(S+31)/32,32>>>(dQ_,dK_,dV_,dP,ddOut,ddQ,ddK,ddV,S,D,SCALE);
  CK(cudaMemcpy(gq,ddQ,sizeof(hQ),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gk,ddK,sizeof(hK),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(gv,ddV,sizeof(hV),cudaMemcpyDeviceToHost));
}

static long seed=101;
static float rnd(){ seed=(seed*1103515245+12345)&0x7fffffff; return (float)seed/2147483648.0f-0.5f; }

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s | causal attention grad-check S=%d D=%d\n",p.name,S,D);
  for(int i=0;i<S*D;i++){ hQ[i]=rnd(); hK[i]=rnd(); hV[i]=rnd(); hT[i]=rnd(); }
  CK(cudaMalloc(&dQ_,sizeof(hQ))); CK(cudaMalloc(&dK_,sizeof(hK))); CK(cudaMalloc(&dV_,sizeof(hV)));
  CK(cudaMalloc(&dP,S*S*sizeof(float))); CK(cudaMalloc(&dOut,sizeof(hO)));
  CK(cudaMalloc(&ddQ,sizeof(hQ))); CK(cudaMalloc(&ddK,sizeof(hK))); CK(cudaMalloc(&ddV,sizeof(hV)));
  CK(cudaMalloc(&ddOut,sizeof(hO)));

  float gq[S*D],gk[S*D],gv[S*D]; analytic(gq,gk,gv);
  //  FP32 central-difference of an O(1) loss has a ~1e-4 absolute roundoff floor;
  //  a larger eps lifts small-gradient signal above it. Tolerance: rel<2e-2 OR
  //  abs<2e-4 (the honest FP32 finite-diff noise floor).
  const float eps=5e-3f; int bad=0; float maxrel=0;
  auto check=[&](const char*tag,float*vec,float*ana,int idx){
    float w0=vec[idx];
    vec[idx]=w0+eps; float Lp=fwd_loss();
    vec[idx]=w0-eps; float Lm=fwd_loss();
    vec[idx]=w0;
    float fd=(Lp-Lm)/(2*eps), ad=fabsf(ana[idx]-fd);
    float rel=ad/(fabsf(ana[idx])+fabsf(fd)+1e-9f);
    bool ok = rel<2e-2f || ad<2e-4f; if(!ok) bad++;
    if(rel>maxrel) maxrel=rel;
    printf("  %s[%d] analytic=% .6e fd=% .6e rel=%.3e %s\n",tag,idx,ana[idx],fd,rel,ok?"ok":"BAD");
  };
  printf(" dQ:\n"); for(int t=0;t<3;t++){ int i=(t*11+2)%(S*D); check("dQ",hQ,gq,i); }
  printf(" dK:\n"); for(int t=0;t<3;t++){ int i=(t*13+5)%(S*D); check("dK",hK,gk,i); }
  printf(" dV:\n"); for(int t=0;t<3;t++){ int i=(t*7+1)%(S*D);  check("dV",hV,gv,i); }

  printf("RESULT: %s (causal attention fwd+bwd grad-checked; max rel=%.3e)\n",
         bad==0?"PASS":"FAIL", maxrel);
  return bad==0?0:1;
}
