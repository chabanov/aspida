// test_celoss_gpu.cu — Step 5h: the last ops toward a GPU-resident Student —
// cross-entropy loss (the training objective) and token embedding (gather/
// scatter). Both forward + backward, grad-checked on an L40S.
//
//  CE:   p = softmax(z);  loss_b = -log p[y_b];  dz[b][k] = p[k] - [k==y_b]
//  Emb:  out[b] = E[id_b];  dE[id_b] += dout[b]   (scatter-add; repeated ids accumulate)
//
//   nvcc -O3 -arch=native test_celoss_gpu.cu -o test_celoss_gpu && ./test_celoss_gpu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

static const int B=4, V=8, D=8;

// ---- cross-entropy ----
__global__ void k_ce(const float*Z,const int*Y,float*loss,float*dZ,int b,int v){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=b) return;
  float mx=-1e30f; for(int k=0;k<v;k++) if(Z[r*v+k]>mx) mx=Z[r*v+k];
  float sum=0; for(int k=0;k<v;k++) sum+=expf(Z[r*v+k]-mx);
  float lse=mx+logf(sum);
  loss[r]=lse-Z[r*v+Y[r]];
  for(int k=0;k<v;k++) dZ[r*v+k]=expf(Z[r*v+k]-lse)-(k==Y[r]?1.0f:0.0f);
}
// ---- embedding ----
__global__ void k_emb_fwd(const float*E,const int*Id,float*Out,int b,int d){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=b*d) return;
  int r=t/d, c=t%d; Out[t]=E[Id[r]*d+c];
}
__global__ void k_emb_bwd(const float*dOut,const int*Id,float*dE,int b,int d){
  int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=b*d) return;
  int r=t/d, c=t%d; atomicAdd(&dE[Id[r]*d+c], dOut[t]);
}

static float *dZ,*dLoss,*ddZ,*dE,*dOut,*ddE,*ddOut;
static int   *dY,*dId;
static float hZ[B*V], hE[V*D], hT[B*D], hOut[B*D];
static int   hY[B], hId[B];

static double ce_loss(){
  CK(cudaMemcpy(dZ,hZ,sizeof(hZ),cudaMemcpyHostToDevice));
  k_ce<<<(B+31)/32,32>>>(dZ,dY,dLoss,ddZ,B,V);
  float hl[B]; CK(cudaMemcpy(hl,dLoss,sizeof(hl),cudaMemcpyDeviceToHost));
  double s=0; for(int i=0;i<B;i++) s+=hl[i]; return s;
}
static double emb_loss(){
  CK(cudaMemcpy(dE,hE,sizeof(hE),cudaMemcpyHostToDevice));
  k_emb_fwd<<<(B*D+63)/64,64>>>(dE,dId,dOut,B,D);
  CK(cudaMemcpy(hOut,dOut,sizeof(hOut),cudaMemcpyDeviceToHost));
  double e=0; for(int i=0;i<B*D;i++){ double dd=(double)hOut[i]-hT[i]; e+=dd*dd; } return 0.5*e;
}

static long seed=211;
static float rnd(){ seed=(seed*1103515245+12345)&0x7fffffff; return (float)seed/2147483648.0f-0.5f; }

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s | CE + embedding grad-check B=%d V=%d D=%d\n",p.name,B,V,D);
  for(int i=0;i<B*V;i++) hZ[i]=rnd()*2.0f;
  for(int i=0;i<V*D;i++) hE[i]=rnd();
  for(int i=0;i<B*D;i++) hT[i]=rnd();
  hY[0]=1; hY[1]=3; hY[2]=0; hY[3]=6;
  hId[0]=2; hId[1]=5; hId[2]=2; hId[3]=7;          // id 2 repeats -> scatter-add accumulates

  CK(cudaMalloc(&dZ,sizeof(hZ))); CK(cudaMalloc(&dLoss,B*sizeof(float)));
  CK(cudaMalloc(&ddZ,sizeof(hZ))); CK(cudaMalloc(&dY,sizeof(hY)));
  CK(cudaMalloc(&dE,sizeof(hE))); CK(cudaMalloc(&dOut,sizeof(hOut)));
  CK(cudaMalloc(&ddE,sizeof(hE))); CK(cudaMalloc(&ddOut,sizeof(hOut)));
  CK(cudaMalloc(&dId,sizeof(hId)));
  CK(cudaMemcpy(dY,hY,sizeof(hY),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dId,hId,sizeof(hId),cudaMemcpyHostToDevice));

  const float eps=5e-3f; int bad=0; float maxrel=0;
  auto probe=[&](const char*tag,float*vec,float ana,int idx,double(*L)()){
    float w0=vec[idx];
    vec[idx]=w0+eps; double Lp=L();
    vec[idx]=w0-eps; double Lm=L();
    vec[idx]=w0;
    float fd=(float)((Lp-Lm)/(2*eps)), ad=fabsf(ana-fd);
    float rel=ad/(fabsf(ana)+fabsf(fd)+1e-9f);
    bool ok = rel<2e-2f || ad<2e-4f; if(!ok) bad++;
    if(rel>maxrel) maxrel=rel;
    printf("  %s[%d] analytic=% .6e fd=% .6e rel=%.3e %s\n",tag,idx,ana,fd,rel,ok?"ok":"BAD");
  };

  // cross-entropy: analytic dZ
  ce_loss(); float gz[B*V]; CK(cudaMemcpy(gz,ddZ,sizeof(gz),cudaMemcpyDeviceToHost));
  printf(" cross-entropy dlogits:\n");
  for(int t=0;t<4;t++){ int i=(t*9+3)%(B*V); probe("dZ",hZ,gz[i],i,ce_loss); }

  // embedding: analytic dE
  emb_loss();
  float hdO[B*D]; for(int i=0;i<B*D;i++) hdO[i]=hOut[i]-hT[i];
  CK(cudaMemcpy(ddOut,hdO,sizeof(hdO),cudaMemcpyHostToDevice));
  CK(cudaMemset(ddE,0,sizeof(hE)));
  k_emb_bwd<<<(B*D+63)/64,64>>>(ddOut,dId,ddE,B,D);
  float ge[V*D]; CK(cudaMemcpy(ge,ddE,sizeof(ge),cudaMemcpyDeviceToHost));
  printf(" embedding dE (incl. repeated id 2):\n");
  probe("dE",hE,ge[2*D+0],2*D+0,emb_loss);    // id 2, dim 0 (accumulates b=0 and b=2)
  probe("dE",hE,ge[5*D+3],5*D+3,emb_loss);    // id 5
  probe("dE",hE,ge[7*D+6],7*D+6,emb_loss);    // id 7
  probe("dE",hE,ge[0*D+0],0*D+0,emb_loss);    // unused id 0 -> grad 0

  printf("RESULT: %s (cross-entropy + embedding grad-checked; max rel=%.3e)\n",
         bad==0?"PASS":"FAIL", maxrel);
  return bad==0?0:1;
}
