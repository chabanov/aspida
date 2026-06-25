// student_shim.cu — Stream B production: the resident multi-head Student exposed
// as a RUNTIME-CONFIGURABLE C-ABI session, so one libaspidastudent.so serves any
// tier (Small/Medium/Large from platform.ads). Architecture is passed to
// stu_create; the forward/backward is the grad-checked math from student_resident.cu.
//
//   void*  stu_create(V,D,F,S,L,H)            -- resident multi-head Student (AdamW)
//   void   stu_set_data(h, ids[S], tgts[S])   -- one next-token example, resident
//   float  stu_step(h, lr)                     -- fwd + bwd + AdamW on device; returns loss
//   void   stu_free(h)
//
//   lib: nvcc -O3 -arch=native -shared -Xcompiler -fPIC student_shim.cu -o libaspidastudent.so

#include <cstdlib>
#include <cmath>
#include "student_kernels.cuh"

#define CKK(x) do{ cudaError_t e_=(x); if(e_){ abort(); } }while(0)
static const int ML = 32;     // max layers
static const float EPS=1e-5f, BASE=10000.0f;

struct Stu {
  int V,D,F,S,L,H,DH; float SC;
  float *Wq[ML],*Wk[ML],*Wv[ML],*Wo[ML],*G1[ML],*G2[ML],*Wg[ML],*Wu[ML],*Wd[ML];
  float *gWq[ML],*gWk[ML],*gWv[ML],*gWo[ML],*gG1[ML],*gG2[ML],*gWg[ML],*gWu[ML],*gWd[ML];
  float *E,*Gf,*Wh,*gE,*gGf,*gWh;
  float *aN1[ML],*aR1[ML],*aQr[ML],*aKr[ML],*aVv[ML],*aP[ML],*aAttn[ML],*aXmid[ML],*aN2[ML],*aR2[ML],*aGate[ML],*aUp[ML],*aH[ML],*aX[ML+1];
  float *aXf,*aRf,*aLogits,*aLoss;
  float *dQ; int soft;   // distillation: teacher distribution [S*V] + soft-target mode
  float *qtmp,*ktmp,*otmp,*mtmp,*dlogits,*dxf,*dhsw,*dgate,*dup,*dn2,*dxmid,*dattn,*dqr,*dkr,*dvv,*dq,*dk,*dn1,*dxn,*t1,*t2,*cA,*cB;
  int *Id,*Yd;
  __half *h16a,*h16b;   // FP16 staging for tensor-core forward matmuls
  float *opt_w[ML*9+8],*opt_g[ML*9+8],*opt_m[ML*9+8],*opt_v[ML*9+8],*opt_a[ML*9+8]; int opt_n[ML*9+8],opt_cnt; long step;
};
static dim3 TB(16,16);
static dim3 gr(int X,int Y){ return dim3((X+15)/16,(Y+15)/16); }
static void ze(float*p,int n){ CKK(cudaMemset(p,0,n*4)); }
static float* M(int n){ float*p; CKK(cudaMalloc(&p,n*4)); return p; }
static long sd=909;
static float rnd(){ sd=(sd*1103515245+12345)&0x7fffffff; return (float)sd/2147483648.0f-0.5f; }

extern "C" void* stu_create(int V,int D,int F,int S,int L,int H){
  if(L>ML||D%H!=0||D/H>MAX_DH) return nullptr;
  sd=909;   // deterministic init so data-parallel replicas start identical
  Stu* s=new Stu(); s->V=V;s->D=D;s->F=F;s->S=S;s->L=L;s->H=H;s->DH=D/H; s->SC=1.0f/sqrtf((float)s->DH);
  int maxw=D*F; if(V*D>maxw)maxw=V*D; if(F*D>maxw)maxw=F*D; if(D*V>maxw)maxw=D*V;
  float* buf=(float*)malloc((size_t)maxw*4);
  auto up=[&](float*dev,int n){ CKK(cudaMemcpy(dev,buf,(size_t)n*4,cudaMemcpyHostToDevice)); };
  for(int l=0;l<L;l++){
    s->Wq[l]=M(D*D);s->Wk[l]=M(D*D);s->Wv[l]=M(D*D);s->Wo[l]=M(D*D);s->G1[l]=M(D);s->G2[l]=M(D);s->Wg[l]=M(D*F);s->Wu[l]=M(D*F);s->Wd[l]=M(F*D);
    s->gWq[l]=M(D*D);s->gWk[l]=M(D*D);s->gWv[l]=M(D*D);s->gWo[l]=M(D*D);s->gG1[l]=M(D);s->gG2[l]=M(D);s->gWg[l]=M(D*F);s->gWu[l]=M(D*F);s->gWd[l]=M(F*D);
    s->aN1[l]=M(S*D);s->aR1[l]=M(S);s->aQr[l]=M(S*D);s->aKr[l]=M(S*D);s->aVv[l]=M(S*D);s->aP[l]=M(H*S*S);s->aAttn[l]=M(S*D);
    s->aXmid[l]=M(S*D);s->aN2[l]=M(S*D);s->aR2[l]=M(S);s->aGate[l]=M(S*F);s->aUp[l]=M(S*F);s->aH[l]=M(S*F);
    auto inW=[&](float*dev,int n,float sc){ for(int i=0;i<n;i++)buf[i]=rnd()*sc; up(dev,n); };
    inW(s->Wq[l],D*D,0.2f);inW(s->Wk[l],D*D,0.2f);inW(s->Wv[l],D*D,0.2f);inW(s->Wo[l],D*D,0.2f);
    inW(s->Wg[l],D*F,0.2f);inW(s->Wu[l],D*F,0.2f);inW(s->Wd[l],F*D,0.2f);
    for(int i=0;i<D;i++)buf[i]=1+rnd()*0.1f; up(s->G1[l],D);
    for(int i=0;i<D;i++)buf[i]=1+rnd()*0.1f; up(s->G2[l],D);
  }
  for(int l=0;l<=L;l++) s->aX[l]=M(S*D);
  s->E=M(V*D);s->Gf=M(D);s->Wh=M(D*V);s->gE=M(V*D);s->gGf=M(D);s->gWh=M(D*V);
  for(int i=0;i<V*D;i++)buf[i]=rnd()*0.1f; up(s->E,V*D);
  for(int i=0;i<D;i++)buf[i]=1+rnd()*0.1f; up(s->Gf,D);
  for(int i=0;i<D*V;i++)buf[i]=rnd()*0.1f; up(s->Wh,D*V);
  s->aXf=M(S*D);s->aRf=M(S);s->aLogits=M(S*V);s->aLoss=M(S);s->dQ=M(S*V);s->soft=0;
  s->qtmp=M(S*D);s->ktmp=M(S*D);s->otmp=M(S*D);s->mtmp=M(S*D);s->dlogits=M(S*V);s->dxf=M(S*D);s->dhsw=M(S*F);s->dgate=M(S*F);s->dup=M(S*F);
  s->dn2=M(S*D);s->dxmid=M(S*D);s->dattn=M(S*D);s->dqr=M(S*D);s->dkr=M(S*D);s->dvv=M(S*D);s->dq=M(S*D);s->dk=M(S*D);s->dn1=M(S*D);s->dxn=M(S*D);
  s->t1=M(S*F);s->t2=M(S*F);s->cA=M(S*D);s->cB=M(S*D);
  s->Id=(int*)M((S+3)/4*4); s->Yd=(int*)M((S+3)/4*4);
  { int mA=S*(D>F?D:F); int mB=(D*F>D*V?D*F:D*V);    // FP16 staging (forward matmuls)
    CKK(cudaMalloc(&s->h16a,(size_t)mA*2)); CKK(cudaMalloc(&s->h16b,(size_t)mB*2)); }
  free(buf);
  s->opt_cnt=0; s->step=0;
  auto reg=[&](float*w,float*g,int n){ int k=s->opt_cnt++; s->opt_w[k]=w;s->opt_g[k]=g;s->opt_n[k]=n;
    s->opt_m[k]=M(n);s->opt_v[k]=M(n);s->opt_a[k]=M(n); ze(s->opt_m[k],n);ze(s->opt_v[k],n);ze(s->opt_a[k],n); };
  for(int l=0;l<L;l++){ reg(s->Wq[l],s->gWq[l],D*D);reg(s->Wk[l],s->gWk[l],D*D);reg(s->Wv[l],s->gWv[l],D*D);reg(s->Wo[l],s->gWo[l],D*D);
    reg(s->G1[l],s->gG1[l],D);reg(s->G2[l],s->gG2[l],D);reg(s->Wg[l],s->gWg[l],D*F);reg(s->Wu[l],s->gWu[l],D*F);reg(s->Wd[l],s->gWd[l],F*D); }
  reg(s->Gf,s->gGf,D); reg(s->Wh,s->gWh,D*V); reg(s->E,s->gE,V*D);
  return s;
}
extern "C" void stu_set_data(void* h,const int* ids,const int* tgts){
  Stu* s=(Stu*)h; CKK(cudaMemcpy(s->Id,ids,(size_t)s->S*4,cudaMemcpyHostToDevice)); CKK(cudaMemcpy(s->Yd,tgts,(size_t)s->S*4,cudaMemcpyHostToDevice));
}
// distillation: set the input ids + a per-position teacher distribution Q[S*V]
// (each row sums to 1); switches the loss to soft-target (KL) cross-entropy.
extern "C" void stu_set_distill(void* h,const int* ids,const float* Q){
  Stu* s=(Stu*)h; CKK(cudaMemcpy(s->Id,ids,(size_t)s->S*4,cudaMemcpyHostToDevice));
  CKK(cudaMemcpy(s->dQ,Q,(size_t)s->S*s->V*4,cudaMemcpyHostToDevice)); s->soft=1;
}
// Forward matmul C[M,N]=A[M,K]·B[K,N]: tensor-core (FP16 in, FP32 out) when the
// dims are 16-aligned (real tiers), else the FP32 tiled path. Weights stay FP32
// (master); only the matmul inputs are cast to FP16 — gradients remain FP32, so
// no loss scaling is needed.
static void mm_w(Stu* s, float* C, const float* A, const float* B, int M, int K, int N){
  if((M%16)||(K%16)||(N%16)){
    k_mm_tiled<<<gr(N,M),TB>>>(A,B,C,M,K,N);
  } else {
    k_f2h<<<(M*K+255)/256,256>>>(A,s->h16a,(long)M*K);
    k_f2h<<<(K*N+255)/256,256>>>(B,s->h16b,(long)K*N);
    dim3 tbw(128,4);
    dim3 gw((M+(16*tbw.x/32)-1)/(16*tbw.x/32),(N+(16*tbw.y)-1)/(16*tbw.y));
    k_mm_wmma<<<gw,tbw>>>(s->h16a,s->h16b,C,M,K,N);
  }
}
static double fwd(Stu* s){
  int V=s->V,D=s->D,F=s->F,S=s->S,L=s->L,H=s->H,DH=s->DH; float SC=s->SC;
  int RW=(S+31)/32, RP=(S*(D/2)+63)/64, BD=(S*D+63)/64, BF=(S*F+63)/64, MH=(H*S+31)/32;
  k_emb_fwd<<<BD,64>>>(s->E,s->Id,s->aX[0],S,D);
  for(int l=0;l<L;l++){
    k_rms_fwd<<<RW,32>>>(s->aX[l],s->G1[l],s->aN1[l],s->aR1[l],S,D,EPS);
    mm_w(s,s->qtmp,s->aN1[l],s->Wq[l],S,D,D); mm_w(s,s->ktmp,s->aN1[l],s->Wk[l],S,D,D); mm_w(s,s->aVv[l],s->aN1[l],s->Wv[l],S,D,D);
    k_rope_mh_fwd<<<RP,64>>>(s->qtmp,s->aQr[l],S,H,DH,BASE);k_rope_mh_fwd<<<RP,64>>>(s->ktmp,s->aKr[l],S,H,DH,BASE);
    k_mha_fwd<<<MH,32>>>(s->aQr[l],s->aKr[l],s->aVv[l],s->aP[l],s->aAttn[l],S,H,DH,SC);
    mm_w(s,s->otmp,s->aAttn[l],s->Wo[l],S,D,D); k_add<<<BD,64>>>(s->aX[l],s->otmp,s->aXmid[l],S*D);
    k_rms_fwd<<<RW,32>>>(s->aXmid[l],s->G2[l],s->aN2[l],s->aR2[l],S,D,EPS);
    mm_w(s,s->aGate[l],s->aN2[l],s->Wg[l],S,D,F); mm_w(s,s->aUp[l],s->aN2[l],s->Wu[l],S,D,F);
    k_swiglu_fwd<<<BF,64>>>(s->aGate[l],s->aUp[l],s->aH[l],S*F);
    mm_w(s,s->mtmp,s->aH[l],s->Wd[l],S,F,D); k_add<<<BD,64>>>(s->aXmid[l],s->mtmp,s->aX[l+1],S*D);
  }
  k_rms_fwd<<<RW,32>>>(s->aX[L],s->Gf,s->aXf,s->aRf,S,D,EPS);
  mm_w(s,s->aLogits,s->aXf,s->Wh,S,D,V);
  if(s->soft) k_ce_soft<<<RW,32>>>(s->aLogits,s->dQ,s->aLoss,s->dlogits,S,V);
  else        k_ce<<<RW,32>>>(s->aLogits,s->Yd,s->aLoss,s->dlogits,S,V);
  float* hl=(float*)malloc((size_t)S*4); CKK(cudaMemcpy(hl,s->aLoss,(size_t)S*4,cudaMemcpyDeviceToHost));
  double e=0; for(int i=0;i<S;i++)e+=hl[i]; free(hl); return e;
}
static void bwd(Stu* s){
  int V=s->V,D=s->D,F=s->F,S=s->S,L=s->L,H=s->H,DH=s->DH; float SC=s->SC;
  int RW=(S+31)/32, RP=(S*(D/2)+63)/64, BD=(S*D+63)/64, BF=(S*F+63)/64, MH=(H*S+31)/32;
  k_mm_AtB<<<gr(V,D),TB>>>(s->aXf,s->dlogits,s->gWh,S,D,V);
  k_mm_ABt<<<gr(D,S),TB>>>(s->dlogits,s->Wh,s->dxf,S,D,V);
  ze(s->gGf,D); k_rms_bwd<<<RW,32>>>(s->aX[L],s->Gf,s->dxf,s->aRf,s->cA,s->gGf,S,D);
  float*carry=s->cA,*cn=s->cB;
  for(int l=L-1;l>=0;l--){
    k_mm_AtB<<<gr(D,F),TB>>>(s->aH[l],carry,s->gWd[l],S,F,D); k_mm_ABt<<<gr(F,S),TB>>>(carry,s->Wd[l],s->dhsw,S,F,D);
    k_swiglu_bwd<<<BF,64>>>(s->aGate[l],s->aUp[l],s->dhsw,s->dgate,s->dup,S*F);
    k_mm_AtB<<<gr(F,D),TB>>>(s->aN2[l],s->dgate,s->gWg[l],S,D,F); k_mm_AtB<<<gr(F,D),TB>>>(s->aN2[l],s->dup,s->gWu[l],S,D,F);
    k_mm_ABt<<<gr(D,S),TB>>>(s->dgate,s->Wg[l],s->t1,S,D,F); k_mm_ABt<<<gr(D,S),TB>>>(s->dup,s->Wu[l],s->t2,S,D,F); k_add<<<BD,64>>>(s->t1,s->t2,s->dn2,S*D);
    ze(s->gG2[l],D); k_rms_bwd<<<RW,32>>>(s->aXmid[l],s->G2[l],s->dn2,s->aR2[l],s->dxn,s->gG2[l],S,D);
    k_add<<<BD,64>>>(carry,s->dxn,s->dxmid,S*D);
    k_mm_AtB<<<gr(D,D),TB>>>(s->aAttn[l],s->dxmid,s->gWo[l],S,D,D); k_mm_ABt<<<gr(D,S),TB>>>(s->dxmid,s->Wo[l],s->dattn,S,D,D);
    ze(s->dkr,S*D); ze(s->dvv,S*D);
    k_mha_bwd<<<MH,32>>>(s->aQr[l],s->aKr[l],s->aVv[l],s->aP[l],s->dattn,s->dqr,s->dkr,s->dvv,S,H,DH,SC);
    k_rope_mh_bwd<<<RP,64>>>(s->dqr,s->dq,S,H,DH,BASE); k_rope_mh_bwd<<<RP,64>>>(s->dkr,s->dk,S,H,DH,BASE);
    k_mm_AtB<<<gr(D,D),TB>>>(s->aN1[l],s->dq,s->gWq[l],S,D,D);k_mm_AtB<<<gr(D,D),TB>>>(s->aN1[l],s->dk,s->gWk[l],S,D,D);k_mm_AtB<<<gr(D,D),TB>>>(s->aN1[l],s->dvv,s->gWv[l],S,D,D);
    k_mm_ABt<<<gr(D,S),TB>>>(s->dq,s->Wq[l],s->t1,S,D,D);k_mm_ABt<<<gr(D,S),TB>>>(s->dk,s->Wk[l],s->t2,S,D,D);k_add<<<BD,64>>>(s->t1,s->t2,s->dn1,S*D);
    k_mm_ABt<<<gr(D,S),TB>>>(s->dvv,s->Wv[l],s->t1,S,D,D); k_add<<<BD,64>>>(s->dn1,s->t1,s->dn1,S*D);
    ze(s->gG1[l],D); k_rms_bwd<<<RW,32>>>(s->aX[l],s->G1[l],s->dn1,s->aR1[l],s->dxn,s->gG1[l],S,D);
    k_add<<<BD,64>>>(s->dxmid,s->dxn,cn,S*D);
    float*t=carry;carry=cn;cn=t;
  }
  ze(s->gE,V*D); k_emb_bwd<<<BD,64>>>(carry,s->Id,s->gE,S,D);
}
extern "C" float stu_step(void* h,float lr){
  Stu* s=(Stu*)h; double loss=fwd(s); bwd(s);
  s->step++; const float b1=0.9f,b2=0.999f;
  float bc1=1.f-powf(b1,(float)s->step), bc2=1.f-powf(b2,(float)s->step);
  for(int k=0;k<s->opt_cnt;k++){ int n=s->opt_n[k];
    k_adamw<<<(n+255)/256,256>>>(s->opt_w[k],s->opt_g[k],s->opt_m[k],s->opt_v[k],n,lr,b1,b2,1e-8f,0.0f,bc1,bc2); }
  return (float)loss;
}
// Gradient accumulation: stu_micro accumulates one micro-batch's grads; stu_apply
// averages over G and does the AdamW update. In data-parallel, the all-reduce sums
// the accumulators across nodes before apply — so one all-reduce amortises G
// micro-steps (the Step-7 benchmark verdict's enabler).
extern "C" float stu_micro(void* h){
  Stu* s=(Stu*)h; double loss=fwd(s); bwd(s);
  for(int k=0;k<s->opt_cnt;k++){ int n=s->opt_n[k];
    k_add<<<(n+255)/256,256>>>(s->opt_a[k],s->opt_g[k],s->opt_a[k],n); }   // acc += grad
  return (float)loss;
}
extern "C" void stu_apply(void* h,float lr,int G){
  Stu* s=(Stu*)h; s->step++; const float b1=0.9f,b2=0.999f;
  float bc1=1.f-powf(b1,(float)s->step), bc2=1.f-powf(b2,(float)s->step), inv=1.0f/(float)G;
  for(int k=0;k<s->opt_cnt;k++){ int n=s->opt_n[k];
    k_scale<<<(n+255)/256,256>>>(s->opt_a[k],n,inv);                       // average over G
    k_adamw<<<(n+255)/256,256>>>(s->opt_w[k],s->opt_a[k],s->opt_m[k],s->opt_v[k],n,lr,b1,b2,1e-8f,0.0f,bc1,bc2);
    ze(s->opt_a[k],n);                                                     // reset accumulator
  }
}
// Data-parallel all-reduce hooks: flatten/restore the gradient accumulators so a
// coordinator can SUM them across nodes between stu_micro and stu_apply.
extern "C" int stu_nparams(void* h){
  Stu* s=(Stu*)h; int t=0; for(int k=0;k<s->opt_cnt;k++) t+=s->opt_n[k]; return t; }
extern "C" void stu_get_acc(void* h,float* out){
  Stu* s=(Stu*)h; int off=0;
  for(int k=0;k<s->opt_cnt;k++){ int n=s->opt_n[k];
    CKK(cudaMemcpy(out+off,s->opt_a[k],(size_t)n*4,cudaMemcpyDeviceToHost)); off+=n; } }
// Read the TRAINED WEIGHTS off the device, flattened in opt-register order
// (per layer: Wq,Wk,Wv,Wo,G1,G2,Wg,Wu,Wd; then Gf,Wh,E) — the same order as
// stu_nparams / stu_get_acc. This is the bridge that lets the Ada side export a
// servable GGUF of the GPU-resident model (each weight is row-major [in,out],
// G1/G2/Gf are [D]). Matches the CPU Student layout exactly.
extern "C" void stu_get_weights(void* h,float* out){
  Stu* s=(Stu*)h; int off=0;
  for(int k=0;k<s->opt_cnt;k++){ int n=s->opt_n[k];
    CKK(cudaMemcpy(out+off,s->opt_w[k],(size_t)n*4,cudaMemcpyDeviceToHost)); off+=n; } }
extern "C" void stu_set_acc(void* h,const float* in){
  Stu* s=(Stu*)h; int off=0;
  for(int k=0;k<s->opt_cnt;k++){ int n=s->opt_n[k];
    CKK(cudaMemcpy(s->opt_a[k],in+off,(size_t)n*4,cudaMemcpyHostToDevice)); off+=n; } }
extern "C" void stu_free(void* h){ delete (Stu*)h; }
