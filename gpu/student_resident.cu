// student_resident.cu — Step 5i: ASSEMBLY of a full GPU-resident Student
// forward+backward, grad-checked end-to-end vs a finite difference of CE loss.
//
//   Stage A: emb -> RMSNorm -> head -> CE.                                (done)
//   Stage B: + one pre-norm transformer layer.                           (done)
//   Stage C (this file): L stacked layers (looped) with inter-layer
//            gradient flow carried through the residual stream.
//   Each op grad-checked in 5d-5h; the chain (transposed backward, softmax
//   jacobian, un-rotate RoPE, two residuals, scatter-add embedding) is
//   grad-checked here across BOTH layers + train-to-zero sanity.
//
//   nvcc -O3 -arch=native student_resident.cu -o student_resident && ./student_resident

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e_=(x); if(e_){ \
  printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); abort(); } }while(0)

static const int V=32, D=16, F=32, S=6, L=2, H=2, DH=D/H;   // 2 heads × 8
static const float EPS=1e-5f, BASE=10000.0f, ASC=0.35355339f; // 1/sqrt(dh=8)

#include "student_kernels.cuh"   // shared kernels (k_emb/k_rms/k_mm/k_rope/k_attn/k_swiglu/k_ce/k_add)

// ---- params (host master) ----
static float hWq[L][D*D],hWk[L][D*D],hWv[L][D*D],hWo[L][D*D],hG1[L][D],hG2[L][D],
             hWg[L][D*F],hWu[L][D*F],hWd[L][F*D];
static float hE[V*D],hGf[D],hWh[D*V];
static int hId[S],hY[S];
// ---- device weights & grads (per layer) ----
static float *Wq[L],*Wk[L],*Wv[L],*Wo[L],*G1[L],*G2[L],*Wg[L],*Wu[L],*Wd[L];
static float *gWq[L],*gWk[L],*gWv[L],*gWo[L],*gG1[L],*gG2[L],*gWg[L],*gWu[L],*gWd[L];
static float *E,*Gf,*Wh, *gE,*gGf,*gWh;
// ---- activations (per layer) ----
static float *aN1[L],*aR1[L],*aQr[L],*aKr[L],*aVv[L],*aP[L],*aAttn[L],
             *aXmid[L],*aN2[L],*aR2[L],*aGate[L],*aUp[L],*aH[L], *aX[L+1];
static float *aXf,*aRf,*aLogits,*aLoss;
// ---- transient + grad-activation scratch (reused per layer) ----
static float *qtmp,*ktmp,*otmp,*mtmp, *dlogits,*dxf,*dhsw,*dgate,*dup,*dn2,
             *dxmid,*dattn,*dqr,*dkr,*dvv,*dq,*dk,*dn1,*dxn,*t1,*t2,*cA,*cB;
static int *Id,*Yd;

static dim3 TB(16,16);
static dim3 gr(int X,int Y){ return dim3((X+15)/16,(Y+15)/16); }
static void ze(float*p,int n){ CK(cudaMemset(p,0,n*4)); }
static int RP=(S*(D/2)+63)/64, RW=(S+31)/32;

static double forward_loss(){
  for(int l=0;l<L;l++){
    CK(cudaMemcpy(Wq[l],hWq[l],D*D*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(Wk[l],hWk[l],D*D*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Wv[l],hWv[l],D*D*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(Wo[l],hWo[l],D*D*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(G1[l],hG1[l],D*4,cudaMemcpyHostToDevice));  CK(cudaMemcpy(G2[l],hG2[l],D*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Wg[l],hWg[l],D*F*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(Wu[l],hWu[l],D*F*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Wd[l],hWd[l],F*D*4,cudaMemcpyHostToDevice));
  }
  CK(cudaMemcpy(E,hE,V*D*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(Gf,hGf,D*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(Wh,hWh,D*V*4,cudaMemcpyHostToDevice));
  k_emb_fwd<<<(S*D+63)/64,64>>>(E,Id,aX[0],S,D);
  for(int l=0;l<L;l++){
    k_rms_fwd<<<RW,32>>>(aX[l],G1[l],aN1[l],aR1[l],S,D,EPS);
    k_mm_tiled<<<gr(D,S),TB>>>(aN1[l],Wq[l],qtmp,S,D,D); k_mm_tiled<<<gr(D,S),TB>>>(aN1[l],Wk[l],ktmp,S,D,D); k_mm_tiled<<<gr(D,S),TB>>>(aN1[l],Wv[l],aVv[l],S,D,D);
    k_rope_mh_fwd<<<RP,64>>>(qtmp,aQr[l],S,H,DH,BASE); k_rope_mh_fwd<<<RP,64>>>(ktmp,aKr[l],S,H,DH,BASE);
    k_mha_fwd<<<(H*S+31)/32,32>>>(aQr[l],aKr[l],aVv[l],aP[l],aAttn[l],S,H,DH,ASC);
    k_mm_tiled<<<gr(D,S),TB>>>(aAttn[l],Wo[l],otmp,S,D,D);
    k_add<<<(S*D+63)/64,64>>>(aX[l],otmp,aXmid[l],S*D);
    k_rms_fwd<<<RW,32>>>(aXmid[l],G2[l],aN2[l],aR2[l],S,D,EPS);
    k_mm_tiled<<<gr(F,S),TB>>>(aN2[l],Wg[l],aGate[l],S,D,F); k_mm_tiled<<<gr(F,S),TB>>>(aN2[l],Wu[l],aUp[l],S,D,F);
    k_swiglu_fwd<<<(S*F+63)/64,64>>>(aGate[l],aUp[l],aH[l],S*F);
    k_mm_tiled<<<gr(D,S),TB>>>(aH[l],Wd[l],mtmp,S,F,D);
    k_add<<<(S*D+63)/64,64>>>(aXmid[l],mtmp,aX[l+1],S*D);
  }
  k_rms_fwd<<<RW,32>>>(aX[L],Gf,aXf,aRf,S,D,EPS);
  k_mm_tiled<<<gr(V,S),TB>>>(aXf,Wh,aLogits,S,D,V);
  k_ce<<<RW,32>>>(aLogits,Yd,aLoss,dlogits,S,V);
  float hl[S]; CK(cudaMemcpy(hl,aLoss,S*4,cudaMemcpyDeviceToHost));
  double s=0; for(int i=0;i<S;i++)s+=hl[i]; return s;
}
static void backward(){
  forward_loss();
  k_mm_AtB<<<gr(V,D),TB>>>(aXf,dlogits,gWh,S,D,V);
  k_mm_ABt<<<gr(D,S),TB>>>(dlogits,Wh,dxf,S,D,V);
  ze(gGf,D); k_rms_bwd<<<RW,32>>>(aX[L],Gf,dxf,aRf,cA,gGf,S,D);   // cA = grad wrt aX[L]
  float *carry=cA, *cnext=cB;
  for(int l=L-1;l>=0;l--){
    // residual2: aX[l+1]=aXmid[l]+m ; dm=carry
    k_mm_AtB<<<gr(D,F),TB>>>(aH[l],carry,gWd[l],S,F,D);
    k_mm_ABt<<<gr(F,S),TB>>>(carry,Wd[l],dhsw,S,F,D);
    k_swiglu_bwd<<<(S*F+63)/64,64>>>(aGate[l],aUp[l],dhsw,dgate,dup,S*F);
    k_mm_AtB<<<gr(F,D),TB>>>(aN2[l],dgate,gWg[l],S,D,F); k_mm_AtB<<<gr(F,D),TB>>>(aN2[l],dup,gWu[l],S,D,F);
    k_mm_ABt<<<gr(D,S),TB>>>(dgate,Wg[l],t1,S,D,F); k_mm_ABt<<<gr(D,S),TB>>>(dup,Wu[l],t2,S,D,F);
    k_add<<<(S*D+63)/64,64>>>(t1,t2,dn2,S*D);
    ze(gG2[l],D); k_rms_bwd<<<RW,32>>>(aXmid[l],G2[l],dn2,aR2[l],dxn,gG2[l],S,D);
    k_add<<<(S*D+63)/64,64>>>(carry,dxn,dxmid,S*D);
    // residual1: aXmid[l]=aX[l]+o ; do=dxmid
    k_mm_AtB<<<gr(D,D),TB>>>(aAttn[l],dxmid,gWo[l],S,D,D);
    k_mm_ABt<<<gr(D,S),TB>>>(dxmid,Wo[l],dattn,S,D,D);
    ze(dkr,S*D); ze(dvv,S*D);
    k_mha_bwd<<<(H*S+31)/32,32>>>(aQr[l],aKr[l],aVv[l],aP[l],dattn,dqr,dkr,dvv,S,H,DH,ASC);
    k_rope_mh_bwd<<<RP,64>>>(dqr,dq,S,H,DH,BASE); k_rope_mh_bwd<<<RP,64>>>(dkr,dk,S,H,DH,BASE);
    k_mm_AtB<<<gr(D,D),TB>>>(aN1[l],dq,gWq[l],S,D,D); k_mm_AtB<<<gr(D,D),TB>>>(aN1[l],dk,gWk[l],S,D,D); k_mm_AtB<<<gr(D,D),TB>>>(aN1[l],dvv,gWv[l],S,D,D);
    k_mm_ABt<<<gr(D,S),TB>>>(dq,Wq[l],t1,S,D,D); k_mm_ABt<<<gr(D,S),TB>>>(dk,Wk[l],t2,S,D,D); k_add<<<(S*D+63)/64,64>>>(t1,t2,dn1,S*D);
    k_mm_ABt<<<gr(D,S),TB>>>(dvv,Wv[l],t1,S,D,D); k_add<<<(S*D+63)/64,64>>>(dn1,t1,dn1,S*D);
    ze(gG1[l],D); k_rms_bwd<<<RW,32>>>(aX[l],G1[l],dn1,aR1[l],dxn,gG1[l],S,D);
    k_add<<<(S*D+63)/64,64>>>(dxmid,dxn,cnext,S*D);   // grad wrt aX[l]
    float*tmp=carry; carry=cnext; cnext=tmp;
  }
  ze(gE,V*D); k_emb_bwd<<<(S*D+63)/64,64>>>(carry,Id,gE,S,D);
}

static long seed=909;
static float rnd(){ seed=(seed*1103515245+12345)&0x7fffffff; return (float)seed/2147483648.0f-0.5f; }

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s | resident Student (Stage C: L=%d layers) V=%d D=%d F=%d S=%d\n",p.name,L,V,D,F,S);
  for(int l=0;l<L;l++){
    for(int i=0;i<D*D;i++){hWq[l][i]=rnd()*0.2f;hWk[l][i]=rnd()*0.2f;hWv[l][i]=rnd()*0.2f;hWo[l][i]=rnd()*0.2f;}
    for(int i=0;i<D;i++){hG1[l][i]=1+rnd()*0.1f;hG2[l][i]=1+rnd()*0.1f;}
    for(int i=0;i<D*F;i++){hWg[l][i]=rnd()*0.2f;hWu[l][i]=rnd()*0.2f;} for(int i=0;i<F*D;i++)hWd[l][i]=rnd()*0.2f;
  }
  for(int i=0;i<D;i++)hGf[i]=1+rnd()*0.1f;
  for(int i=0;i<V*D;i++)hE[i]=rnd()*0.1f; for(int i=0;i<D*V;i++)hWh[i]=rnd()*0.1f;
  hId[0]=3;hId[1]=7;hId[2]=3;hId[3]=11;hId[4]=20;hId[5]=7;
  hY[0]=5;hY[1]=1;hY[2]=9;hY[3]=0;hY[4]=14;hY[5]=2;

  auto M=[&](float**p,int n){CK(cudaMalloc(p,n*4));};
  for(int l=0;l<L;l++){
    M(&Wq[l],D*D);M(&Wk[l],D*D);M(&Wv[l],D*D);M(&Wo[l],D*D);M(&G1[l],D);M(&G2[l],D);M(&Wg[l],D*F);M(&Wu[l],D*F);M(&Wd[l],F*D);
    M(&gWq[l],D*D);M(&gWk[l],D*D);M(&gWv[l],D*D);M(&gWo[l],D*D);M(&gG1[l],D);M(&gG2[l],D);M(&gWg[l],D*F);M(&gWu[l],D*F);M(&gWd[l],F*D);
    M(&aN1[l],S*D);M(&aR1[l],S);M(&aQr[l],S*D);M(&aKr[l],S*D);M(&aVv[l],S*D);M(&aP[l],H*S*S);M(&aAttn[l],S*D);
    M(&aXmid[l],S*D);M(&aN2[l],S*D);M(&aR2[l],S);M(&aGate[l],S*F);M(&aUp[l],S*F);M(&aH[l],S*F);
  }
  for(int l=0;l<=L;l++) M(&aX[l],S*D);
  M(&E,V*D);M(&Gf,D);M(&Wh,D*V);M(&gE,V*D);M(&gGf,D);M(&gWh,D*V);
  M(&aXf,S*D);M(&aRf,S);M(&aLogits,S*V);M(&aLoss,S);
  M(&qtmp,S*D);M(&ktmp,S*D);M(&otmp,S*D);M(&mtmp,S*D);M(&dlogits,S*V);M(&dxf,S*D);M(&dhsw,S*F);M(&dgate,S*F);M(&dup,S*F);
  M(&dn2,S*D);M(&dxmid,S*D);M(&dattn,S*D);M(&dqr,S*D);M(&dkr,S*D);M(&dvv,S*D);M(&dq,S*D);M(&dk,S*D);M(&dn1,S*D);M(&dxn,S*D);
  M(&t1,S*F);M(&t2,S*F);M(&cA,S*D);M(&cB,S*D);
  CK(cudaMalloc(&Id,sizeof(hId)));CK(cudaMalloc(&Yd,sizeof(hY)));
  CK(cudaMemcpy(Id,hId,sizeof(hId),cudaMemcpyHostToDevice));CK(cudaMemcpy(Yd,hY,sizeof(hY),cudaMemcpyHostToDevice));

  backward();
  const float eps=5e-3f; int bad=0; float maxrel=0;
  auto probe=[&](const char*tag,float*vec,float*gdev,int idx){
    float ana; CK(cudaMemcpy(&ana,gdev+idx,4,cudaMemcpyDeviceToHost));
    float w0=vec[idx]; vec[idx]=w0+eps; double Lp=forward_loss(); vec[idx]=w0-eps; double Lm=forward_loss(); vec[idx]=w0;
    float fd=(float)((Lp-Lm)/(2*eps)),ad=fabsf(ana-fd),rel=ad/(fabsf(ana)+fabsf(fd)+1e-9f);
    bool ok=rel<2e-2f||ad<2e-4f; if(!ok)bad++; if(rel>maxrel)maxrel=rel;
    printf("  %-7s analytic=% .6e fd=% .6e rel=%.3e %s\n",tag,ana,fd,rel,ok?"ok":"BAD"); };
  printf(" grad-check across both layers:\n");
  probe("L0.Wq",hWq[0],gWq[0],5);  probe("L0.Wo",hWo[0],gWo[0],100); probe("L0.Wd",hWd[0],gWd[0],50);
  probe("L1.Wq",hWq[1],gWq[1],5);  probe("L1.g2",hG2[1],gG2[1],4);   probe("L1.Wu",hWu[1],gWu[1],77);
  probe("Wh",   hWh,   gWh,   101); probe("E",    hE,    gE,    3*D+0);

  double L0=forward_loss();
  for(int it=0;it<400;it++){ backward();
    const float lr=0.03f;
    auto upd=[&](float*h,float*gdev,int n){ float g[D*F]; CK(cudaMemcpy(g,gdev,n*4,cudaMemcpyDeviceToHost)); for(int i=0;i<n;i++)h[i]-=lr*g[i]; };
    for(int l=0;l<L;l++){ upd(hWq[l],gWq[l],D*D);upd(hWk[l],gWk[l],D*D);upd(hWv[l],gWv[l],D*D);upd(hWo[l],gWo[l],D*D);
      upd(hG1[l],gG1[l],D);upd(hG2[l],gG2[l],D);upd(hWg[l],gWg[l],D*F);upd(hWu[l],gWu[l],D*F);upd(hWd[l],gWd[l],F*D); }
    upd(hGf,gGf,D); upd(hWh,gWh,D*V); { float g[V*D]; CK(cudaMemcpy(g,gE,V*D*4,cudaMemcpyDeviceToHost)); for(int i=0;i<V*D;i++)hE[i]-=lr*g[i]; }
  }
  double Lf=forward_loss();
  printf(" train sanity: loss %.4f -> %.4f (%s)\n",L0,Lf,Lf<L0?"down":"UP");
  printf("RESULT: %s (Stage C: %d-layer Student grad-checked end-to-end; max rel=%.3e)\n",
         (bad==0&&Lf<L0)?"PASS":"FAIL",L,maxrel);
  return (bad==0&&Lf<L0)?0:1;
}
