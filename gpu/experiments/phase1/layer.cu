// Stream B · Phase 1 — full Llama transformer layer on GPU (blk.0), assembled
// from the validated kernels, checked against the CPU-engine reference
// (gen_ref.adb, pos 0 / seq 1). Proves the kernels COMPOSE into a correct layer
// on real Llama-3.3-70B weights: rmsnorm x2, matvec x6, SwiGLU, GQA + residuals.
// (RoPE is identity at pos 0 and single-pos attention is softmax=1 -> V, so q/k
//  don't affect the output; both are validated at scale in ops.cu.)
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

__device__ __forceinline__ float f16(const uint8_t *p){ __half h; *reinterpret_cast<uint16_t*>(&h)=(uint16_t)p[0]|((uint16_t)p[1]<<8); return __half2float(h);}
__device__ __forceinline__ void gsm(const uint8_t*sc,int j,int*d,int*m){ if(j<4){*d=sc[j]&63;*m=sc[j+4]&63;} else {*d=(sc[j+4]&0x0F)|((sc[j-4]>>6)<<4);*m=(sc[j+4]>>4)|((sc[j]>>6)<<4);} }
__device__ void deq_block(const uint8_t*b,float*o){ float d=f16(b),dm=f16(b+2); const uint8_t*sc=b+4,*qs=b+16;
  for(int g=0;g<4;++g){int s1,m1,s2,m2;gsm(sc,2*g,&s1,&m1);gsm(sc,2*g+1,&s2,&m2);float d1=d*s1,mm1=dm*m1,d2=d*s2,mm2=dm*m2;const uint8_t*q=qs+g*32;
    for(int l=0;l<32;++l)o[64*g+l]=d1*(q[l]&0x0F)-mm1; for(int l=0;l<32;++l)o[64*g+32+l]=d2*(q[l]>>4)-mm2;}}
__global__ void k_matvec(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*144;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_block(r+(size_t)b*144,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}
// Q6_K (210 B/256): ql[128] qh[64] scales[16 int8] d(fp16). Mixed Q4_K_M uses
// this for attn_v + ffn_down. Mirrors llama.cpp dequantize_row_q6_K.
__device__ void deq_block_q6k(const uint8_t*b,float*out){const uint8_t*ql=b;const uint8_t*qh=b+128;const int8_t*sc=(const int8_t*)(b+192);float d=f16(b+208);
  for(int h=0;h<2;++h){const uint8_t*QL=ql+h*64;const uint8_t*QH=qh+h*32;const int8_t*SC=sc+h*8;float*Y=out+h*128;
    for(int l=0;l<32;++l){int is=l/16;
      int q1=(int)((QL[l]&0xF)|(((QH[l]>>0)&3)<<4))-32; int q2=(int)((QL[l+32]&0xF)|(((QH[l]>>2)&3)<<4))-32;
      int q3=(int)((QL[l]>>4)|(((QH[l]>>4)&3)<<4))-32;  int q4=(int)((QL[l+32]>>4)|(((QH[l]>>6)&3)<<4))-32;
      Y[l]=d*SC[is+0]*q1; Y[l+32]=d*SC[is+2]*q2; Y[l+64]=d*SC[is+4]*q3; Y[l+96]=d*SC[is+6]*q4;}}}
__global__ void k_matvec_q6k(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*210;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_block_q6k(r+(size_t)b*210,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}
__global__ void k_rmsnorm(const float*x,const float*w,float*y,int n){double ss=0;for(int i=0;i<n;++i)ss+=(double)x[i]*x[i];float r=sqrtf((float)(ss/n)+1e-6f);for(int i=0;i<n;++i)y[i]=x[i]/r*w[i];}
__global__ void k_gqa(const float*v,float*ctx,int nh,int nkv,int hd){int h=blockIdx.x;int kv=h/(nh/nkv);for(int j=0;j<hd;++j)ctx[h*hd+j]=v[kv*hd+j];}
__global__ void k_add(const float*a,const float*b,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]+b[i];}
__global__ void k_swiglu(const float*g,const float*u,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=g[i]/(1.0f+expf(-g[i]))*u[i];}

static void* slurp(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open %s\n",p);exit(2);}fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);void*b=malloc(s);if(fread(b,1,s,f)!=(size_t)s)exit(2);fclose(f);if(n)*n=s;return b;}
static uint8_t* upw(const char*p){size_t n;uint8_t*h=(uint8_t*)slurp(p,&n);uint8_t*d;cudaMalloc(&d,n);cudaMemcpy(d,h,n,cudaMemcpyHostToDevice);free(h);return d;}
static float* upf(const char*p,int n){float*h=(float*)slurp(p,0);float*d;cudaMalloc(&d,n*4);cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice);free(h);return d;}

int main(){
  const int D=8192, FFN=28672, KV=1024, HD=128, NH=D/HD, NKV=KV/HD;
  float *dx=upf("layer_x.bin",D), *dan=upf("layer_an.bin",D), *dfn=upf("layer_fn.bin",D);
  uint8_t *wq=upw("layer_wq.bin"),*wk=upw("layer_wk.bin"),*wv=upw("layer_wv.bin"),*wo=upw("layer_wo.bin");
  uint8_t *wg=upw("layer_wg.bin"),*wu=upw("layer_wu.bin"),*wd=upw("layer_wd.bin"); (void)wq;(void)wk;
  float *xn,*v,*ctx,*attn,*x1,*xn2,*g,*u,*gu,*dn;
  cudaMalloc(&xn,D*4);cudaMalloc(&v,KV*4);cudaMalloc(&ctx,D*4);cudaMalloc(&attn,D*4);cudaMalloc(&x1,D*4);
  cudaMalloc(&xn2,D*4);cudaMalloc(&g,FFN*4);cudaMalloc(&u,FFN*4);cudaMalloc(&gu,FFN*4);cudaMalloc(&dn,D*4);
  auto MV =[&](uint8_t*w,float*x,float*y,int in,int out){k_matvec<<<(out+127)/128,128>>>(w,x,y,in,out);};       // Q4_K
  auto MV6=[&](uint8_t*w,float*x,float*y,int in,int out){k_matvec_q6k<<<(out+127)/128,128>>>(w,x,y,in,out);};   // Q6_K

  k_rmsnorm<<<1,1>>>(dx,dan,xn,D);
  MV6(wv,xn,v,D,KV);                             // attn_v is Q6_K; q,k skipped (no effect at pos0)
  k_gqa<<<NH,1>>>(v,ctx,NH,NKV,HD);
  MV(wo,ctx,attn,D,D);
  k_add<<<(D+127)/128,128>>>(dx,attn,x1,D);      // residual
  k_rmsnorm<<<1,1>>>(x1,dfn,xn2,D);
  MV(wg,xn2,g,D,FFN); MV(wu,xn2,u,D,FFN);
  k_swiglu<<<(FFN+127)/128,128>>>(g,u,gu,FFN);
  MV6(wd,gu,dn,FFN,D);                           // ffn_down is Q6_K
  k_add<<<(D+127)/128,128>>>(x1,dn,x1,D);        // x1 += dn -> y
  cudaDeviceSynchronize();

  float *got=(float*)malloc(D*4); cudaMemcpy(got,x1,D*4,cudaMemcpyDeviceToHost);
  float *exp=(float*)slurp("layer_y.bin",0);
  double ma=0,mr=0; for(int i=0;i<D;++i){double a=fabs((double)got[i]-exp[i]);if(a>ma)ma=a;double r=a/(fabs((double)exp[i])+1e-6);if(r>mr)mr=r;}
  printf("[layer] D=%d FFN=%d nh=%d nkv=%d  max_abs=%.3e max_rel=%.3e -> %s\n",D,FFN,NH,NKV,ma,mr, ma<1e-3?"OK":"FAIL");
  printf(ma<1e-3?"PHASE1 LAYER: PASS\n":"PHASE1 LAYER: FAIL\n");
  return ma<1e-3?0:1;
}
