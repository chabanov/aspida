// Stream B — GPU matvec shim for the Ada engine (LLM_GPU dlopen's this).
// Exposes one C entry point; weights are uploaded to VRAM once (cached by host
// pointer) and stay resident across tokens. All five K-quants (Q4_K/Q5_K/Q6_K/
// Q3_K/Q2_K), bit-exact vs the CPU engine (build with --fmad=false). Build:
//   nvcc -O3 --fmad=false -arch=native -shared -Xcompiler -fPIC gpu_matvec.cu -o libaspidagpu.so
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include "fattn_ggml.cuh"   // Phase B: prefill full-attn via llama.cpp fattn-mma (ggml link-and-call)
#include "moe_ggml.cuh"     // MoE Phase B: prefill MoE via llama.cpp mul_mat_id (MMQ int8, ggml link-and-call)

// Decode issues thousands of tiny kernels + small blocking copies per token;
// the default ScheduleAuto wait can take a yield/sleep path that costs ~0.5 ms
// per implicit sync. Spin-waiting (we own the box, 8 vCPUs, 1 context) makes
// every blocking copy return as soon as the GPU finishes. Must run before the
// CUDA context exists -> constructor at dlopen time.
__attribute__((constructor)) static void aspida_gpu_setflags(void) {
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
}

__device__ __forceinline__ float f16(const uint8_t *p){ __half h; *reinterpret_cast<uint16_t*>(&h)=(uint16_t)p[0]|((uint16_t)p[1]<<8); return __half2float(h);}
__device__ __forceinline__ void gsm(const uint8_t*sc,int j,int*d,int*m){ if(j<4){*d=sc[j]&63;*m=sc[j+4]&63;} else {*d=(sc[j+4]&0x0F)|((sc[j-4]>>6)<<4);*m=(sc[j+4]>>4)|((sc[j]>>6)<<4);} }
__device__ void deq_q4k(const uint8_t*b,float*o){ float d=f16(b),dm=f16(b+2); const uint8_t*sc=b+4,*qs=b+16;
  for(int g=0;g<4;++g){int s1,m1,s2,m2;gsm(sc,2*g,&s1,&m1);gsm(sc,2*g+1,&s2,&m2);float d1=d*s1,mm1=dm*m1,d2=d*s2,mm2=dm*m2;const uint8_t*q=qs+g*32;
    for(int l=0;l<32;++l)o[64*g+l]=d1*(q[l]&0x0F)-mm1; for(int l=0;l<32;++l)o[64*g+32+l]=d2*(q[l]>>4)-mm2;}}
__device__ void deq_q6k(const uint8_t*b,float*out){const uint8_t*ql=b;const uint8_t*qh=b+128;const int8_t*sc=(const int8_t*)(b+192);float d=f16(b+208);
  for(int h=0;h<2;++h){const uint8_t*QL=ql+h*64;const uint8_t*QH=qh+h*32;const int8_t*SC=sc+h*8;float*Y=out+h*128;
    for(int l=0;l<32;++l){int is=l/16;
      int q1=(int)((QL[l]&0xF)|(((QH[l]>>0)&3)<<4))-32; int q2=(int)((QL[l+32]&0xF)|(((QH[l]>>2)&3)<<4))-32;
      int q3=(int)((QL[l]>>4)|(((QH[l]>>4)&3)<<4))-32;  int q4=(int)((QL[l+32]>>4)|(((QH[l]>>6)&3)<<4))-32;
      Y[l]=d*SC[is+0]*q1; Y[l+32]=d*SC[is+2]*q2; Y[l+64]=d*SC[is+4]*q3; Y[l+96]=d*SC[is+6]*q4;}}}

// ---- Legacy scalar kernels: one thread per output row (uncoalesced reads,
//      low occupancy). Bit-exact vs the CPU engine. Kept for validation /
//      debugging behind ASPIDA_GPU_SCALAR. ----
// Q5_K super-block: 176 B / 256 vals. d(f16) dmin(f16) scales[12] qh[32] qs[128].
// 5-bit weight = low nibble (qs) | high bit (qh) << 4; same 6-bit scales as Q4_K.
__device__ void deq_q5k(const uint8_t*b,float*o){ float d=f16(b),dm=f16(b+2); const uint8_t*sc=b+4,*qh=b+16,*qs=b+48;
  for(int g=0;g<4;++g){int s1,m1,s2,m2;gsm(sc,2*g,&s1,&m1);gsm(sc,2*g+1,&s2,&m2);float d1=d*s1,mm1=dm*m1,d2=d*s2,mm2=dm*m2;
    for(int l=0;l<32;++l){unsigned q=qs[32*g+l];int lo=(q&0xF)+(((qh[l]>>(2*g))&1)<<4);int hi=(q>>4)+(((qh[l]>>(2*g+1))&1)<<4);
      o[64*g+l]=d1*lo-mm1; o[64*g+32+l]=d2*hi-mm2;}}}

__device__ __forceinline__ uint32_t ld32(const uint8_t*p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
// Q3_K: 110 B/256. hmask[32] qs[64](2-bit) scales[12](16 signed 6-bit) d(f16).
// 16 scales unpacked (kmask1=0x03030303, kmask2=0x0f0f0f0f) into sca[is]-32.
__device__ __forceinline__ void q3k_scales(const uint8_t*sc,int*sca){
  uint32_t a0=ld32(sc),a1=ld32(sc+4),tmp=ld32(sc+8),KM1=0x03030303u,KM2=0x0f0f0f0fu;
  uint32_t S0=(a0&KM2)|(((tmp>>0)&KM1)<<4), S1=(a1&KM2)|(((tmp>>2)&KM1)<<4);
  uint32_t S2=((a0>>4)&KM2)|(((tmp>>4)&KM1)<<4), S3=((a1>>4)&KM2)|(((tmp>>6)&KM1)<<4);
  for(int k=0;k<4;++k){sca[k]=(int)((S0>>(8*k))&0xFF)-32; sca[4+k]=(int)((S1>>(8*k))&0xFF)-32;
                       sca[8+k]=(int)((S2>>(8*k))&0xFF)-32; sca[12+k]=(int)((S3>>(8*k))&0xFF)-32;}}
__device__ void deq_q3k(const uint8_t*b,float*o){
  const uint8_t*hm=b,*qs=b+32,*sc=b+96; float d=f16(b+108); int sca[16]; q3k_scales(sc,sca);
  for(int P=0;P<256;++P){int nh=P>>7,pp=P&127,j=pp>>5,rm=pp&31,g=rm>>4,l=rm&15;
    int low2=(qs[nh*32+g*16+l]>>(2*j))&3, hbit=(hm[g*16+l]>>(nh*4+j))&1, is=nh*8+j*2+g;
    o[P]=d*sca[is]*(float)(low2+4*hbit-4);}}
// Q2_K: 84 B/256. scales[16](4-bit scale|4-bit min) qs[64](2-bit) d(f16) dmin(f16).
__device__ void deq_q2k(const uint8_t*b,float*o){
  const uint8_t*sc=b,*qs=b+16; float d=f16(b+80),dm=f16(b+82);
  for(int P=0;P<256;++P){int nh=P>>7,pp=P&127,j=pp>>5,rm=pp&31,g=rm>>4,l=rm&15;
    int q2=(qs[nh*32+g*16+l]>>(2*j))&3, is=nh*8+j*2+g;
    o[P]=d*(sc[is]&0xF)*(float)q2 - dm*(sc[is]>>4);}}

__global__ void k_q4k(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*144;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_q4k(r+(size_t)b*144,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}
__global__ void k_q3k(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*110;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_q3k(r+(size_t)b*110,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}
__global__ void k_q2k(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*84;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_q2k(r+(size_t)b*84,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}
__global__ void k_q6k(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*210;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_q6k(r+(size_t)b*210,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}
__global__ void k_q5k(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*176;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_q5k(r+(size_t)b*176,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}

// ---- Fast kernels: one warp (32 lanes) per output row. The 32 lanes
//      cooperatively stream each quant block with COALESCED global loads,
//      accumulate partial dot products in registers, then warp-shuffle reduce.
//      Same math as the scalar kernels; sum order differs (not bit-exact). ----
__device__ __forceinline__ float warp_reduce(float a){
  #pragma unroll
  for(int o=16;o>0;o>>=1) a += __shfl_down_sync(0xffffffffu, a, o);
  return a;
}

__global__ void k_q4k_w(const uint8_t* __restrict__ w, const float* __restrict__ x,
                        float* __restrict__ y, int in, int out){
  int row  = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;   // one warp per row
  int lane = threadIdx.x & 31;
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*144;
  const uint8_t* r = w + (size_t)row*bpr;
  float acc=0.f;
  for(int b=0;b<nb;++b){
    const uint8_t* blk = r + (size_t)b*144;
    float d=f16(blk), dm=f16(blk+2);
    const uint8_t* sc=blk+4; const uint8_t* qs=blk+16;
    int bs=b*256;
    #pragma unroll
    for(int t=0;t<4;++t){              // lane handles bytes lane, lane+32, +64, +96
      int j=lane+32*t;                 // 0..127  (coalesced read of qs[j])
      int g=j>>5, l=j&31;
      int s1,m1,s2,m2; gsm(sc,2*g,&s1,&m1); gsm(sc,2*g+1,&s2,&m2);
      uint8_t q=qs[j];
      acc += (d*s1*(q&0x0F) - dm*m1) * x[bs+64*g+l];
      acc += (d*s2*(q>>4)   - dm*m2) * x[bs+64*g+32+l];
    }
  }
  acc=warp_reduce(acc);
  if(lane==0) y[row]=acc;
}

__global__ void k_q6k_w(const uint8_t* __restrict__ w, const float* __restrict__ x,
                        float* __restrict__ y, int in, int out){
  int row  = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;         // l in 0..31
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*210;
  const uint8_t* r = w + (size_t)row*bpr;
  float acc=0.f;
  for(int b=0;b<nb;++b){
    const uint8_t* blk=r+(size_t)b*210;
    const uint8_t* ql=blk; const uint8_t* qh=blk+128;
    const int8_t* sc=(const int8_t*)(blk+192); float d=f16(blk+208);
    int bs=b*256, l=lane, is=l/16;
    #pragma unroll
    for(int h=0;h<2;++h){
      const uint8_t* QL=ql+h*64; const uint8_t* QH=qh+h*32; const int8_t* SC=sc+h*8;
      int base=bs+h*128;
      int q1=(int)((QL[l]&0xF)   |(((QH[l]>>0)&3)<<4))-32;
      int q2=(int)((QL[l+32]&0xF)|(((QH[l]>>2)&3)<<4))-32;
      int q3=(int)((QL[l]>>4)    |(((QH[l]>>4)&3)<<4))-32;
      int q4=(int)((QL[l+32]>>4) |(((QH[l]>>6)&3)<<4))-32;
      acc += d*SC[is+0]*q1 * x[base+l];
      acc += d*SC[is+2]*q2 * x[base+l+32];
      acc += d*SC[is+4]*q3 * x[base+l+64];
      acc += d*SC[is+6]*q4 * x[base+l+96];
    }
  }
  acc=warp_reduce(acc);
  if(lane==0) y[row]=acc;
}

__global__ void k_q5k_w(const uint8_t* __restrict__ w, const float* __restrict__ x,
                        float* __restrict__ y, int in, int out){
  int row  = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;          // L in 0..31
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*176;
  const uint8_t* r = w + (size_t)row*bpr;
  float acc=0.f;
  for(int b=0;b<nb;++b){
    const uint8_t* blk=r+(size_t)b*176;
    float d=f16(blk), dm=f16(blk+2);
    const uint8_t* sc=blk+4; const uint8_t* qh=blk+16; const uint8_t* qs=blk+48;
    int bs=b*256, L=lane; unsigned hbits=qh[L];
    #pragma unroll
    for(int g=0;g<4;++g){
      int s1,m1,s2,m2; gsm(sc,2*g,&s1,&m1); gsm(sc,2*g+1,&s2,&m2);
      unsigned q=qs[32*g+L];
      int lo=(q&0xF)+(((hbits>>(2*g))&1)<<4);
      int hi=(q>>4) +(((hbits>>(2*g+1))&1)<<4);
      acc += (d*s1*lo - dm*m1) * x[bs+64*g+L];
      acc += (d*s2*hi - dm*m2) * x[bs+64*g+32+L];
    }
  }
  acc=warp_reduce(acc);
  if(lane==0) y[row]=acc;
}

// ---- Batched warp-per-row kernels: X[B,in] @ W^T -> Y[B,out], row-major
//      (x[b*in+i], y[b*out+o]). One warp owns output row `o`: it reads that
//      row's quant weight ONCE and accumulates into B sequence sums, so the
//      weight-byte traffic of B tokens equals that of one (the batching win —
//      decode is weight-bandwidth-bound). Dequant math is done once per block;
//      only the multiply-add is per-sequence. Caps batch at MAXB. ----
#define MAXB 32
__global__ void k_q4k_wb(const uint8_t* __restrict__ w, const float* __restrict__ x,
                         float* __restrict__ y, int in, int out, int B){
  int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*144; const uint8_t* r=w+(size_t)row*bpr;
  float acc[MAXB];
  for(int b=0;b<B;++b) acc[b]=0.f;
  for(int blk=0;blk<nb;++blk){
    const uint8_t* bl=r+(size_t)blk*144;
    float d=f16(bl),dm=f16(bl+2); const uint8_t*sc=bl+4,*qs=bl+16; int bs=blk*256;
    #pragma unroll
    for(int t=0;t<4;++t){
      int j=lane+32*t,g=j>>5,l=j&31; int s1,m1,s2,m2;
      gsm(sc,2*g,&s1,&m1); gsm(sc,2*g+1,&s2,&m2);
      unsigned q=qs[j];
      float lo=d*s1*(q&0x0F)-dm*m1, hi=d*s2*(q>>4)-dm*m2;
      int ilo=bs+64*g+l, ihi=bs+64*g+32+l;
      for(int b=0;b<B;++b) acc[b]+=lo*x[b*in+ilo]+hi*x[b*in+ihi];
    }
  }
  for(int b=0;b<B;++b){ float a=warp_reduce(acc[b]); if(lane==0) y[b*out+row]=a; }
}

__global__ void k_q6k_wb(const uint8_t* __restrict__ w, const float* __restrict__ x,
                         float* __restrict__ y, int in, int out, int B){
  int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*210; const uint8_t* r=w+(size_t)row*bpr;
  float acc[MAXB];
  for(int b=0;b<B;++b) acc[b]=0.f;
  for(int blk=0;blk<nb;++blk){
    const uint8_t* bl=r+(size_t)blk*210;
    const uint8_t*ql=bl; const uint8_t*qh=bl+128; const int8_t*sc=(const int8_t*)(bl+192);
    float d=f16(bl+208); int bs=blk*256,l=lane,is=l/16;
    #pragma unroll
    for(int h=0;h<2;++h){
      const uint8_t*QL=ql+h*64; const uint8_t*QH=qh+h*32; const int8_t*SC=sc+h*8; int base=bs+h*128;
      int q1=(int)((QL[l]&0xF)|(((QH[l]>>0)&3)<<4))-32, q2=(int)((QL[l+32]&0xF)|(((QH[l]>>2)&3)<<4))-32;
      int q3=(int)((QL[l]>>4)|(((QH[l]>>4)&3)<<4))-32,  q4=(int)((QL[l+32]>>4)|(((QH[l]>>6)&3)<<4))-32;
      float w1=d*SC[is+0]*q1,w2=d*SC[is+2]*q2,w3=d*SC[is+4]*q3,w4=d*SC[is+6]*q4;
      int i1=base+l,i2=base+l+32,i3=base+l+64,i4=base+l+96;
      for(int b=0;b<B;++b){const float*xb=x+b*in; acc[b]+=w1*xb[i1]+w2*xb[i2]+w3*xb[i3]+w4*xb[i4];}
    }
  }
  for(int b=0;b<B;++b){ float a=warp_reduce(acc[b]); if(lane==0) y[b*out+row]=a; }
}

__global__ void k_q5k_wb(const uint8_t* __restrict__ w, const float* __restrict__ x,
                         float* __restrict__ y, int in, int out, int B){
  int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*176; const uint8_t* r=w+(size_t)row*bpr;
  float acc[MAXB];
  for(int b=0;b<B;++b) acc[b]=0.f;
  for(int blk=0;blk<nb;++blk){
    const uint8_t* bl=r+(size_t)blk*176;
    float d=f16(bl),dm=f16(bl+2); const uint8_t*sc=bl+4,*qh=bl+16,*qs=bl+48;
    int bs=blk*256,L=lane; unsigned hbits=qh[L];
    #pragma unroll
    for(int g=0;g<4;++g){
      int s1,m1,s2,m2; gsm(sc,2*g,&s1,&m1); gsm(sc,2*g+1,&s2,&m2);
      unsigned q=qs[32*g+L];
      int lo=(q&0xF)+(((hbits>>(2*g))&1)<<4), hi=(q>>4)+(((hbits>>(2*g+1))&1)<<4);
      float w1=d*s1*lo-dm*m1, w2=d*s2*hi-dm*m2; int i1=bs+64*g+L, i2=bs+64*g+32+L;
      for(int b=0;b<B;++b){const float*xb=x+b*in; acc[b]+=w1*xb[i1]+w2*xb[i2];}
    }
  }
  for(int b=0;b<B;++b){ float a=warp_reduce(acc[b]); if(lane==0) y[b*out+row]=a; }
}

// ---- Q3_K / Q2_K warp + batched kernels. lane (0..31) == the qs byte index
//      within a 128-half (= g*16+l, g=lane/16); each lane decodes its 4 j-fields
//      (j=0..3) for both halves and accumulates. Coalesced qs/hmask reads. ----
__global__ void k_q3k_w(const uint8_t* __restrict__ w, const float* __restrict__ x,
                        float* __restrict__ y, int in, int out){
  int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*110; const uint8_t* r=w+(size_t)row*bpr;
  int g=lane>>4; float acc=0.f;
  for(int blk=0;blk<nb;++blk){
    const uint8_t* bl=r+(size_t)blk*110; const uint8_t*hm=bl,*qs=bl+32,*sc=bl+96;
    float d=f16(bl+108); int sca[16]; q3k_scales(sc,sca);
    int bs=blk*256; unsigned hb=hm[lane];
    #pragma unroll
    for(int nh=0;nh<2;++nh){ unsigned qb=qs[nh*32+lane];
      #pragma unroll
      for(int j=0;j<4;++j){ int low2=(qb>>(2*j))&3, hbit=(hb>>(nh*4+j))&1, is=nh*8+j*2+g;
        acc += d*sca[is]*(float)(low2+4*hbit-4) * x[bs+nh*128+j*32+lane]; }
    }
  }
  acc=warp_reduce(acc); if(lane==0) y[row]=acc;
}
__global__ void k_q2k_w(const uint8_t* __restrict__ w, const float* __restrict__ x,
                        float* __restrict__ y, int in, int out){
  int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*84; const uint8_t* r=w+(size_t)row*bpr;
  int g=lane>>4; float acc=0.f;
  for(int blk=0;blk<nb;++blk){
    const uint8_t* bl=r+(size_t)blk*84; const uint8_t*sc=bl,*qs=bl+16;
    float d=f16(bl+80),dm=f16(bl+82); int bs=blk*256;
    #pragma unroll
    for(int nh=0;nh<2;++nh){ unsigned qb=qs[nh*32+lane];
      #pragma unroll
      for(int j=0;j<4;++j){ int q2=(qb>>(2*j))&3, is=nh*8+j*2+g;
        acc += (d*(sc[is]&0xF)*(float)q2 - dm*(sc[is]>>4)) * x[bs+nh*128+j*32+lane]; }
    }
  }
  acc=warp_reduce(acc); if(lane==0) y[row]=acc;
}
__global__ void k_q3k_wb(const uint8_t* __restrict__ w, const float* __restrict__ x,
                         float* __restrict__ y, int in, int out, int B){
  int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*110; const uint8_t* r=w+(size_t)row*bpr;
  int g=lane>>4; float acc[MAXB]; for(int b=0;b<B;++b) acc[b]=0.f;
  for(int blk=0;blk<nb;++blk){
    const uint8_t* bl=r+(size_t)blk*110; const uint8_t*hm=bl,*qs=bl+32,*sc=bl+96;
    float d=f16(bl+108); int sca[16]; q3k_scales(sc,sca);
    int bs=blk*256; unsigned hb=hm[lane];
    #pragma unroll
    for(int nh=0;nh<2;++nh){ unsigned qb=qs[nh*32+lane];
      #pragma unroll
      for(int j=0;j<4;++j){ int low2=(qb>>(2*j))&3, hbit=(hb>>(nh*4+j))&1, is=nh*8+j*2+g;
        float wv=d*sca[is]*(float)(low2+4*hbit-4); int i=bs+nh*128+j*32+lane;
        for(int b=0;b<B;++b) acc[b]+=wv*x[b*in+i]; }
    }
  }
  for(int b=0;b<B;++b){ float a=warp_reduce(acc[b]); if(lane==0) y[b*out+row]=a; }
}
__global__ void k_q2k_wb(const uint8_t* __restrict__ w, const float* __restrict__ x,
                         float* __restrict__ y, int in, int out, int B){
  int row=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31;
  if(row>=out) return;
  int nb=in/256; size_t bpr=(size_t)nb*84; const uint8_t* r=w+(size_t)row*bpr;
  int g=lane>>4; float acc[MAXB]; for(int b=0;b<B;++b) acc[b]=0.f;
  for(int blk=0;blk<nb;++blk){
    const uint8_t* bl=r+(size_t)blk*84; const uint8_t*sc=bl,*qs=bl+16;
    float d=f16(bl+80),dm=f16(bl+82); int bs=blk*256;
    #pragma unroll
    for(int nh=0;nh<2;++nh){ unsigned qb=qs[nh*32+lane];
      #pragma unroll
      for(int j=0;j<4;++j){ int q2=(qb>>(2*j))&3, is=nh*8+j*2+g;
        float wv=d*(sc[is]&0xF)*(float)q2 - dm*(sc[is]>>4); int i=bs+nh*128+j*32+lane;
        for(int b=0;b<B;++b) acc[b]+=wv*x[b*in+i]; }
    }
  }
  for(int b=0;b<B;++b){ float a=warp_reduce(acc[b]); if(lane==0) y[b*out+row]=a; }
}

// ---- Q8_0 (kind 5): 34-byte blocks of 32 values [f16 d | int8 qs[32]],
//      w[i] = d * qs[i]. Block size 32 == warp width -> one value per lane.
//      Matches the CPU oracle Dequant_Q8_0. Used by the prod model (Hura
//      Q8_0 served on ollama). ----
__global__ void k_q8_0_w(const uint8_t *__restrict__ w, const float *__restrict__ x,
                         float *__restrict__ y, int in, int out) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= out) return;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *r = w + (size_t) row * bpr; float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = r + (size_t) b * 34; float d = f16(blk);
        const int8_t *qs = (const int8_t *) (blk + 2);
        acc += d * (float) qs[lane] * x[b * 32 + lane];
    }
    acc = warp_reduce(acc); if (lane == 0) y[row] = acc;
}
__global__ void k_q8_0_wb(const uint8_t *__restrict__ w, const float *__restrict__ x,
                          float *__restrict__ y, int in, int out, int B) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= out) return;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *r = w + (size_t) row * bpr; float acc[MAXB];
    for (int b = 0; b < B; ++b) acc[b] = 0.f;
    for (int blk = 0; blk < nb; ++blk) {
        const uint8_t *bl = r + (size_t) blk * 34; float d = f16(bl);
        const int8_t *qs = (const int8_t *) (bl + 2);
        float wv = d * (float) qs[lane]; int i = blk * 32 + lane;
        for (int b = 0; b < B; ++b) acc[b] += wv * x[(size_t) b * in + i];
    }
    for (int b = 0; b < B; ++b) { float a = warp_reduce(acc[b]); if (lane == 0) y[(size_t) b * out + row] = a; }
}

//  Decode-regime variant: B is a COMPILE-TIME constant so acc[BT] uses exactly
//  BT registers instead of the fixed acc[MAXB] (32) the runtime kernel above
//  reserves regardless of B. At B<=8 (the batch-serve lane count) that array
//  destroys occupancy — measured 2.1-3.8x slower than this templated form at
//  the real Ornith attention-projection shapes on H200, bit-identical output
//  (NMSE 2.6e-14). Instantiated for the decode lane counts and dispatched by
//  launch_mv_b's small-B branch; larger B keeps the tensor-core / runtime paths.
template <int BT>
__global__ void k_q8_0_wb_T(const uint8_t *__restrict__ w, const float *__restrict__ x,
                            float *__restrict__ y, int in, int out) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= out) return;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *r = w + (size_t) row * bpr; float acc[BT];
    #pragma unroll
    for (int b = 0; b < BT; ++b) acc[b] = 0.f;
    for (int blk = 0; blk < nb; ++blk) {
        const uint8_t *bl = r + (size_t) blk * 34; float d = f16(bl);
        const int8_t *qs = (const int8_t *) (bl + 2);
        float wv = d * (float) qs[lane]; int i = blk * 32 + lane;
        #pragma unroll
        for (int b = 0; b < BT; ++b) acc[b] += wv * x[(size_t) b * in + i];
    }
    #pragma unroll
    for (int b = 0; b < BT; ++b) { float a = warp_reduce(acc[b]); if (lane == 0) y[(size_t) b * out + row] = a; }
}

#include <cstdlib>

//  Shared weight VRAM cache: each distinct host weight pointer is uploaded
//  once and reused by both matvec and matmul (same model weights).
static std::unordered_map<const void *, uint8_t *> g_wcache;
static uint8_t *upload_weight(const void *w, long wbytes) {
    auto it = g_wcache.find(w);
    if (it != g_wcache.end()) return it->second;
    uint8_t *dw; cudaMalloc(&dw, wbytes);
    cudaMemcpy(dw, w, wbytes, cudaMemcpyHostToDevice);
    g_wcache[w] = dw; return dw;
}

//  Phase 1b eviction: drop the VRAM mirror of one host weight pointer. Called
//  by the engine when a model is unloaded, BEFORE its host bytes are freed (so
//  the pointer is still a valid cache key). Without this the device buffer
//  leaks AND a later model whose host bytes are reallocated at the same address
//  would be served this model's stale weights. No-op if w was never uploaded.
extern "C" void aspida_gpu_free_weight(const void *w) {
    aspida_ggml_free_weight(w);   // MoE Phase B: evict ggml-owned weights too
    auto it = g_wcache.find(w);
    if (it == g_wcache.end()) return;
    cudaFree(it->second);
    g_wcache.erase(it);
}

extern "C" void aspida_gpu_matvec(const void *w, long wbytes, int kind,
                                  int in_dim, int out_dim, const float *x, float *y) {
    static float *dx = nullptr, *dy = nullptr; static long cx = 0, cy = 0;
    static int scalar = (getenv("ASPIDA_GPU_SCALAR") != nullptr) ? 1 : 0;

    uint8_t *dw = upload_weight(w, wbytes);
    if (in_dim > cx)  { if (dx) cudaFree(dx); cudaMalloc(&dx, (size_t) in_dim * 4);  cx = in_dim; }
    if (out_dim > cy) { if (dy) cudaFree(dy); cudaMalloc(&dy, (size_t) out_dim * 4); cy = out_dim; }
    cudaMemcpy(dx, x, (size_t) in_dim * 4, cudaMemcpyHostToDevice);
    if (scalar) {                            // legacy one-thread-per-row path
        int g = (out_dim + 127) / 128;
        if (kind == 0)      k_q4k<<<g, 128>>>(dw, dx, dy, in_dim, out_dim);
        else if (kind == 1) k_q6k<<<g, 128>>>(dw, dx, dy, in_dim, out_dim);
        else if (kind == 2) k_q5k<<<g, 128>>>(dw, dx, dy, in_dim, out_dim);
        else if (kind == 3) k_q3k<<<g, 128>>>(dw, dx, dy, in_dim, out_dim);
        else                k_q2k<<<g, 128>>>(dw, dx, dy, in_dim, out_dim);
    } else {                                 // fast warp-per-row path (default)
        const int TPB = 256, WPB = TPB / 32; // 8 warps (=rows) per block
        int blocks = (out_dim + WPB - 1) / WPB;
        if (kind == 0)      k_q4k_w<<<blocks, TPB>>>(dw, dx, dy, in_dim, out_dim);
        else if (kind == 1) k_q6k_w<<<blocks, TPB>>>(dw, dx, dy, in_dim, out_dim);
        else if (kind == 2) k_q5k_w<<<blocks, TPB>>>(dw, dx, dy, in_dim, out_dim);
        else if (kind == 3) k_q3k_w<<<blocks, TPB>>>(dw, dx, dy, in_dim, out_dim);
        else                k_q2k_w<<<blocks, TPB>>>(dw, dx, dy, in_dim, out_dim);
    }
    // D2H cudaMemcpy on the default stream blocks until the kernel finishes,
    // so the explicit device sync was redundant — dropped (one less stall/call).
    cudaMemcpy(y, dy, (size_t) out_dim * 4, cudaMemcpyDeviceToHost);
}

//  Batched: X[B,in] @ W -> Y[B,out], row-major. Weight read once per row,
//  reused across B (the continuous-batching throughput win). batch <= MAXB.
extern "C" void aspida_gpu_matmul(const void *w, long wbytes, int kind,
                                  int in_dim, int out_dim, int batch,
                                  const float *x, float *y) {
    static float *dxb = nullptr, *dyb = nullptr; static long cxb = 0, cyb = 0;
    if (batch < 1) return;
    if (batch > MAXB) batch = MAXB;          // caller must keep B <= MAXB
    uint8_t *dw = upload_weight(w, wbytes);
    long nx = (long) in_dim * batch, ny = (long) out_dim * batch;
    if (nx > cxb) { if (dxb) cudaFree(dxb); cudaMalloc(&dxb, (size_t) nx * 4); cxb = nx; }
    if (ny > cyb) { if (dyb) cudaFree(dyb); cudaMalloc(&dyb, (size_t) ny * 4); cyb = ny; }
    cudaMemcpy(dxb, x, (size_t) nx * 4, cudaMemcpyHostToDevice);
    const int TPB = 256, WPB = TPB / 32;
    int blocks = (out_dim + WPB - 1) / WPB;
    if (kind == 0)      k_q4k_wb<<<blocks, TPB>>>(dw, dxb, dyb, in_dim, out_dim, batch);
    else if (kind == 1) k_q6k_wb<<<blocks, TPB>>>(dw, dxb, dyb, in_dim, out_dim, batch);
    else if (kind == 2) k_q5k_wb<<<blocks, TPB>>>(dw, dxb, dyb, in_dim, out_dim, batch);
    else if (kind == 3) k_q3k_wb<<<blocks, TPB>>>(dw, dxb, dyb, in_dim, out_dim, batch);
    else                k_q2k_wb<<<blocks, TPB>>>(dw, dxb, dyb, in_dim, out_dim, batch);
    cudaMemcpy(y, dyb, (size_t) ny * 4, cudaMemcpyDeviceToHost);
}

// =====================================================================
// Increment 1 — fused resident MoE FFN decode (declared in qwen_resident.cu;
// implemented here where the K-quant warp kernels + g_wcache live).
// Router GEMV -> softmax/top-k (on host, matching
// LLM_MoE.Forward) -> K SwiGLU experts -> weighted combine -> shared expert,
// every matvec on RESIDENT device buffers. Per MoE layer: one H2D of x, one
// tiny D2H of the router logits, one D2H of the result — vs the 28 activation
// round-trips of the per-matvec path. Same warp kernels => matches the
// per-matvec GPU MoE to fp precision; the softmax/top-k/SwiGLU/gate math
// mirrors LLM_MoE exactly.
// =====================================================================
#include <cmath>

__global__ void k_swiglu(const float *__restrict__ g, const float *__restrict__ u,
                         float *__restrict__ h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float a = g[i]; h[i] = (a / (1.0f + expf(-a))) * u[i];   // silu(gate)*up
}
__global__ void k_axpy(float *__restrict__ acc, float w,
                       const float *__restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    acc[i] += w * y[i];
}

// ---- Split-K: ONE block per output row, WARPS warps splitting the block loop
//      so many more warps are resident (the ncu fix: the tiny matvecs were
//      occupancy-bound, ~64 warps of 142 SMs' worth). Each warp strides the
//      quant blocks, warp-reduces its partial, then warp 0 sums the WARPS
//      partials. out blocks => far higher occupancy for small `out`. ----
#define SKW 8
__global__ void k_q4k_sk(const uint8_t *__restrict__ w, const float *__restrict__ x,
                         float *__restrict__ y, int in, int out) {
    int row = blockIdx.x; if (row >= out) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, WARPS = blockDim.x >> 5;
    int nb = in / 256; size_t bpr = (size_t) nb * 144;
    const uint8_t *r = w + (size_t) row * bpr;
    float acc = 0.f;
    for (int b = warp; b < nb; b += WARPS) {
        const uint8_t *blk = r + (size_t) b * 144;
        float d = f16(blk), dm = f16(blk + 2);
        const uint8_t *sc = blk + 4; const uint8_t *qs = blk + 16; int bs = b * 256;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int j = lane + 32 * t, g = j >> 5, l = j & 31;
            int s1, m1, s2, m2; gsm(sc, 2 * g, &s1, &m1); gsm(sc, 2 * g + 1, &s2, &m2);
            uint8_t q = qs[j];
            acc += (d * s1 * (q & 0x0F) - dm * m1) * x[bs + 64 * g + l];
            acc += (d * s2 * (q >> 4)   - dm * m2) * x[bs + 64 * g + 32 + l];
        }
    }
    acc = warp_reduce(acc);
    __shared__ float part[32];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) { float sacc = 0.f; for (int i = 0; i < WARPS; ++i) sacc += part[i]; y[row] = sacc; }
}
__global__ void k_q6k_sk(const uint8_t *__restrict__ w, const float *__restrict__ x,
                         float *__restrict__ y, int in, int out) {
    int row = blockIdx.x; if (row >= out) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, WARPS = blockDim.x >> 5;
    int nb = in / 256; size_t bpr = (size_t) nb * 210;
    const uint8_t *r = w + (size_t) row * bpr; int l = lane, is = l / 16;
    float acc = 0.f;
    for (int b = warp; b < nb; b += WARPS) {
        const uint8_t *blk = r + (size_t) b * 210;
        const uint8_t *ql = blk; const uint8_t *qh = blk + 128;
        const int8_t *sc = (const int8_t *) (blk + 192); float d = f16(blk + 208); int bs = b * 256;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint8_t *QL = ql + h * 64; const uint8_t *QH = qh + h * 32;
            const int8_t *SC = sc + h * 8; int base = bs + h * 128;
            int q1 = (int) ((QL[l] & 0xF)      | (((QH[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int) ((QL[l + 32] & 0xF) | (((QH[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int) ((QL[l] >> 4)       | (((QH[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int) ((QL[l + 32] >> 4)  | (((QH[l] >> 6) & 3) << 4)) - 32;
            acc += d * SC[is + 0] * q1 * x[base + l];
            acc += d * SC[is + 2] * q2 * x[base + l + 32];
            acc += d * SC[is + 4] * q3 * x[base + l + 64];
            acc += d * SC[is + 6] * q4 * x[base + l + 96];
        }
    }
    acc = warp_reduce(acc);
    __shared__ float part[32];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) { float sacc = 0.f; for (int i = 0; i < WARPS; ++i) sacc += part[i]; y[row] = sacc; }
}

__global__ void k_q8_0_sk(const uint8_t *__restrict__ w, const float *__restrict__ x,
                         float *__restrict__ y, int in, int out) {
    int row = blockIdx.x; if (row >= out) return;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, WARPS = blockDim.x >> 5;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *r = w + (size_t) row * bpr; float acc = 0.f;
    for (int b = warp; b < nb; b += WARPS) {
        const uint8_t *blk = r + (size_t) b * 34; float d = f16(blk);
        const int8_t *qs = (const int8_t *) (blk + 2);
        acc += d * (float) qs[lane] * x[b * 32 + lane];
    }
    acc = warp_reduce(acc);
    __shared__ float part[32];
    if (lane == 0) part[warp] = acc; __syncthreads();
    if (threadIdx.x == 0) { float sacc = 0.f; for (int i = 0; i < WARPS; ++i) sacc += part[i]; y[row] = sacc; }
}

// Launch the right warp-per-row K-quant matvec into a DEVICE buffer:
//   y[out] = W[out,in] . x[in].  No host copy — dx/dy are device-resident.
static inline void launch_matvec(const uint8_t *dw, int kind, int in, int out,
                                 const float *dx, float *dy) {
    const int TPB = 256, WPB = TPB / 32; int blocks = (out + WPB - 1) / WPB;
    if (kind == 0)      k_q4k_sk<<<out, SKW * 32>>>(dw, dx, dy, in, out);
    else if (kind == 1) k_q6k_sk<<<out, SKW * 32>>>(dw, dx, dy, in, out);
    else if (kind == 2) k_q5k_w<<<blocks, TPB>>>(dw, dx, dy, in, out);
    else if (kind == 3) k_q3k_w<<<blocks, TPB>>>(dw, dx, dy, in, out);
    else if (kind == 5) k_q8_0_sk<<<out, SKW * 32>>>(dw, dx, dy, in, out);
    else                k_q2k_w<<<blocks, TPB>>>(dw, dx, dy, in, out);
}

// ---- Warp-cooperative single-row quant dot products (device functions).
//      Same math/lane layout as the k_*_w kernels; every lane returns the
//      reduced value. `r` points at the row's first quant block, `in` is the
//      row length. Used by the fused MoE kernels (one warp = one (row,expert)).
__device__ __forceinline__ float wrow_q4k(const uint8_t *r, const float *x, int in, int lane) {
  int nb = in / 256; float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const uint8_t *blk = r + (size_t) b * 144;
    float d = f16(blk), dm = f16(blk + 2);
    const uint8_t *sc = blk + 4; const uint8_t *qs = blk + 16;
    int bs = b * 256;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
      int j = lane + 32 * t, g = j >> 5, l = j & 31;
      int s1, m1, s2, m2; gsm(sc, 2 * g, &s1, &m1); gsm(sc, 2 * g + 1, &s2, &m2);
      uint8_t q = qs[j];
      acc += (d * s1 * (q & 0x0F) - dm * m1) * x[bs + 64 * g + l];
      acc += (d * s2 * (q >> 4)   - dm * m2) * x[bs + 64 * g + 32 + l];
    }
  }
  acc = warp_reduce(acc);
  return __shfl_sync(0xffffffffu, acc, 0);
}
__device__ __forceinline__ float wrow_q6k(const uint8_t *r, const float *x, int in, int lane) {
  int nb = in / 256; float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const uint8_t *blk = r + (size_t) b * 210;
    const uint8_t *ql = blk; const uint8_t *qh = blk + 128;
    const int8_t *sc = (const int8_t *) (blk + 192); float d = f16(blk + 208);
    int bs = b * 256, l = lane, is = l / 16;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint8_t *QL = ql + h * 64; const uint8_t *QH = qh + h * 32;
      const int8_t *SC = sc + h * 8; int base = bs + h * 128;
      int q1 = (int) ((QL[l] & 0xF)      | (((QH[l] >> 0) & 3) << 4)) - 32;
      int q2 = (int) ((QL[l + 32] & 0xF) | (((QH[l] >> 2) & 3) << 4)) - 32;
      int q3 = (int) ((QL[l] >> 4)       | (((QH[l] >> 4) & 3) << 4)) - 32;
      int q4 = (int) ((QL[l + 32] >> 4)  | (((QH[l] >> 6) & 3) << 4)) - 32;
      acc += d * SC[is + 0] * q1 * x[base + l];
      acc += d * SC[is + 2] * q2 * x[base + l + 32];
      acc += d * SC[is + 4] * q3 * x[base + l + 64];
      acc += d * SC[is + 6] * q4 * x[base + l + 96];
    }
  }
  acc = warp_reduce(acc);
  return __shfl_sync(0xffffffffu, acc, 0);
}
__device__ __forceinline__ float wrow_q5k(const uint8_t *r, const float *x, int in, int lane) {
  int nb = in / 256; float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const uint8_t *blk = r + (size_t) b * 176;
    float d = f16(blk), dm = f16(blk + 2);
    const uint8_t *sc = blk + 4; const uint8_t *qh = blk + 16; const uint8_t *qs = blk + 48;
    int bs = b * 256, L = lane; unsigned hbits = qh[L];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
      int s1, m1, s2, m2; gsm(sc, 2 * g, &s1, &m1); gsm(sc, 2 * g + 1, &s2, &m2);
      unsigned q = qs[32 * g + L];
      int lo = (q & 0xF) + (((hbits >> (2 * g)) & 1) << 4);
      int hi = (q >> 4)  + (((hbits >> (2 * g + 1)) & 1) << 4);
      acc += (d * s1 * lo - dm * m1) * x[bs + 64 * g + L];
      acc += (d * s2 * hi - dm * m2) * x[bs + 64 * g + 32 + L];
    }
  }
  acc = warp_reduce(acc);
  return __shfl_sync(0xffffffffu, acc, 0);
}
__device__ __forceinline__ float wrow_q3k(const uint8_t *r, const float *x, int in, int lane) {
  int nb = in / 256; int g = lane >> 4; float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const uint8_t *bl = r + (size_t) b * 110; const uint8_t *hm = bl, *qs = bl + 32, *sc = bl + 96;
    float d = f16(bl + 108); int sca[16]; q3k_scales(sc, sca);
    int bs = b * 256; unsigned hb = hm[lane];
    #pragma unroll
    for (int nh = 0; nh < 2; ++nh) { unsigned qb = qs[nh * 32 + lane];
      #pragma unroll
      for (int j = 0; j < 4; ++j) { int low2 = (qb >> (2 * j)) & 3, hbit = (hb >> (nh * 4 + j)) & 1, is = nh * 8 + j * 2 + g;
        acc += d * sca[is] * (float) (low2 + 4 * hbit - 4) * x[bs + nh * 128 + j * 32 + lane]; }
    }
  }
  acc = warp_reduce(acc);
  return __shfl_sync(0xffffffffu, acc, 0);
}
__device__ __forceinline__ float wrow_q2k(const uint8_t *r, const float *x, int in, int lane) {
  int nb = in / 256; int g = lane >> 4; float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const uint8_t *bl = r + (size_t) b * 84; const uint8_t *sc = bl, *qs = bl + 16;
    float d = f16(bl + 80), dm = f16(bl + 82); int bs = b * 256;
    #pragma unroll
    for (int nh = 0; nh < 2; ++nh) { unsigned qb = qs[nh * 32 + lane];
      #pragma unroll
      for (int j = 0; j < 4; ++j) { int q2 = (qb >> (2 * j)) & 3, is = nh * 8 + j * 2 + g;
        acc += (d * (sc[is] & 0xF) * (float) q2 - dm * (sc[is] >> 4)) * x[bs + nh * 128 + j * 32 + lane]; }
    }
  }
  acc = warp_reduce(acc);
  return __shfl_sync(0xffffffffu, acc, 0);
}
__device__ __forceinline__ float wrow_q8_0(const uint8_t *r, const float *x, int in, int lane) {
  int nb = in / 32; float acc = 0.f;
  for (int b = 0; b < nb; ++b) {
    const uint8_t *blk = r + (size_t) b * 34; float d = f16(blk);
    const int8_t *qs = (const int8_t *) (blk + 2);
    acc += d * (float) qs[lane] * x[b * 32 + lane];
  }
  acc = warp_reduce(acc);
  return __shfl_sync(0xffffffffu, acc, 0);
}
// cp.async double-buffered Q8_0 row dot (decode MoE lever). Stages aligned
// 8-block (272B) chunks into a per-warp shared buffer one chunk ahead, so the
// next chunk's HBM read overlaps the current chunk's dequant+FMA. cp.async
// needs 16B-aligned src+size: a single 34B block is NOT aligned, but a row
// start and an 8-block 272B chunk ARE (272 % 16 == 0), copied as 17x16B. Math
// is bit-exact to wrow_q8_0 (same d/qs/x, same ascending-block order). ~1.23x
// on decode MoE at real dims (nb divisible by 8). Requires stg >= 2*272B/warp.
#define MOE_ASYNC_CHUNK_BLK 8
#define MOE_ASYNC_CHUNK_B  (MOE_ASYNC_CHUNK_BLK * 34)   /* 272, 16B-aligned */
#define MOE_ASYNC_CHUNK_16 (MOE_ASYNC_CHUNK_B / 16)     /* 17 */
#define MOE_ASYNC_SHMEM(tpb) ((size_t) ((tpb) / 32) * (2 * MOE_ASYNC_CHUNK_B))
__device__ __forceinline__ float wrow_q8_async(const uint8_t *r, const float *x, int in,
                                               int lane, uint8_t *stg) {
  int nb = in / 32, nch = nb / MOE_ASYNC_CHUNK_BLK; float acc = 0.f;
  if (lane < MOE_ASYNC_CHUNK_16) __pipeline_memcpy_async(stg + lane * 16, r + lane * 16, 16);
  __pipeline_commit();
  for (int c = 0; c < nch; ++c) {
    int cur = (c & 1) * MOE_ASYNC_CHUNK_B, nxt = ((c + 1) & 1) * MOE_ASYNC_CHUNK_B;
    if (c + 1 < nch) {
      if (lane < MOE_ASYNC_CHUNK_16)
        __pipeline_memcpy_async(stg + nxt + lane * 16,
                                r + (size_t) (c + 1) * MOE_ASYNC_CHUNK_B + lane * 16, 16);
      __pipeline_commit();
    }
    __pipeline_wait_prior(c + 1 < nch ? 1 : 0);
    __syncwarp();
    const uint8_t *base = stg + cur;
    #pragma unroll
    for (int bb = 0; bb < MOE_ASYNC_CHUNK_BLK; ++bb) {
      const uint8_t *blk = base + bb * 34; float d = f16(blk);
      const int8_t *qs = (const int8_t *) (blk + 2);
      acc += d * (float) qs[lane] * x[(c * MOE_ASYNC_CHUNK_BLK + bb) * 32 + lane];
    }
    __syncwarp();
  }
  acc = warp_reduce(acc);
  return __shfl_sync(0xffffffffu, acc, 0);
}
// Bytes per 256-value super-block, by kind code.
__host__ __device__ __forceinline__ int kq_bpb(int kind) {
  return kind == 0 ? 144 : kind == 1 ? 210 : kind == 2 ? 176 : kind == 3 ? 110 : kind == 5 ? 272 : 84;
}
// Warp-cooperative dot of one quant row (branch is warp-uniform).
__device__ __forceinline__ float wrow(const uint8_t *r, int kind, const float *x, int in, int lane) {
  switch (kind) {
    case 0:  return wrow_q4k(r, x, in, lane);
    case 1:  return wrow_q6k(r, x, in, lane);
    case 2:  return wrow_q5k(r, x, in, lane);
    case 3:  return wrow_q3k(r, x, in, lane);
    case 5:  return wrow_q8_0(r, x, in, lane);
    default: return wrow_q2k(r, x, in, lane);
  }
}
// Same as wrow, but for a chunk-aligned Q8_0 row (the decode MoE case) it takes
// the cp.async double-buffered path through the per-warp shared `stg`. Any
// other kind or a non-8-block-divisible length falls back to the direct dot,
// so it is a safe drop-in for the generic MoE kernels.
__device__ __forceinline__ float wrow_maybe_async(const uint8_t *r, int kind, const float *x,
                                                  int in, int lane, uint8_t *stg) {
  if (kind == 5 && (in / 32) % MOE_ASYNC_CHUNK_BLK == 0)
    return wrow_q8_async(r, x, in, lane, stg);
  return wrow(r, kind, x, in, lane);
}

// Expert routing packed into kernel params (no extra H2D): idx[k] = expert id,
// w[k] = combine weight; slot MOE_MAXK holds the shared expert's gate weight.
#define MOE_MAXK 8
struct MoeRoute { int idx[MOE_MAXK]; float w[MOE_MAXK + 1]; };

// Fused kernel 1: h[k][r] = silu(gate_k_row_r . x) * (up_k_row_r . x) for the
// top_k routed experts AND the shared expert (slot top_k). One warp per
// (expert, row); both dots share the same x reads. Replaces 18 launches.
__global__ void k_moe_gu(const uint8_t *__restrict__ gdw, const uint8_t *__restrict__ udw,
                         const uint8_t *__restrict__ sgdw, const uint8_t *__restrict__ sudw,
                         const float *__restrict__ x, float *__restrict__ h,
                         MoeRoute route, int top_k, int dim, int intermed,
                         long g_bpe, long u_bpe, int gk, int uk, int sgk, int suk) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int total = (top_k + 1) * intermed;
    if (wid >= total) return;
    int k = wid / intermed, r = wid % intermed;
    const uint8_t *grow, *urow; int gkind, ukind;
    if (k < top_k) {
        int e = route.idx[k];
        grow = gdw + (size_t) e * g_bpe + (size_t) r * (dim / 256) * kq_bpb(gk);
        urow = udw + (size_t) e * u_bpe + (size_t) r * (dim / 256) * kq_bpb(uk);
        gkind = gk; ukind = uk;
    } else {                                  // shared expert
        grow = sgdw + (size_t) r * (dim / 256) * kq_bpb(sgk);
        urow = sudw + (size_t) r * (dim / 256) * kq_bpb(suk);
        gkind = sgk; ukind = suk;
    }
    float g = wrow(grow, gkind, x, dim, lane);
    float u = wrow(urow, ukind, x, dim, lane);
    if (lane == 0)
        h[(size_t) k * intermed + r] = (g / (1.f + expf(-g))) * u;
}

// Fused kernel 2: y[i] = sum_k route.w[k] * (down_k_row_i . h[k]) + the shared
// expert (slot top_k, weight route.w[MOE_MAXK]). One warp per output row i,
// looping the experts inside — single launch, single write. Replaces 12.
__global__ void k_moe_down(const uint8_t *__restrict__ ddw, const uint8_t *__restrict__ sddw,
                           const float *__restrict__ h, float *__restrict__ y,
                           MoeRoute route, int top_k, int intermed, int dim,
                           long d_bpe, int dk, int sdk) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= dim) return;
    size_t bpr_d = (size_t) (intermed / 256) * kq_bpb(dk);
    size_t bpr_s = (size_t) (intermed / 256) * kq_bpb(sdk);
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        const uint8_t *row = ddw + (size_t) route.idx[k] * d_bpe + (size_t) wid * bpr_d;
        acc += route.w[k] * wrow(row, dk, h + (size_t) k * intermed, intermed, lane);
    }
    acc += route.w[MOE_MAXK]
           * wrow(sddw + (size_t) wid * bpr_s, sdk, h + (size_t) top_k * intermed, intermed, lane);
    if (lane == 0) y[wid] = acc;
}

// Dense F32 matvec (the LM head + token-embedding endpoints bypass the K-quant
// path). y[out] = W[out,in] . x[in], W row-major dense float. Warp-per-row,
// weight cached resident by host pointer (same g_wcache). One H2D + one D2H.
__global__ void k_dense_mv(const float *__restrict__ w, const float *__restrict__ x,
                           float *__restrict__ y, int in, int out) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= out) return;
    const float *r = w + (size_t) row * in;
    float acc = 0.f;
    if ((in & 3) == 0 && (((uintptr_t) r) & 15) == 0) {
        //  Vectorized: 16B loads saturate DRAM far better than 4B strides —
        //  this is the LM-head hot path (a [vocab, dim] read every token).
        const float4 *r4 = (const float4 *) r;
        const float4 *x4 = (const float4 *) x;
        int n4 = in >> 2;
        for (int i = lane; i < n4; i += 32) {
            float4 a = r4[i], b = x4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = lane; i < in; i += 32) acc += r[i] * x[i];
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}
extern "C" void aspida_gpu_dense_matvec(const void *w, long wbytes, int in_dim,
                                        int out_dim, const float *x, float *y) {
    static float *dx = nullptr, *dy = nullptr; static long cx = 0, cy = 0;
    uint8_t *dw = upload_weight(w, wbytes);
    if (in_dim > cx)  { if (dx) cudaFree(dx); cudaMalloc(&dx, (size_t) in_dim * 4);  cx = in_dim; }
    if (out_dim > cy) { if (dy) cudaFree(dy); cudaMalloc(&dy, (size_t) out_dim * 4); cy = out_dim; }
    cudaMemcpy(dx, x, (size_t) in_dim * 4, cudaMemcpyHostToDevice);
    const int TPB = 256, WPB = TPB / 32;
    k_dense_mv<<<(out_dim + WPB - 1) / WPB, TPB>>>((const float *) dw, dx, dy, in_dim, out_dim);
    cudaMemcpy(y, dy, (size_t) out_dim * 4, cudaMemcpyDeviceToHost);
}

// =====================================================================
// Increment 2 — resident delta-net per-head recurrence + gated RMSNorm.
// Oracle: LLM_DeltaNet.Step + the gated-norm in LLM_DeltaNet_Blk.Step. The
// per-head recurrent state S_All stays RESIDENT on the device across tokens
// (allocated per layer via aspida_gpu_dnet_new, updated in place here). Only
// the small per-token vectors move host<->device (cq/gate/beta/z in, o_row out).
//
// One block per head; blockDim = khd (== vhd). Each thread owns column v of
// this head's state block and loops k ASCENDING — the same order as the CPU
// sums (Retr, output, L2/RMS), so results track the CPU oracle to fp precision.
// Key_Head_Dim == Value_Head_Dim (set in Create), so one thread index serves
// as both the k and v index.
// =====================================================================
// Per-layer resident delta-net state: the recurrent S_All AND the causal-conv
// history window both live on the device across tokens.
struct DnetState { float *S; float *hist; int qo; int kernel; size_t sn; size_t hn; };
static std::vector<DnetState> g_dnet;
static std::vector<int> g_dnet_free;      // freed slots for reuse

//  Stream-ordered pool allocator for per-generation KV/recurrent state. The
//  old cudaMalloc/cudaFree here were SYNCHRONOUS + DEVICE-WIDE: every request
//  start/finish drained the entire GPU, stalling all in-flight decode/prefill
//  lanes — the dominant cause of the concurrent-load blowup (1.3s -> 38s).
//  cudaMallocAsync pulls from a retained pool (reuses freed blocks, no OS
//  round-trip, no device drain) and is stream-ordered; one sync on the
//  dedicated alloc stream makes the block usable on any stream. Frees are
//  stream-ordered too — safe because Free_States runs at teardown, after the
//  generation's forwards have already synced their streams (state is idle).
static cudaStream_t g_astream = nullptr;
static void ensure_astream() {
    if (g_astream) return;
    cudaStreamCreate(&g_astream);
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
        uint64_t thr = ~0ull;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
    }
}

extern "C" int aspida_gpu_dnet_new(int nv, int khd, int vhd, int qo, int kernel) {
    ensure_astream();
    DnetState st; size_t n = (size_t) nv * khd * vhd;
    if (cudaMallocAsync(&st.S, n * 4, g_astream) != cudaSuccess) return -1;
    cudaMemsetAsync(st.S, 0, n * 4, g_astream);
    size_t hn = (size_t) (kernel > 1 ? kernel - 1 : 1) * qo;
    if (cudaMallocAsync(&st.hist, hn * 4, g_astream) != cudaSuccess) { cudaFreeAsync(st.S, g_astream); cudaStreamSynchronize(g_astream); return -1; }
    cudaMemsetAsync(st.hist, 0, hn * 4, g_astream);
    cudaStreamSynchronize(g_astream);   // block + memset ready for use on any stream
    st.qo = qo; st.kernel = kernel; st.sn = n; st.hn = hn;
    if (!g_dnet_free.empty()) {
        int h = g_dnet_free.back(); g_dnet_free.pop_back();
        g_dnet[h] = st; return h;
    }
    g_dnet.push_back(st);
    return (int) g_dnet.size() - 1;
}

// States are per-generation (allocated in Init_State) — without this they
// leak ~8 MB VRAM per delta-net layer per request. Slot is reused by _new.
extern "C" void aspida_gpu_dnet_free(int handle) {
    if (handle < 0 || handle >= (int) g_dnet.size()) return;
    DnetState &st = g_dnet[handle];
    if (st.S)    { cudaFreeAsync(st.S, g_astream); st.S = nullptr; }
    if (st.hist) { cudaFreeAsync(st.hist, g_astream); st.hist = nullptr; }
    g_dnet_free.push_back(handle);
}

//  ---- Prefix-state snapshots (RETAINED across requests) --------------------
//  A prefix KV-cache lets a repeated request (identical system prompt) skip
//  re-prefilling the shared prefix: after prefilling it once we snapshot each
//  layer's device state into a retained slot, and on a later cache hit we
//  restore it into that request's fresh state and prefill only the suffix.
//  DeltaNet is a RECURRENT accumulator (unlike positional attention KV), so the
//  snapshot must be the full S matrix + conv history captured EXACTLY at the
//  end-of-prefix position. Snapshots live in their own pools and are NOT touched
//  by Free_States (per-request teardown) — only aspida_gpu_prefix_reset frees
//  them (on model change). Copies are device→device on the alloc stream and
//  synced, so the snapshot is a stable, byte-identical image of the state.
struct DnetSnap { float *S; float *hist; size_t sn; size_t hn; };
static std::vector<DnetSnap> g_dnet_snap;
static std::vector<int> g_dnet_snap_free;

//  Serialises all snapshot-pool access. Under BATCH_SERVE several handler tasks
//  prefill concurrently; without this a concurrent snapshot (vector push_back /
//  slot reuse) would race a restore reading the same vectors. Snapshots are
//  rare (once per distinct prefix) and restores are a few small device copies,
//  so the critical section is short relative to the seconds of decode it saves.
static std::mutex g_snap_mtx;

//  Snapshot delta-net state `handle` into slot `slot` (-1 = allocate a new
//  slot). Returns the slot id, or -1 on error. Reusing a slot overwrites it.
extern "C" int aspida_gpu_dnet_snapshot(int handle, int slot) {
    std::lock_guard<std::mutex> lk(g_snap_mtx);
    if (handle < 0 || handle >= (int) g_dnet.size()) return -1;
    DnetState &st = g_dnet[handle];
    if (!st.S || !st.hist) return -1;
    ensure_astream();
    DnetSnap sn;
    if (slot >= 0 && slot < (int) g_dnet_snap.size()
        && g_dnet_snap[slot].S != nullptr) {
        sn = g_dnet_snap[slot];
        if (sn.sn != st.sn || sn.hn != st.hn) return -1;  // size mismatch
    } else {
        if (cudaMallocAsync(&sn.S, st.sn * 4, g_astream) != cudaSuccess) return -1;
        if (cudaMallocAsync(&sn.hist, st.hn * 4, g_astream) != cudaSuccess) {
            cudaFreeAsync(sn.S, g_astream); cudaStreamSynchronize(g_astream); return -1; }
        sn.sn = st.sn; sn.hn = st.hn;
    }
    cudaMemcpyAsync(sn.S, st.S, st.sn * 4, cudaMemcpyDeviceToDevice, g_astream);
    cudaMemcpyAsync(sn.hist, st.hist, st.hn * 4, cudaMemcpyDeviceToDevice, g_astream);
    cudaStreamSynchronize(g_astream);
    if (slot >= 0 && slot < (int) g_dnet_snap.size()) { g_dnet_snap[slot] = sn; return slot; }
    if (!g_dnet_snap_free.empty()) {
        int h = g_dnet_snap_free.back(); g_dnet_snap_free.pop_back();
        g_dnet_snap[h] = sn; return h;
    }
    g_dnet_snap.push_back(sn);
    return (int) g_dnet_snap.size() - 1;
}

//  Restore snapshot `slot` into a fresh delta-net state `handle`.
extern "C" int aspida_gpu_dnet_restore(int handle, int slot) {
    std::lock_guard<std::mutex> lk(g_snap_mtx);
    if (handle < 0 || handle >= (int) g_dnet.size()) return -1;
    if (slot < 0 || slot >= (int) g_dnet_snap.size()) return -1;
    DnetState &st = g_dnet[handle];
    DnetSnap &sn = g_dnet_snap[slot];
    if (!st.S || !sn.S || st.sn != sn.sn || st.hn != sn.hn) return -1;
    ensure_astream();
    cudaMemcpyAsync(st.S, sn.S, st.sn * 4, cudaMemcpyDeviceToDevice, g_astream);
    cudaMemcpyAsync(st.hist, sn.hist, st.hn * 4, cudaMemcpyDeviceToDevice, g_astream);
    cudaStreamSynchronize(g_astream);
    return 0;
}

// Matvec with kind >= 0 -> K-quant warp kernel; kind == -1 -> the weight's raw
// bytes are a dense row-major [out,in] F32 matrix (GGUF F32) -> dense kernel.
static inline void launch_mv_any(const uint8_t *dw, int kind, int in, int out,
                                 const float *dx, float *dy) {
    if (kind >= 0) { launch_matvec(dw, kind, in, out, dx, dy); return; }
    const int TPB = 256, WPB = TPB / 32;
    k_dense_mv<<<(out + WPB - 1) / WPB, TPB>>>((const float *) dw, dx, dy, in, out);
}

// Stream-aware matvec dispatch (for graph capture). Mirrors launch_matvec +
// the dense F32 fallback, but every launch goes on stream `st`.
static inline void launch_mv_st(const uint8_t *dw, int kind, int in, int out,
                                const float *dx, float *dy, cudaStream_t st) {
    const int TPB = 256, WPB = TPB / 32;
    if (kind == 0)      k_q4k_sk<<<out, SKW * 32, 0, st>>>(dw, dx, dy, in, out);
    else if (kind == 1) k_q6k_sk<<<out, SKW * 32, 0, st>>>(dw, dx, dy, in, out);
    else if (kind == 2) k_q5k_w<<<(out + WPB - 1) / WPB, TPB, 0, st>>>(dw, dx, dy, in, out);
    else if (kind == 3) k_q3k_w<<<(out + WPB - 1) / WPB, TPB, 0, st>>>(dw, dx, dy, in, out);
    else if (kind == 4) k_q2k_w<<<(out + WPB - 1) / WPB, TPB, 0, st>>>(dw, dx, dy, in, out);
    else if (kind == 5) k_q8_0_w<<<(out + WPB - 1) / WPB, TPB, 0, st>>>(dw, dx, dy, in, out);
    else k_dense_mv<<<(out + WPB - 1) / WPB, TPB, 0, st>>>((const float *) dw, dx, dy, in, out);
}

// Causal conv1d + SiLU over the resident history window, then advance it.
// One thread per channel c: cq[c] = silu(qkv[c]*w[c,K-1] + sum_k hist[k,c]*w[c,k]),
// then shift this channel's history down one row and append qkv[c].
__global__ void k_dnet_conv(const float *__restrict__ qkv, float *__restrict__ hist,
                            const float *__restrict__ convw, float *__restrict__ cq,
                            int qo, int kernel) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= qo) return;
    float acc = qkv[c] * convw[c * kernel + (kernel - 1)];
    for (int k = 0; k < kernel - 1; ++k) acc += hist[(size_t) k * qo + c] * convw[c * kernel + k];
    cq[c] = acc / (1.f + expf(-acc));
    for (int k = 0; k + 1 < kernel - 1; ++k) hist[(size_t) k * qo + c] = hist[(size_t) (k + 1) * qo + c];
    if (kernel >= 2) hist[(size_t) (kernel - 2) * qo + c] = qkv[c];
}

// Per-head decay/beta transform: gate[h] = exp(a[h] * softplus(ar[h] + dt[h])),
// beta[h] = sigmoid(br[h]). Same softplus branches as the CPU (LLM_DeltaNet_Blk).
__global__ void k_dnet_gates(const float *__restrict__ ar, const float *__restrict__ br,
                             const float *__restrict__ a, const float *__restrict__ dt,
                             float *__restrict__ gate, float *__restrict__ beta, int nv) {
    int h = blockIdx.x * blockDim.x + threadIdx.x; if (h >= nv) return;
    float xx = ar[h] + dt[h];
    float sp = xx > 20.f ? xx : (xx < -20.f ? expf(xx) : logf(1.f + expf(xx)));
    gate[h] = expf(a[h] * sp);
    beta[h] = 1.f / (1.f + expf(-br[h]));
}

__global__ void k_dnet_recur(float *S, const float *cq, const float *gate,
    const float *beta, const float *z, const float *norm_w, float *o_row,
    int khd, int vhd, int q_dim, int n_k_heads) {
    int h = blockIdx.x;
    int v = threadIdx.x;                  // 0..khd-1 (== vhd-1)
    extern __shared__ float sh[];
    float *Qr = sh, *Kr = sh + khd, *QN = sh + 2 * khd,
          *KN = sh + 3 * khd, *Vv = sh + 4 * khd, *osh = sh + 4 * khd + vhd;
    int k_head = h % n_k_heads;
    Qr[v] = cq[k_head * khd + v];
    Kr[v] = cq[q_dim + k_head * khd + v];
    Vv[v] = cq[2 * q_dim + h * vhd + v];
    __syncthreads();
    // L2 normalise Q, K (each thread sums ascending, matching the CPU order).
    float ssq = 0.f, ssk = 0.f;
    for (int i = 0; i < khd; ++i) { ssq += Qr[i] * Qr[i]; ssk += Kr[i] * Kr[i]; }
    QN[v] = Qr[v] * (1.f / (sqrtf(ssq) + 1e-6f));
    KN[v] = Kr[v] * (1.f / (sqrtf(ssk) + 1e-6f));
    __syncthreads();
    float g = gate[h], b = beta[h], scale = 1.f / sqrtf((float) khd);
    int base = h * khd;                   // row offset into S_All [nv*khd, vhd]
    // Thread v owns column v: retrieval, correction, gated write, output.
    float retr = 0.f;
    for (int k = 0; k < khd; ++k) retr += g * S[(size_t)(base + k) * vhd + v] * KN[k];
    float corr = b * (Vv[v] - retr);
    for (int k = 0; k < khd; ++k)
        S[(size_t)(base + k) * vhd + v] = g * S[(size_t)(base + k) * vhd + v] + KN[k] * corr;
    float o = 0.f;
    for (int k = 0; k < khd; ++k) o += S[(size_t)(base + k) * vhd + v] * QN[k];
    osh[v] = o * scale;
    __syncthreads();
    // Gated RMSNorm over vhd (ascending sum, CPU order) + SiLU(z) gate.
    float ss = 0.f;
    for (int i = 0; i < vhd; ++i) ss += osh[i] * osh[i];
    float rms = sqrtf(ss / (float) vhd + 1e-6f);
    float zz = z[h * vhd + v];
    o_row[h * vhd + v] = (osh[v] / rms) * norm_w[v] * (zz / (1.f + expf(-zz)));
}

// Batched delta-net recurrence: all B lanes in ONE launch (B*nv blocks) instead
// of B separate nv-block launches. Each lane keeps its own resident state via
// the S_arr pointer table; the per-lane inputs are contiguous [B, ...] buffers.
// Removes B-1 launch overheads per layer AND lifts occupancy (nv=32 blocks was
// far below the SM count; B*nv fills the device).
__global__ void k_dnet_recur_b(float *const *__restrict__ S_arr, const float *cq,
    const float *gate, const float *beta, const float *z, const float *norm_w,
    float *o_row, int khd, int vhd, int q_dim, int n_k_heads, int nv, int qo, int v_dim) {
    int lane = blockIdx.x / nv, h = blockIdx.x % nv, v = threadIdx.x;
    float *S = S_arr[lane];
    const float *cq_l = cq + (size_t) lane * qo;
    const float *gate_l = gate + (size_t) lane * nv;
    const float *beta_l = beta + (size_t) lane * nv;
    const float *z_l = z + (size_t) lane * v_dim;
    float *o_l = o_row + (size_t) lane * v_dim;
    extern __shared__ float sh[];
    float *Qr = sh, *Kr = sh + khd, *QN = sh + 2 * khd,
          *KN = sh + 3 * khd, *Vv = sh + 4 * khd, *osh = sh + 4 * khd + vhd;
    int k_head = h % n_k_heads;
    Qr[v] = cq_l[k_head * khd + v];
    Kr[v] = cq_l[q_dim + k_head * khd + v];
    Vv[v] = cq_l[2 * q_dim + h * vhd + v];
    __syncthreads();
    float ssq = 0.f, ssk = 0.f;
    for (int i = 0; i < khd; ++i) { ssq += Qr[i] * Qr[i]; ssk += Kr[i] * Kr[i]; }
    QN[v] = Qr[v] * (1.f / (sqrtf(ssq) + 1e-6f));
    KN[v] = Kr[v] * (1.f / (sqrtf(ssk) + 1e-6f));
    __syncthreads();
    float g = gate_l[h], b = beta_l[h], scale = 1.f / sqrtf((float) khd);
    int base = h * khd;
    float retr = 0.f;
    for (int k = 0; k < khd; ++k) retr += g * S[(size_t)(base + k) * vhd + v] * KN[k];
    float corr = b * (Vv[v] - retr);
    for (int k = 0; k < khd; ++k)
        S[(size_t)(base + k) * vhd + v] = g * S[(size_t)(base + k) * vhd + v] + KN[k] * corr;
    float o = 0.f;
    for (int k = 0; k < khd; ++k) o += S[(size_t)(base + k) * vhd + v] * QN[k];
    osh[v] = o * scale;
    __syncthreads();
    float ss = 0.f;
    for (int i = 0; i < vhd; ++i) ss += osh[i] * osh[i];
    float rms = sqrtf(ss / (float) vhd + 1e-6f);
    float zz = z_l[h * vhd + v];
    o_l[h * vhd + v] = (osh[v] / rms) * norm_w[v] * (zz / (1.f + expf(-zz)));
}

// Batched causal conv (per-lane hist via pointer table) and gate transform.
__global__ void k_dnet_conv_b(const float *__restrict__ qkv, float *const *__restrict__ hist_arr,
                              const float *__restrict__ convw, float *__restrict__ cq,
                              int qo, int kernel, int B) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x, lane = gid / qo, c = gid % qo;
    if (lane >= B) return;
    float *hist = hist_arr[lane];
    const float *qkv_l = qkv + (size_t) lane * qo;
    float *cq_l = cq + (size_t) lane * qo;
    float acc = qkv_l[c] * convw[c * kernel + (kernel - 1)];
    for (int k = 0; k < kernel - 1; ++k) acc += hist[(size_t) k * qo + c] * convw[c * kernel + k];
    cq_l[c] = acc / (1.f + expf(-acc));
    for (int k = 0; k + 1 < kernel - 1; ++k) hist[(size_t) k * qo + c] = hist[(size_t) (k + 1) * qo + c];
    if (kernel >= 2) hist[(size_t) (kernel - 2) * qo + c] = qkv_l[c];
}

__global__ void k_dnet_gates_b(const float *__restrict__ ar, const float *__restrict__ br,
                               const float *__restrict__ a, const float *__restrict__ dt,
                               float *__restrict__ gate, float *__restrict__ beta, int nv, int B) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x, lane = gid / nv, h = gid % nv;
    if (lane >= B) return;
    const float *ar_l = ar + (size_t) lane * nv, *br_l = br + (size_t) lane * nv;
    float *gate_l = gate + (size_t) lane * nv, *beta_l = beta + (size_t) lane * nv;
    float xx = ar_l[h] + dt[h];
    float sp = xx > 20.f ? xx : (xx < -20.f ? expf(xx) : logf(1.f + expf(xx)));
    gate_l[h] = expf(a[h] * sp);
    beta_l[h] = 1.f / (1.f + expf(-br_l[h]));
}

// One full delta-net decode layer on the device: h2d(x) -> qkv/alpha/beta/z
// projections -> causal conv+SiLU (resident hist) -> gate/beta transform ->
// per-head recurrence + gated RMSNorm (resident S_All) -> out projection ->
// d2h(out). Replaces ~5 blocking per-matvec round-trips with 1 H2D + 1 D2H.
// Small dense weights (conv_w/a/dt/norm_w, host F32) ride the resident weight
// cache keyed by their host pointers, so they upload once.
extern "C" void aspida_gpu_dnet_step(
    int handle, const float *x, int dim,
    const void *qkv_w,  long qkv_b,  int qkv_k,     // rows = qo
    const void *al_w,   long al_b,   int al_k,      // rows = nv
    const void *be_w,   long be_b,   int be_k,      // rows = nv
    const void *ga_w,   long ga_b,   int ga_k,      // rows = v_dim (Z gate)
    const void *out_w,  long out_b,  int out_k,     // [dim, v_dim]
    const float *conv_w, long conv_b,                // [qo, kernel] host F32
    const float *a_w,   long a_b,                    // [nv] host F32
    const float *dt_w,  long dt_b,                   // [nv] host F32
    const float *norm_w, long norm_b,                // [vhd] host F32
    int nv, int khd, int vhd, int qo, int q_dim, int n_k_heads, int v_dim,
    int kernel, float *out) {
    if (handle < 0 || handle >= (int) g_dnet.size()) return;
    DnetState st = g_dnet[handle];

    static float *dx = nullptr, *dqkv = nullptr, *dcq = nullptr, *dar = nullptr,
                 *dbr = nullptr, *dz = nullptr, *dg = nullptr, *db = nullptr,
                 *dor = nullptr, *dout = nullptr;
    static int c_dim = 0, c_qo = 0, c_nv = 0, c_vd = 0;
    if (dim > c_dim) { if (dx) cudaFree(dx); cudaMalloc(&dx, (size_t) dim * 4);
                       if (dout) cudaFree(dout); cudaMalloc(&dout, (size_t) dim * 4); c_dim = dim; }
    if (qo > c_qo)   { if (dqkv) cudaFree(dqkv); cudaMalloc(&dqkv, (size_t) qo * 4);
                       if (dcq) cudaFree(dcq); cudaMalloc(&dcq, (size_t) qo * 4); c_qo = qo; }
    if (nv > c_nv)   { if (dar) cudaFree(dar); cudaMalloc(&dar, (size_t) nv * 4);
                       if (dbr) cudaFree(dbr); cudaMalloc(&dbr, (size_t) nv * 4);
                       if (dg) cudaFree(dg); cudaMalloc(&dg, (size_t) nv * 4);
                       if (db) cudaFree(db); cudaMalloc(&db, (size_t) nv * 4); c_nv = nv; }
    if (v_dim > c_vd) { if (dz) cudaFree(dz); cudaMalloc(&dz, (size_t) v_dim * 4);
                        if (dor) cudaFree(dor); cudaMalloc(&dor, (size_t) v_dim * 4); c_vd = v_dim; }

    uint8_t *dqw = upload_weight(qkv_w, qkv_b);
    uint8_t *daw = upload_weight(al_w, al_b);
    uint8_t *dbw = upload_weight(be_w, be_b);
    uint8_t *dgw = upload_weight(ga_w, ga_b);
    uint8_t *dow = upload_weight(out_w, out_b);
    float *dconv = (float *) upload_weight(conv_w, conv_b);
    float *da    = (float *) upload_weight(a_w, a_b);
    float *ddt   = (float *) upload_weight(dt_w, dt_b);
    float *dnw   = (float *) upload_weight(norm_w, norm_b);

    cudaMemcpy(dx, x, (size_t) dim * 4, cudaMemcpyHostToDevice);

    launch_mv_any(dqw, qkv_k, dim, qo, dx, dqkv);
    launch_mv_any(daw, al_k, dim, nv, dx, dar);
    launch_mv_any(dbw, be_k, dim, nv, dx, dbr);
    launch_mv_any(dgw, ga_k, dim, v_dim, dx, dz);
    k_dnet_conv<<<(qo + 255) / 256, 256>>>(dqkv, st.hist, dconv, dcq, qo, kernel);
    k_dnet_gates<<<(nv + 255) / 256, 256>>>(dar, dbr, da, ddt, dg, db, nv);
    size_t shmem = (size_t) (4 * khd + 2 * vhd) * 4;
    k_dnet_recur<<<nv, khd, shmem>>>(st.S, dcq, dg, db, dz, dnw, dor,
                                     khd, vhd, q_dim, n_k_heads);
    launch_mv_any(dow, out_k, v_dim, dim, dor, dout);
    cudaDeviceSynchronize();   // drain via spin BEFORE the blocking copy
    cudaMemcpy(out, dout, (size_t) dim * 4, cudaMemcpyDeviceToHost);
}

// =====================================================================
// Phase B2 — resident full-attention (GQA) decode layer.
// Oracle: LLM_FullAttn.Step. K/V caches live on the device across tokens;
// per token: 1 H2D of x, QKV projections, per-head QK-RMSNorm + partial RoPE
// (NeoX split-half or interleaved, YaRN/PI/freq-factor aware), K/V append,
// causal GQA softmax over the whole cache, per-dim sigmoid gate, out
// projection, 1 D2H. Scores use a resident per-layer scratch (no length cap).
// =====================================================================
struct FattnState { __half *K, *V; float *scores; int max_len, kvd; };  // K/V fp16 (Phase B)
static std::vector<FattnState> g_fattn;
static std::vector<int> g_fattn_free;     // freed slots for reuse

extern "C" int aspida_gpu_fattn_new(int max_len, int kvd, int nq) {
    ensure_astream();
    FattnState st; st.max_len = max_len; st.kvd = kvd;
    if (cudaMallocAsync(&st.K, (size_t) max_len * kvd * 2, g_astream) != cudaSuccess) return -1;  // fp16
    if (cudaMallocAsync(&st.V, (size_t) max_len * kvd * 2, g_astream) != cudaSuccess) { cudaFreeAsync(st.K, g_astream); cudaStreamSynchronize(g_astream); return -1; }  // fp16
    if (cudaMallocAsync(&st.scores, (size_t) nq * max_len * 4, g_astream) != cudaSuccess) {
        cudaFreeAsync(st.K, g_astream); cudaFreeAsync(st.V, g_astream); cudaStreamSynchronize(g_astream); return -1; }
    cudaStreamSynchronize(g_astream);   // blocks ready for use on any stream
    if (!g_fattn_free.empty()) {
        int h = g_fattn_free.back(); g_fattn_free.pop_back();
        g_fattn[h] = st; return h;
    }
    g_fattn.push_back(st);
    return (int) g_fattn.size() - 1;
}

// Per-generation KV caches — must be released when the generation ends or
// they leak tens of MB VRAM per request. Slot is reused by _new.
extern "C" void aspida_gpu_fattn_free(int handle) {
    if (handle < 0 || handle >= (int) g_fattn.size()) return;
    FattnState &st = g_fattn[handle];
    if (st.K)      { cudaFreeAsync(st.K, g_astream); st.K = nullptr; }
    if (st.V)      { cudaFreeAsync(st.V, g_astream); st.V = nullptr; }
    if (st.scores) { cudaFreeAsync(st.scores, g_astream); st.scores = nullptr; }
    g_fattn_free.push_back(handle);
}

//  ---- Full-attention prefix snapshots (RETAINED) --------------------------
//  Unlike DeltaNet, attention state is positional: the first N K/V rows ARE the
//  prefix's cache. We snapshot exactly those N rows (N = prefix length); the
//  scores buffer is scratch and is not saved. On restore the rows are copied
//  back into a fresh state and the Ada layer sets its Len := N so decode
//  appends at row N. See the DeltaNet snapshot header for lifetime rules.
struct FattnSnap { __half *K; __half *V; int rows; int kvd; };  // fp16 (Phase B)
static std::vector<FattnSnap> g_fattn_snap;
static std::vector<int> g_fattn_snap_free;

//  Snapshot the first `rows` K/V rows of state `handle` into `slot` (-1 = new).
extern "C" int aspida_gpu_fattn_snapshot(int handle, int rows, int slot) {
    std::lock_guard<std::mutex> lk(g_snap_mtx);
    if (handle < 0 || handle >= (int) g_fattn.size()) return -1;
    FattnState &st = g_fattn[handle];
    if (!st.K || !st.V || rows <= 0 || rows > st.max_len) return -1;
    ensure_astream();
    size_t bytes = (size_t) rows * st.kvd * 2;   // fp16
    FattnSnap sn;
    if (slot >= 0 && slot < (int) g_fattn_snap.size()
        && g_fattn_snap[slot].K != nullptr
        && g_fattn_snap[slot].rows == rows && g_fattn_snap[slot].kvd == st.kvd) {
        sn = g_fattn_snap[slot];
    } else {
        if (cudaMallocAsync(&sn.K, bytes, g_astream) != cudaSuccess) return -1;
        if (cudaMallocAsync(&sn.V, bytes, g_astream) != cudaSuccess) {
            cudaFreeAsync(sn.K, g_astream); cudaStreamSynchronize(g_astream); return -1; }
        sn.rows = rows; sn.kvd = st.kvd;
    }
    cudaMemcpyAsync(sn.K, st.K, bytes, cudaMemcpyDeviceToDevice, g_astream);
    cudaMemcpyAsync(sn.V, st.V, bytes, cudaMemcpyDeviceToDevice, g_astream);
    cudaStreamSynchronize(g_astream);
    if (slot >= 0 && slot < (int) g_fattn_snap.size()) { g_fattn_snap[slot] = sn; return slot; }
    if (!g_fattn_snap_free.empty()) {
        int h = g_fattn_snap_free.back(); g_fattn_snap_free.pop_back();
        g_fattn_snap[h] = sn; return h;
    }
    g_fattn_snap.push_back(sn);
    return (int) g_fattn_snap.size() - 1;
}

//  Restore snapshot `slot` (its `rows` K/V rows) into fresh state `handle`.
extern "C" int aspida_gpu_fattn_restore(int handle, int slot) {
    std::lock_guard<std::mutex> lk(g_snap_mtx);
    if (handle < 0 || handle >= (int) g_fattn.size()) return -1;
    if (slot < 0 || slot >= (int) g_fattn_snap.size()) return -1;
    FattnState &st = g_fattn[handle];
    FattnSnap &sn = g_fattn_snap[slot];
    if (!st.K || !sn.K || sn.kvd != st.kvd || sn.rows > st.max_len) return -1;
    ensure_astream();
    size_t bytes = (size_t) sn.rows * st.kvd * 2;   // fp16
    cudaMemcpyAsync(st.K, sn.K, bytes, cudaMemcpyDeviceToDevice, g_astream);
    cudaMemcpyAsync(st.V, sn.V, bytes, cudaMemcpyDeviceToDevice, g_astream);
    cudaStreamSynchronize(g_astream);
    return 0;
}

//  Free ALL retained prefix snapshots (delta-net + full-attn). Called on model
//  change / chain reset, when cached prefixes are no longer valid.
extern "C" void aspida_gpu_prefix_reset(void) {
    std::lock_guard<std::mutex> lk(g_snap_mtx);
    ensure_astream();
    for (auto &sn : g_dnet_snap) {
        if (sn.S)    cudaFreeAsync(sn.S, g_astream);
        if (sn.hist) cudaFreeAsync(sn.hist, g_astream);
        sn.S = nullptr; sn.hist = nullptr;
    }
    for (auto &sn : g_fattn_snap) {
        if (sn.K) cudaFreeAsync(sn.K, g_astream);
        if (sn.V) cudaFreeAsync(sn.V, g_astream);
        sn.K = nullptr; sn.V = nullptr;
    }
    cudaStreamSynchronize(g_astream);
    g_dnet_snap.clear();  g_dnet_snap_free.clear();
    g_fattn_snap.clear(); g_fattn_snap_free.clear();
}

// theta for RoPE pair i (matches LLM_RoPE.Apply_Sections text path exactly).
__device__ __forceinline__ float rope_theta(
    int i, int pos_eff, int rd, float base, float freq_scale,
    int yarn_on, float corr_lo, float corr_hi,
    const float *ff, int use_ff) {
    float extrap = (float) pos_eff / powf(base, (float) (2 * i) / (float) rd);
    float th;
    if (yarn_on) {
        float interp = freq_scale * extrap;
        float yv = ((float) i - corr_lo) / fmaxf(0.001f, corr_hi - corr_lo);
        float ramp = 1.f - fminf(1.f, fmaxf(0.f, yv));
        th = interp * (1.f - ramp) + extrap * ramp;
    } else {
        th = extrap * freq_scale;
    }
    if (use_ff) th = th / ff[i];
    return th;
}

// Per-head QK-RMSNorm + partial RoPE + cache append. Block b < nq: query head
// (also splits out the raw gate). Block b >= nq: kv head — writes K/V cache
// row `pos`. blockDim = hd. Shared: normed[hd].
__global__ void k_fattn_prep(
    const float *__restrict__ qg, const float *__restrict__ kt, const float *__restrict__ vt,
    const float *__restrict__ q_norm, const float *__restrict__ k_norm,
    float *__restrict__ q_all, float *__restrict__ g_all,
    __half *__restrict__ Kc, __half *__restrict__ Vc,     // fp16 cache (Phase B)
    int nq, int nkv, int hd, int kvd, const int *__restrict__ posp,
    int rd, float base, float freq_scale, float m_scale,
    int yarn_on, float corr_lo, float corr_hi,
    const float *__restrict__ ff, int use_ff, int interleaved, int sec_total) {
    extern __shared__ float sh[];
    float *nrm = sh;                       // [hd]
    int pos = *posp;
    int b = blockIdx.x, d = threadIdx.x;
    int half = rd / 2;
    bool is_q = b < nq;
    int head = is_q ? b : b - nq;
    float v;
    if (is_q) {
        v = qg[head * 2 * hd + d];
        g_all[head * hd + d] = qg[head * 2 * hd + hd + d];
    } else {
        v = kt[head * hd + d];
        Vc[(size_t) pos * kvd + head * hd + d] = __float2half(vt[head * hd + d]);
    }
    nrm[d] = v;
    __syncthreads();
    // RMSNorm over hd (each thread sums ascending — CPU order)
    float ss = 0.f;
    for (int i = 0; i < hd; ++i) ss += nrm[i] * nrm[i];
    float rms = sqrtf(ss / (float) hd + 1e-6f);
    float w = is_q ? q_norm[d] : k_norm[d];
    float nv = (v / rms) * w;
    __syncthreads();
    nrm[d] = nv;                            // normed value
    __syncthreads();
    // partial RoPE on the first rd dims
    float out = nv;
    if (d < rd) {
        int i, other; bool first;
        if (interleaved) { i = d / 2; first = (d % 2) == 0; other = first ? d + 1 : d - 1; }
        else if (d < half) { i = d; first = true;  other = d + half; }
        else               { i = d - half; first = false; other = d - half; }
        int pos_eff = (i < sec_total) ? pos : 0;
        float th = rope_theta(i, pos_eff, rd, base, freq_scale,
                              yarn_on, corr_lo, corr_hi, ff, use_ff);
        float c = cosf(th) * m_scale, s = sinf(th) * m_scale;
        float x1 = first ? nrm[d] : nrm[other];
        float x2 = first ? nrm[other] : nrm[d];
        out = first ? (x1 * c - x2 * s) : (x2 * c + x1 * s);
    }
    if (is_q) q_all[head * hd + d] = out;
    else      Kc[(size_t) pos * kvd + head * hd + d] = __float2half(out);
}

// Causal GQA softmax attention over the resident cache + per-dim sigmoid gate.
// One block (256 threads) per q head; len = pos+1 positions.
__global__ void k_fattn_attend(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const __half *__restrict__ Kc, const __half *__restrict__ Vc,     // fp16 cache (Phase B)
    float *__restrict__ scores, float *__restrict__ attn,
    int nq, int nkv, int hd, int kvd, const int *__restrict__ posp, int max_len) {
    int len = *posp + 1;
    int h = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int rep = nq / nkv, kvh = h / rep;
    int q_off = h * hd, kv_off = kvh * hd;
    float *sc = scores + (size_t) h * max_len;
    __shared__ float red[256];
    float scale = rsqrtf((float) hd);       // 1/sqrt(hd)
    // Phase 1: scores + local max (each thread strides positions)
    float lmax = -3.402823466e38f;
    for (int s = tid; s < len; s += nt) {
        float dot = 0.f;
        const __half *k = Kc + (size_t) s * kvd + kv_off;   // fp16
        const float *q = q_all + q_off;
        for (int d = 0; d < hd; ++d) dot += q[d] * __half2float(k[d]);
        dot *= scale;
        sc[s] = dot;
        if (dot > lmax) lmax = dot;
    }
    red[tid] = lmax; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    float bmax = red[0]; __syncthreads();
    // Phase 2: exp + sum
    float lsum = 0.f;
    for (int s = tid; s < len; s += nt) { float e = expf(sc[s] - bmax); sc[s] = e; lsum += e; }
    red[tid] = lsum; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    float inv = 1.f / red[0];
    __syncthreads();
    // Phase 3: weighted V + sigmoid gate (threads 0..hd-1)
    if (tid < hd) {
        float acc = 0.f;
        for (int s = 0; s < len; ++s)
            acc += sc[s] * inv * __half2float(Vc[(size_t) s * kvd + kv_off + tid]);
        float g = g_all[q_off + tid];
        attn[q_off + tid] = acc * (1.f / (1.f + expf(-g)));
    }
}

// One full-attention decode layer on the device. pos = 0-based position (the
// Ada St.Len before this token); the caller advances its Len afterwards.
extern "C" void aspida_gpu_fattn_step(
    int handle, const float *x, int dim,
    const void *q_w, long q_b, int q_k,       // rows = nq*2*hd (query|gate)
    const void *k_w, long k_b, int k_k,       // rows = nkv*hd
    const void *v_w, long v_b, int v_k,       // rows = nkv*hd
    const void *o_w, long o_b, int o_k,       // [dim, nq*hd]
    const float *q_norm, long qn_b,            // [hd] host F32
    const float *k_norm, long kn_b,            // [hd] host F32
    int nq, int nkv, int hd, int pos,
    int rd, float base, float freq_scale, float m_scale,
    int yarn_on, float corr_lo, float corr_hi,
    const float *ff, long ff_b, int use_ff, int interleaved, int sec_total,
    float *out) {
    if (handle < 0 || handle >= (int) g_fattn.size()) return;
    FattnState st = g_fattn[handle];
    int kvd = nkv * hd, att = nq * hd, qgd = nq * 2 * hd;

    static float *dx = nullptr, *dqg = nullptr, *dkt = nullptr, *dvt = nullptr,
                 *dqa = nullptr, *dga = nullptr, *datt = nullptr, *dout = nullptr;
    static int c_dim = 0, c_qgd = 0, c_kvd = 0, c_att = 0;
    if (dim > c_dim) { if (dx) cudaFree(dx); cudaMalloc(&dx, (size_t) dim * 4);
                       if (dout) cudaFree(dout); cudaMalloc(&dout, (size_t) dim * 4); c_dim = dim; }
    if (qgd > c_qgd) { if (dqg) cudaFree(dqg); cudaMalloc(&dqg, (size_t) qgd * 4); c_qgd = qgd; }
    if (kvd > c_kvd) { if (dkt) cudaFree(dkt); cudaMalloc(&dkt, (size_t) kvd * 4);
                       if (dvt) cudaFree(dvt); cudaMalloc(&dvt, (size_t) kvd * 4); c_kvd = kvd; }
    if (att > c_att) { if (dqa) cudaFree(dqa); cudaMalloc(&dqa, (size_t) att * 4);
                       if (dga) cudaFree(dga); cudaMalloc(&dga, (size_t) att * 4);
                       if (datt) cudaFree(datt); cudaMalloc(&datt, (size_t) att * 4); c_att = att; }

    uint8_t *dqw = upload_weight(q_w, q_b);
    uint8_t *dkw = upload_weight(k_w, k_b);
    uint8_t *dvw = upload_weight(v_w, v_b);
    uint8_t *dow = upload_weight(o_w, o_b);
    float *dqn = (float *) upload_weight(q_norm, qn_b);
    float *dkn = (float *) upload_weight(k_norm, kn_b);
    float *dff = (use_ff && ff) ? (float *) upload_weight(ff, ff_b) : nullptr;

    static int *d_pos1 = nullptr; if (!d_pos1) cudaMalloc(&d_pos1, 4);
    cudaMemcpy(d_pos1, &pos, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t) dim * 4, cudaMemcpyHostToDevice);
    launch_mv_any(dqw, q_k, dim, qgd, dx, dqg);
    launch_mv_any(dkw, k_k, dim, kvd, dx, dkt);
    launch_mv_any(dvw, v_k, dim, kvd, dx, dvt);
    k_fattn_prep<<<nq + nkv, hd, (size_t) hd * 4>>>(
        dqg, dkt, dvt, dqn, dkn, dqa, dga, st.K, st.V,
        nq, nkv, hd, kvd, d_pos1, rd, base, freq_scale, m_scale,
        yarn_on, corr_lo, corr_hi, dff, use_ff, interleaved, sec_total);
    k_fattn_attend<<<nq, 256>>>(dqa, dga, st.K, st.V, st.scores, datt,
                                nq, nkv, hd, kvd, d_pos1, st.max_len);
    launch_mv_any(dow, o_k, att, dim, datt, dout);
    cudaMemcpy(out, dout, (size_t) dim * 4, cudaMemcpyDeviceToHost);
}

// Router + stable-softmax + greedy top-k stay on the CPU (they are tiny, and
// the router is often non-K-quant in mixed quants like Q4_K_M): LLM_MoE.Forward
// computes top_idx (0-based) + renormalised top_w and hands them here. This
// kernel does ONLY the expensive part — the 3D routed experts and the 2D shared
// expert (all K-quant), fully on resident device buffers.
extern "C" void aspida_gpu_moe_experts(
    const float *x, int dim, int top_k, int intermed, int n_exp,
    const int *top_idx, const float *top_w,
    const void *gate_w, long gate_bytes, int gate_kind,
    const void *up_w,   long up_bytes,   int up_kind,
    const void *down_w, long down_bytes, int down_kind,
    const void *shg_w,  long shg_bytes,  int shg_kind,
    const void *shu_w,  long shu_bytes,  int shu_kind,
    const void *shd_w,  long shd_bytes,  int shd_kind,
    const float *shared_gate_inp, int gate_inp_len,
    float *y) {
    // Resident scratch, grown-only across tokens/layers. d_h holds all
    // (MOE_MAXK+1) experts' SwiGLU outputs for the fused path.
    static float *dx = nullptr, *d_h = nullptr, *d_acc = nullptr;
    static int cdim = 0; static size_t chn = 0;
    size_t hn = (size_t) (MOE_MAXK + 1) * intermed;
    if (dim > cdim) {
        if (dx) cudaFree(dx);       cudaMalloc(&dx,    (size_t) dim * 4);
        if (d_acc) cudaFree(d_acc); cudaMalloc(&d_acc, (size_t) dim * 4);
        cdim = dim;
    }
    if (hn > chn) { if (d_h) cudaFree(d_h); cudaMalloc(&d_h, hn * 4); chn = hn; }

    uint8_t *gdw = upload_weight(gate_w, gate_bytes);
    uint8_t *udw = upload_weight(up_w, up_bytes);
    uint8_t *ddw = upload_weight(down_w, down_bytes);
    uint8_t *sgdw = upload_weight(shg_w, shg_bytes);
    uint8_t *sudw = upload_weight(shu_w, shu_bytes);
    uint8_t *sddw = upload_weight(shd_w, shd_bytes);
    size_t g_bpe = (size_t)(gate_bytes / n_exp), u_bpe = (size_t)(up_bytes / n_exp),
           d_bpe = (size_t)(down_bytes / n_exp);

    // Segment profiler (env ASPIDA_MOE_PROF): wall time per phase, sync after
    // each, accumulated across calls, dumped every 36*20 calls (~20 tokens).
    static int prof = getenv("ASPIDA_MOE_PROF") ? 1 : 0;
    static double t_h2d = 0, t_gu = 0, t_down = 0, t_d2h = 0;
    static long n_calls = 0;
    struct timespec t0, t1;
    #define SEG(acc, ...) do { if (prof) { cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC, &t0); } \
        __VA_ARGS__; \
        if (prof) { cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC, &t1); \
            acc += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9; } } while (0)

    if (top_k > MOE_MAXK) top_k = MOE_MAXK;   // route struct capacity (top_k is 8)

    // Routing packed into kernel params; slot MOE_MAXK = shared expert gate.
    MoeRoute route;
    for (int k = 0; k < top_k; ++k) { route.idx[k] = top_idx[k]; route.w[k] = top_w[k]; }
    for (int k = top_k; k < MOE_MAXK; ++k) { route.idx[k] = 0; route.w[k] = 0.f; }
    float shared_gate = 1.0f;
    if (gate_inp_len > 1 && shared_gate_inp) {
        float gs = 0.f;
        for (int d = 0; d < dim; ++d) gs += shared_gate_inp[d] * x[d];
        shared_gate = 1.0f / (1.0f + expf(-gs));
    }
    route.w[MOE_MAXK] = shared_gate;

    SEG(t_h2d, cudaMemcpy(dx, x, (size_t) dim * 4, cudaMemcpyHostToDevice));

    // Fused: 1 launch for all (top_k+1) experts' gate+up+SwiGLU, 1 launch for
    // all down projections + weighted combine (replaces ~30 launches/layer).
    int w1 = (top_k + 1) * intermed, b1 = (w1 * 32 + 255) / 256;
    int b2 = (dim * 32 + 255) / 256;
    SEG(t_gu, k_moe_gu<<<b1, 256>>>(gdw, udw, sgdw, sudw, dx, d_h, route, top_k,
                                    dim, intermed, (long) g_bpe, (long) u_bpe,
                                    gate_kind, up_kind, shg_kind, shu_kind));
    SEG(t_down, k_moe_down<<<b2, 256>>>(ddw, sddw, d_h, d_acc, route, top_k,
                                        intermed, dim, (long) d_bpe,
                                        down_kind, shd_kind));

    cudaDeviceSynchronize();   // drain via spin BEFORE the blocking copy
    SEG(t_d2h, cudaMemcpy(y, d_acc, (size_t) dim * 4, cudaMemcpyDeviceToHost));
    #undef SEG

    if (prof && ++n_calls % (36 * 20) == 0) {
        fprintf(stderr,
            "[MOEPROF] dims: dim=%d intermed=%d top_k=%d kinds g/u/d=%d/%d/%d "
            "shg/shu/shd=%d/%d/%d\n"
            "[MOEPROF] per-layer-call ms: h2d=%.3f gu_fused=%.3f down_fused=%.3f "
            "d2h=%.3f (n=%ld)\n",
            dim, intermed, top_k, gate_kind, up_kind, down_kind,
            shg_kind, shu_kind, shd_kind,
            1e3 * t_h2d / n_calls, 1e3 * t_gu / n_calls,
            1e3 * t_down / n_calls, 1e3 * t_d2h / n_calls, n_calls);
    }
}

// =====================================================================
// Phase C — full resident forward chain. The hidden state H lives on the
// device across ALL layers; per token the host sends the embedding row id
// and receives the logits. Layers are registered once at model load (device
// weight pointers resolved through the same g_wcache); per-generation attn
// states are passed as a handle array per call. Every kernel matches its CPU
// oracle's summation order (single-thread exact norms / routing).
// =====================================================================

// Exact single-block RMSNorm: thread 0 sums ascending (CPU order), all scale.
__global__ void k_norm1(const float *__restrict__ x, const float *__restrict__ w,
                        float *__restrict__ y, int n) {
    __shared__ float red[256]; __shared__ float rms;
    int tid = threadIdx.x, nt = blockDim.x;
    float ls = 0.f;
    for (int i = tid; i < n; i += nt) ls += x[i] * x[i];
    red[tid] = ls; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) rms = sqrtf(red[0] / (float) n + 1e-6f);
    __syncthreads();
    for (int i = tid; i < n; i += nt) y[i] = (x[i] / rms) * w[i];
}

// Embedding row gather: H = embed[row].
__global__ void k_embed(const float *__restrict__ embed, const int *__restrict__ rowp,
                        float *__restrict__ h, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) h[i] = embed[(size_t) (*rowp) * dim + i];
}

// Router softmax + greedy top-k + renorm + shared sigmoid gate, single thread
// (byte-for-byte the LLM_MoE.Forward order: ascending scans, strict >, first
// max wins). Writes the MoeRoute struct in device memory.
// 256-thread router: parallel stable softmax + gate-dot reductions; the greedy
// top-k stays a single-thread serial scan (order-exact, matches LLM_MoE, keeps
// the expert selection robust). Launch <<<1,256>>>.
__global__ void k_moe_route(const float *__restrict__ rl, const float *__restrict__ x,
                            const float *__restrict__ sgi, int sgi_len,
                            int n_exp, int top_k, int dim, MoeRoute *route) {
    __shared__ float r[512]; __shared__ float red[256]; __shared__ int redi[256];
    __shared__ float mx, sm;
    int tid = threadIdx.x, nt = blockDim.x;
    if (n_exp > 512) return;
    float lmax = -3.402823466e38f;
    for (int e = tid; e < n_exp; e += nt) { r[e] = rl[e]; if (r[e] > lmax) lmax = r[e]; }
    red[tid] = lmax; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    if (tid == 0) mx = red[0]; __syncthreads();
    float ls = 0.f;
    for (int e = tid; e < n_exp; e += nt) { r[e] = expf(r[e] - mx); ls += r[e]; }
    red[tid] = ls; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) sm = red[0]; __syncthreads();
    for (int e = tid; e < n_exp; e += nt) r[e] /= sm;
    __syncthreads();
    // Parallel top-k: repeatedly argmax over r[] (block-wide reduction), then
    // knock out the winner by setting it to -inf. All 256 threads participate,
    // vs the previous single-thread O(top_k*n_exp) scan with a local used[] array.
    for (int k = 0; k < top_k; ++k) {
        float bv = -3.402823466e38f; int bi = 0;
        for (int e = tid; e < n_exp; e += nt)
            if (r[e] > bv) { bv = r[e]; bi = e; }
        red[tid] = bv; redi[tid] = bi; __syncthreads();
        for (int o = nt / 2; o > 0; o >>= 1) {
            if (tid < o && red[tid + o] > red[tid]) { red[tid] = red[tid + o]; redi[tid] = redi[tid + o]; }
            __syncthreads();
        }
        if (tid == 0) { route->idx[k] = redi[0]; route->w[k] = red[0]; r[redi[0]] = -3.402823466e38f; }
        __syncthreads();
    }
    if (tid == 0) {
        float sw = 0.f;
        for (int k = 0; k < top_k; ++k) sw += route->w[k];
        for (int k = 0; k < top_k; ++k) route->w[k] /= sw;
        for (int k = top_k; k < MOE_MAXK; ++k) { route->idx[k] = 0; route->w[k] = 0.f; }
    }
    __syncthreads();
    float gd = 0.f;
    if (sgi_len > 1 && sgi) for (int d = tid; d < dim; d += nt) gd += sgi[d] * x[d];
    red[tid] = gd; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) route->w[MOE_MAXK] = (sgi_len > 1 && sgi) ? 1.0f / (1.0f + expf(-red[0])) : 1.0f;
}

// Device-route variants of the fused MoE kernels (route in device memory so
// the whole layer chain runs without host knowledge of the selection).
__global__ void k_moe_gu_p(const uint8_t *__restrict__ gdw, const uint8_t *__restrict__ udw,
                           const uint8_t *__restrict__ sgdw, const uint8_t *__restrict__ sudw,
                           const float *__restrict__ x, float *__restrict__ h,
                           const MoeRoute *__restrict__ route, int top_k, int dim, int intermed,
                           long g_bpe, long u_bpe, int gk, int uk, int sgk, int suk) {
    extern __shared__ uint8_t moe_async_smem[];
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int total = (top_k + 1) * intermed;
    if (wid >= total) return;
    uint8_t *stg = moe_async_smem + (size_t) (threadIdx.x >> 5) * (2 * MOE_ASYNC_CHUNK_B);
    int k = wid / intermed, r = wid % intermed;
    const uint8_t *grow, *urow; int gkind, ukind;
    if (k < top_k) {
        int e = route->idx[k];
        grow = gdw + (size_t) e * g_bpe + (size_t) r * (dim / 256) * kq_bpb(gk);
        urow = udw + (size_t) e * u_bpe + (size_t) r * (dim / 256) * kq_bpb(uk);
        gkind = gk; ukind = uk;
    } else {
        grow = sgdw + (size_t) r * (dim / 256) * kq_bpb(sgk);
        urow = sudw + (size_t) r * (dim / 256) * kq_bpb(suk);
        gkind = sgk; ukind = suk;
    }
    float g = wrow_maybe_async(grow, gkind, x, dim, lane, stg);
    float u = wrow_maybe_async(urow, ukind, x, dim, lane, stg);
    if (lane == 0)
        h[(size_t) k * intermed + r] = (g / (1.f + expf(-g))) * u;
}

__global__ void k_moe_down_p(const uint8_t *__restrict__ ddw, const uint8_t *__restrict__ sddw,
                             const float *__restrict__ h, float *__restrict__ y,
                             const MoeRoute *__restrict__ route, int top_k, int intermed, int dim,
                             long d_bpe, int dk, int sdk) {
    extern __shared__ uint8_t moe_async_smem[];
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= dim) return;
    uint8_t *stg = moe_async_smem + (size_t) (threadIdx.x >> 5) * (2 * MOE_ASYNC_CHUNK_B);
    size_t bpr_d = (size_t) (intermed / 256) * kq_bpb(dk);
    size_t bpr_s = (size_t) (intermed / 256) * kq_bpb(sdk);
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        const uint8_t *row = ddw + (size_t) route->idx[k] * d_bpe + (size_t) wid * bpr_d;
        acc += route->w[k] * wrow_maybe_async(row, dk, h + (size_t) k * intermed, intermed, lane, stg);
    }
    acc += route->w[MOE_MAXK]
           * wrow_maybe_async(sddw + (size_t) wid * bpr_s, sdk, h + (size_t) top_k * intermed, intermed, lane, stg);
    if (lane == 0) y[wid] = acc;
}

// ---- Batched MoE: all B lanes in one launch (lane = warp / rows-per-lane) ----
// Each lane keeps its own routing (route_b[lane]) and its own I/O slice, so the
// per-lane expert gather is unchanged — this only removes the B-way launch loop
// and lifts occupancy (B x the blocks).
__global__ void k_moe_gu_p_b(const uint8_t *__restrict__ gdw, const uint8_t *__restrict__ udw,
                             const uint8_t *__restrict__ sgdw, const uint8_t *__restrict__ sudw,
                             const float *__restrict__ x_b, float *__restrict__ h_b,
                             const MoeRoute *__restrict__ route_b, int top_k, int dim, int intermed,
                             long g_bpe, long u_bpe, int gk, int uk, int sgk, int suk, int B) {
    int total = (top_k + 1) * intermed;
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int bl = gw / total, wid = gw % total;
    if (bl >= B) return;
    const MoeRoute *route = route_b + bl;
    const float *x = x_b + (size_t) bl * dim;
    float *h = h_b + (size_t) bl * (MOE_MAXK + 1) * intermed;
    int k = wid / intermed, r = wid % intermed;
    const uint8_t *grow, *urow; int gkind, ukind;
    if (k < top_k) {
        int e = route->idx[k];
        grow = gdw + (size_t) e * g_bpe + (size_t) r * (dim / 256) * kq_bpb(gk);
        urow = udw + (size_t) e * u_bpe + (size_t) r * (dim / 256) * kq_bpb(uk);
        gkind = gk; ukind = uk;
    } else {
        grow = sgdw + (size_t) r * (dim / 256) * kq_bpb(sgk);
        urow = sudw + (size_t) r * (dim / 256) * kq_bpb(suk);
        gkind = sgk; ukind = suk;
    }
    float g = wrow(grow, gkind, x, dim, lane);
    float u = wrow(urow, ukind, x, dim, lane);
    if (lane == 0) h[(size_t) k * intermed + r] = (g / (1.f + expf(-g))) * u;
}

__global__ void k_moe_down_p_b(const uint8_t *__restrict__ ddw, const uint8_t *__restrict__ sddw,
                               const float *__restrict__ h_b, float *__restrict__ y_b,
                               const MoeRoute *__restrict__ route_b, int top_k, int intermed, int dim,
                               long d_bpe, int dk, int sdk, int B) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int bl = gw / dim, wid = gw % dim;
    if (bl >= B) return;
    const MoeRoute *route = route_b + bl;
    const float *h = h_b + (size_t) bl * (MOE_MAXK + 1) * intermed;
    size_t bpr_d = (size_t) (intermed / 256) * kq_bpb(dk);
    size_t bpr_s = (size_t) (intermed / 256) * kq_bpb(sdk);
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        const uint8_t *row = ddw + (size_t) route->idx[k] * d_bpe + (size_t) wid * bpr_d;
        acc += route->w[k] * wrow(row, dk, h + (size_t) k * intermed, intermed, lane);
    }
    acc += route->w[MOE_MAXK]
           * wrow(sddw + (size_t) wid * bpr_s, sdk, h + (size_t) top_k * intermed, intermed, lane);
    if (lane == 0) y_b[(size_t) bl * dim + wid] = acc;
}

__global__ void k_moe_route_b(const float *__restrict__ rl_b, const float *__restrict__ x_b,
                              const float *__restrict__ sgi, int sgi_len, int n_exp, int top_k,
                              int dim, MoeRoute *__restrict__ route_b, int B) {
    int bl = blockIdx.x; if (bl >= B) return;
    const float *rl = rl_b + (size_t) bl * n_exp;
    const float *x = x_b + (size_t) bl * dim;
    MoeRoute *route = route_b + bl;
    __shared__ float r[512]; __shared__ float red[256]; __shared__ int redi[256];
    __shared__ float mx, sm;
    int tid = threadIdx.x, nt = blockDim.x;
    if (n_exp > 512) return;
    float lmax = -3.402823466e38f;
    for (int e = tid; e < n_exp; e += nt) { r[e] = rl[e]; if (r[e] > lmax) lmax = r[e]; }
    red[tid] = lmax; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    if (tid == 0) mx = red[0]; __syncthreads();
    float ls = 0.f;
    for (int e = tid; e < n_exp; e += nt) { r[e] = expf(r[e] - mx); ls += r[e]; }
    red[tid] = ls; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) sm = red[0]; __syncthreads();
    for (int e = tid; e < n_exp; e += nt) r[e] /= sm;
    __syncthreads();
    for (int kk = 0; kk < top_k; ++kk) {
        float bv = -3.402823466e38f; int bi = 0;
        for (int e = tid; e < n_exp; e += nt) if (r[e] > bv) { bv = r[e]; bi = e; }
        red[tid] = bv; redi[tid] = bi; __syncthreads();
        for (int o = nt / 2; o > 0; o >>= 1) {
            if (tid < o && red[tid + o] > red[tid]) { red[tid] = red[tid + o]; redi[tid] = redi[tid + o]; }
            __syncthreads();
        }
        if (tid == 0) { route->idx[kk] = redi[0]; route->w[kk] = red[0]; r[redi[0]] = -3.402823466e38f; }
        __syncthreads();
    }
    if (tid == 0) {
        float sw = 0.f;
        for (int kk = 0; kk < top_k; ++kk) sw += route->w[kk];
        for (int kk = 0; kk < top_k; ++kk) route->w[kk] /= sw;
        for (int kk = top_k; kk < MOE_MAXK; ++kk) { route->idx[kk] = 0; route->w[kk] = 0.f; }
    }
    __syncthreads();
    float gd = 0.f;
    if (sgi_len > 1 && sgi) for (int d = tid; d < dim; d += nt) gd += sgi[d] * x[d];
    red[tid] = gd; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) route->w[MOE_MAXK] = (sgi_len > 1 && sgi) ? 1.0f / (1.0f + expf(-red[0])) : 1.0f;
}

struct ChainLayer {
    int is_fattn;
    float *attn_norm, *post_norm;
    // delta-net
    uint8_t *qkv, *al, *be, *ga, *ow; int qkv_k, al_k, be_k, ga_k, ow_k;
    // qkv+alpha+beta+gate concatenated into one Q8_0 weight → single matvec
    // (fewer kernels in the latency-bound decode chain). Only when all 4 share a kind.
    uint8_t *proj; int proj_out, proj_fused;
    float *conv, *aw, *dtw, *nw;
    int nv, khd, vhd, qo, q_dim, nkh, v_dim, kernel;
    // full-attn
    uint8_t *qw, *kw, *vw, *fow; int qw_k, kw_k, vw_k, fow_k;
    float *qn, *kn, *ffp;
    int nq, nkv, hd, rd, yarn_on, use_ff, interleaved, sec_total;
    float base, freq_scale, m_scale, corr_lo, corr_hi;
    // moe
    uint8_t *rw; int rk;
    uint8_t *gdw, *udw, *ddw, *sgdw, *sudw, *sddw;
    int gk, uk, dk, sgk, suk, sdk;
    long g_bpe, u_bpe, d_bpe;
    float *sgi; int sgi_len;
    int n_exp, top_k, intermed;
    int has_moe;
    //  MoE Phase B: routed expert weights as ggml tensors (mul_mat_id prefill).
    //  gdw/udw/ddw above point at THESE tensors' device data when set — the
    //  decode kernels keep reading the same bytes.  null => grouped fallback.
    void *ggt, *ugt, *dgt;
};
static std::vector<ChainLayer> g_chain;
static float *g_ch_embed = nullptr, *g_ch_fnorm = nullptr;
static uint8_t *g_ch_lm = nullptr; static int g_ch_lm_k = -1;  // LM head kept in native quant
static int g_ch_dim = 0, g_ch_vocab = 0, g_ch_ready = 0;
static int g_mx_qo = 0, g_mx_nv = 0, g_mx_vd = 0, g_mx_qgd = 0, g_mx_kvd = 0,
           g_mx_att = 0, g_mx_hbuf = 0, g_mx_nexp = 0;

extern "C" void aspida_gpu_chain_reset(void) {
    g_chain.clear(); g_ch_embed = nullptr; g_ch_ready = 0;
    g_mx_qo = g_mx_nv = g_mx_vd = g_mx_qgd = g_mx_kvd = g_mx_att = g_mx_hbuf = g_mx_nexp = 0;
}

extern "C" void aspida_gpu_chain_dnet(
    const float *attn_norm, long an_b, const float *post_norm, long pn_b,
    const void *qkv_w, long qkv_b, int qkv_k,
    const void *al_w, long al_b, int al_k,
    const void *be_w, long be_b, int be_k,
    const void *ga_w, long ga_b, int ga_k,
    const void *out_w, long out_b, int out_k,
    const float *conv_w, long conv_b, const float *a_w, long a_b,
    const float *dt_w, long dt_b, const float *norm_w, long norm_b,
    int nv, int khd, int vhd, int qo, int q_dim, int n_k_heads, int v_dim, int kernel) {
    ChainLayer L = {}; L.is_fattn = 0;
    L.attn_norm = (float *) upload_weight(attn_norm, an_b);
    L.post_norm = (float *) upload_weight(post_norm, pn_b);
    L.qkv = upload_weight(qkv_w, qkv_b); L.qkv_k = qkv_k;
    L.al = upload_weight(al_w, al_b); L.al_k = al_k;
    L.be = upload_weight(be_w, be_b); L.be_k = be_k;
    L.ga = upload_weight(ga_w, ga_b); L.ga_k = ga_k;
    L.ow = upload_weight(out_w, out_b); L.ow_k = out_k;
    // Fuse the four input projections (qkv|alpha|beta|gate) into one contiguous
    // Q8_0 weight so the decode chain issues 1 matvec instead of 4. Q8_0 is
    // row-major, so concatenating the byte blobs == stacking the output rows.
    L.proj = nullptr; L.proj_fused = 0; L.proj_out = qo + nv + nv + v_dim;
    if (qkv_k >= 0 && qkv_k == al_k && al_k == be_k && be_k == ga_k) {
        size_t tot = (size_t) qkv_b + al_b + be_b + ga_b;
        if (cudaMalloc(&L.proj, tot) == cudaSuccess) {
            uint8_t *p = L.proj;
            cudaMemcpy(p, L.qkv, qkv_b, cudaMemcpyDeviceToDevice); p += qkv_b;
            cudaMemcpy(p, L.al, al_b, cudaMemcpyDeviceToDevice);   p += al_b;
            cudaMemcpy(p, L.be, be_b, cudaMemcpyDeviceToDevice);   p += be_b;
            cudaMemcpy(p, L.ga, ga_b, cudaMemcpyDeviceToDevice);
            L.proj_fused = 1;
        }
    }
    L.conv = (float *) upload_weight(conv_w, conv_b);
    L.aw = (float *) upload_weight(a_w, a_b);
    L.dtw = (float *) upload_weight(dt_w, dt_b);
    L.nw = (float *) upload_weight(norm_w, norm_b);
    L.nv = nv; L.khd = khd; L.vhd = vhd; L.qo = qo; L.q_dim = q_dim;
    L.nkh = n_k_heads; L.v_dim = v_dim; L.kernel = kernel;
    if (qo > g_mx_qo) g_mx_qo = qo;
    if (nv > g_mx_nv) g_mx_nv = nv;
    if (v_dim > g_mx_vd) g_mx_vd = v_dim;
    g_chain.push_back(L);
}

extern "C" void aspida_gpu_chain_fattn(
    const float *attn_norm, long an_b, const float *post_norm, long pn_b,
    const void *q_w, long q_b, int q_k, const void *k_w, long k_b, int k_k,
    const void *v_w, long v_b, int v_k, const void *o_w, long o_b, int o_k,
    const float *q_norm, long qn_b, const float *k_norm, long kn_b,
    int nq, int nkv, int hd,
    int rd, float base, float freq_scale, float m_scale,
    int yarn_on, float corr_lo, float corr_hi,
    const float *ff, long ff_b, int use_ff, int interleaved, int sec_total) {
    ChainLayer L = {}; L.is_fattn = 1;
    L.attn_norm = (float *) upload_weight(attn_norm, an_b);
    L.post_norm = (float *) upload_weight(post_norm, pn_b);
    L.qw = upload_weight(q_w, q_b); L.qw_k = q_k;
    L.kw = upload_weight(k_w, k_b); L.kw_k = k_k;
    L.vw = upload_weight(v_w, v_b); L.vw_k = v_k;
    L.fow = upload_weight(o_w, o_b); L.fow_k = o_k;
    L.qn = (float *) upload_weight(q_norm, qn_b);
    L.kn = (float *) upload_weight(k_norm, kn_b);
    L.ffp = (use_ff && ff) ? (float *) upload_weight(ff, ff_b) : nullptr;
    L.nq = nq; L.nkv = nkv; L.hd = hd; L.rd = rd; L.base = base;
    L.freq_scale = freq_scale; L.m_scale = m_scale; L.yarn_on = yarn_on;
    L.corr_lo = corr_lo; L.corr_hi = corr_hi; L.use_ff = use_ff;
    L.interleaved = interleaved; L.sec_total = sec_total;
    int qgd = nq * 2 * hd, kvd = nkv * hd, att = nq * hd;
    if (qgd > g_mx_qgd) g_mx_qgd = qgd;
    if (kvd > g_mx_kvd) g_mx_kvd = kvd;
    if (att > g_mx_att) g_mx_att = att;
    g_chain.push_back(L);
}

extern "C" void aspida_gpu_chain_moe(
    const void *router_w, long router_b, int router_k,
    const void *gate_w, long gate_b, int gate_k,
    const void *up_w, long up_b, int up_k,
    const void *down_w, long down_b, int down_k,
    const void *shg_w, long shg_b, int shg_k,
    const void *shu_w, long shu_b, int shu_k,
    const void *shd_w, long shd_b, int shd_k,
    const float *sgi, long sgi_b, int sgi_len,
    int n_exp, int top_k, int intermed) {
    if (g_chain.empty()) return;
    ChainLayer &L = g_chain.back();
    L.rw = upload_weight(router_w, router_b); L.rk = router_k;
    //  MoE Phase B: allocate the ROUTED expert weights as ggml tensors so the
    //  prefill can run llama.cpp's mul_mat_id (MMQ int8 tensor cores, measured
    //  2.9x the grouped kernels).  ggml CUDA buffers are plain device memory,
    //  so gdw/udw/ddw keep pointing at the same bytes for the decode kernels.
    //  Bytes upload ONCE (into ggml memory) — no double residency.  Q8_0 only;
    //  any mismatch (or ASPIDA_MOE_NOGGML=1) falls back to upload_weight and
    //  the old grouped prefill path.
    static int moe_noggml_ld = getenv("ASPIDA_MOE_NOGGML") ? 1 : 0;
    L.ggt = L.ugt = L.dgt = nullptr;
    if (!moe_noggml_ld && gate_k == 5 && up_k == 5 && down_k == 5 && n_exp > 0 && intermed > 0) {
        long gbpe = gate_b / n_exp, ubpe = up_b / n_exp, dbpe = down_b / n_exp;
        //  gate/up: m=intermed rows of k=dim cols; down: k=intermed, m=dim.
        long gbpr = gbpe / intermed;                    //  bytes per row
        int64_t kdim = (gbpr % 34 == 0) ? (gbpr / 34) * 32 : 0;
        long dnb = intermed / 32, dbpr = dnb * 34;
        int64_t mdim = (dbpr > 0 && dbpe % dbpr == 0) ? dbpe / dbpr : 0;
        if (kdim > 0 && mdim > 0 && gbpe == ubpe) {
            ggml_tensor *gt = aspida_ggml_upload_q8(gate_w, gate_b, kdim, intermed, n_exp);
            ggml_tensor *ut = gt ? aspida_ggml_upload_q8(up_w, up_b, kdim, intermed, n_exp) : nullptr;
            ggml_tensor *dt = ut ? aspida_ggml_upload_q8(down_w, down_b, intermed, mdim, n_exp) : nullptr;
            if (gt && ut && dt) {
                L.ggt = gt; L.ugt = ut; L.dgt = dt;
                L.gdw = (uint8_t *) gt->data; L.udw = (uint8_t *) ut->data; L.ddw = (uint8_t *) dt->data;
            }
        }
    }
    if (!L.ggt) {
        L.gdw = upload_weight(gate_w, gate_b);
        L.udw = upload_weight(up_w, up_b);
        L.ddw = upload_weight(down_w, down_b);
    }
    L.gk = gate_k; L.uk = up_k; L.dk = down_k;
    L.sgdw = upload_weight(shg_w, shg_b); L.sgk = shg_k;
    L.sudw = upload_weight(shu_w, shu_b); L.suk = shu_k;
    L.sddw = upload_weight(shd_w, shd_b); L.sdk = shd_k;
    L.sgi = (sgi_len > 1 && sgi) ? (float *) upload_weight(sgi, sgi_b) : nullptr;
    L.sgi_len = sgi_len;
    L.g_bpe = gate_b / n_exp; L.u_bpe = up_b / n_exp; L.d_bpe = down_b / n_exp;
    L.n_exp = n_exp; L.top_k = top_k > MOE_MAXK ? MOE_MAXK : top_k;
    L.intermed = intermed; L.has_moe = 1;
    int hbuf = (MOE_MAXK + 1) * intermed;
    if (hbuf > g_mx_hbuf) g_mx_hbuf = hbuf;
    if (n_exp > g_mx_nexp) g_mx_nexp = n_exp;
}

extern "C" void aspida_gpu_chain_model(
    const float *embed, long embed_b, const float *fnorm, long fnorm_b,
    const void *lm, long lm_b, int lm_k, int dim, int vocab) {
    g_ch_embed = (float *) upload_weight(embed, embed_b);
    g_ch_fnorm = (float *) upload_weight(fnorm, fnorm_b);
    // Keep the output projection in its native quant (Q8_0 = 4x fewer bytes than
    // dequantized F32) — it is read in full every token, so this is ~1.7ms/token.
    g_ch_lm = upload_weight(lm, lm_b); g_ch_lm_k = lm_k;
    g_ch_dim = dim; g_ch_vocab = vocab; g_ch_ready = 1;
}

extern "C" int aspida_gpu_chain_ready(void) { return g_ch_ready && !g_chain.empty(); }

// Persistent chain scratch + graph state.
static float *H = nullptr, *nx = nullptr, *ao = nullptr, *dlog = nullptr,
             *dqkv = nullptr, *dcq = nullptr, *dar = nullptr, *dbr = nullptr,
             *dz = nullptr, *dg = nullptr, *db = nullptr, *dor = nullptr, *dproj = nullptr,
             *dqg = nullptr, *dkt = nullptr, *dvt = nullptr, *dqa = nullptr,
             *dga = nullptr, *datt = nullptr, *drl = nullptr, *dhb = nullptr;
static MoeRoute *droute = nullptr;
static int ch_inited = 0;
static int *g_d_row = nullptr, *g_d_pos = nullptr;   // device-side per-token inputs
static const int *g_handles = nullptr;               // per-generation state handles
static cudaStream_t g_cstream = 0;
//  Stream priority: DECODE (interactive, latency-critical) gets the highest
//  priority, PREFILL (throughput, bursty, saturates HBM for tens of ms) the
//  lowest — so a short decode token can slip through while a big prefill runs,
//  instead of being stuck behind it. high=1 returns the highest-priority value
//  (most negative on CUDA), high=0 the lowest.
static int aspida_stream_prio(int high) {
    int lo = 0, hi = 0; cudaDeviceGetStreamPriorityRange(&lo, &hi);
    return high ? hi : lo;
}
static cudaGraph_t g_graph = nullptr;
static cudaGraphExec_t g_gexec = nullptr;
static int g_captured = 0;

static void chain_alloc(void) {
    int dim = g_ch_dim;
    if (ch_inited) return;
    cudaMalloc(&H, (size_t) dim * 4); cudaMalloc(&nx, (size_t) dim * 4);
    cudaMalloc(&ao, (size_t) dim * 4);
    cudaMalloc(&dlog, (size_t) g_ch_vocab * 4);
    if (g_mx_qo) { cudaMalloc(&dqkv, (size_t) g_mx_qo * 4); cudaMalloc(&dcq, (size_t) g_mx_qo * 4); }
    if (g_mx_nv) { cudaMalloc(&dar, (size_t) g_mx_nv * 4); cudaMalloc(&dbr, (size_t) g_mx_nv * 4);
                   cudaMalloc(&dg, (size_t) g_mx_nv * 4); cudaMalloc(&db, (size_t) g_mx_nv * 4); }
    if (g_mx_vd) { cudaMalloc(&dz, (size_t) g_mx_vd * 4); cudaMalloc(&dor, (size_t) g_mx_vd * 4); }
    // Combined projection output buffer: [qkv | alpha | beta | gate] contiguous.
    cudaMalloc(&dproj, (size_t) (g_mx_qo + 2 * g_mx_nv + g_mx_vd) * 4);
    if (g_mx_qgd) cudaMalloc(&dqg, (size_t) g_mx_qgd * 4);
    if (g_mx_kvd) { cudaMalloc(&dkt, (size_t) g_mx_kvd * 4); cudaMalloc(&dvt, (size_t) g_mx_kvd * 4); }
    if (g_mx_att) { cudaMalloc(&dqa, (size_t) g_mx_att * 4); cudaMalloc(&dga, (size_t) g_mx_att * 4);
                    cudaMalloc(&datt, (size_t) g_mx_att * 4); }
    if (g_mx_nexp) cudaMalloc(&drl, (size_t) g_mx_nexp * 4);
    if (g_mx_hbuf) cudaMalloc(&dhb, (size_t) g_mx_hbuf * 4);
    cudaMalloc(&droute, sizeof(MoeRoute));
    cudaMalloc(&g_d_row, 4); cudaMalloc(&g_d_pos, 4);
    cudaStreamCreateWithPriority(&g_cstream, cudaStreamDefault, aspida_stream_prio(1));
    ch_inited = 1;
}

// Record the fixed per-token kernel sequence onto `st` (a capture stream or the
// default stream). embed_row/pos are read from device memory (g_d_row/g_d_pos),
// so the identical sequence serves every token — capturable into a CUDA graph.
static void chain_record(cudaStream_t st) {
    int dim = g_ch_dim;
    int gblk = (dim + 255) / 256;
    k_embed<<<gblk, 256, 0, st>>>(g_ch_embed, g_d_row, H, dim);
    for (size_t li = 0; li < g_chain.size(); ++li) {
        ChainLayer &L = g_chain[li];
        k_norm1<<<1, 256, 0, st>>>(H, L.attn_norm, nx, dim);
        if (!L.is_fattn) {
            DnetState ds = g_dnet[g_handles[li]];
            float *uqkv = dqkv, *uar = dar, *ubr = dbr, *uz = dz;
            if (L.proj_fused) {
                // one matvec over the stacked [qkv|alpha|beta|gate] rows, then slice
                launch_mv_st(L.proj, L.qkv_k, dim, L.proj_out, nx, dproj, st);
                uqkv = dproj; uar = dproj + L.qo; ubr = dproj + L.qo + L.nv;
                uz = dproj + L.qo + 2 * L.nv;
            } else {
                launch_mv_st(L.qkv, L.qkv_k, dim, L.qo, nx, dqkv, st);
                launch_mv_st(L.al, L.al_k, dim, L.nv, nx, dar, st);
                launch_mv_st(L.be, L.be_k, dim, L.nv, nx, dbr, st);
                launch_mv_st(L.ga, L.ga_k, dim, L.v_dim, nx, dz, st);
            }
            k_dnet_conv<<<(L.qo + 255) / 256, 256, 0, st>>>(uqkv, ds.hist, L.conv, dcq, L.qo, L.kernel);
            k_dnet_gates<<<(L.nv + 255) / 256, 256, 0, st>>>(uar, ubr, L.aw, L.dtw, dg, db, L.nv);
            size_t shmem = (size_t) (4 * L.khd + 2 * L.vhd) * 4;
            k_dnet_recur<<<L.nv, L.khd, shmem, st>>>(ds.S, dcq, dg, db, uz, L.nw, dor,
                                                     L.khd, L.vhd, L.q_dim, L.nkh);
            launch_mv_st(L.ow, L.ow_k, L.v_dim, dim, dor, ao, st);
        } else {
            FattnState fs = g_fattn[g_handles[li]];
            int kvd = L.nkv * L.hd, att = L.nq * L.hd, qgd = L.nq * 2 * L.hd;
            launch_mv_st(L.qw, L.qw_k, dim, qgd, nx, dqg, st);
            launch_mv_st(L.kw, L.kw_k, dim, kvd, nx, dkt, st);
            launch_mv_st(L.vw, L.vw_k, dim, kvd, nx, dvt, st);
            k_fattn_prep<<<L.nq + L.nkv, L.hd, (size_t) L.hd * 4, st>>>(
                dqg, dkt, dvt, L.qn, L.kn, dqa, dga, fs.K, fs.V,
                L.nq, L.nkv, L.hd, kvd, g_d_pos, L.rd, L.base, L.freq_scale, L.m_scale,
                L.yarn_on, L.corr_lo, L.corr_hi, L.ffp, L.use_ff, L.interleaved, L.sec_total);
            k_fattn_attend<<<L.nq, 256, 0, st>>>(dqa, dga, fs.K, fs.V, fs.scores, datt,
                                                 L.nq, L.nkv, L.hd, kvd, g_d_pos, fs.max_len);
            launch_mv_st(L.fow, L.fow_k, att, dim, datt, ao, st);
        }
        k_axpy<<<gblk, 256, 0, st>>>(H, 1.0f, ao, dim);
        if (L.has_moe) {
            k_norm1<<<1, 256, 0, st>>>(H, L.post_norm, nx, dim);
            launch_mv_st(L.rw, L.rk, dim, L.n_exp, nx, drl, st);
            k_moe_route<<<1, 256, 0, st>>>(drl, nx, L.sgi, L.sgi_len, L.n_exp, L.top_k, dim, droute);
            int w1 = (L.top_k + 1) * L.intermed, b1 = (w1 * 32 + 255) / 256;
            int b2 = (dim * 32 + 255) / 256;
            k_moe_gu_p<<<b1, 256, MOE_ASYNC_SHMEM(256), st>>>(L.gdw, L.udw, L.sgdw, L.sudw, nx, dhb, droute,
                                           L.top_k, dim, L.intermed, L.g_bpe, L.u_bpe,
                                           L.gk, L.uk, L.sgk, L.suk);
            k_moe_down_p<<<b2, 256, MOE_ASYNC_SHMEM(256), st>>>(L.ddw, L.sddw, dhb, ao, droute, L.top_k,
                                             L.intermed, dim, L.d_bpe, L.dk, L.sdk);
            k_axpy<<<gblk, 256, 0, st>>>(H, 1.0f, ao, dim);
        }
    }
    k_norm1<<<1, 256, 0, st>>>(H, g_ch_fnorm, nx, dim);
    launch_mv_st(g_ch_lm, g_ch_lm_k, dim, g_ch_vocab, nx, dlog, st);
}

// Begin a generation: bind the per-generation state handles and force a fresh
// graph capture (state device pointers changed vs the previous generation).
extern "C" void aspida_gpu_chain_begin(const int *attn_handles) {
    chain_alloc();
    g_handles = attn_handles;
    if (g_gexec) { cudaGraphExecDestroy(g_gexec); g_gexec = nullptr; }
    if (g_graph) { cudaGraphDestroy(g_graph); g_graph = nullptr; }
    g_captured = 0;
}

extern "C" void aspida_gpu_chain_end(void) {
    if (g_gexec) { cudaGraphExecDestroy(g_gexec); g_gexec = nullptr; }
    if (g_graph) { cudaGraphDestroy(g_graph); g_graph = nullptr; }
    g_captured = 0; g_handles = nullptr;
}

// Return (and clear) the last CUDA error as an int (0 == cudaSuccess). Called
// by the Ada side right after a forward: a failed GPU op (e.g. an allocation
// failure under VRAM pressure, or an illegal access) must ABORT the generation
// and release the inference lock, not leave a handler wedged holding the lock
// while the GPU sits idle — the shape of the 2026-07-13 prod GPU-0% wedge.
extern "C" int aspida_gpu_last_error(void) {
    return (int) cudaGetLastError();
}

// One decode step. First call of a generation captures the graph; the rest
// replay it — one launch for the whole ~350-kernel token instead of 350.
extern "C" void aspida_gpu_chain_forward(int embed_row, int pos,
                                         const int *attn_handles, float *logits) {
    // CUDA graph capture speeds the per-token chain (one launch vs ~350) but is
    // FRAGILE when another CUDA context shares the device: a co-resident Ollama
    // model (bge-m3) doing GPU work during the first-token capture corrupts the
    // captured graph — garbage logits (immediate stop, 0 tokens) on Ada RTX, and
    // a hard hang on an NVIDIA GPU (the 2026-07-13 prod first-inference hang). Set
    // ASPIDA_NO_GRAPH to take the robust direct-launch path (same kernels, one
    // launch each, no capture): slower but correct when the GPU is shared.
    static int no_graph = getenv("ASPIDA_NO_GRAPH") ? 1 : 0;
    if (g_handles == nullptr) aspida_gpu_chain_begin(attn_handles);
    cudaMemcpy(g_d_row, &embed_row, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_pos, &pos, 4, cudaMemcpyHostToDevice);
    if (!no_graph && !g_captured) {
        cudaStreamBeginCapture(g_cstream, cudaStreamCaptureModeThreadLocal);
        chain_record(g_cstream);
        cudaStreamEndCapture(g_cstream, &g_graph);
        cudaGraphInstantiate(&g_gexec, g_graph, nullptr, nullptr, 0);
        g_captured = 1;
    }
    static int cprof = getenv("ASPIDA_CHAIN_PROF") ? 1 : 0;
    static cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (cprof && !ev0) { cudaEventCreate(&ev0); cudaEventCreate(&ev1); }
    if (cprof) cudaEventRecord(ev0, g_cstream);
    if (no_graph) {
        chain_record(g_cstream);          // launch the kernels directly
    } else {
        cudaGraphLaunch(g_gexec, g_cstream);
    }
    cudaMemcpyAsync(logits, dlog, (size_t) g_ch_vocab * 4, cudaMemcpyDeviceToHost, g_cstream);
    if (cprof) cudaEventRecord(ev1, g_cstream);
    cudaStreamSynchronize(g_cstream);
    if (cprof) {
        static double gpu_acc = 0, wall_acc = 0; static long n = 0;
        static struct timespec t0; static int have = 0;
        float gms = 0; cudaEventElapsedTime(&gms, ev0, ev1);
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        if (have) {
            wall_acc += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            gpu_acc += gms * 1e-3; n++;
        }
        have = 1; t0 = t1;
        if (n && n % 50 == 0)
            fprintf(stderr, "[CHAINPROF] GPU-forward %.3f ms | wall/tok %.3f ms | E2EE+host tail %.3f ms (n=%ld)\n",
                    1e3 * gpu_acc / n, 1e3 * wall_acc / n, 1e3 * (wall_acc - gpu_acc) / n, n);
    }
}


// =====================================================================
// Phase E — batched (continuous-batching) decode. The single-request chain
// above is untouched. Here B lanes share one forward: the shared-weight
// matvecs (projections, router, attn q/k/v/o, final norm, LM head) go through
// the batched warp kernels (weight read ONCE, B sequence accumulators — the
// proven throughput win); the stateful/routing parts (delta-net recurrence,
// full-attn over per-lane KV, MoE expert selection) loop over the B lanes.
// =====================================================================
__global__ void k_embed_b(const float *__restrict__ embed, const int *__restrict__ rows,
                          float *__restrict__ H, int dim, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; int n = dim * B;
    if (i >= n) return;
    int b = i / dim, d = i % dim;
    H[(size_t) b * dim + d] = embed[(size_t) rows[b] * dim + d];
}
__global__ void k_norm1_b(const float *__restrict__ x, const float *__restrict__ w,
                          float *__restrict__ y, int n, int B) {
    int b = blockIdx.x; if (b >= B) return;
    const float *xb = x + (size_t) b * n; float *yb = y + (size_t) b * n;
    __shared__ float red[256]; __shared__ float rms;
    int tid = threadIdx.x, nt = blockDim.x; float ls = 0.f;
    for (int i = tid; i < n; i += nt) ls += xb[i] * xb[i];
    red[tid] = ls; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) rms = sqrtf(red[0] / (float) n + 1e-6f);
    __syncthreads();
    for (int i = tid; i < n; i += nt) yb[i] = (xb[i] / rms) * w[i];
}
// H[b] += res[b] then y[b] = norm(H[b])*w  (fused residual, batched).
__global__ void k_norm_res_b(float *__restrict__ H, const float *__restrict__ res,
                             const float *__restrict__ w, float *__restrict__ y, int n, int B) {
    int b = blockIdx.x; if (b >= B) return;
    float *Hb = H + (size_t) b * n; const float *rb = res + (size_t) b * n; float *yb = y + (size_t) b * n;
    __shared__ float red[256]; __shared__ float rms;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < n; i += nt) Hb[i] += rb[i];
    __syncthreads();
    float ls = 0.f;
    for (int i = tid; i < n; i += nt) ls += Hb[i] * Hb[i];
    red[tid] = ls; __syncthreads();
    for (int o = nt / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) rms = sqrtf(red[0] / (float) n + 1e-6f);
    __syncthreads();
    for (int i = tid; i < n; i += nt) yb[i] = (Hb[i] / rms) * w[i];
}
// Batched dense F32 matvec: Y[b,out] = W[out,in] . X[b,in], weight read once.
__global__ void k_dense_mv_b(const float *__restrict__ w, const float *__restrict__ x,
                             float *__restrict__ y, int in, int out, int B) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= out) return;
    const float *r = w + (size_t) row * in;
    float acc[MAXB]; for (int b = 0; b < B; ++b) acc[b] = 0.f;
    for (int i = lane; i < in; i += 32) { float wv = r[i]; for (int b = 0; b < B; ++b) acc[b] += wv * x[(size_t) b * in + i]; }
    for (int b = 0; b < B; ++b) { float a = warp_reduce(acc[b]); if (lane == 0) y[(size_t) b * out + row] = a; }
}
// Batched matvec dispatch (weight read once for B lanes).
// Tensor-core batched Q8_0 matmul: Y[B,out] = X[B,in] @ W[out,in]^T.
// One warp computes a 16(M)x16(N) output tile, looping K in 32-wide Q8 blocks;
// W stays Q8 in HBM (read once), dequantized into a shared FP16 tile, multiplied
// by FP16 tensor cores. Weight-stationary in the true sense — per-token cost
// falls with B (12x over the warp-per-row kernel at B=128), which is what makes
// chunked prefill competitive. Handles any B/out (partial tiles guarded).
#define WMM_TM 16
#define WMM_TN 16
#define WMM_TK 32
__global__ void k_q8_wmma(const uint8_t *__restrict__ w, const float *__restrict__ x,
                          float *__restrict__ y, int in, int out, int B) {
    int n0 = blockIdx.x * WMM_TN, m0 = blockIdx.y * WMM_TM;
    int lane = threadIdx.x & 31;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ half As[WMM_TM * WMM_TK];   // x tile [16 M][32 K] row-major, ld=32
    __shared__ half Bs[WMM_TK * WMM_TN];   // W tile [32 K][16 N] row-major, ld=16; Bs[k][n]=W[n,k]
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
    nvcuda::wmma::fill_fragment(cf, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int n = 0; n < WMM_TN; ++n) {
            int gn = n0 + n;
            const uint8_t *bl = w + (size_t) gn * bpr + (size_t) kb * 34;
            float d = f16(bl); const int8_t *qs = (const int8_t *) (bl + 2);
            Bs[(size_t) lane * WMM_TN + n] = __float2half(d * (float) qs[lane]);
        }
        for (int m = 0; m < WMM_TM; ++m) {
            int gm = m0 + m;
            float xv = (gm < B) ? x[(size_t) gm * in + kb * 32 + lane] : 0.0f;
            As[(size_t) m * WMM_TK + lane] = __float2half(xv);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, WMM_TK);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * WMM_TN, WMM_TN);
            nvcuda::wmma::mma_sync(cf, af, bf, cf);
        }
        __syncwarp();
    }
    __shared__ float Cs[WMM_TM * WMM_TN];
    nvcuda::wmma::store_matrix_sync(Cs, cf, WMM_TN, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
        int m = idx / WMM_TN, n = idx % WMM_TN, gm = m0 + m, gn = n0 + n;
        if (gm < B && gn < out) y[(size_t) gm * out + gn] = Cs[idx];
    }
}

//  Register-tiled Q8 tensor-core matmul (moe-lever, 2026-07-15): each warp
//  computes an MT x NT grid of 16x16 output fragments (e.g. 4x4 = a 64x64
//  tile), issuing MT*NT mma per A/B fragment-load instead of 2. The 16x16
//  k_q8_wmma was TC-UNDERUTILIZED (22.9 TFLOP/s; NOT memory- or dequant-
//  bound — measured via share-weight and fp16-weight diagnostics); reg4x4
//  reaches 62.3 TFLOP/s = 2.72x, BIT-EXACT (same dequant values, same
//  per-tile K order). Caveats: needs out % (16*NT) == 0 (a tail guard in
//  the hot dequant loop collapses perf ~3x — gate + fall back instead) and
//  enough blocks to fill the GPU (starves below ~512 blocks: qkv at B=256
//  was 0.48x) — hence the block-count dispatch gates in launch_mv_b.
template<int MT, int NT>
__global__ void k_q8_reg(const uint8_t *__restrict__ w, const float *__restrict__ x,
                         float *__restrict__ y, int in, int out, int B) {
    int n0 = blockIdx.x * (WMM_TN * NT), m0 = blockIdx.y * (WMM_TM * MT);
    int lane = threadIdx.x & 31;
    int nb = in / 32; size_t bpr = (size_t) nb * 34;
    __shared__ half As[MT][WMM_TM * WMM_TK];
    __shared__ half Bs[NT][WMM_TK * WMM_TN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf[MT][NT];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j) nvcuda::wmma::fill_fragment(cf[i][j], 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        //  No tail guard here (branching in this hot loop collapses perf ~3x);
        //  the caller guarantees out % (16*NT) == 0.
        #pragma unroll
        for (int j = 0; j < NT; ++j) for (int n = 0; n < WMM_TN; ++n) {
            const uint8_t *bl = w + (size_t) (n0 + j * WMM_TN + n) * bpr + (size_t) kb * 34;
            Bs[j][(size_t) lane * WMM_TN + n] = __float2half(f16(bl) * (float) ((const int8_t *) (bl + 2))[lane]);
        }
        #pragma unroll
        for (int i = 0; i < MT; ++i) for (int m = 0; m < WMM_TM; ++m) {
            int gm = m0 + i * WMM_TM + m;
            float xv = (gm < B) ? x[(size_t) gm * in + kb * 32 + lane] : 0.0f;
            As[i][(size_t) m * WMM_TK + lane] = __float2half(xv);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af[MT];
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf[NT];
            #pragma unroll
            for (int i = 0; i < MT; ++i) nvcuda::wmma::load_matrix_sync(af[i], As[i] + k16 * 16, WMM_TK);
            #pragma unroll
            for (int j = 0; j < NT; ++j) nvcuda::wmma::load_matrix_sync(bf[j], Bs[j] + (size_t) k16 * 16 * WMM_TN, WMM_TN);
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j) nvcuda::wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
        }
        __syncwarp();
    }
    __shared__ float Cs[WMM_TM * WMM_TN];
    #pragma unroll
    for (int i = 0; i < MT; ++i) for (int j = 0; j < NT; ++j) {
        __syncwarp();
        nvcuda::wmma::store_matrix_sync(Cs, cf[i][j], WMM_TN, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < WMM_TM * WMM_TN; idx += 32) {
            int m = idx / WMM_TN, n = idx % WMM_TN;
            int gm = m0 + i * WMM_TM + m, gn = n0 + j * WMM_TN + n;
            if (gm < B && gn < out) y[(size_t) gm * out + gn] = Cs[idx];
        }
    }
}

static inline void launch_mv_b(const uint8_t *dw, int kind, int in, int out,
                               const float *dx, float *dy, int B, cudaStream_t st) {
    const int TPB = 256, WPB = TPB / 32; int blocks = (out + WPB - 1) / WPB;
    //  Q8_0 (the hura weight format): tensor-core path for LARGE batches only.
    //  k_q8_wmma has a fixed cost (a 16x16 tile over all of `out`) independent
    //  of B, so it only wins once B fills the 16-row M-tile — crossover ~B=12
    //  (microbench: B=1 it's 4.7x SLOWER, B>=16 ~1.8x faster). Prefill runs at
    //  B=PCH (256) -> WMMA; batched DECODE runs at B<=8 lanes -> the scalar
    //  warp-per-row kernel, which is weight-read-bound and far faster at low B.
    if (kind == 5) {
        if (B == 1) {
            //  Single-lane decode: k_q8_0_wb declares acc[MAXB] (32 regs)
            //  regardless of B, which tanks occupancy at B=1 (213 vs 768 GB/s
            //  measured). The single-vector kernel has no acc array and hits
            //  ~80-100% of peak bandwidth — 3.6x faster, bit-identical.
            k_q8_0_w<<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out);
        } else if (out % 64 == 0 && B >= 64
                   && (size_t)((out + 63) / 64) * ((B + 63) / 64) >= 512) {
            //  Large prefill projections (B=PCH): register-tiled 64x64 warp
            //  tiles, 2.72x over the 16x16 wmma, bit-exact. Block-count gate:
            //  below ~512 blocks the wide tile starves the GPU (measured
            //  0.48x at qkv B=256) — those shapes stay on the paths below.
            dim3 grid(out / 64, (B + 63) / 64);
            k_q8_reg<4, 4><<<grid, 32, 0, st>>>(dw, dx, dy, in, out, B);
        } else if (out % 32 == 0 && B >= 32
                   && (size_t)((out + 31) / 32) * ((B + 31) / 32) >= 2048) {
            //  Medium tier: 32x32 warp tiles (1.71x at the big shapes). Block
            //  threshold 2048, NOT 512: the 32x32 tile does a quarter the
            //  work/block of reg4x4, so at ~512 blocks it LOSES (measured
            //  0.58x at dkt/dvt out=512 B=1024) — those shapes fall through
            //  to k_q8_wmma instead. reg4x4's 512 floor is fine (verified
            //  1.69x at o_proj's exactly-512 blocks).
            dim3 grid(out / 32, (B + 31) / 32);
            k_q8_reg<2, 2><<<grid, 32, 0, st>>>(dw, dx, dy, in, out, B);
        } else if (B >= 16) {
            dim3 grid((out + WMM_TN - 1) / WMM_TN, (B + WMM_TM - 1) / WMM_TM);
            k_q8_wmma<<<grid, 32, 0, st>>>(dw, dx, dy, in, out, B);
        } else {
            //  Decode (B=2..15): compile-time-B kernel — acc[B] not acc[MAXB],
            //  2.1-3.8x faster at these shapes on H200, bit-identical. Cases
            //  cover up to 16 lanes so raising Max_Lanes needs no kernel edit;
            //  any B outside falls back to the runtime-B kernel.
            switch (B) {
                case  2: k_q8_0_wb_T< 2><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case  3: k_q8_0_wb_T< 3><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case  4: k_q8_0_wb_T< 4><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case  5: k_q8_0_wb_T< 5><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case  6: k_q8_0_wb_T< 6><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case  7: k_q8_0_wb_T< 7><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case  8: k_q8_0_wb_T< 8><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case  9: k_q8_0_wb_T< 9><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case 10: k_q8_0_wb_T<10><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case 11: k_q8_0_wb_T<11><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case 12: k_q8_0_wb_T<12><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case 13: k_q8_0_wb_T<13><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case 14: k_q8_0_wb_T<14><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case 15: k_q8_0_wb_T<15><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                case 16: k_q8_0_wb_T<16><<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out); break;
                default: k_q8_0_wb<<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out, B); break;
            }
        }
        return;
    }
    //  Other quant kinds keep the warp-per-row kernels, whose acc[MAXB] caps the
    //  batch at MAXB — sub-batch so B > MAXB (large prefill chunks) stays correct.
    for (int b0 = 0; b0 < B; b0 += MAXB) {
        int bb = B - b0; if (bb > MAXB) bb = MAXB;
        const float *xb = dx + (size_t) b0 * in; float *yb = dy + (size_t) b0 * out;
        if (kind == 0)      k_q4k_wb<<<blocks, TPB, 0, st>>>(dw, xb, yb, in, out, bb);
        else if (kind == 1) k_q6k_wb<<<blocks, TPB, 0, st>>>(dw, xb, yb, in, out, bb);
        else if (kind == 2) k_q5k_wb<<<blocks, TPB, 0, st>>>(dw, xb, yb, in, out, bb);
        else if (kind == 3) k_q3k_wb<<<blocks, TPB, 0, st>>>(dw, xb, yb, in, out, bb);
        else if (kind == 4) k_q2k_wb<<<blocks, TPB, 0, st>>>(dw, xb, yb, in, out, bb);
        else                k_dense_mv_b<<<blocks, TPB, 0, st>>>((const float *) dw, xb, yb, in, out, bb);
    }
}

__global__ void k_axpy_b(float *__restrict__ acc, const float *__restrict__ y, int n, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= (size_t) n * B) return;
    acc[i] += y[i];
}

#define BMAX 16
// Batched scratch (×BMAX of the single-lane buffers).
static float *Hb=nullptr,*nxb=nullptr,*aob=nullptr,*dlogb=nullptr,*dqkvb=nullptr,
             *dcqb=nullptr,*darb=nullptr,*dbrb=nullptr,*dzb=nullptr,*dgb=nullptr,
             *dbb=nullptr,*dorb=nullptr,*dqgb=nullptr,*dktb=nullptr,*dvtb=nullptr,
             *dqab=nullptr,*dgab=nullptr,*dattb=nullptr,*drlb=nullptr,*dhbb=nullptr;
static MoeRoute *drouteb=nullptr;
static int *gb_rows=nullptr,*gb_pos=nullptr;
static void **gb_Sptr=nullptr;    // device table of B per-lane delta-net S pointers
static void **gb_histptr=nullptr; // device table of B per-lane conv hist pointers
static int chb_inited=0;
static cudaStream_t gb_stream=0;

static void chain_alloc_b(void) {
    if (chb_inited) return;
    int dim=g_ch_dim; size_t Bd=(size_t)BMAX;
    cudaMalloc(&Hb, Bd*dim*4); cudaMalloc(&nxb, Bd*dim*4); cudaMalloc(&aob, Bd*dim*4);
    cudaMalloc(&dlogb, Bd*g_ch_vocab*4);
    if (g_mx_qo){ cudaMalloc(&dqkvb, Bd*g_mx_qo*4); cudaMalloc(&dcqb, Bd*g_mx_qo*4); }
    if (g_mx_nv){ cudaMalloc(&darb, Bd*g_mx_nv*4); cudaMalloc(&dbrb, Bd*g_mx_nv*4);
                  cudaMalloc(&dgb, Bd*g_mx_nv*4); cudaMalloc(&dbb, Bd*g_mx_nv*4); }
    if (g_mx_vd){ cudaMalloc(&dzb, Bd*g_mx_vd*4); cudaMalloc(&dorb, Bd*g_mx_vd*4); }
    if (g_mx_qgd) cudaMalloc(&dqgb, Bd*g_mx_qgd*4);
    if (g_mx_kvd){ cudaMalloc(&dktb, Bd*g_mx_kvd*4); cudaMalloc(&dvtb, Bd*g_mx_kvd*4); }
    if (g_mx_att){ cudaMalloc(&dqab, Bd*g_mx_att*4); cudaMalloc(&dgab, Bd*g_mx_att*4);
                   cudaMalloc(&dattb, Bd*g_mx_att*4); }
    if (g_mx_nexp) cudaMalloc(&drlb, Bd*g_mx_nexp*4);
    if (g_mx_hbuf) cudaMalloc(&dhbb, Bd*g_mx_hbuf*4);
    cudaMalloc(&drouteb, Bd*sizeof(MoeRoute));
    cudaMalloc(&gb_rows, Bd*4); cudaMalloc(&gb_pos, Bd*4);
    cudaMalloc(&gb_Sptr, (size_t) g_chain.size() * Bd * sizeof(void*));
    cudaMalloc(&gb_histptr, (size_t) g_chain.size() * Bd * sizeof(void*));
    cudaStreamCreateWithPriority(&gb_stream, cudaStreamDefault, aspida_stream_prio(1));
    chb_inited=1;
}

// Batched decode step for B lanes. handles[b*NL + li] = lane b's layer-li state.
// rows[b], pos[b] = lane b's embedding row and position. logits[b*vocab] out.
extern "C" void aspida_gpu_chain_forward_batch(int B, const int *rows, const int *pos,
                                               const int *handles, float *logits) {
    if (B < 1) return; if (B > BMAX) B = BMAX;
    chain_alloc_b();
    int dim=g_ch_dim, NL=(int)g_chain.size(); cudaStream_t st=gb_stream;
    //  Opt-in decode profiler (env ASPIDA_DEC_PROF): per-phase GPU time summed
    //  over layers, printed every 50 tokens. Per-phase syncs perturb wall so
    //  measurement-only. Zero cost when unset.
    static int dprof = getenv("ASPIDA_DEC_PROF") ? 1 : 0;
    static cudaEvent_t dpa=nullptr,dpb=nullptr;
    static double a_attn=0,a_moe=0,a_lm=0; static long a_n=0;
    double l_attn=0,l_moe=0,l_lm=0;
    if (dprof && !dpa) { cudaEventCreate(&dpa); cudaEventCreate(&dpb); }
    #define DSEG(acc) do{ if(dprof){ cudaEventRecord(dpb,st); cudaEventSynchronize(dpb); float _m=0; cudaEventElapsedTime(&_m,dpa,dpb); (acc)+=_m; cudaEventRecord(dpa,st);} }while(0)
    cudaMemcpy(gb_rows, rows, (size_t)B*4, cudaMemcpyHostToDevice);
    cudaMemcpy(gb_pos, pos, (size_t)B*4, cudaMemcpyHostToDevice);
    //  Per-lane delta-net S pointers for every layer, assembled once (the host
    //  staging is untouched for the rest of this forward, so the async copy is
    //  race-free). k_dnet_recur_b then reads gb_Sptr[li*B + lane].
    static void *hSptr[BMAX * 64], *hHptr[BMAX * 64];
    for (int li = 0; li < NL; ++li)
        if (!g_chain[li].is_fattn)
            for (int b = 0; b < B; ++b) {
                DnetState ds = g_dnet[handles[b * NL + li]];
                hSptr[(size_t) li * B + b] = ds.S;
                hHptr[(size_t) li * B + b] = ds.hist;
            }
    cudaMemcpyAsync(gb_Sptr, hSptr, (size_t) NL * B * sizeof(void*),
                    cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(gb_histptr, hHptr, (size_t) NL * B * sizeof(void*),
                    cudaMemcpyHostToDevice, st);
    k_embed_b<<<((size_t)dim*B+255)/256,256,0,st>>>(g_ch_embed, gb_rows, Hb, dim, B);
    if (dprof) cudaEventRecord(dpa,st);
    for (int li=0; li<NL; ++li) {
        ChainLayer &L=g_chain[li];
        k_norm1_b<<<B,256,0,st>>>(Hb, L.attn_norm, nxb, dim, B);
        if (!L.is_fattn) {
            launch_mv_b(L.qkv, L.qkv_k, dim, L.qo, nxb, dqkvb, B, st);
            launch_mv_b(L.al, L.al_k, dim, L.nv, nxb, darb, B, st);
            launch_mv_b(L.be, L.be_k, dim, L.nv, nxb, dbrb, B, st);
            launch_mv_b(L.ga, L.ga_k, dim, L.v_dim, nxb, dzb, B, st);
            size_t shmem=(size_t)(4*L.khd+2*L.vhd)*4;
            k_dnet_conv_b<<<((size_t)B*L.qo+255)/256,256,0,st>>>(
                dqkvb, (float *const *) (gb_histptr + (size_t) li * B), L.conv, dcqb, L.qo, L.kernel, B);
            k_dnet_gates_b<<<((size_t)B*L.nv+255)/256,256,0,st>>>(
                darb, dbrb, L.aw, L.dtw, dgb, dbb, L.nv, B);
            //  All B lanes' recurrence in one launch (B*nv blocks) — per-lane S
            //  via the pointer table, per-lane I/O via the [B,...] buffers.
            k_dnet_recur_b<<<B*L.nv, L.khd, shmem, st>>>(
                (float *const *) (gb_Sptr + (size_t) li * B), dcqb, dgb, dbb, dzb,
                L.nw, dorb, L.khd, L.vhd, L.q_dim, L.nkh, L.nv, L.qo, L.v_dim);
            launch_mv_b(L.ow, L.ow_k, L.v_dim, dim, dorb, aob, B, st);
        } else {
            int kvd=L.nkv*L.hd, att=L.nq*L.hd, qgd=L.nq*2*L.hd;
            launch_mv_b(L.qw, L.qw_k, dim, qgd, nxb, dqgb, B, st);
            launch_mv_b(L.kw, L.kw_k, dim, kvd, nxb, dktb, B, st);
            launch_mv_b(L.vw, L.vw_k, dim, kvd, nxb, dvtb, B, st);
            for (int b=0;b<B;++b) {
                FattnState fs=g_fattn[handles[b*NL+li]];
                k_fattn_prep<<<L.nq+L.nkv,L.hd,(size_t)L.hd*4,st>>>(
                    dqgb+(size_t)b*qgd, dktb+(size_t)b*kvd, dvtb+(size_t)b*kvd, L.qn, L.kn,
                    dqab+(size_t)b*att, dgab+(size_t)b*att, fs.K, fs.V,
                    L.nq,L.nkv,L.hd,kvd, gb_pos+b, L.rd,L.base,L.freq_scale,L.m_scale,
                    L.yarn_on,L.corr_lo,L.corr_hi, L.ffp,L.use_ff,L.interleaved,L.sec_total);
                k_fattn_attend<<<L.nq,256,0,st>>>(dqab+(size_t)b*att, dgab+(size_t)b*att, fs.K, fs.V, fs.scores, dattb+(size_t)b*att,
                    L.nq,L.nkv,L.hd,kvd, gb_pos+b, fs.max_len);
            }
            launch_mv_b(L.fow, L.fow_k, att, dim, dattb, aob, B, st);
        }
        k_axpy_b<<<((size_t)dim*B+255)/256,256,0,st>>>(Hb, aob, dim, B);
        DSEG(l_attn);
        if (L.has_moe) {
            k_norm1_b<<<B,256,0,st>>>(Hb, L.post_norm, nxb, dim, B);
            launch_mv_b(L.rw, L.rk, dim, L.n_exp, nxb, drlb, B, st);
            //  All B lanes' MoE in three launches instead of 3*B (route, gate+up,
            //  down). Per-lane routing/gather is unchanged — this removes the loop
            //  overhead and fills the device.
            k_moe_route_b<<<B,256,0,st>>>(drlb, nxb, L.sgi, L.sgi_len, L.n_exp, L.top_k, dim, drouteb, B);
            int w1=(L.top_k+1)*L.intermed;
            int b1b=((size_t)B*w1*32+255)/256, b2b=((size_t)B*dim*32+255)/256;
            k_moe_gu_p_b<<<b1b,256,0,st>>>(L.gdw,L.udw,L.sgdw,L.sudw, nxb, dhbb, drouteb, L.top_k,dim,L.intermed,L.g_bpe,L.u_bpe,L.gk,L.uk,L.sgk,L.suk, B);
            k_moe_down_p_b<<<b2b,256,0,st>>>(L.ddw,L.sddw, dhbb, aob, drouteb, L.top_k,L.intermed,dim,L.d_bpe,L.dk,L.sdk, B);
            k_axpy_b<<<((size_t)dim*B+255)/256,256,0,st>>>(Hb, aob, dim, B);
            DSEG(l_moe);
        }
    }
    k_norm1_b<<<B,256,0,st>>>(Hb, g_ch_fnorm, nxb, dim, B);
    launch_mv_b(g_ch_lm, g_ch_lm_k, dim, g_ch_vocab, nxb, dlogb, B, st);
    DSEG(l_lm);
    if (dprof) {
        a_attn+=l_attn; a_moe+=l_moe; a_lm+=l_lm; ++a_n;
        if (a_n%50==0) fprintf(stderr,"[DECPROF B=%d] per-tok: attn=%.3f moe=%.3f lm=%.3f sum=%.3f ms (n=%ld)\n",
            B, a_attn/a_n, a_moe/a_n, a_lm/a_n, (a_attn+a_moe+a_lm)/a_n, a_n);
    }
    cudaMemcpyAsync(logits, dlogb, (size_t)B*g_ch_vocab*4, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
}

//===========================================================================
//  Chunked PREFILL — process P sequence positions per launch instead of one.
//
//  The per-token forward is dominated (~90%) by weight-bandwidth-bound matmuls
//  (projections + MoE); each reads the whole weight for ONE position. Running
//  the prompt one token at a time made prefill of a real agent request (5-25k
//  tokens) take 1-2 minutes and time out. Here the matmuls are batched over a
//  chunk of PCH positions via the same launch_mv_b that batches decode lanes
//  (weight read once for up to MAXB rows), while the genuinely sequential parts
//  — the delta-net conv (sliding hist) and recurrence (state S carried
//  position→position) and the causal attention (K/V written then attended) —
//  reuse the EXACT single-position math, looped over the chunk. So the output
//  is bit-identical to the sequential path; only the matmul bandwidth is
//  amortised (~MAXB×).
//
//  Runs on its OWN stream + scratch, touching only this generation's per-lane
//  resident state, so it overlaps the batch Driver's concurrent decode forwards
//  (read-only shared weights) without a lock.
//===========================================================================
#define PCH 1024 // chunk size. The Q8 matmuls use the tensor-core k_q8_wmma
                 // (no MAXB cap; weight-stationary, per-token cost falls with
                 // the chunk), so a big chunk amortises the weight read. Non-Q8
                 // kinds are sub-batched by MAXB inside launch_mv_b.
                 //
                 // 256 -> 512 -> 1024 (moe-lever, 2026-07-15): at P=256
                 // essentially ALL 256 experts fire every chunk, so the
                 // ~570MB/layer gate+up weight stream is a FIXED cost the chunk
                 // amortises: MoE per-256-tok-equivalent 155.8ms (P=256) ->
                 // 86ms (512) -> 57.7ms (1024).
                 //
                 // PROD VRAM budget (2026-07-15): the deployed the GPU host is a
                 // 46GB (not 48GB) an NVIDIA GPU that ALSO co-hosts a 3.2GB voice_server,
                 // so free VRAM over the resident 35B-Q8 (~38.4GB) is ~4.4GB.
                 // We run PCH=1024 with ASPIDA_PREFILL_SETS=1 (ONE scratch set,
                 // ~440MB): vs 512 (~220MB) the extra 220MB buys ~10% faster
                 // prefill (5619 tok 3.9s->3.5s) via MoE weight-stream
                 // amortisation, at negligible headroom cost. Measured prompt
                 // ceiling on prod is >=20k tokens (20017 tok = 15.4s, no OOM),
                 // comfortably covering the platform's 25-100KB prompts — so KV
                 // headroom is NOT the binding constraint at SETS=1 (an earlier
                 // ~9k estimate was a wrong analytical KV model; the real per-gen
                 // KV is far smaller). Use SETS>1 only with the full 48GB free
                 // (voice relocated). MUST match PCHUNK in src/llm/llm_qwen.adb
                 // (the Ada side sends the chunks).

// Delta-net causal depthwise conv over a chunk: thread owns channel c and walks
// t = 0..P-1, maintaining the sliding hist window exactly as k_dnet_conv does.
__global__ void k_dnet_conv_chunk(const float *__restrict__ qkv, float *__restrict__ hist,
                                  const float *__restrict__ convw, float *__restrict__ cq,
                                  int qo, int kernel, int P) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= qo) return;
    for (int t = 0; t < P; ++t) {
        const float *x = qkv + (size_t) t * qo;
        float *o = cq + (size_t) t * qo;
        float acc = x[c] * convw[c * kernel + (kernel - 1)];
        for (int k = 0; k < kernel - 1; ++k)
            acc += hist[(size_t) k * qo + c] * convw[c * kernel + k];
        o[c] = acc / (1.f + expf(-acc));
        for (int k = 0; k + 1 < kernel - 1; ++k)
            hist[(size_t) k * qo + c] = hist[(size_t) (k + 1) * qo + c];
        if (kernel >= 2) hist[(size_t) (kernel - 2) * qo + c] = x[c];
    }
}

//===========================================================================
//  Warp-parallel delta-net recurrence (ASPIDA_DNET_WARP) — same math as
//  k_dnet_recur_chunk, restructured for occupancy + register-resident state.
//
//  The sequential kernel launches <<<nv, khd>>> = 32*128 = 4096 threads on a
//  142-SM card (~2% occupancy: 110 SMs idle) and reads/writes the recurrent
//  state S to GLOBAL memory 3*khd times PER position PER column (strided by
//  vhd) — so it is both occupancy- and bandwidth-starved. Profiled ~150 ms /
//  256-chunk at 25k, the #2 prefill cost after full-attn.
//
//  Here each WARP owns one (head, column) and holds that column's khd-slice of
//  S in REGISTERS (khd/32 = 4 floats/lane) for the whole chunk — S touches
//  HBM only twice per chunk (load at start, store at end) instead of per step.
//  Columns are independent (the state update S[k][v] and outputs only read
//  column v), so this is exact; the RMS output-norm that DOES couple columns
//  is split into k_dnet_out_norm below. The per-position khd reductions use
//  warp shfl (butterfly), differing from the sequential k-loop only at the
//  float level — so like the tiled attention this path is OPT-IN pending
//  eval-harness sign-off; the sequential kernel stays the bit-exact default.
//===========================================================================

//  L2-normalise Q and K per (position, k-head). block = (t, k_head), threads =
//  khd. Matches the in-kernel norm of k_dnet_recur_chunk exactly (1/(√ss+1e-6)).
__global__ void k_dnet_qk_norm(const float *__restrict__ cq,
    float *__restrict__ kn, float *__restrict__ qn,
    int khd, int q_dim, int n_k_heads, int qo, int P) {
    int t = blockIdx.x / n_k_heads, kh = blockIdx.x % n_k_heads, v = threadIdx.x;
    const float *cq_t = cq + (size_t) t * qo;
    float qv = cq_t[kh * khd + v], kv = cq_t[q_dim + kh * khd + v];
    __shared__ float sq[256], sk[256];
    sq[v] = qv * qv; sk[v] = kv * kv;
    __syncthreads();
    for (int o = khd >> 1; o > 0; o >>= 1) {
        if (v < o) { sq[v] += sq[v + o]; sk[v] += sk[v + o]; }
        __syncthreads();
    }
    float rq = 1.f / (sqrtf(sq[0]) + 1e-6f), rk = 1.f / (sqrtf(sk[0]) + 1e-6f);
    qn[(size_t) t * q_dim + kh * khd + v] = qv * rq;
    kn[(size_t) t * q_dim + kh * khd + v] = kv * rk;
}

//  Recurrence: grid = (nv heads, col-tiles of WARPS_PER_BLOCK), one warp per
//  column. S[:,v] lives in `RQ` registers per lane across the chunk. Writes the
//  scaled output o*(1/√khd) to `osh` [P, v_dim]; the RMS + z-gate is applied by
//  k_dnet_out_norm. Inactive warps (v >= vhd) still drive the cooperative KN/QN
//  tile load + __syncthreads.
__global__ void k_dnet_recur_warp(float *__restrict__ S,
    const float *__restrict__ kn, const float *__restrict__ qn,
    const float *__restrict__ cq, const float *__restrict__ gate,
    const float *__restrict__ beta, float *__restrict__ osh,
    int khd, int vhd, int q_dim, int n_k_heads, int nv, int qo, int v_dim, int P) {
    int h = blockIdx.x;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int v = blockIdx.y * (blockDim.x >> 5) + warp;   // column
    int k_head = h % n_k_heads;
    int RQ = khd >> 5;                                // regs per lane (khd/32)
    extern __shared__ float sh[];                     // KN[khd], QN[khd]
    float *KN = sh, *QN = sh + khd;
    bool active = v < vhd;
    float Sreg[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) Sreg[j] = 0.f;
    if (active)
        for (int j = 0; j < RQ; ++j)
            Sreg[j] = S[(size_t)(h * khd + lane + 32 * j) * vhd + v];
    float scale = 1.f / sqrtf((float) khd);
    for (int t = 0; t < P; ++t) {
        for (int i = threadIdx.x; i < khd; i += blockDim.x) {
            KN[i] = kn[(size_t) t * q_dim + k_head * khd + i];
            QN[i] = qn[(size_t) t * q_dim + k_head * khd + i];
        }
        __syncthreads();
        if (active) {
            float g = gate[(size_t) t * nv + h], b = beta[(size_t) t * nv + h];
            float Vv = cq[(size_t) t * qo + 2 * q_dim + h * vhd + v];
            float pr = 0.f;
            for (int j = 0; j < RQ; ++j) pr += Sreg[j] * KN[lane + 32 * j];
            for (int o = 16; o > 0; o >>= 1) pr += __shfl_xor_sync(0xffffffffu, pr, o);
            float retr = g * pr;
            float corr = b * (Vv - retr);
            for (int j = 0; j < RQ; ++j) Sreg[j] = g * Sreg[j] + KN[lane + 32 * j] * corr;
            float po = 0.f;
            for (int j = 0; j < RQ; ++j) po += Sreg[j] * QN[lane + 32 * j];
            for (int o = 16; o > 0; o >>= 1) po += __shfl_xor_sync(0xffffffffu, po, o);
            if (lane == 0) osh[(size_t) t * v_dim + h * vhd + v] = po * scale;
        }
        __syncthreads();
    }
    if (active)
        for (int j = 0; j < RQ; ++j)
            S[(size_t)(h * khd + lane + 32 * j) * vhd + v] = Sreg[j];
}

//  Output RMS-norm + z-gate per (position, head), matching the tail of
//  k_dnet_recur_chunk. block = (t, head), threads = vhd.
__global__ void k_dnet_out_norm(const float *__restrict__ osh,
    const float *__restrict__ z, const float *__restrict__ norm_w,
    float *__restrict__ o_row, int vhd, int nv, int v_dim, int P) {
    int t = blockIdx.x / nv, h = blockIdx.x % nv, v = threadIdx.x;
    float o = osh[(size_t) t * v_dim + h * vhd + v];
    __shared__ float sr[256];
    sr[v] = o * o;
    __syncthreads();
    for (int k = vhd >> 1; k > 0; k >>= 1) {
        if (v < k) sr[v] += sr[v + k];
        __syncthreads();
    }
    float rms = sqrtf(sr[0] / (float) vhd + 1e-6f);
    float zz = z[(size_t) t * v_dim + h * vhd + v];
    o_row[(size_t) t * v_dim + h * vhd + v] = (o / rms) * norm_w[v] * (zz / (1.f + expf(-zz)));
}

// Delta-net recurrence over a chunk: block = head, thread = column v, walks
// t = 0..P-1 keeping the state matrix S resident (identical math + ascending
// sum order to k_dnet_recur). Chunk buffers are [P, ...] row-major.
__global__ void k_dnet_recur_chunk(float *__restrict__ S, const float *__restrict__ cq,
    const float *__restrict__ gate, const float *__restrict__ beta,
    const float *__restrict__ z, const float *__restrict__ norm_w, float *__restrict__ o_row,
    int khd, int vhd, int q_dim, int n_k_heads, int nv, int qo, int v_dim, int P) {
    int h = blockIdx.x;
    int v = threadIdx.x;                  // 0..khd-1 (== vhd-1)
    extern __shared__ float sh[];
    float *Qr = sh, *Kr = sh + khd, *QN = sh + 2 * khd,
          *KN = sh + 3 * khd, *Vv = sh + 4 * khd, *osh = sh + 4 * khd + vhd;
    int k_head = h % n_k_heads;
    int base = h * khd;
    float scale = 1.f / sqrtf((float) khd);
    for (int t = 0; t < P; ++t) {
        const float *cq_t = cq + (size_t) t * qo;
        Qr[v] = cq_t[k_head * khd + v];
        Kr[v] = cq_t[q_dim + k_head * khd + v];
        Vv[v] = cq_t[2 * q_dim + h * vhd + v];
        __syncthreads();
        //  L2 norms via ONE parallel reduction each (was: every thread summed
        //  the whole khd vector — khd-way redundant). Reuse QN/KN as scratch
        //  (not yet written). Tree order differs from the sequential sum only
        //  at the float-rounding level. Assumes blockDim.x (=khd) a power of 2.
        QN[v] = Qr[v] * Qr[v]; KN[v] = Kr[v] * Kr[v];
        __syncthreads();
        for (int o = khd >> 1; o > 0; o >>= 1) {
            if (v < o) { QN[v] += QN[v + o]; KN[v] += KN[v + o]; }
            __syncthreads();
        }
        float ssq = QN[0], ssk = KN[0];
        __syncthreads();
        QN[v] = Qr[v] * (1.f / (sqrtf(ssq) + 1e-6f));
        KN[v] = Kr[v] * (1.f / (sqrtf(ssk) + 1e-6f));
        __syncthreads();
        float g = gate[(size_t) t * nv + h], b = beta[(size_t) t * nv + h];
        float retr = 0.f;
        for (int k = 0; k < khd; ++k) retr += g * S[(size_t)(base + k) * vhd + v] * KN[k];
        float corr = b * (Vv[v] - retr);
        for (int k = 0; k < khd; ++k)
            S[(size_t)(base + k) * vhd + v] = g * S[(size_t)(base + k) * vhd + v] + KN[k] * corr;
        float o = 0.f;
        for (int k = 0; k < khd; ++k) o += S[(size_t)(base + k) * vhd + v] * QN[k];
        osh[v] = o * scale;
        __syncthreads();
        //  RMS via one reduction (was vhd-way redundant). Reuse Qr as scratch.
        Qr[v] = osh[v] * osh[v];
        __syncthreads();
        for (int o2 = vhd >> 1; o2 > 0; o2 >>= 1) {
            if (v < o2) Qr[v] += Qr[v + o2];
            __syncthreads();
        }
        float rms = sqrtf(Qr[0] / (float) vhd + 1e-6f);
        __syncthreads();
        float zz = z[(size_t) t * v_dim + h * vhd + v];
        o_row[(size_t) t * v_dim + h * vhd + v] =
            (osh[v] / rms) * norm_w[v] * (zz / (1.f + expf(-zz)));
        __syncthreads();
    }
}

// Batched prep for a chunk of P positions: block = (bt) where bt encodes
// (which head/kv, position t). Writes K/V into the resident cache at
// pos_start+t and the rotated Q into q_all_chunk[t]. Identical math to
// k_fattn_prep, position by position, but all P in ONE launch.
__global__ void k_fattn_prep_chunk(
    const float *__restrict__ qg, const float *__restrict__ kt, const float *__restrict__ vt,
    const float *__restrict__ q_norm, const float *__restrict__ k_norm,
    float *__restrict__ q_all, float *__restrict__ g_all,
    __half *__restrict__ Kc, __half *__restrict__ Vc,     // fp16 cache (Phase B)
    int nq, int nkv, int hd, int kvd, int pos_start,
    int rd, float base, float freq_scale, float m_scale,
    int yarn_on, float corr_lo, float corr_hi,
    const float *__restrict__ ff, int use_ff, int interleaved, int sec_total, int P) {
    extern __shared__ float sh[];
    float *nrm = sh;                       // [hd]
    int nbt = nq + nkv;
    int t = blockIdx.x / nbt, bb = blockIdx.x % nbt, d = threadIdx.x;
    int pos = pos_start + t;
    int qgd = nq * 2 * hd, att = nq * hd;
    const float *qg_t = qg + (size_t) t * qgd;
    const float *kt_t = kt + (size_t) t * kvd;
    const float *vt_t = vt + (size_t) t * kvd;
    int half = rd / 2;
    bool is_q = bb < nq;
    int head = is_q ? bb : bb - nq;
    float v;
    if (is_q) {
        v = qg_t[head * 2 * hd + d];
        g_all[(size_t) t * att + head * hd + d] = qg_t[head * 2 * hd + hd + d];
    } else {
        v = kt_t[head * hd + d];
        Vc[(size_t) pos * kvd + head * hd + d] = __float2half(vt_t[head * hd + d]);
    }
    nrm[d] = v;
    __syncthreads();
    float ss = 0.f;
    for (int i = 0; i < hd; ++i) ss += nrm[i] * nrm[i];
    float rms = sqrtf(ss / (float) hd + 1e-6f);
    float w = is_q ? q_norm[d] : k_norm[d];
    float nv = (v / rms) * w;
    __syncthreads();
    nrm[d] = nv;
    __syncthreads();
    float out = nv;
    if (d < rd) {
        int i, other; bool first;
        if (interleaved) { i = d / 2; first = (d % 2) == 0; other = first ? d + 1 : d - 1; }
        else if (d < half) { i = d; first = true;  other = d + half; }
        else               { i = d - half; first = false; other = d - half; }
        int pos_eff = (i < sec_total) ? pos : 0;
        float th = rope_theta(i, pos_eff, rd, base, freq_scale,
                              yarn_on, corr_lo, corr_hi, ff, use_ff);
        float c = cosf(th) * m_scale, s = sinf(th) * m_scale;
        float x1 = first ? nrm[d] : nrm[other];
        float x2 = first ? nrm[other] : nrm[d];
        out = first ? (x1 * c - x2 * s) : (x2 * c + x1 * s);
    }
    if (is_q) q_all[(size_t) t * att + head * hd + d] = out;
    else      Kc[(size_t) pos * kvd + head * hd + d] = __float2half(out);
}

//  Tiled causal attention for chunked prefill (flash-attention style).
//  Grid = (query tiles of TQW=16 warps) x (head h); each WARP owns one query
//  row, K/V stream through shared memory in tiles of TK positions, and the
//  softmax is online in registers. Same ascending-s update order as the naive
//  k_fattn_attend_chunk below, so the only float-level difference is the dot
//  product's butterfly-shfl reduce order (vs the shared-memory tree there).
//
//  Why: the naive kernel does a block-wide tree reduction with __syncthreads
//  INSIDE the per-position loop — at 25k context that is 25k serialized
//  block-wide reductions per query, and every query block re-reads the entire
//  K/V cache (P x len reads per layer). Here the inner loop is warp-
//  synchronous (shfl only, no block sync) and each K/V tile is read from HBM
//  once per 16 queries. Profiled 2026-07-14 (RTX 6000 Ada, P=256, pos~4k):
//  fattn was 180-226 ms/chunk (~40%, growing linearly with position) — the
//  dominant prefill cost on long prompts.
__global__ void k_fattn_attend_tile(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn,
    int nq, int nkv, int hd, int kvd, int pos_start, int P, int TK) {
    int h = blockIdx.y;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nwarp = blockDim.x >> 5;
    int t = blockIdx.x * nwarp + warp;      //  query position in chunk
    int rep = nq / nkv, kvh = h / rep;
    int kv_off = kvh * hd, att = nq * hd;
    int RQ = hd >> 5;                       //  q/acc registers per lane (<= 8)
    extern __shared__ float shkv[];         //  K tile [TK][hd], then V tile
    float *Ksh = shkv, *Vsh = shkv + (size_t) TK * hd;
    float q[8], acc[8];
    bool active = t < P;
    int len = active ? (pos_start + t + 1) : 0;
    int t_last = min(P - 1, (blockIdx.x + 1) * nwarp - 1);
    int len_max = pos_start + t_last + 1;   //  block-wide causal bound
    const float *qp = q_all + (size_t) t * att + h * hd;
    #pragma unroll
    for (int j = 0; j < 8; ++j) { q[j] = 0.f; acc[j] = 0.f; }
    for (int j = 0; j < RQ; ++j) q[j] = active ? qp[lane + 32 * j] : 0.f;
    float scale = rsqrtf((float) hd), m = -3.402823466e38f, l = 0.f;
    for (int s0 = 0; s0 < len_max; s0 += TK) {
        int tk = min(TK, len_max - s0);
        //  Cooperative tile load: ALL warps participate (idle query warps
        //  included — they must reach the __syncthreads below).
        for (int i = threadIdx.x; i < tk * hd; i += blockDim.x) {
            int srow = i / hd, sd = i - srow * hd;
            Ksh[(size_t) srow * hd + sd] = Kc[(size_t)(s0 + srow) * kvd + kv_off + sd];
            Vsh[(size_t) srow * hd + sd] = Vc[(size_t)(s0 + srow) * kvd + kv_off + sd];
        }
        __syncthreads();
        int smax = min(tk, len - s0);       //  warp-local causal bound
        for (int s = 0; s < smax; ++s) {
            float part = 0.f;
            for (int j = 0; j < RQ; ++j) part += q[j] * Ksh[(size_t) s * hd + lane + 32 * j];
            //  Butterfly reduce: every lane ends with the full dot product.
            for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
            float dot = part * scale;
            float m_new = fmaxf(m, dot);
            float corr = expf(m - m_new);
            float p = expf(dot - m_new);
            l = l * corr + p;
            for (int j = 0; j < RQ; ++j)
                acc[j] = acc[j] * corr + p * Vsh[(size_t) s * hd + lane + 32 * j];
            m = m_new;
        }
        __syncthreads();
    }
    if (active) {
        const float *gp = g_all + (size_t) t * att + h * hd;
        float *op = attn + (size_t) t * att + h * hd;
        for (int j = 0; j < RQ; ++j) {
            float g = gp[lane + 32 * j];
            op[lane + 32 * j] = (acc[j] / l) * (1.f / (1.f + expf(-g)));
        }
    }
}

//  GQA K/V-reuse variant of the tiled kernel (fattn-gqa lever, 2026-07-15).
//  The tiled kernel launches one independent block per q-head (grid.y = nq), so
//  each kv-head's K/V streams from HBM rep = nq/nkv times — the long-context
//  bottleneck (hura: rep=8). Here block = (query-tile, head-group): each warp
//  carries online-softmax state for HPB group heads at once, sharing one
//  Ksh/Vsh tile -> K/V HBM traffic drops HPB x. Same fp32 arithmetic as the
//  tiled kernel, only the head loop is reorganised — outputs match it to ~1e-6
//  (oracle-identical). Measured on real hura dims (hd=256, nq=16, nkv=2):
//  1.77-1.94x over tiled at pos>=4k; HPB=2 with 16 query-warps (512 thr, 80
//  regs, no spills) beats HPB=4/8 (register-occupancy trade). HPB must divide
//  rep. Escape hatch: ASPIDA_FATTN_NOGQA=1 forces the plain tiled kernel.
template<int HPB, int RQ>
__global__ void k_fattn_tile_gqa(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn, int nq, int nkv, int hd, int kvd, int pos_start, int P, int TK) {
    int hg = blockIdx.y;                     //  head-group index
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nwarp = blockDim.x >> 5;
    int t = blockIdx.x * nwarp + warp;
    int rep = nq / nkv;
    int h_base = hg * HPB, kvh = h_base / rep;
    int kv_off = kvh * hd, att = nq * hd;
    extern __shared__ float shkv[];
    float *Ksh = shkv, *Vsh = shkv + (size_t) TK * hd;
    float q[HPB][RQ], acc[HPB][RQ], m[HPB], l[HPB];
    bool active = t < P;
    int len = active ? (pos_start + t + 1) : 0;
    int t_last = min(P - 1, (blockIdx.x + 1) * nwarp - 1);
    int len_max = pos_start + t_last + 1;
    #pragma unroll
    for (int hh = 0; hh < HPB; ++hh) {
        const float *qp = q_all + (size_t) t * att + (h_base + hh) * hd;
        #pragma unroll
        for (int j = 0; j < RQ; ++j) { q[hh][j] = active ? qp[lane + 32 * j] : 0.f; acc[hh][j] = 0.f; }
        m[hh] = -3.402823466e38f; l[hh] = 0.f;
    }
    float scale = rsqrtf((float) hd);
    for (int s0 = 0; s0 < len_max; s0 += TK) {
        int tk = min(TK, len_max - s0);
        for (int i = threadIdx.x; i < tk * hd; i += blockDim.x) {
            int srow = i / hd, sd = i - srow * hd;
            Ksh[(size_t) srow * hd + sd] = Kc[(size_t)(s0 + srow) * kvd + kv_off + sd];
            Vsh[(size_t) srow * hd + sd] = Vc[(size_t)(s0 + srow) * kvd + kv_off + sd];
        }
        __syncthreads();
        int smax = min(tk, len - s0);
        for (int s = 0; s < smax; ++s) {
            const float *ks = Ksh + (size_t) s * hd + lane;
            const float *vs = Vsh + (size_t) s * hd + lane;
            float kreg[RQ], vreg[RQ];
            #pragma unroll
            for (int j = 0; j < RQ; ++j) { kreg[j] = ks[32 * j]; vreg[j] = vs[32 * j]; }
            #pragma unroll
            for (int hh = 0; hh < HPB; ++hh) {
                float part = 0.f;
                #pragma unroll
                for (int j = 0; j < RQ; ++j) part += q[hh][j] * kreg[j];
                for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
                float dot = part * scale;
                float m_new = fmaxf(m[hh], dot);
                float corr = expf(m[hh] - m_new), p = expf(dot - m_new);
                l[hh] = l[hh] * corr + p;
                #pragma unroll
                for (int j = 0; j < RQ; ++j) acc[hh][j] = acc[hh][j] * corr + p * vreg[j];
                m[hh] = m_new;
            }
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int hh = 0; hh < HPB; ++hh) {
            const float *gp = g_all + (size_t) t * att + (h_base + hh) * hd;
            float *op = attn + (size_t) t * att + (h_base + hh) * hd;
            #pragma unroll
            for (int j = 0; j < RQ; ++j) {
                float g = gp[lane + 32 * j];
                op[lane + 32 * j] = (acc[hh][j] / l[hh]) * (1.f / (1.f + expf(-g)));
            }
        }
    }
}

//  Row of accumulator element e for this lane (sm_70+ m16n16k16 f32 layout):
//    row = (lane>>2) + ((e & 2) ? 8 : 0)   (probe-verified on the an NVIDIA GPU)
__device__ __forceinline__ int accrow(int lane, int e) { return (lane >> 2) + ((e & 2) ? 8 : 0); }

//  Occupancy-aware WMMA attend (fattn-lever wmma3, 2026-07-15). W query-warps
//  per block share ONE fp16 K/V tile; each warp owns a 16-query tile for head
//  blockIdx.y; O lives in 16 accumulator FRAGMENTS rescaled in-place per
//  key-tile (no shared roundtrip — the epilogue that killed wmma v1). Reads
//  the EXISTING fp32 fs.K/fs.V (fp16 convert in the tile load): no cache-
//  format, prep or snapshot changes. 95232 B shared (needs the >48KB opt-in),
//  186 regs, 1 block/SM = 8 warps/SM — the occupancy v1 lacked (2 warps/SM).
//  Measured vs GQA-2 at hd=256: pos12k 1.66x, pos25k 1.47x, pos37k 1.53x;
//  rel err 3.8e-4 (fp16 QK/PV + __expf) — NOT bit-exact, so this path is
//  length-gated to long chunks and must pass the eval-hura gate like the
//  tiled path did. Requires hd==256. ASPIDA_FATTN_NOW3=1 disables.
//  Two traps fixed during bring-up (recorded): lanes>=16 writing Psh out of
//  bounds, and a divergent __syncthreads caused by a PER-WARP key bound —
//  the loop bound below must be block-uniform.
template<int W>
__global__ void k_fattn_wmma3(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn, int nq, int nkv, int hd, int kvd, int pos_start, int P) {
    const int HT = 16;                    //  hd/16 (hd=256)
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.y, qtile = blockIdx.x * W + warp, q0 = qtile * 16;
    int rep = nq / nkv, kvh = h / rep, kv_off = kvh * hd, att = nq * hd;
    extern __shared__ char w3smem[];
    half *Ksh = (half *) w3smem;                     //  [16*hd] block-shared
    half *Vsh = Ksh + 16 * hd;                       //  [16*hd] block-shared
    half *Qb  = Vsh + 16 * hd;                       //  [W][16*hd]
    float *Sb = (float *) (Qb + (size_t) W * 16 * hd);   //  [W][16*16]
    half  *Pb = (half *) (Sb + (size_t) W * 16 * 16);    //  [W][16*16]
    float *Cb = (float *) (Pb + (size_t) W * 16 * 16);   //  [W][16] corr
    float *Lb = Cb + (size_t) W * 16;                    //  [W][16] l
    half *Qsh = Qb + (size_t) warp * 16 * hd;
    float *Ssh = Sb + (size_t) warp * 16 * 16;
    half *Psh = Pb + (size_t) warp * 16 * 16;
    float *Csh = Cb + (size_t) warp * 16;
    float *Lsh = Lb + (size_t) warp * 16;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> O[HT];
    #pragma unroll
    for (int n = 0; n < HT; ++n) nvcuda::wmma::fill_fragment(O[n], 0.f);
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd;
        Qsh[i] = (q0 + r < P) ? __float2half(q_all[(size_t)(q0 + r) * att + h * hd + d]) : __float2half(0.f); }
    float m = -3.402823466e38f, l = 0.f;         //  per-row (lane<16 owns row=lane)
    float scale = rsqrtf((float) hd);
    //  Block-UNIFORM key bound: every warp must run the SAME number of key-
    //  tiles or the __syncthreads below diverges (warps have different q0).
    //  Exhausted warps see all-masked keys (corr=1, p=0 -> O unchanged).
    int qlast_blk = min(P - 1, (int)((blockIdx.x + 1) * W * 16 - 1));
    int len_max = pos_start + qlast_blk + 1;
    __syncthreads();
    for (int k0 = 0; k0 < len_max; k0 += 16) {
        int kn = min(16, len_max - k0);
        for (int i = threadIdx.x; i < 16 * hd; i += blockDim.x) { int r = i / hd, d = i % hd;
            Ksh[i] = (r < kn) ? __float2half(Kc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f);
            Vsh[i] = (r < kn) ? __float2half(Vc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f); }
        __syncthreads();
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> Sf;
        nvcuda::wmma::fill_fragment(Sf, 0.f);
        #pragma unroll
        for (int kt = 0; kt < HT; ++kt) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> bf;
            nvcuda::wmma::load_matrix_sync(af, Qsh + kt * 16, hd);
            nvcuda::wmma::load_matrix_sync(bf, Ksh + kt * 16, hd);   //  col_major => K^T
            nvcuda::wmma::mma_sync(Sf, af, bf, Sf);
        }
        nvcuda::wmma::store_matrix_sync(Ssh, Sf, 16, nvcuda::wmma::mem_row_major);
        __syncwarp();
        if (lane < 16) {
            int qpos = pos_start + q0 + lane; float rmax = m;
            float s[16];
            #pragma unroll
            for (int k = 0; k < 16; ++k) { float sv = (k < kn && (k0 + k) <= qpos) ? Ssh[lane * 16 + k] * scale : -3.402823466e38f; s[k] = sv; rmax = fmaxf(rmax, sv); }
            float m_new = rmax; float corr = (m > -3.0e38f) ? __expf(m - m_new) : 0.f;
            float lnew = l * corr;
            #pragma unroll
            for (int k = 0; k < 16; ++k) { float p = (s[k] > -3.0e38f) ? __expf(s[k] - m_new) : 0.f; Psh[lane * 16 + k] = __float2half(p); lnew += p; }
            m = (m_new > -3.0e38f) ? m_new : m; l = lnew; Csh[lane] = corr;
        }
        __syncwarp();
        #pragma unroll
        for (int n = 0; n < HT; ++n) {
            #pragma unroll
            for (int e = 0; e < O[n].num_elements; ++e) O[n].x[e] *= Csh[accrow(lane, e)];
        }
        #pragma unroll
        for (int n = 0; n < HT; ++n) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, Psh, 16);
            nvcuda::wmma::load_matrix_sync(bf, Vsh + n * 16, hd);
            nvcuda::wmma::mma_sync(O[n], af, bf, O[n]);
        }
        __syncthreads();      //  before the next K/V tile overwrites Ksh/Vsh
    }
    if (lane < 16) Lsh[lane] = l;
    __syncwarp();
    #pragma unroll
    for (int n = 0; n < HT; ++n) {
        #pragma unroll
        for (int e = 0; e < O[n].num_elements; ++e) O[n].x[e] /= Lsh[accrow(lane, e)];
        nvcuda::wmma::store_matrix_sync(Ssh, O[n], 16, nvcuda::wmma::mem_row_major);   //  reuse as scratch
        __syncwarp();
        for (int i = lane; i < 16 * 16; i += 32) { int r = i / 16, c = i % 16; int d = n * 16 + c;
            if (q0 + r < P) { float g = g_all[(size_t)(q0 + r) * att + h * hd + d];
                attn[(size_t)(q0 + r) * att + h * hd + d] = Ssh[i] * (1.f / (1.f + expf(-g))); } }
        __syncwarp();
    }
}

//  Tensor-core (wmma) FlashAttention for chunked prefill. One warp = one 16-query
//  tile x head; Q@K^T and P@V via 16x16x16 wmma (fp16 in, fp32 accum — the same
//  nvcuda::wmma pattern as k_q8_wmma), online softmax in shared, O rescaled between
//  key tiles. ~1.6x the tiled kernel at long context (25k: 22 vs 36 ms/chunk),
//  fp16-precision (rel ~3e-4 vs the naive oracle) — so OPT-IN via ASPIDA_FATTN_WMMA
//  pending the eval-harness gate, like the tiled path was. Requires (hd % 16)==0.
//  1 warp/block (~22 KB shared): WQT>1 sharing K/V measured WORSE (occupancy-bound).
__global__ void k_fattn_wmma(const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc, float *__restrict__ attn,
    int nq, int nkv, int hd, int kvd, int pos_start, int P) {
    int qt = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    int rep = nq / nkv, kvh = h / rep, kv_off = kvh * hd, att = nq * hd, HT = hd / 16, q0 = qt * 16;
    extern __shared__ char smem[];
    half *Qsh = (half *) smem;                       // [16*hd]
    half *Ksh = Qsh + 16 * hd, *Vsh = Ksh + 16 * hd, *Psh = Vsh + 16 * hd;   // [16*hd],[16*hd],[16*16]
    float *Osh = (float *) (Psh + 16 * 16);          // [16*hd]
    float *Ssh = Osh + 16 * hd;                      // [16*16]
    __shared__ float m[16], l[16];
    int qmax = min(16, P - q0);
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd;
        Qsh[i] = (q0 + r < P) ? __float2half(q_all[(size_t)(q0 + r) * att + h * hd + d]) : __float2half(0.f);
        Osh[i] = 0.f; }
    if (lane < 16) { m[lane] = -3.402823466e38f; l[lane] = 0.f; }
    __syncwarp();
    float scale = rsqrtf((float) hd);
    int len_max = pos_start + q0 + qmax;
    for (int k0 = 0; k0 < len_max; k0 += 16) {
        int kn = min(16, len_max - k0);
        for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd;
            Ksh[i] = (r < kn) ? __float2half(Kc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f);
            Vsh[i] = (r < kn) ? __float2half(Vc[(size_t)(k0 + r) * kvd + kv_off + d]) : __float2half(0.f); }
        __syncwarp();
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cf;
        nvcuda::wmma::fill_fragment(cf, 0.f);
        for (int kt = 0; kt < HT; ++kt) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> bf;
            nvcuda::wmma::load_matrix_sync(af, Qsh + kt * 16, hd);
            nvcuda::wmma::load_matrix_sync(bf, Ksh + kt * 16, hd);   // col_major => K^T
            nvcuda::wmma::mma_sync(cf, af, bf, cf);
        }
        nvcuda::wmma::store_matrix_sync(Ssh, cf, 16, nvcuda::wmma::mem_row_major);
        __syncwarp();
        if (lane < 16) {
            if (lane < qmax) {
                int qpos = pos_start + q0 + lane; float rmax = m[lane]; float s[16];
                for (int k = 0; k < 16; ++k) { float sv = (k < kn && (k0 + k) <= qpos) ? Ssh[lane * 16 + k] * scale : -3.402823466e38f; s[k] = sv; rmax = fmaxf(rmax, sv); }
                float corr = expf(m[lane] - rmax), lnew = l[lane] * corr;
                for (int k = 0; k < 16; ++k) { float p = (s[k] > -3.0e38f) ? expf(s[k] - rmax) : 0.f; Psh[lane * 16 + k] = __float2half(p); lnew += p; }
                for (int d = 0; d < hd; ++d) Osh[lane * hd + d] *= corr;
                m[lane] = rmax; l[lane] = lnew;
            } else for (int k = 0; k < 16; ++k) Psh[lane * 16 + k] = __float2half(0.f);
        }
        __syncwarp();
        for (int n = 0; n < HT; ++n) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> of;
            nvcuda::wmma::load_matrix_sync(of, Osh + n * 16, hd, nvcuda::wmma::mem_row_major);
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, Psh, 16);
            nvcuda::wmma::load_matrix_sync(bf, Vsh + n * 16, hd);
            nvcuda::wmma::mma_sync(of, af, bf, of);
            nvcuda::wmma::store_matrix_sync(Osh + n * 16, of, hd, nvcuda::wmma::mem_row_major);
        }
        __syncwarp();
    }
    for (int i = lane; i < 16 * hd; i += 32) { int r = i / hd, d = i % hd; if (q0 + r < P) {
        float g = g_all[(size_t)(q0 + r) * att + h * hd + d];
        attn[(size_t)(q0 + r) * att + h * hd + d] = (Osh[i] / l[r]) * (1.f / (1.f + expf(-g))); } }
}

// Batched causal attention for a chunk: block = (h, t), threads = hd. Query
// (head h, chunk-position t) attends over cache 0..pos_start+t with an ONLINE
// (flash) softmax — no per-head scores scratch, one launch for all nq*P queries
// instead of one per position. Numerically the running-max softmax differs from
// the two-pass one only at the ~1e-6 float level; validated bit-exact end to
// end. threads = hd (each owns output dim d and cooperates on each dot).
// KEPT as the fallback path (env ASPIDA_FATTN_NAIVE=1, or hd not a multiple
// of 32 / hd > 256) — k_fattn_attend_tile above is the serving default.
__global__ void k_fattn_attend_chunk(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
    float *__restrict__ attn,
    int nq, int nkv, int hd, int kvd, int pos_start) {
    int t = blockIdx.x / nq, h = blockIdx.x % nq, d = threadIdx.x;
    int len = pos_start + t + 1;
    int rep = nq / nkv, kvh = h / rep;
    int q_off = h * hd, kv_off = kvh * hd, att = nq * hd;
    const float *q = q_all + (size_t) t * att + q_off;
    float scale = rsqrtf((float) hd);
    __shared__ float red[256];
    __shared__ float qsh[256];
    qsh[d] = q[d];
    __syncthreads();
    float m = -3.402823466e38f, l = 0.f, acc = 0.f;
    for (int s = 0; s < len; ++s) {
        const float *k = Kc + (size_t) s * kvd + kv_off;
        float part = qsh[d] * k[d];
        // block reduce (hd threads) -> dot in red[0], broadcast
        red[d] = part; __syncthreads();
        for (int o = blockDim.x / 2; o > 0; o >>= 1) {
            if (d < o) red[d] += red[d + o];
            __syncthreads();
        }
        float dot = red[0] * scale;
        __syncthreads();
        float m_new = fmaxf(m, dot);
        float corr = expf(m - m_new);
        float p = expf(dot - m_new);
        l = l * corr + p;
        acc = acc * corr + p * Vc[(size_t) s * kvd + kv_off + d];
        m = m_new;
    }
    float g = g_all[(size_t) t * att + q_off + d];
    attn[(size_t) t * att + q_off + d] = (acc / l) * (1.f / (1.f + expf(-g)));
}

// Prefill scratch (×PCH), separate from decode buffers so prefill can overlap
// the Driver. Held PER LANE (PMAXLANE): several generations can prefill at once
// (the batcher's whole point). A SINGLE shared set raced when two requests
// prefilled concurrently — garbage output (2026-07-13). Only the LM head runs
// once per prefill (last position), so no [PCH,vocab] per position.
#define PMAXLANE 8
// Split fused input-projection output comb[P, proj] (row layout per position:
//   [ qkv(qo) | alpha(nv) | beta(nv) | z(v_dim) ] )
// into the separate contiguous buffers the delta-net kernels read. Bit-exact
// vs the 4 separate launches (same k_q8_wmma, byte-concatenated weights).
__global__ void k_proj_scatter(
        const float* __restrict__ comb, int P, int proj,
        float* __restrict__ qkv, int qo,
        float* __restrict__ ar,  float* __restrict__ br, int nv,
        float* __restrict__ z,   int vdim) {
    size_t gid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (size_t) P * proj) return;
    int t = (int)(gid / proj), c = (int)(gid % proj);
    float v = comb[(size_t) t * proj + c];
    if      (c < qo)        qkv[(size_t) t * qo   + c]              = v;
    else if (c < qo + nv)   ar [(size_t) t * nv   + (c - qo)]       = v;
    else if (c < qo + 2*nv) br [(size_t) t * nv   + (c - qo - nv)]  = v;
    else                    z  [(size_t) t * vdim + (c - qo - 2*nv)]= v;
}

struct PScr {
    float *Hb,*nxb,*aob,*dqkv,*dcq,*dar,*dbr,*dz,*dg,*db,*dor,*dqg,*dkt,*dvt,
          *dqa,*dga,*datt,*drl,*dhb,*dlog;
    //  Warp-recurrence scratch (ASPIDA_DNET_WARP): normalised Q/K per chunk
    //  and the pre-output-norm delta-net output (o*scale). Sized like q/v dims.
    float *dkn,*dqn,*dosh;
    //  Fused input-projection output [P, qo+2*nv+v_dim] (dnet lever): one
    //  k_q8_wmma over the loader's concatenated L.proj, scattered back into
    //  dqkv/dar/dbr/dz. Saves ~5.3 ms/chunk vs 4 separate launches (the tiny
    //  alpha/beta matmuls are occupancy-starved on their own).
    float *dcomb;
    MoeRoute *droute;
    int *rows;
    int *mg_pos, *mg_k, *mg_cnt;   // expert-grouped MoE: positions/slots per expert + counts
    float *dmoe;                   // grouped-down per-(position,slot) outputs [P][MOE_MAXK+1][dim]
    cudaStream_t stream;
};
static PScr g_ps[PMAXLANE];
//  Prefill scratch is a bounded LAZY POOL, not eager per-lane arrays (VRAM
//  audit 2026-07-15): per-set scratch is 436,040 B x PCH + ~1 MB fixed, so
//  eager 8 x PCH=1024 would need 3.4 GB — OOM in the ~2.5 GB post-model
//  budget. A pool of ASPIDA_PREFILL_SETS sets (default 4, clamp 1..8),
//  lazily allocated and blocking-acquired, caps the peak: 4 sets @1024 =
//  1.7 GB = the SAME footprint as the old 8 x 512. Prefill is already
//  GPU-saturating, so capping concurrent prefills costs ~no wall-clock;
//  the (N+1)th prefill waits a few ms instead of OOMing. Decode buffers
//  (chain_alloc_b) are untouched.
static bool g_prefill_oom = false;
static inline void ckmalloc(void **p, size_t n, const char *what) {
    cudaError_t e = cudaMalloc(p, n);
    if (e != cudaSuccess) { *p = nullptr; g_prefill_oom = true;
        fprintf(stderr, "[PREFILL] cudaMalloc(%s,%zuB) FAILED: %s\n", what, n, cudaGetErrorString(e)); }
}

//  Allocate ONE prefill scratch set (buffers + stream). ckmalloc flags OOM
//  instead of leaving kernels to dereference null pointers.
static void alloc_one(PScr &S) {
    int dim=g_ch_dim; size_t Pd=(size_t)PCH;
    ckmalloc((void**)&S.Hb,Pd*dim*4,"Hb"); ckmalloc((void**)&S.nxb,Pd*dim*4,"nxb"); ckmalloc((void**)&S.aob,Pd*dim*4,"aob");
    ckmalloc((void**)&S.dlog,(size_t)g_ch_vocab*4,"dlog");
    if (g_mx_qo){ ckmalloc((void**)&S.dqkv,Pd*g_mx_qo*4,"dqkv"); ckmalloc((void**)&S.dcq,Pd*g_mx_qo*4,"dcq"); }
    if (g_mx_nv){ ckmalloc((void**)&S.dar,Pd*g_mx_nv*4,"dar"); ckmalloc((void**)&S.dbr,Pd*g_mx_nv*4,"dbr");
                  ckmalloc((void**)&S.dg,Pd*g_mx_nv*4,"dg"); ckmalloc((void**)&S.db,Pd*g_mx_nv*4,"db"); }
    if (g_mx_vd){ ckmalloc((void**)&S.dz,Pd*g_mx_vd*4,"dz"); ckmalloc((void**)&S.dor,Pd*g_mx_vd*4,"dor");
                  ckmalloc((void**)&S.dosh,Pd*g_mx_vd*4,"dosh"); }
    if (g_mx_qo){ ckmalloc((void**)&S.dkn,Pd*g_mx_qo*4,"dkn"); ckmalloc((void**)&S.dqn,Pd*g_mx_qo*4,"dqn"); }
    if (g_mx_qo) ckmalloc((void**)&S.dcomb,Pd*(size_t)(g_mx_qo + 2*g_mx_nv + g_mx_vd)*4,"dcomb");
    if (g_mx_qgd) ckmalloc((void**)&S.dqg,Pd*g_mx_qgd*4,"dqg");
    if (g_mx_kvd){ ckmalloc((void**)&S.dkt,Pd*g_mx_kvd*4,"dkt"); ckmalloc((void**)&S.dvt,Pd*g_mx_kvd*4,"dvt"); }
    if (g_mx_att){ ckmalloc((void**)&S.dqa,Pd*g_mx_att*4,"dqa"); ckmalloc((void**)&S.dga,Pd*g_mx_att*4,"dga");
                   ckmalloc((void**)&S.datt,Pd*g_mx_att*4,"datt"); }
    if (g_mx_nexp) ckmalloc((void**)&S.drl,Pd*g_mx_nexp*4,"drl");
    if (g_mx_hbuf) ckmalloc((void**)&S.dhb,Pd*g_mx_hbuf*4,"dhb");
    ckmalloc((void**)&S.droute,Pd*sizeof(MoeRoute),"droute");
    ckmalloc((void**)&S.rows,Pd*4,"rows");
    if (g_mx_nexp){ ckmalloc((void**)&S.mg_pos,(size_t)g_mx_nexp*Pd*4,"mg_pos");
                    ckmalloc((void**)&S.mg_k,(size_t)g_mx_nexp*Pd*4,"mg_k");
                    ckmalloc((void**)&S.mg_cnt,(size_t)g_mx_nexp*4,"mg_cnt"); }
    ckmalloc((void**)&S.dmoe,Pd*(size_t)(MOE_MAXK+1)*dim*4,"dmoe");
    cudaStreamCreateWithPriority(&S.stream, cudaStreamDefault, aspida_stream_prio(0));
}
static bool g_ps_inited[PMAXLANE]={false}, g_ps_busy[PMAXLANE]={false};
static std::mutex g_ps_mtx; static std::condition_variable g_ps_cv;
static int g_prefill_sets=-1;
static int prefill_sets(){ if(g_prefill_sets<0){ const char*e=getenv("ASPIDA_PREFILL_SETS");
    int n=e?atoi(e):4; if(n<1)n=1; if(n>PMAXLANE)n=PMAXLANE; g_prefill_sets=n; } return g_prefill_sets; }
//  Blocks until one of prefill_sets() sets is free; lazily allocs on first
//  use. Returns slot idx, or -1 on OOM (slot released).
static int pscr_acquire(){
    int n=prefill_sets(); std::unique_lock<std::mutex> lk(g_ps_mtx); int idx=-1;
    g_ps_cv.wait(lk,[&]{ for(int i=0;i<n;++i) if(!g_ps_busy[i]){ idx=i; return true; } return false; });
    g_ps_busy[idx]=true; bool need=!g_ps_inited[idx]; g_ps_inited[idx]=true; lk.unlock();
    if(need) alloc_one(g_ps[idx]);
    if(g_prefill_oom){ std::lock_guard<std::mutex> l2(g_ps_mtx); g_ps_busy[idx]=false; g_ps_cv.notify_one(); return -1; }
    return idx;
}
static void pscr_release(int idx){ { std::lock_guard<std::mutex> lk(g_ps_mtx); g_ps_busy[idx]=false; } g_ps_cv.notify_one(); }

// Expert-grouped MoE for chunked prefill. The per-position kernel k_moe_gu_p_b
// re-reads each expert's weight once PER position that routed to it; at a large
// chunk that is ~chunk/n_exp x redundant. Instead: bucket the chunk's (position,
// slot) pairs by expert, then run ONE tensor-core GEMM per expert over its
// bucket — the expert weight is read once and reused across all its positions.
//
// k_moe_group builds the buckets: mg_pos[e*Pstride + s] / mg_k[...] list the
// positions (and their routing slot) that chose expert e; mg_cnt[e] the count.
__global__ void k_moe_group(const MoeRoute *__restrict__ route_b, int P, int top_k,
                            int *__restrict__ mg_pos, int *__restrict__ mg_k,
                            int *__restrict__ mg_cnt, int Pstride) {
    int p = blockIdx.x * blockDim.x + threadIdx.x; if (p >= P) return;
    const MoeRoute *route = route_b + p;
    for (int k = 0; k < top_k; ++k) {
        int e = route->idx[k];
        int s = atomicAdd(&mg_cnt[e], 1);
        if (s < Pstride) { mg_pos[(size_t) e * Pstride + s] = p; mg_k[(size_t) e * Pstride + s] = k; }
    }
}

// Grouped gate+up+SwiGLU on tensor cores. Each warp computes a 16(position) x
// 16(intermed) tile for one expert (blockIdx.z), gathering its bucket's x rows
// and dequantizing the Q8 gate/up weights into shared FP16 tiles. mode 0 =
// routed experts (buckets from mg_*); mode 1 = the shared expert (all P
// positions, slot MOE_MAXK, gdw/udw already point at the shared weights).
// Writes h_b[p][slot][r] = silu(gate)*up — the exact layout k_moe_down_p_b reads.
__global__ void k_moe_gu_grouped(
    const uint8_t *__restrict__ gdw, const uint8_t *__restrict__ udw,
    const float *__restrict__ x_b, float *__restrict__ h_b,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k,
    const int *__restrict__ mg_cnt,
    long g_bpe, long u_bpe, int dim, int intermed, int Pstride,
    int top_k, int P, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : P;
    int m0 = blockIdx.y * 16; if (m0 >= cnt) return;
    int n0 = blockIdx.x * 16;
    int lane = threadIdx.x & 31;
    int nb = dim / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *gW = (mode == 0) ? gdw + (size_t) e * g_bpe : gdw;
    const uint8_t *uW = (mode == 0) ? udw + (size_t) e * u_bpe : udw;
    __shared__ half As[16 * 32];
    __shared__ half Gs[32 * 16], Us[32 * 16];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cg, cu;
    nvcuda::wmma::fill_fragment(cg, 0.0f); nvcuda::wmma::fill_fragment(cu, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int m = 0; m < 16; ++m) {
            int gm = m0 + m;
            int p = (gm < cnt) ? ((mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm) : -1;
            float xv = (p >= 0) ? x_b[(size_t) p * dim + kb * 32 + lane] : 0.0f;
            As[(size_t) m * 32 + lane] = __float2half(xv);
        }
        for (int n = 0; n < 16; ++n) {
            int gn = n0 + n;
            const uint8_t *blG = gW + (size_t) gn * bpr + (size_t) kb * 34;
            float dG = f16(blG); const int8_t *qG = (const int8_t *) (blG + 2);
            Gs[(size_t) lane * 16 + n] = __float2half(dG * (float) qG[lane]);
            const uint8_t *blU = uW + (size_t) gn * bpr + (size_t) kb * 34;
            float dU = f16(blU); const int8_t *qU = (const int8_t *) (blU + 2);
            Us[(size_t) lane * 16 + n] = __float2half(dU * (float) qU[lane]);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bg, bu;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, 32);
            nvcuda::wmma::load_matrix_sync(bg, Gs + (size_t) k16 * 16 * 16, 16);
            nvcuda::wmma::load_matrix_sync(bu, Us + (size_t) k16 * 16 * 16, 16);
            nvcuda::wmma::mma_sync(cg, af, bg, cg);
            nvcuda::wmma::mma_sync(cu, af, bu, cu);
        }
        __syncwarp();
    }
    __shared__ float Cg[16 * 16], Cu[16 * 16];
    nvcuda::wmma::store_matrix_sync(Cg, cg, 16, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(Cu, cu, 16, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < 256; idx += 32) {
        int m = idx / 16, n = idx % 16, gm = m0 + m;
        if (gm < cnt && (n0 + n) < intermed) {
            int p = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
            int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
            float g = Cg[idx], u = Cu[idx];
            h_b[((size_t) p * (MOE_MAXK + 1) + k) * intermed + n0 + n] = (g / (1.f + expf(-g))) * u;
        }
    }
}

// Grouped down projection on tensor cores. Like k_moe_gu_grouped but the input
// is the SwiGLU output h_b[p][slot] (K=intermed) and the output goes to a
// per-(position,slot) buffer d_buf[p][slot][dim] — NOT summed here, so the
// cross-expert combine can run in a fixed, deterministic order afterwards.
__global__ void k_moe_down_grouped(
    const uint8_t *__restrict__ ddw, const float *__restrict__ h_b, float *__restrict__ d_buf,
    const int *__restrict__ mg_pos, const int *__restrict__ mg_k, const int *__restrict__ mg_cnt,
    long d_bpe, int intermed, int dim, int Pstride, int top_k, int P, int mode) {
    int e = blockIdx.z;
    int cnt = (mode == 0) ? mg_cnt[e] : P;
    int m0 = blockIdx.y * 16; if (m0 >= cnt) return;
    int n0 = blockIdx.x * 16;
    int lane = threadIdx.x & 31;
    int nb = intermed / 32; size_t bpr = (size_t) nb * 34;
    const uint8_t *W = (mode == 0) ? ddw + (size_t) e * d_bpe : ddw;
    __shared__ half As[16 * 32];
    __shared__ half Bs[32 * 16];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> cd;
    nvcuda::wmma::fill_fragment(cd, 0.0f);
    for (int kb = 0; kb < nb; ++kb) {
        for (int m = 0; m < 16; ++m) {
            int gm = m0 + m; float hv = 0.0f;
            if (gm < cnt) {
                int p = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
                int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
                hv = h_b[((size_t) p * (MOE_MAXK + 1) + k) * intermed + kb * 32 + lane];
            }
            As[(size_t) m * 32 + lane] = __float2half(hv);
        }
        for (int n = 0; n < 16; ++n) {
            const uint8_t *bl = W + (size_t) (n0 + n) * bpr + (size_t) kb * 34;
            float d = f16(bl); const int8_t *q = (const int8_t *) (bl + 2);
            Bs[(size_t) lane * 16 + n] = __float2half(d * (float) q[lane]);
        }
        __syncwarp();
        #pragma unroll
        for (int k16 = 0; k16 < 2; ++k16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> af;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> bf;
            nvcuda::wmma::load_matrix_sync(af, As + k16 * 16, 32);
            nvcuda::wmma::load_matrix_sync(bf, Bs + (size_t) k16 * 16 * 16, 16);
            nvcuda::wmma::mma_sync(cd, af, bf, cd);
        }
        __syncwarp();
    }
    __shared__ float Cs[16 * 16];
    nvcuda::wmma::store_matrix_sync(Cs, cd, 16, nvcuda::wmma::mem_row_major);
    for (int idx = lane; idx < 256; idx += 32) {
        int m = idx / 16, n = idx % 16, gm = m0 + m;
        if (gm < cnt && (n0 + n) < dim) {
            int p = (mode == 0) ? mg_pos[(size_t) e * Pstride + gm] : gm;
            int k = (mode == 0) ? mg_k[(size_t) e * Pstride + gm] : top_k;
            d_buf[((size_t) p * (MOE_MAXK + 1) + k) * dim + n0 + n] = Cs[idx];
        }
    }
}

// Deterministic expert combine: y[p] = sum_k route.w[k] * d_buf[p][k] (routed,
// ascending k) + route.w[shared] * d_buf[p][shared]. Fixed order → bit-stable.
__global__ void k_moe_combine(const float *__restrict__ d_buf, const MoeRoute *__restrict__ route_b,
                              float *__restrict__ y_b, int top_k, int dim, int P) {
    int p = blockIdx.y; if (p >= P) return;
    int d = blockIdx.x * blockDim.x + threadIdx.x; if (d >= dim) return;
    const MoeRoute *r = route_b + p;
    const float *D = d_buf + (size_t) p * (MOE_MAXK + 1) * dim;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) acc += r->w[k] * D[(size_t) k * dim + d];
    acc += r->w[MOE_MAXK] * D[(size_t) top_k * dim + d];
    y_b[(size_t) p * dim + d] = acc;
}

//  MoE Phase B helpers (ggml mul_mat_id prefill) -----------------------------
//  ids for ggml: i32 [P][top_k] from the routing.
__global__ void k_moe_ids(const MoeRoute *__restrict__ route_b, int32_t *__restrict__ ids,
                          int top_k, int P) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * top_k) return;
    int p = i / top_k, k = i - p * top_k;
    ids[i] = route_b[p].idx[k];
}
//  Combine for the ggml path: routed slots come from ggml's mm_id output
//  [p][k][dim] (fp32, contiguous), the shared expert from d_buf slot top_k
//  (written by the mode-1 grouped kernels).  Same fixed order as k_moe_combine.
__global__ void k_moe_combine_ggml(const float *__restrict__ g_out,
                                   const float *__restrict__ d_buf,
                                   const MoeRoute *__restrict__ route_b,
                                   float *__restrict__ y_b, int top_k, int dim, int P) {
    int p = blockIdx.y; if (p >= P) return;
    int d = blockIdx.x * blockDim.x + threadIdx.x; if (d >= dim) return;
    const MoeRoute *r = route_b + p;
    const float *G = g_out + (size_t) p * top_k * dim;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) acc += r->w[k] * G[(size_t) k * dim + d];
    acc += r->w[MOE_MAXK] * d_buf[((size_t) p * (MOE_MAXK + 1) + top_k) * dim + d];
    y_b[(size_t) p * dim + d] = acc;
}

// Prefill P positions [pos_start .. pos_start+P-1] of ONE generation on LANE
// `lane`, updating its resident per-layer state (handles[li]); return the
// LM-head logits of the LAST position in last_logits (what the caller samples).
// Each lane has private scratch, so concurrent lanes never race.
extern "C" void aspida_gpu_chain_prefill(int lane, int P, const int *rows, int pos_start,
                                         const int *handles, float *last_logits) {
    if (P < 1) return; if (P > PCH) P = PCH;
    (void) lane;   //  kept for ABI; the pool assigns the scratch set now
    int slot = pscr_acquire();
    if (slot < 0) { fprintf(stderr, "[PREFILL] scratch OOM — chunk aborted\n"); return; }
    PScr &S = g_ps[slot];
    float *pHb=S.Hb,*pnxb=S.nxb,*paob=S.aob,*pdqkv=S.dqkv,*pdcq=S.dcq,*pdar=S.dar,
          *pdbr=S.dbr,*pdz=S.dz,*pdg=S.dg,*pdb=S.db,*pdor=S.dor,*pdqg=S.dqg,
          *pdkt=S.dkt,*pdvt=S.dvt,*pdqa=S.dqa,*pdga=S.dga,*pdatt=S.datt,*pdrl=S.drl,
          *pdhb=S.dhb,*pdlog=S.dlog,*pdkn=S.dkn,*pdqn=S.dqn,*pdosh=S.dosh;
    MoeRoute *pdroute=S.droute;
    int *p_rows=S.rows;
    int *pmg_pos=S.mg_pos,*pmg_k=S.mg_k,*pmg_cnt=S.mg_cnt;
    float *pdmoe=S.dmoe;
    int dim=g_ch_dim, NL=(int)g_chain.size(); cudaStream_t st=S.stream;
    //  Opt-in per-phase profiler (env ASPIDA_PREFILL_PROF): splits GPU time
    //  into the attention/dnet block vs the MoE block, aggregated over layers.
    //  Zero cost when the env is unset. Per-layer event syncs perturb wall
    //  time (~2x) so use it only for measurement, never in a serving config.
    static int pprof = getenv("ASPIDA_PREFILL_PROF") ? 1 : 0;
    cudaEvent_t pe0=0,pe1=0,pe2=0; double acc_attn=0, acc_moe=0, acc_dnet=0, acc_fattn=0, acc_dproj=0, acc_recur=0;
    if (pprof) { cudaEventCreate(&pe0); cudaEventCreate(&pe1); cudaEventCreate(&pe2); }
    cudaMemcpyAsync(p_rows, rows, (size_t)P*4, cudaMemcpyHostToDevice, st);
    k_embed_b<<<((size_t)dim*P+255)/256,256,0,st>>>(g_ch_embed, p_rows, pHb, dim, P);
    for (int li=0; li<NL; ++li) {
        ChainLayer &L=g_chain[li];
        if (pprof) cudaEventRecord(pe0, st);
        k_norm1_b<<<P,256,0,st>>>(pHb, L.attn_norm, pnxb, dim, P);
        if (!L.is_fattn) {
            DnetState ds = g_dnet[handles[li]];
            //  Fused input projection (dnet lever): one k_q8_wmma over the
            //  loader's concatenated qkv|alpha|beta|gate weight + a cheap
            //  scatter. Bit-exact; ~1.3x over 4 separate launches (the 2048x32
            //  alpha/beta matmuls alone are occupancy-starved: 32 warps/GPU).
            if (L.proj_fused && S.dcomb) {
                launch_mv_b(L.proj, L.qkv_k, dim, L.proj_out, pnxb, S.dcomb, P, st);
                k_proj_scatter<<<((size_t)P*L.proj_out+255)/256,256,0,st>>>(
                    S.dcomb, P, L.proj_out, pdqkv, L.qo, pdar, pdbr, L.nv, pdz, L.v_dim);
            } else {
                launch_mv_b(L.qkv, L.qkv_k, dim, L.qo, pnxb, pdqkv, P, st);
                launch_mv_b(L.al, L.al_k, dim, L.nv, pnxb, pdar, P, st);
                launch_mv_b(L.be, L.be_k, dim, L.nv, pnxb, pdbr, P, st);
                launch_mv_b(L.ga, L.ga_k, dim, L.v_dim, pnxb, pdz, P, st);
            }
            size_t shmem=(size_t)(4*L.khd+2*L.vhd)*4;
            k_dnet_conv_chunk<<<(L.qo+255)/256,256,0,st>>>(pdqkv, ds.hist, L.conv, pdcq, L.qo, L.kernel, P);
            k_dnet_gates_b<<<((size_t)P*L.nv+255)/256,256,0,st>>>(pdar, pdbr, L.aw, L.dtw, pdg, pdb, L.nv, P);
            if (pprof) { cudaEventRecord(pe1, st); cudaEventSynchronize(pe1); float p=0; cudaEventElapsedTime(&p, pe0, pe1); acc_dproj += p; }
            //  Warp-parallel register-resident recurrence (~14x): DEFAULT after
            //  the eval gate (no quality regression). ASPIDA_DNET_SEQ=1 forces
            //  the bit-exact sequential kernel (A/B, debugging).
            static int dnet_seq = getenv("ASPIDA_DNET_SEQ") ? 1 : 0;
            if (!dnet_seq && (L.khd & 31) == 0 && L.khd <= 256 && L.vhd <= 256) {
                //  Register-resident warp recurrence: normalise Q/K, run the
                //  column-parallel scan (S in registers), then RMS+z-gate.
                k_dnet_qk_norm<<<P * L.nkh, L.khd, 0, st>>>(
                    pdcq, pdkn, pdqn, L.khd, L.q_dim, L.nkh, L.qo, P);
                int RW = 8;   //  warps (=columns) per block
                dim3 rg((unsigned) L.nv, (unsigned) ((L.vhd + RW - 1) / RW));
                k_dnet_recur_warp<<<rg, RW * 32, (size_t) 2 * L.khd * 4, st>>>(
                    ds.S, pdkn, pdqn, pdcq, pdg, pdb, pdosh,
                    L.khd, L.vhd, L.q_dim, L.nkh, L.nv, L.qo, L.v_dim, P);
                k_dnet_out_norm<<<P * L.nv, L.vhd, 0, st>>>(
                    pdosh, pdz, L.nw, pdor, L.vhd, L.nv, L.v_dim, P);
            } else {
                k_dnet_recur_chunk<<<L.nv, L.khd, shmem, st>>>(
                    ds.S, pdcq, pdg, pdb, pdz, L.nw, pdor,
                    L.khd, L.vhd, L.q_dim, L.nkh, L.nv, L.qo, L.v_dim, P);
            }
            if (pprof) { cudaEventRecord(pe2, st); cudaEventSynchronize(pe2); float r=0; cudaEventElapsedTime(&r, pe1, pe2); acc_recur += r; }
            launch_mv_b(L.ow, L.ow_k, L.v_dim, dim, pdor, paob, P, st);
        } else {
            FattnState fs = g_fattn[handles[li]];
            int kvd=L.nkv*L.hd, att=L.nq*L.hd, qgd=L.nq*2*L.hd;
            launch_mv_b(L.qw, L.qw_k, dim, qgd, pnxb, pdqg, P, st);
            launch_mv_b(L.kw, L.kw_k, dim, kvd, pnxb, pdkt, P, st);
            launch_mv_b(L.vw, L.vw_k, dim, kvd, pnxb, pdvt, P, st);
            //  Write ALL P positions' K/V + rotated Q in one launch, then attend
            //  all nq*P queries in one launch (causal by pos_start+t). K/V of
            //  earlier chunk positions are resident before later queries read
            //  them because the prep launch completes before the attend launch.
            k_fattn_prep_chunk<<<(size_t)(L.nq+L.nkv)*P, L.hd, (size_t)L.hd*4, st>>>(
                pdqg, pdkt, pdvt, L.qn, L.kn, pdqa, pdga, fs.K, fs.V,
                L.nq,L.nkv,L.hd,kvd, pos_start, L.rd,L.base,L.freq_scale,L.m_scale,
                L.yarn_on,L.corr_lo,L.corr_hi, L.ffp,L.use_ff,L.interleaved,L.sec_total, P);
            //  Phase B: prefill full-attention via llama.cpp fattn-mma
            //  (ncols2=8 GQA column-packing + cp_async pipelining), through the
            //  ggml public API. Measured ~6.5x the previous wmma3 kernel at
            //  hd=256/40k on the an NVIDIA GPU (139 TFLOPS, ~38% of peak). fs.K/fs.V are
            //  the resident fp16 cache; pdqa the rotated Q [t][h][d]; the
            //  per-dim sigmoid gate (pdga) is folded in the epilogue. fp16-KV
            //  precision ~8e-4 vs fp32 ref (must clear eval-hura, like wmma3
            //  did). The old wmma3/tile_gqa/attend_chunk kernels remain compiled
            //  but unused; roll back by redeploying the previous .so.
            //  See gpu/fattn_ggml.cuh for the layout mapping + repack.
            aspida_ggml_fattn_prefill(pdqa, fs.K, fs.V, pdga, pdatt,
                                      L.nq, L.nkv, L.hd, P, pos_start, st);
            launch_mv_b(L.fow, L.fow_k, att, dim, pdatt, paob, P, st);
        }
        k_axpy_b<<<((size_t)dim*P+255)/256,256,0,st>>>(pHb, paob, dim, P);
        if (pprof) { cudaEventRecord(pe1, st); cudaEventSynchronize(pe1);
            float a=0; cudaEventElapsedTime(&a, pe0, pe1);
            acc_attn += a; if (L.is_fattn) acc_fattn += a; else acc_dnet += a; }
        if (L.has_moe) {
            k_norm1_b<<<P,256,0,st>>>(pHb, L.post_norm, pnxb, dim, P);
            launch_mv_b(L.rw, L.rk, dim, L.n_exp, pnxb, pdrl, P, st);
            k_moe_route_b<<<P,256,0,st>>>(pdrl, pnxb, L.sgi, L.sgi_len, L.n_exp, L.top_k, dim, pdroute, P);
            int w1=(L.top_k+1)*L.intermed;
            int b1b=((size_t)P*w1*32+255)/256, b2b=((size_t)P*dim*32+255)/256;
            bool grouped = (L.gk==5 && L.uk==5 && L.sgk==5 && L.suk==5 &&
                            L.dk==5 && L.sdk==5 && pmg_cnt);
            //  MoE Phase B: routed experts via llama.cpp mul_mat_id (MMQ int8
            //  tensor cores) — measured 2.9x the grouped kernels at hura dims
            //  (5.71 -> ~1.95 ms/layer-chunk).  The shared expert + the
            //  deterministic combine stay on the aspida side.  Falls back to
            //  the grouped path on any failure (latched) or ASPIDA_MOE_NOGGML=1.
            static int moe_noggml = getenv("ASPIDA_MOE_NOGGML") ? 1 : 0;
            static int moe_ggml_failed = 0;
            const float *moe_gout = nullptr;
            if (grouped && !moe_noggml && !moe_ggml_failed && L.ggt && L.top_k <= MOE_MAXK) {
                //  ids for mul_mat_id — pmg_pos is unused by the ggml path and
                //  the mode-1 (shared) kernels never read mg_*, so reuse it.
                k_moe_ids<<<((size_t)P*L.top_k+255)/256,256,0,st>>>(pdroute, (int32_t*)pmg_pos, L.top_k, P);
                //  shared expert via the existing mode-1 grouped kernels
                //  (h_b/d_buf slot top_k), overlapped on the aspida stream.
                dim3 grSh((L.intermed+15)/16,(P+15)/16,1);
                k_moe_gu_grouped<<<grSh,32,0,st>>>(L.sgdw,L.sudw, pnxb, pdhb, pmg_pos,pmg_k,pmg_cnt,
                    0,0, dim,L.intermed, PCH, L.top_k, P, 1);
                dim3 grDs((dim+15)/16,(P+15)/16,1);
                k_moe_down_grouped<<<grDs,32,0,st>>>(L.sddw, pdhb, pdmoe, pmg_pos,pmg_k,pmg_cnt,
                    0, L.intermed, dim, PCH, L.top_k, P, 1);
                moe_gout = aspida_ggml_moe_prefill(
                    (ggml_tensor *) L.ggt, (ggml_tensor *) L.ugt, (ggml_tensor *) L.dgt,
                    pnxb, (const int32_t *) pmg_pos, dim, L.intermed, L.top_k, P, st);
                if (!moe_gout) { moe_ggml_failed = 1;
                    fprintf(stderr, "[MOE] ggml mul_mat_id failed — grouped fallback\n"); }
            }
            if (moe_gout) {
                dim3 grC((dim+255)/256, P, 1);
                k_moe_combine_ggml<<<grC,256,0,st>>>(moe_gout, pdmoe, pdroute, paob, L.top_k, dim, P);
            } else if (grouped) {
                //  Expert-grouped tensor-core MoE: bucket positions by expert
                //  (once), then one GEMM per expert reading its weight once.
                cudaMemsetAsync(pmg_cnt, 0, (size_t)L.n_exp*4, st);
                k_moe_group<<<(P+255)/256,256,0,st>>>(pdroute, P, L.top_k, pmg_pos, pmg_k, pmg_cnt, PCH);
                //  gate+up+SwiGLU -> h_b[p][slot]
                dim3 grR((L.intermed+15)/16, (PCH+15)/16, L.n_exp);
                k_moe_gu_grouped<<<grR,32,0,st>>>(L.gdw,L.udw, pnxb, pdhb, pmg_pos,pmg_k,pmg_cnt,
                    L.g_bpe,L.u_bpe, dim,L.intermed, PCH, L.top_k, P, 0);
                dim3 grSh((L.intermed+15)/16, (P+15)/16, 1);
                k_moe_gu_grouped<<<grSh,32,0,st>>>(L.sgdw,L.sudw, pnxb, pdhb, pmg_pos,pmg_k,pmg_cnt,
                    0,0, dim,L.intermed, PCH, L.top_k, P, 1);
                //  down -> per-(position,slot) d_buf, then deterministic combine
                dim3 grD((dim+15)/16, (PCH+15)/16, L.n_exp);
                k_moe_down_grouped<<<grD,32,0,st>>>(L.ddw, pdhb, pdmoe, pmg_pos,pmg_k,pmg_cnt,
                    L.d_bpe, L.intermed, dim, PCH, L.top_k, P, 0);
                dim3 grDs((dim+15)/16, (P+15)/16, 1);
                k_moe_down_grouped<<<grDs,32,0,st>>>(L.sddw, pdhb, pdmoe, pmg_pos,pmg_k,pmg_cnt,
                    0, L.intermed, dim, PCH, L.top_k, P, 1);
                dim3 grC((dim+255)/256, P, 1);
                k_moe_combine<<<grC,256,0,st>>>(pdmoe, pdroute, paob, L.top_k, dim, P);
            } else {
                k_moe_gu_p_b<<<b1b,256,0,st>>>(L.gdw,L.udw,L.sgdw,L.sudw, pnxb, pdhb, pdroute, L.top_k,dim,L.intermed,L.g_bpe,L.u_bpe,L.gk,L.uk,L.sgk,L.suk, P);
                k_moe_down_p_b<<<b2b,256,0,st>>>(L.ddw,L.sddw, pdhb, paob, pdroute, L.top_k,L.intermed,dim,L.d_bpe,L.dk,L.sdk, P);
            }
            k_axpy_b<<<((size_t)dim*P+255)/256,256,0,st>>>(pHb, paob, dim, P);
            if (pprof) { cudaEventRecord(pe2, st); cudaEventSynchronize(pe2);
                float m=0; cudaEventElapsedTime(&m, pe1, pe2); acc_moe += m; }
        }
    }
    if (pprof) {
        fprintf(stderr, "[PREFILLPROF] P=%d NL=%d | attn-block=%.2fms (dnet=%.2f [dproj=%.2f recur=%.2f] fattn=%.2f) "
            "moe=%.2fms total=%.2fms | per-tok: attn=%.3f moe=%.3f sum=%.3f ms\n",
            P, NL, acc_attn, acc_dnet, acc_dproj, acc_recur, acc_fattn, acc_moe, acc_attn+acc_moe,
            acc_attn/P, acc_moe/P, (acc_attn+acc_moe)/P);
        cudaEventDestroy(pe0); cudaEventDestroy(pe1); cudaEventDestroy(pe2);
    }
    // LM head on the LAST position only.
    const float *last_h = pHb + (size_t)(P - 1) * dim;
    k_norm1<<<1,256,0,st>>>((float *) last_h, g_ch_fnorm, pnxb, dim);
    launch_mv_st(g_ch_lm, g_ch_lm_k, dim, g_ch_vocab, pnxb, pdlog, st);
    cudaMemcpyAsync(last_logits, pdlog, (size_t)g_ch_vocab*4, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    pscr_release(slot);
}

// Per-position variant of chain_prefill: same resident forward, but projects
// EVERY position through the LM head into all_logits[p*vocab + k], not just
// the last. This is the verify primitive speculative decoding needs: score
// gamma draft tokens in one pass and read the target's prediction at each.
// The P hidden states already exist in pHb (the forward computes them all);
// only the projection differs. Reuses the vocab-sized pdlog scratch per
// position in a loop, so no P*vocab device buffer is needed.
extern "C" void aspida_gpu_chain_prefill_logits(int lane, int P, const int *rows, int pos_start,
                                                const int *handles, float *all_logits) {
    if (P < 1) return; if (P > PCH) P = PCH;
    (void) lane;   //  kept for ABI; the pool assigns the scratch set now
    int slot = pscr_acquire();
    if (slot < 0) { fprintf(stderr, "[PREFILL] scratch OOM — chunk aborted\n"); return; }
    PScr &S = g_ps[slot];
    float *pHb=S.Hb,*pnxb=S.nxb,*paob=S.aob,*pdqkv=S.dqkv,*pdcq=S.dcq,*pdar=S.dar,
          *pdbr=S.dbr,*pdz=S.dz,*pdg=S.dg,*pdb=S.db,*pdor=S.dor,*pdqg=S.dqg,
          *pdkt=S.dkt,*pdvt=S.dvt,*pdqa=S.dqa,*pdga=S.dga,*pdatt=S.datt,*pdrl=S.drl,
          *pdhb=S.dhb,*pdlog=S.dlog,*pdkn=S.dkn,*pdqn=S.dqn,*pdosh=S.dosh;
    MoeRoute *pdroute=S.droute;
    int *p_rows=S.rows;
    int *pmg_pos=S.mg_pos,*pmg_k=S.mg_k,*pmg_cnt=S.mg_cnt;
    float *pdmoe=S.dmoe;
    int dim=g_ch_dim, NL=(int)g_chain.size(); cudaStream_t st=S.stream;
    //  Opt-in per-phase profiler (env ASPIDA_PREFILL_PROF): splits GPU time
    //  into the attention/dnet block vs the MoE block, aggregated over layers.
    //  Zero cost when the env is unset. Per-layer event syncs perturb wall
    //  time (~2x) so use it only for measurement, never in a serving config.
    static int pprof = getenv("ASPIDA_PREFILL_PROF") ? 1 : 0;
    cudaEvent_t pe0=0,pe1=0,pe2=0; double acc_attn=0, acc_moe=0, acc_dnet=0, acc_fattn=0, acc_dproj=0, acc_recur=0;
    if (pprof) { cudaEventCreate(&pe0); cudaEventCreate(&pe1); cudaEventCreate(&pe2); }
    cudaMemcpyAsync(p_rows, rows, (size_t)P*4, cudaMemcpyHostToDevice, st);
    k_embed_b<<<((size_t)dim*P+255)/256,256,0,st>>>(g_ch_embed, p_rows, pHb, dim, P);
    for (int li=0; li<NL; ++li) {
        ChainLayer &L=g_chain[li];
        if (pprof) cudaEventRecord(pe0, st);
        k_norm1_b<<<P,256,0,st>>>(pHb, L.attn_norm, pnxb, dim, P);
        if (!L.is_fattn) {
            DnetState ds = g_dnet[handles[li]];
            //  Fused input projection (dnet lever): one k_q8_wmma over the
            //  loader's concatenated qkv|alpha|beta|gate weight + a cheap
            //  scatter. Bit-exact; ~1.3x over 4 separate launches (the 2048x32
            //  alpha/beta matmuls alone are occupancy-starved: 32 warps/GPU).
            if (L.proj_fused && S.dcomb) {
                launch_mv_b(L.proj, L.qkv_k, dim, L.proj_out, pnxb, S.dcomb, P, st);
                k_proj_scatter<<<((size_t)P*L.proj_out+255)/256,256,0,st>>>(
                    S.dcomb, P, L.proj_out, pdqkv, L.qo, pdar, pdbr, L.nv, pdz, L.v_dim);
            } else {
                launch_mv_b(L.qkv, L.qkv_k, dim, L.qo, pnxb, pdqkv, P, st);
                launch_mv_b(L.al, L.al_k, dim, L.nv, pnxb, pdar, P, st);
                launch_mv_b(L.be, L.be_k, dim, L.nv, pnxb, pdbr, P, st);
                launch_mv_b(L.ga, L.ga_k, dim, L.v_dim, pnxb, pdz, P, st);
            }
            size_t shmem=(size_t)(4*L.khd+2*L.vhd)*4;
            k_dnet_conv_chunk<<<(L.qo+255)/256,256,0,st>>>(pdqkv, ds.hist, L.conv, pdcq, L.qo, L.kernel, P);
            k_dnet_gates_b<<<((size_t)P*L.nv+255)/256,256,0,st>>>(pdar, pdbr, L.aw, L.dtw, pdg, pdb, L.nv, P);
            if (pprof) { cudaEventRecord(pe1, st); cudaEventSynchronize(pe1); float p=0; cudaEventElapsedTime(&p, pe0, pe1); acc_dproj += p; }
            //  Warp-parallel register-resident recurrence (~14x): DEFAULT after
            //  the eval gate (no quality regression). ASPIDA_DNET_SEQ=1 forces
            //  the bit-exact sequential kernel (A/B, debugging).
            static int dnet_seq = getenv("ASPIDA_DNET_SEQ") ? 1 : 0;
            if (!dnet_seq && (L.khd & 31) == 0 && L.khd <= 256 && L.vhd <= 256) {
                //  Register-resident warp recurrence: normalise Q/K, run the
                //  column-parallel scan (S in registers), then RMS+z-gate.
                k_dnet_qk_norm<<<P * L.nkh, L.khd, 0, st>>>(
                    pdcq, pdkn, pdqn, L.khd, L.q_dim, L.nkh, L.qo, P);
                int RW = 8;   //  warps (=columns) per block
                dim3 rg((unsigned) L.nv, (unsigned) ((L.vhd + RW - 1) / RW));
                k_dnet_recur_warp<<<rg, RW * 32, (size_t) 2 * L.khd * 4, st>>>(
                    ds.S, pdkn, pdqn, pdcq, pdg, pdb, pdosh,
                    L.khd, L.vhd, L.q_dim, L.nkh, L.nv, L.qo, L.v_dim, P);
                k_dnet_out_norm<<<P * L.nv, L.vhd, 0, st>>>(
                    pdosh, pdz, L.nw, pdor, L.vhd, L.nv, L.v_dim, P);
            } else {
                k_dnet_recur_chunk<<<L.nv, L.khd, shmem, st>>>(
                    ds.S, pdcq, pdg, pdb, pdz, L.nw, pdor,
                    L.khd, L.vhd, L.q_dim, L.nkh, L.nv, L.qo, L.v_dim, P);
            }
            if (pprof) { cudaEventRecord(pe2, st); cudaEventSynchronize(pe2); float r=0; cudaEventElapsedTime(&r, pe1, pe2); acc_recur += r; }
            launch_mv_b(L.ow, L.ow_k, L.v_dim, dim, pdor, paob, P, st);
        } else {
            FattnState fs = g_fattn[handles[li]];
            int kvd=L.nkv*L.hd, att=L.nq*L.hd, qgd=L.nq*2*L.hd;
            launch_mv_b(L.qw, L.qw_k, dim, qgd, pnxb, pdqg, P, st);
            launch_mv_b(L.kw, L.kw_k, dim, kvd, pnxb, pdkt, P, st);
            launch_mv_b(L.vw, L.vw_k, dim, kvd, pnxb, pdvt, P, st);
            //  Write ALL P positions' K/V + rotated Q in one launch, then attend
            //  all nq*P queries in one launch (causal by pos_start+t). K/V of
            //  earlier chunk positions are resident before later queries read
            //  them because the prep launch completes before the attend launch.
            k_fattn_prep_chunk<<<(size_t)(L.nq+L.nkv)*P, L.hd, (size_t)L.hd*4, st>>>(
                pdqg, pdkt, pdvt, L.qn, L.kn, pdqa, pdga, fs.K, fs.V,
                L.nq,L.nkv,L.hd,kvd, pos_start, L.rd,L.base,L.freq_scale,L.m_scale,
                L.yarn_on,L.corr_lo,L.corr_hi, L.ffp,L.use_ff,L.interleaved,L.sec_total, P);
            //  Phase B: prefill full-attention via llama.cpp fattn-mma
            //  (ncols2=8 GQA column-packing + cp_async pipelining), through the
            //  ggml public API. Measured ~6.5x the previous wmma3 kernel at
            //  hd=256/40k on the an NVIDIA GPU (139 TFLOPS, ~38% of peak). fs.K/fs.V are
            //  the resident fp16 cache; pdqa the rotated Q [t][h][d]; the
            //  per-dim sigmoid gate (pdga) is folded in the epilogue. fp16-KV
            //  precision ~8e-4 vs fp32 ref (must clear eval-hura, like wmma3
            //  did). The old wmma3/tile_gqa/attend_chunk kernels remain compiled
            //  but unused; roll back by redeploying the previous .so.
            //  See gpu/fattn_ggml.cuh for the layout mapping + repack.
            aspida_ggml_fattn_prefill(pdqa, fs.K, fs.V, pdga, pdatt,
                                      L.nq, L.nkv, L.hd, P, pos_start, st);
            launch_mv_b(L.fow, L.fow_k, att, dim, pdatt, paob, P, st);
        }
        k_axpy_b<<<((size_t)dim*P+255)/256,256,0,st>>>(pHb, paob, dim, P);
        if (pprof) { cudaEventRecord(pe1, st); cudaEventSynchronize(pe1);
            float a=0; cudaEventElapsedTime(&a, pe0, pe1);
            acc_attn += a; if (L.is_fattn) acc_fattn += a; else acc_dnet += a; }
        if (L.has_moe) {
            k_norm1_b<<<P,256,0,st>>>(pHb, L.post_norm, pnxb, dim, P);
            launch_mv_b(L.rw, L.rk, dim, L.n_exp, pnxb, pdrl, P, st);
            k_moe_route_b<<<P,256,0,st>>>(pdrl, pnxb, L.sgi, L.sgi_len, L.n_exp, L.top_k, dim, pdroute, P);
            int w1=(L.top_k+1)*L.intermed;
            int b1b=((size_t)P*w1*32+255)/256, b2b=((size_t)P*dim*32+255)/256;
            bool grouped = (L.gk==5 && L.uk==5 && L.sgk==5 && L.suk==5 &&
                            L.dk==5 && L.sdk==5 && pmg_cnt);
            //  MoE Phase B: routed experts via llama.cpp mul_mat_id (MMQ int8
            //  tensor cores) — measured 2.9x the grouped kernels at hura dims
            //  (5.71 -> ~1.95 ms/layer-chunk).  The shared expert + the
            //  deterministic combine stay on the aspida side.  Falls back to
            //  the grouped path on any failure (latched) or ASPIDA_MOE_NOGGML=1.
            static int moe_noggml = getenv("ASPIDA_MOE_NOGGML") ? 1 : 0;
            static int moe_ggml_failed = 0;
            const float *moe_gout = nullptr;
            if (grouped && !moe_noggml && !moe_ggml_failed && L.ggt && L.top_k <= MOE_MAXK) {
                //  ids for mul_mat_id — pmg_pos is unused by the ggml path and
                //  the mode-1 (shared) kernels never read mg_*, so reuse it.
                k_moe_ids<<<((size_t)P*L.top_k+255)/256,256,0,st>>>(pdroute, (int32_t*)pmg_pos, L.top_k, P);
                //  shared expert via the existing mode-1 grouped kernels
                //  (h_b/d_buf slot top_k), overlapped on the aspida stream.
                dim3 grSh((L.intermed+15)/16,(P+15)/16,1);
                k_moe_gu_grouped<<<grSh,32,0,st>>>(L.sgdw,L.sudw, pnxb, pdhb, pmg_pos,pmg_k,pmg_cnt,
                    0,0, dim,L.intermed, PCH, L.top_k, P, 1);
                dim3 grDs((dim+15)/16,(P+15)/16,1);
                k_moe_down_grouped<<<grDs,32,0,st>>>(L.sddw, pdhb, pdmoe, pmg_pos,pmg_k,pmg_cnt,
                    0, L.intermed, dim, PCH, L.top_k, P, 1);
                moe_gout = aspida_ggml_moe_prefill(
                    (ggml_tensor *) L.ggt, (ggml_tensor *) L.ugt, (ggml_tensor *) L.dgt,
                    pnxb, (const int32_t *) pmg_pos, dim, L.intermed, L.top_k, P, st);
                if (!moe_gout) { moe_ggml_failed = 1;
                    fprintf(stderr, "[MOE] ggml mul_mat_id failed — grouped fallback\n"); }
            }
            if (moe_gout) {
                dim3 grC((dim+255)/256, P, 1);
                k_moe_combine_ggml<<<grC,256,0,st>>>(moe_gout, pdmoe, pdroute, paob, L.top_k, dim, P);
            } else if (grouped) {
                //  Expert-grouped tensor-core MoE: bucket positions by expert
                //  (once), then one GEMM per expert reading its weight once.
                cudaMemsetAsync(pmg_cnt, 0, (size_t)L.n_exp*4, st);
                k_moe_group<<<(P+255)/256,256,0,st>>>(pdroute, P, L.top_k, pmg_pos, pmg_k, pmg_cnt, PCH);
                //  gate+up+SwiGLU -> h_b[p][slot]
                dim3 grR((L.intermed+15)/16, (PCH+15)/16, L.n_exp);
                k_moe_gu_grouped<<<grR,32,0,st>>>(L.gdw,L.udw, pnxb, pdhb, pmg_pos,pmg_k,pmg_cnt,
                    L.g_bpe,L.u_bpe, dim,L.intermed, PCH, L.top_k, P, 0);
                dim3 grSh((L.intermed+15)/16, (P+15)/16, 1);
                k_moe_gu_grouped<<<grSh,32,0,st>>>(L.sgdw,L.sudw, pnxb, pdhb, pmg_pos,pmg_k,pmg_cnt,
                    0,0, dim,L.intermed, PCH, L.top_k, P, 1);
                //  down -> per-(position,slot) d_buf, then deterministic combine
                dim3 grD((dim+15)/16, (PCH+15)/16, L.n_exp);
                k_moe_down_grouped<<<grD,32,0,st>>>(L.ddw, pdhb, pdmoe, pmg_pos,pmg_k,pmg_cnt,
                    L.d_bpe, L.intermed, dim, PCH, L.top_k, P, 0);
                dim3 grDs((dim+15)/16, (P+15)/16, 1);
                k_moe_down_grouped<<<grDs,32,0,st>>>(L.sddw, pdhb, pdmoe, pmg_pos,pmg_k,pmg_cnt,
                    0, L.intermed, dim, PCH, L.top_k, P, 1);
                dim3 grC((dim+255)/256, P, 1);
                k_moe_combine<<<grC,256,0,st>>>(pdmoe, pdroute, paob, L.top_k, dim, P);
            } else {
                k_moe_gu_p_b<<<b1b,256,0,st>>>(L.gdw,L.udw,L.sgdw,L.sudw, pnxb, pdhb, pdroute, L.top_k,dim,L.intermed,L.g_bpe,L.u_bpe,L.gk,L.uk,L.sgk,L.suk, P);
                k_moe_down_p_b<<<b2b,256,0,st>>>(L.ddw,L.sddw, pdhb, paob, pdroute, L.top_k,L.intermed,dim,L.d_bpe,L.dk,L.sdk, P);
            }
            k_axpy_b<<<((size_t)dim*P+255)/256,256,0,st>>>(pHb, paob, dim, P);
            if (pprof) { cudaEventRecord(pe2, st); cudaEventSynchronize(pe2);
                float m=0; cudaEventElapsedTime(&m, pe1, pe2); acc_moe += m; }
        }
    }
    if (pprof) {
        fprintf(stderr, "[PREFILLPROF] P=%d NL=%d | attn-block=%.2fms (dnet=%.2f [dproj=%.2f recur=%.2f] fattn=%.2f) "
            "moe=%.2fms total=%.2fms | per-tok: attn=%.3f moe=%.3f sum=%.3f ms\n",
            P, NL, acc_attn, acc_dnet, acc_dproj, acc_recur, acc_fattn, acc_moe, acc_attn+acc_moe,
            acc_attn/P, acc_moe/P, (acc_attn+acc_moe)/P);
        cudaEventDestroy(pe0); cudaEventDestroy(pe1); cudaEventDestroy(pe2);
    }
    // LM head on EVERY position. pHb holds all P hidden states; k_norm1 only
    // reads its input (x is const __restrict__), so looping never corrupts pHb.
    for (int p = 0; p < P; p++) {
        const float *hp = pHb + (size_t) p * dim;
        k_norm1<<<1,256,0,st>>>((float *) hp, g_ch_fnorm, pnxb, dim);
        launch_mv_st(g_ch_lm, g_ch_lm_k, dim, g_ch_vocab, pnxb, pdlog, st);
        cudaMemcpyAsync(all_logits + (size_t) p * g_ch_vocab, pdlog,
                        (size_t) g_ch_vocab*4, cudaMemcpyDeviceToHost, st);
    }
    cudaStreamSynchronize(st);
    pscr_release(slot);
}
