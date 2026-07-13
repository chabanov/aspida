// Stream B — GPU matvec shim for the Ada engine (LLM_GPU dlopen's this).
// Exposes one C entry point; weights are uploaded to VRAM once (cached by host
// pointer) and stay resident across tokens. All five K-quants (Q4_K/Q5_K/Q6_K/
// Q3_K/Q2_K), bit-exact vs the CPU engine (build with --fmad=false). Build:
//   nvcc -O3 --fmad=false -arch=native -shared -Xcompiler -fPIC gpu_matvec.cu -o libaspidagpu.so
#include <cuda_fp16.h>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <ctime>

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
// See GPU_RESIDENT_FORWARD.md. Router GEMV -> softmax/top-k (on host, matching
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
struct DnetState { float *S; float *hist; int qo; int kernel; };
static std::vector<DnetState> g_dnet;
static std::vector<int> g_dnet_free;      // freed slots for reuse

extern "C" int aspida_gpu_dnet_new(int nv, int khd, int vhd, int qo, int kernel) {
    DnetState st; size_t n = (size_t) nv * khd * vhd;
    if (cudaMalloc(&st.S, n * 4) != cudaSuccess) return -1;
    cudaMemset(st.S, 0, n * 4);
    size_t hn = (size_t) (kernel > 1 ? kernel - 1 : 1) * qo;
    if (cudaMalloc(&st.hist, hn * 4) != cudaSuccess) { cudaFree(st.S); return -1; }
    cudaMemset(st.hist, 0, hn * 4);
    st.qo = qo; st.kernel = kernel;
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
    if (st.S)    { cudaFree(st.S); st.S = nullptr; }
    if (st.hist) { cudaFree(st.hist); st.hist = nullptr; }
    g_dnet_free.push_back(handle);
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
struct FattnState { float *K, *V, *scores; int max_len, kvd; };
static std::vector<FattnState> g_fattn;
static std::vector<int> g_fattn_free;     // freed slots for reuse

extern "C" int aspida_gpu_fattn_new(int max_len, int kvd, int nq) {
    FattnState st; st.max_len = max_len; st.kvd = kvd;
    if (cudaMalloc(&st.K, (size_t) max_len * kvd * 4) != cudaSuccess) return -1;
    if (cudaMalloc(&st.V, (size_t) max_len * kvd * 4) != cudaSuccess) { cudaFree(st.K); return -1; }
    if (cudaMalloc(&st.scores, (size_t) nq * max_len * 4) != cudaSuccess) {
        cudaFree(st.K); cudaFree(st.V); return -1; }
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
    if (st.K)      { cudaFree(st.K); st.K = nullptr; }
    if (st.V)      { cudaFree(st.V); st.V = nullptr; }
    if (st.scores) { cudaFree(st.scores); st.scores = nullptr; }
    g_fattn_free.push_back(handle);
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
    float *__restrict__ Kc, float *__restrict__ Vc,
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
        Vc[(size_t) pos * kvd + head * hd + d] = vt[head * hd + d];
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
    else      Kc[(size_t) pos * kvd + head * hd + d] = out;
}

// Causal GQA softmax attention over the resident cache + per-dim sigmoid gate.
// One block (256 threads) per q head; len = pos+1 positions.
__global__ void k_fattn_attend(
    const float *__restrict__ q_all, const float *__restrict__ g_all,
    const float *__restrict__ Kc, const float *__restrict__ Vc,
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
        const float *k = Kc + (size_t) s * kvd + kv_off;
        const float *q = q_all + q_off;
        for (int d = 0; d < hd; ++d) dot += q[d] * k[d];
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
            acc += sc[s] * inv * Vc[(size_t) s * kvd + kv_off + tid];
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
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    int total = (top_k + 1) * intermed;
    if (wid >= total) return;
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
    if (lane == 0)
        h[(size_t) k * intermed + r] = (g / (1.f + expf(-g))) * u;
}

__global__ void k_moe_down_p(const uint8_t *__restrict__ ddw, const uint8_t *__restrict__ sddw,
                             const float *__restrict__ h, float *__restrict__ y,
                             const MoeRoute *__restrict__ route, int top_k, int intermed, int dim,
                             long d_bpe, int dk, int sdk) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= dim) return;
    size_t bpr_d = (size_t) (intermed / 256) * kq_bpb(dk);
    size_t bpr_s = (size_t) (intermed / 256) * kq_bpb(sdk);
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        const uint8_t *row = ddw + (size_t) route->idx[k] * d_bpe + (size_t) wid * bpr_d;
        acc += route->w[k] * wrow(row, dk, h + (size_t) k * intermed, intermed, lane);
    }
    acc += route->w[MOE_MAXK]
           * wrow(sddw + (size_t) wid * bpr_s, sdk, h + (size_t) top_k * intermed, intermed, lane);
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
    L.gdw = upload_weight(gate_w, gate_b); L.gk = gate_k;
    L.udw = upload_weight(up_w, up_b); L.uk = up_k;
    L.ddw = upload_weight(down_w, down_b); L.dk = down_k;
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
    cudaStreamCreate(&g_cstream);
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
            k_moe_gu_p<<<b1, 256, 0, st>>>(L.gdw, L.udw, L.sgdw, L.sudw, nx, dhb, droute,
                                           L.top_k, dim, L.intermed, L.g_bpe, L.u_bpe,
                                           L.gk, L.uk, L.sgk, L.suk);
            k_moe_down_p<<<b2, 256, 0, st>>>(L.ddw, L.sddw, dhb, ao, droute, L.top_k,
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
static inline void launch_mv_b(const uint8_t *dw, int kind, int in, int out,
                               const float *dx, float *dy, int B, cudaStream_t st) {
    const int TPB = 256, WPB = TPB / 32; int blocks = (out + WPB - 1) / WPB;
    if (B > MAXB) B = MAXB;
    if (kind == 0)      k_q4k_wb<<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out, B);
    else if (kind == 1) k_q6k_wb<<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out, B);
    else if (kind == 2) k_q5k_wb<<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out, B);
    else if (kind == 3) k_q3k_wb<<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out, B);
    else if (kind == 4) k_q2k_wb<<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out, B);
    else if (kind == 5) k_q8_0_wb<<<blocks, TPB, 0, st>>>(dw, dx, dy, in, out, B);
    else                k_dense_mv_b<<<blocks, TPB, 0, st>>>((const float *) dw, dx, dy, in, out, B);
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
    cudaStreamCreate(&gb_stream);
    chb_inited=1;
}

// Batched decode step for B lanes. handles[b*NL + li] = lane b's layer-li state.
// rows[b], pos[b] = lane b's embedding row and position. logits[b*vocab] out.
extern "C" void aspida_gpu_chain_forward_batch(int B, const int *rows, const int *pos,
                                               const int *handles, float *logits) {
    if (B < 1) return; if (B > BMAX) B = BMAX;
    chain_alloc_b();
    int dim=g_ch_dim, NL=(int)g_chain.size(); cudaStream_t st=gb_stream;
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
        }
    }
    k_norm1_b<<<B,256,0,st>>>(Hb, g_ch_fnorm, nxb, dim, B);
    launch_mv_b(g_ch_lm, g_ch_lm_k, dim, g_ch_vocab, nxb, dlogb, B, st);
    cudaMemcpyAsync(logits, dlogb, (size_t)B*g_ch_vocab*4, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
}
