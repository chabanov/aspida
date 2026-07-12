// Stream B — GPU matvec shim for the Ada engine (LLM_GPU dlopen's this).
// Exposes one C entry point; weights are uploaded to VRAM once (cached by host
// pointer) and stay resident across tokens. All five K-quants (Q4_K/Q5_K/Q6_K/
// Q3_K/Q2_K), bit-exact vs the CPU engine (build with --fmad=false). Build:
//   nvcc -O3 --fmad=false -arch=native -shared -Xcompiler -fPIC gpu_matvec.cu -o libaspidagpu.so
#include <cuda_fp16.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

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

// Launch the right warp-per-row K-quant matvec into a DEVICE buffer:
//   y[out] = W[out,in] . x[in].  No host copy — dx/dy are device-resident.
static inline void launch_matvec(const uint8_t *dw, int kind, int in, int out,
                                 const float *dx, float *dy) {
    const int TPB = 256, WPB = TPB / 32; int blocks = (out + WPB - 1) / WPB;
    if (kind == 0)      k_q4k_w<<<blocks, TPB>>>(dw, dx, dy, in, out);
    else if (kind == 1) k_q6k_w<<<blocks, TPB>>>(dw, dx, dy, in, out);
    else if (kind == 2) k_q5k_w<<<blocks, TPB>>>(dw, dx, dy, in, out);
    else if (kind == 3) k_q3k_w<<<blocks, TPB>>>(dw, dx, dy, in, out);
    else                k_q2k_w<<<blocks, TPB>>>(dw, dx, dy, in, out);
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
    for (int i = lane; i < in; i += 32) acc += r[i] * x[i];
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
static std::vector<float *> g_dnet;   // resident S_All per layer, by handle

extern "C" int aspida_gpu_dnet_new(int nv, int khd, int vhd) {
    float *s; size_t n = (size_t) nv * khd * vhd;
    if (cudaMalloc(&s, n * 4) != cudaSuccess) return -1;
    cudaMemset(s, 0, n * 4);
    g_dnet.push_back(s);
    return (int) g_dnet.size() - 1;
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

extern "C" void aspida_gpu_dnet_recur(int handle, const float *cq, const float *gate,
    const float *beta, const float *z, const float *norm_w, float *o_row,
    int nv, int khd, int vhd, int qo, int q_dim, int n_k_heads, int v_dim) {
    if (handle < 0 || handle >= (int) g_dnet.size()) return;
    float *S = g_dnet[handle];
    static float *dcq = nullptr, *dg = nullptr, *db = nullptr, *dz = nullptr,
                 *dnw = nullptr, *dor = nullptr;
    static int c_qo = 0, c_nv = 0, c_vd = 0, c_vhd = 0;
    if (qo > c_qo)    { if (dcq) cudaFree(dcq); cudaMalloc(&dcq, (size_t) qo * 4);    c_qo = qo; }
    if (nv > c_nv)    { if (dg) cudaFree(dg); cudaMalloc(&dg, (size_t) nv * 4);
                        if (db) cudaFree(db); cudaMalloc(&db, (size_t) nv * 4);       c_nv = nv; }
    if (v_dim > c_vd) { if (dz) cudaFree(dz); cudaMalloc(&dz, (size_t) v_dim * 4);
                        if (dor) cudaFree(dor); cudaMalloc(&dor, (size_t) v_dim * 4); c_vd = v_dim; }
    if (vhd > c_vhd)  { if (dnw) cudaFree(dnw); cudaMalloc(&dnw, (size_t) vhd * 4);   c_vhd = vhd; }
    cudaMemcpy(dcq, cq, (size_t) qo * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, gate, (size_t) nv * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, beta, (size_t) nv * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dz, z, (size_t) v_dim * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dnw, norm_w, (size_t) vhd * 4, cudaMemcpyHostToDevice);
    size_t shmem = (size_t) (4 * khd + 2 * vhd) * 4;
    k_dnet_recur<<<nv, khd, shmem>>>(S, dcq, dg, db, dz, dnw, dor, khd, vhd, q_dim, n_k_heads);
    cudaMemcpy(o_row, dor, (size_t) v_dim * 4, cudaMemcpyDeviceToHost);
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
    // Resident scratch, grown-only across tokens/layers.
    static float *dx = nullptr, *d_gate = nullptr, *d_up = nullptr,
                 *d_h = nullptr, *d_y = nullptr, *d_acc = nullptr;
    static int cdim = 0, cint = 0;
    if (dim > cdim) {
        if (dx) cudaFree(dx);   cudaMalloc(&dx,   (size_t) dim * 4);
        if (d_y) cudaFree(d_y); cudaMalloc(&d_y,  (size_t) dim * 4);
        if (d_acc) cudaFree(d_acc); cudaMalloc(&d_acc, (size_t) dim * 4);
        cdim = dim;
    }
    if (intermed > cint) {
        if (d_gate) cudaFree(d_gate); cudaMalloc(&d_gate, (size_t) intermed * 4);
        if (d_up) cudaFree(d_up);     cudaMalloc(&d_up,   (size_t) intermed * 4);
        if (d_h) cudaFree(d_h);       cudaMalloc(&d_h,    (size_t) intermed * 4);
        cint = intermed;
    }

    uint8_t *gdw = upload_weight(gate_w, gate_bytes);
    uint8_t *udw = upload_weight(up_w, up_bytes);
    uint8_t *ddw = upload_weight(down_w, down_bytes);
    uint8_t *sgdw = upload_weight(shg_w, shg_bytes);
    uint8_t *sudw = upload_weight(shu_w, shu_bytes);
    uint8_t *sddw = upload_weight(shd_w, shd_bytes);
    size_t g_bpe = (size_t)(gate_bytes / n_exp), u_bpe = (size_t)(up_bytes / n_exp),
           d_bpe = (size_t)(down_bytes / n_exp);

    cudaMemcpy(dx, x, (size_t) dim * 4, cudaMemcpyHostToDevice);
    int gblk = (dim + 255) / 256, iblk = (intermed + 255) / 256;

    // Selected experts — gate/up (3D slice e*bpe) -> SwiGLU -> down -> combine.
    cudaMemset(d_acc, 0, (size_t) dim * 4);
    for (int k = 0; k < top_k; ++k) {
        int e = top_idx[k];
        launch_matvec(gdw + (size_t) e * g_bpe, gate_kind, dim, intermed, dx, d_gate);
        launch_matvec(udw + (size_t) e * u_bpe, up_kind,   dim, intermed, dx, d_up);
        k_swiglu<<<iblk, 256>>>(d_gate, d_up, d_h, intermed);
        launch_matvec(ddw + (size_t) e * d_bpe, down_kind, intermed, dim, d_h, d_y);
        k_axpy<<<gblk, 256>>>(d_acc, top_w[k], d_y, dim);
    }

    // Shared expert (+ optional sigmoid gate; gate dot on host over dim).
    launch_matvec(sgdw, shg_kind, dim, intermed, dx, d_gate);
    launch_matvec(sudw, shu_kind, dim, intermed, dx, d_up);
    k_swiglu<<<iblk, 256>>>(d_gate, d_up, d_h, intermed);
    launch_matvec(sddw, shd_kind, intermed, dim, d_h, d_y);
    float shared_gate = 1.0f;
    if (gate_inp_len > 1 && shared_gate_inp) {
        float gs = 0.f;
        for (int d = 0; d < dim; ++d) gs += shared_gate_inp[d] * x[d];
        shared_gate = 1.0f / (1.0f + expf(-gs));
    }
    k_axpy<<<gblk, 256>>>(d_acc, shared_gate, d_y, dim);

    cudaMemcpy(y, d_acc, (size_t) dim * 4, cudaMemcpyDeviceToHost);
}
