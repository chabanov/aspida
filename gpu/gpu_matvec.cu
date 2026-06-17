// Stream B — GPU matvec shim for the Ada engine (LLM_GPU dlopen's this).
// Exposes one C entry point; weights are uploaded to VRAM once (cached by host
// pointer) and stay resident across tokens. Q4_K + Q6_K, bit-exact vs the CPU
// engine (build with --fmad=false). Build:
//   nvcc -O3 --fmad=false -arch=native -shared -Xcompiler -fPIC gpu_matvec.cu -o libaspidagpu.so
#include <cuda_fp16.h>
#include <unordered_map>
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

__global__ void k_q4k(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*144;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_q4k(r+(size_t)b*144,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}
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
        else                k_q5k<<<g, 128>>>(dw, dx, dy, in_dim, out_dim);
    } else {                                 // fast warp-per-row path (default)
        const int TPB = 256, WPB = TPB / 32; // 8 warps (=rows) per block
        int blocks = (out_dim + WPB - 1) / WPB;
        if (kind == 0)      k_q4k_w<<<blocks, TPB>>>(dw, dx, dy, in_dim, out_dim);
        else if (kind == 1) k_q6k_w<<<blocks, TPB>>>(dw, dx, dy, in_dim, out_dim);
        else                k_q5k_w<<<blocks, TPB>>>(dw, dx, dy, in_dim, out_dim);
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
    else                k_q5k_wb<<<blocks, TPB>>>(dw, dxb, dyb, in_dim, out_dim, batch);
    cudaMemcpy(y, dyb, (size_t) ny * 4, cudaMemcpyDeviceToHost);
}
