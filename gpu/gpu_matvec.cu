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

__global__ void k_q4k(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*144;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_q4k(r+(size_t)b*144,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}
__global__ void k_q6k(const uint8_t*w,const float*x,float*y,int in,int out){int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=out)return;int nb=in/256;size_t bpr=(size_t)nb*210;const uint8_t*r=w+(size_t)o*bpr;float t[256],a=0;for(int b=0;b<nb;++b){deq_q6k(r+(size_t)b*210,t);int bs=b*256;for(int l=0;l<256;++l)a+=t[l]*x[bs+l];}y[o]=a;}

extern "C" void aspida_gpu_matvec(const void *w, long wbytes, int kind,
                                  int in_dim, int out_dim, const float *x, float *y) {
    static std::unordered_map<const void *, uint8_t *> cache;
    static float *dx = nullptr, *dy = nullptr; static long cx = 0, cy = 0;

    uint8_t *dw;
    auto it = cache.find(w);
    if (it == cache.end()) {                 // upload this weight to VRAM once
        cudaMalloc(&dw, wbytes);
        cudaMemcpy(dw, w, wbytes, cudaMemcpyHostToDevice);
        cache[w] = dw;
    } else {
        dw = it->second;
    }
    if (in_dim > cx)  { if (dx) cudaFree(dx); cudaMalloc(&dx, (size_t) in_dim * 4);  cx = in_dim; }
    if (out_dim > cy) { if (dy) cudaFree(dy); cudaMalloc(&dy, (size_t) out_dim * 4); cy = out_dim; }
    cudaMemcpy(dx, x, (size_t) in_dim * 4, cudaMemcpyHostToDevice);
    if (kind == 0) k_q4k<<<(out_dim + 127) / 128, 128>>>(dw, dx, dy, in_dim, out_dim);
    else           k_q6k<<<(out_dim + 127) / 128, 128>>>(dw, dx, dy, in_dim, out_dim);
    cudaDeviceSynchronize();
    cudaMemcpy(y, dy, (size_t) out_dim * 4, cudaMemcpyDeviceToHost);
}
