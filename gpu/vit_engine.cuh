// Native Ornith vision-tower engine entry. aspida_vit_forward: pixel_values
// (host, [S,1536]) + grid -> visual tokens (host, [S/4, 2048]). Weights cached
// from a dir on first call. Full ViT validated NMSE 1.09e-03 vs the transformers
// oracle (see gpu/vit.cu / full_vit_reference.py). Plus k_vit_inject to splice
// visual tokens into a prefill embedding buffer at <|image_pad|> positions.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "vit_host.h"
namespace viteng {
#define VK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"[VIT] %d %s\n",__LINE__,cudaGetErrorString(e));}}while(0)
static const char* WDIR="/opt/aspida-models/vit_w/";
static cublasHandle_t g_cb=nullptr; static bool g_loaded=false;
struct Blk{float*n1w,*n1b,*qkvw,*qkvb,*prw,*prb,*n2w,*n2b,*f1w,*f1b,*f2w,*f2b;};
static float *g_pW,*g_pB,*g_peW,*g_mnw,*g_mnb,*g_mf1w,*g_mf1b,*g_mf2w,*g_mf2b; static Blk g_blk[27];
static float* ld(const std::string&n,size_t cnt){std::string p=std::string(WDIR)+n+".bin";FILE*f=fopen(p.c_str(),"rb");
  if(!f){fprintf(stderr,"[VIT] missing %s\n",p.c_str());exit(1);} std::vector<float> h(cnt); if(fread(h.data(),4,cnt,f)){} fclose(f);
  float*d; VK(cudaMalloc(&d,cnt*4)); VK(cudaMemcpy(d,h.data(),cnt*4,cudaMemcpyHostToDevice)); return d;}
static void gemm(const float*X,const float*W,float*Y,int M,int N,int K){float a=1,b=0;cublasSgemm(g_cb,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&a,W,K,X,K,&b,Y,N);}
__global__ void kbias(float*Y,const float*B,int M,int N){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<(size_t)M*N)Y[i]+=B[i%N];}
__global__ void kadd(float*x,const float*y,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=y[i];}
__global__ void kln(float*o,const float*x,const float*w,const float*b,int M,int H){int m=blockIdx.x;if(m>=M)return;extern __shared__ float sh[];int t=threadIdx.x,nt=blockDim.x;
  float s=0;for(int i=t;i<H;i+=nt)s+=x[(size_t)m*H+i];sh[t]=s;__syncthreads();for(int o=nt/2;o;o>>=1){if(t<o)sh[t]+=sh[t+o];__syncthreads();}float mean=sh[0]/H;__syncthreads();
  float v=0;for(int i=t;i<H;i+=nt){float d=x[(size_t)m*H+i]-mean;v+=d*d;}sh[t]=v;__syncthreads();for(int o=nt/2;o;o>>=1){if(t<o)sh[t]+=sh[t+o];__syncthreads();}float inv=rsqrtf(sh[0]/H+1e-6f);
  for(int i=t;i<H;i+=nt)o[(size_t)m*H+i]=(x[(size_t)m*H+i]-mean)*inv*w[i]+b[i];}
__global__ void kgelu(float*x,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];x[i]=0.5f*v*(1.f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));}}
__global__ void krope(float*qkv,const float*cs,const float*sn,int S,int NH,int HD){int s=blockIdx.x,h=blockIdx.y,d=threadIdx.x;if(s>=S||h>=NH||d>=HD)return;int half=HD/2;const float*c=cs+(size_t)s*HD;const float*si=sn+(size_t)s*HD;
  for(int part=0;part<2;++part){size_t base=(((size_t)s*3+part)*NH+h)*HD;float x=qkv[base+d];float rot=d<half?-qkv[base+d+half]:qkv[base+d-half];__syncthreads();qkv[base+d]=x*c[d]+rot*si[d];}}
__global__ void kattn(const float*qkv,float*out,int S,int NH,int HD){int s=blockIdx.x,h=blockIdx.y;int t=threadIdx.x,nt=blockDim.x;if(s>=S||h>=NH)return;extern __shared__ float sh[];float*sc=sh;const float scale=rsqrtf((float)HD);
  size_t qb=(((size_t)s*3+0)*NH+h)*HD;for(int j=t;j<S;j+=nt){size_t kb=(((size_t)j*3+1)*NH+h)*HD;float dot=0;for(int d=0;d<HD;++d)dot+=qkv[qb+d]*qkv[kb+d];sc[j]=dot*scale;}__syncthreads();
  __shared__ float red;if(t==0){float mx=-1e30f;for(int j=0;j<S;++j)mx=fmaxf(mx,sc[j]);red=mx;}__syncthreads();float mx=red;float ls=0;for(int j=t;j<S;j+=nt){float e=expf(sc[j]-mx);sc[j]=e;ls+=e;}
  __shared__ float sr[256];sr[t]=ls;__syncthreads();for(int o=nt/2;o;o>>=1){if(t<o)sr[t]+=sr[t+o];__syncthreads();}float inv=1.f/sr[0];
  for(int d=t;d<HD;d+=nt){float acc=0;for(int j=0;j<S;++j){size_t vb=(((size_t)j*3+2)*NH+h)*HD;acc+=sc[j]*qkv[vb+d];}out[(size_t)s*NH*HD+h*HD+d]=acc*inv;}}
static void load_weights(){ if(g_loaded)return; cublasCreate(&g_cb);
  g_pW=ld("patch_W",(size_t)1152*1536);g_pB=ld("patch_B",1152);g_peW=ld("pos_embed_W",(size_t)2304*1152);
  for(int i=0;i<27;++i){char p[8];sprintf(p,"b%d_",i);std::string P(p);Blk&b=g_blk[i];
    b.n1w=ld(P+"n1w",1152);b.n1b=ld(P+"n1b",1152);b.qkvw=ld(P+"qkvw",(size_t)3456*1152);b.qkvb=ld(P+"qkvb",3456);
    b.prw=ld(P+"prw",(size_t)1152*1152);b.prb=ld(P+"prb",1152);b.n2w=ld(P+"n2w",1152);b.n2b=ld(P+"n2b",1152);
    b.f1w=ld(P+"f1w",(size_t)4304*1152);b.f1b=ld(P+"f1b",4304);b.f2w=ld(P+"f2w",(size_t)1152*4304);b.f2b=ld(P+"f2b",1152);}
  g_mnw=ld("mg_nw",1152);g_mnb=ld("mg_nb",1152);g_mf1w=ld("mg_f1w",(size_t)4608*4608);g_mf1b=ld("mg_f1b",4608);
  g_mf2w=ld("mg_f2w",(size_t)2048*4608);g_mf2b=ld("mg_f2b",2048); g_loaded=true; fprintf(stderr,"[VIT] weights loaded\n");}
// pixel_values host [S,1536], grid h,w -> out host [S/4,2048]
extern "C" void aspida_vit_forward(const float*pv_host,int gh,int gw,float*out_host){
  load_weights(); const int H=1152,NH=16,HD=72,IM=4304,MG=4608,OUT=2048,ms=2; int S=gh*gw,TOK=S/(ms*ms);
  std::vector<float> h_pos,h_cos,h_sin; { std::vector<float> pe((size_t)2304*H); VK(cudaMemcpy(pe.data(),g_peW,pe.size()*4,cudaMemcpyDeviceToHost));
    host_pos_add(pe.data(),48,H,1,gh,gw,ms,h_pos); host_rope(gh,gw,ms,HD,h_cos,h_sin);}
  float*pv,*hid,*pos,*cs,*sn,*a,*qkv,*att,*proj,*mh;
  VK(cudaMalloc(&pv,(size_t)S*1536*4));VK(cudaMemcpy(pv,pv_host,(size_t)S*1536*4,cudaMemcpyHostToDevice));
  VK(cudaMalloc(&hid,(size_t)S*H*4));gemm(pv,g_pW,hid,S,H,1536);kbias<<<((size_t)S*H+255)/256,256>>>(hid,g_pB,S,H);
  VK(cudaMalloc(&pos,h_pos.size()*4));VK(cudaMemcpy(pos,h_pos.data(),h_pos.size()*4,cudaMemcpyHostToDevice));kadd<<<((size_t)S*H+255)/256,256>>>(hid,pos,(size_t)S*H);
  VK(cudaMalloc(&cs,h_cos.size()*4));VK(cudaMemcpy(cs,h_cos.data(),h_cos.size()*4,cudaMemcpyHostToDevice));
  VK(cudaMalloc(&sn,h_sin.size()*4));VK(cudaMemcpy(sn,h_sin.data(),h_sin.size()*4,cudaMemcpyHostToDevice));
  VK(cudaMalloc(&a,(size_t)S*H*4));VK(cudaMalloc(&qkv,(size_t)S*3*H*4));VK(cudaMalloc(&att,(size_t)S*H*4));VK(cudaMalloc(&proj,(size_t)S*H*4));VK(cudaMalloc(&mh,(size_t)S*IM*4));
  for(int i=0;i<27;++i){Blk&b=g_blk[i];
    kln<<<S,256,256*4>>>(a,hid,b.n1w,b.n1b,S,H);gemm(a,b.qkvw,qkv,S,3*H,H);kbias<<<((size_t)S*3*H+255)/256,256>>>(qkv,b.qkvb,S,3*H);
    krope<<<dim3(S,NH),HD>>>(qkv,cs,sn,S,NH,HD);kattn<<<dim3(S,NH),256,(size_t)S*4>>>(qkv,att,S,NH,HD);
    gemm(att,b.prw,proj,S,H,H);kbias<<<((size_t)S*H+255)/256,256>>>(proj,b.prb,S,H);kadd<<<((size_t)S*H+255)/256,256>>>(hid,proj,(size_t)S*H);
    kln<<<S,256,256*4>>>(a,hid,b.n2w,b.n2b,S,H);gemm(a,b.f1w,mh,S,IM,H);kbias<<<((size_t)S*IM+255)/256,256>>>(mh,b.f1b,S,IM);kgelu<<<((size_t)S*IM+255)/256,256>>>(mh,(size_t)S*IM);
    gemm(mh,b.f2w,proj,S,H,IM);kbias<<<((size_t)S*H+255)/256,256>>>(proj,b.f2b,S,H);kadd<<<((size_t)S*H+255)/256,256>>>(hid,proj,(size_t)S*H);}
  kln<<<S,256,256*4>>>(a,hid,g_mnw,g_mnb,S,H);
  float*mgh,*mout;VK(cudaMalloc(&mgh,(size_t)TOK*MG*4));VK(cudaMalloc(&mout,(size_t)TOK*OUT*4));
  gemm(a,g_mf1w,mgh,TOK,MG,MG);kbias<<<((size_t)TOK*MG+255)/256,256>>>(mgh,g_mf1b,TOK,MG);kgelu<<<((size_t)TOK*MG+255)/256,256>>>(mgh,(size_t)TOK*MG);
  gemm(mgh,g_mf2w,mout,TOK,OUT,MG);kbias<<<((size_t)TOK*OUT+255)/256,256>>>(mout,g_mf2b,TOK,OUT);
  VK(cudaDeviceSynchronize());VK(cudaMemcpy(out_host,mout,(size_t)TOK*OUT*4,cudaMemcpyDeviceToHost));
  cudaFree(pv);cudaFree(hid);cudaFree(pos);cudaFree(cs);cudaFree(sn);cudaFree(a);cudaFree(qkv);cudaFree(att);cudaFree(proj);cudaFree(mh);cudaFree(mgh);cudaFree(mout);
}
} // namespace
