#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "/root/vit_host.h"
#define CK(x) do{cudaError_t e=(x);if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static const char* D="/root/vit_w/";
static float* load(const std::string&n,size_t cnt){
  std::string p=std::string(D)+n+".bin"; FILE*f=fopen(p.c_str(),"rb");
  if(!f){printf("missing %s\n",p.c_str());exit(1);}
  std::vector<float> h(cnt); fread(h.data(),4,cnt,f); fclose(f);
  float*d; CK(cudaMalloc(&d,cnt*4)); CK(cudaMemcpy(d,h.data(),cnt*4,cudaMemcpyHostToDevice)); return d;
}
static cublasHandle_t CB;
// Y(M,N) = X(M,K) @ W(N,K)^T   (row-major)
static void gemm(const float*X,const float*W,float*Y,int M,int N,int K){
  float a=1,b=0; cublasSgemm(CB,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&a,W,K,X,K,&b,Y,N);
}
__global__ void kbias(float*Y,const float*B,int M,int N){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<(size_t)M*N)Y[i]+=B[i%N];}
__global__ void kadd(float*x,const float*y,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=y[i];}
__global__ void kln(float*o,const float*x,const float*w,const float*b,int M,int H){
  int m=blockIdx.x; if(m>=M)return; extern __shared__ float sh[]; int t=threadIdx.x,nt=blockDim.x;
  float s=0; for(int i=t;i<H;i+=nt)s+=x[(size_t)m*H+i]; sh[t]=s; __syncthreads();
  for(int o=nt/2;o;o>>=1){if(t<o)sh[t]+=sh[t+o];__syncthreads();} float mean=sh[0]/H; __syncthreads();
  float v=0; for(int i=t;i<H;i+=nt){float d=x[(size_t)m*H+i]-mean;v+=d*d;} sh[t]=v;__syncthreads();
  for(int o=nt/2;o;o>>=1){if(t<o)sh[t]+=sh[t+o];__syncthreads();} float inv=rsqrtf(sh[0]/H+1e-6f);
  for(int i=t;i<H;i+=nt)o[(size_t)m*H+i]=(x[(size_t)m*H+i]-mean)*inv*w[i]+b[i];
}
__global__ void kgelu(float*x,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];x[i]=0.5f*v*(1.f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));}}
// qkv (S,3,NH,HD); apply RoPE to q(0),k(1). cos/sin (S,HD). rot_half: [-x2,x1] halves of HD.
__global__ void krope(float*qkv,const float*cos,const float*sin,int S,int NH,int HD){
  int s=blockIdx.x, h=blockIdx.y, d=threadIdx.x; if(s>=S||h>=NH||d>=HD)return;
  int half=HD/2; const float*c=cos+(size_t)s*HD; const float*sn=sin+(size_t)s*HD;
  for(int part=0;part<2;++part){ size_t base=(((size_t)s*3+part)*NH+h)*HD;
    float x=qkv[base+d]; float rot = d<half ? -qkv[base+d+half] : qkv[base+d-half];
    __syncthreads(); qkv[base+d]=x*c[d]+rot*sn[d]; }
}
// attention: out(S,NH*HD). block=(query s, head h). full softmax over S keys.
__global__ void kattn(const float*qkv,float*out,int S,int NH,int HD){
  int s=blockIdx.x, h=blockIdx.y; int t=threadIdx.x,nt=blockDim.x; if(s>=S||h>=NH)return;
  extern __shared__ float sh[]; float*sc=sh; // S scores
  const float scale=rsqrtf((float)HD);
  size_t qb=(((size_t)s*3+0)*NH+h)*HD;
  for(int j=t;j<S;j+=nt){ size_t kb=(((size_t)j*3+1)*NH+h)*HD; float dot=0;
    for(int d=0;d<HD;++d)dot+=qkv[qb+d]*qkv[kb+d]; sc[j]=dot*scale; }
  __syncthreads();
  __shared__ float red; if(t==0){float mx=-1e30f;for(int j=0;j<S;++j)mx=fmaxf(mx,sc[j]);red=mx;} __syncthreads();
  float mx=red; float ls=0; for(int j=t;j<S;j+=nt){float e=expf(sc[j]-mx);sc[j]=e;ls+=e;}
  __shared__ float sred[256]; sred[t]=ls;__syncthreads();
  for(int o=nt/2;o;o>>=1){if(t<o)sred[t]+=sred[t+o];__syncthreads();} float inv=1.f/sred[0];
  for(int d=t;d<HD;d+=nt){ float acc=0; for(int j=0;j<S;++j){size_t vb=(((size_t)j*3+2)*NH+h)*HD;acc+=sc[j]*qkv[vb+d];}
    out[(size_t)s*NH*HD + h*HD + d]=acc*inv; }
}
int main(){
  cublasCreate(&CB);
  const int S=784,H=1152,NH=16,HD=72,IM=4304,MG=4608,OUT=2048,NB=27,TOK=196;
  float*x=load("pixel_values",(size_t)S*1536);
  float*pW=load("patch_W",(size_t)H*1536),*pB=load("patch_B",H);
  float*hid; CK(cudaMalloc(&hid,(size_t)S*H*4));
  gemm(x,pW,hid,S,H,1536); kbias<<<((size_t)S*H+255)/256,256>>>(hid,pB,S,H);
  std::vector<float> h_pe((size_t)2304*H); {FILE*f=fopen("/root/vit_w/pos_embed_W.bin","rb"); if(fread(h_pe.data(),4,h_pe.size(),f)){} fclose(f);}
  std::vector<float> h_pos,h_cos,h_sin; host_pos_add(h_pe.data(),48,H,1,28,28,2,h_pos); host_rope(28,28,2,HD,h_cos,h_sin);
  float*pos,*cosb,*sinb;
  CK(cudaMalloc(&pos,h_pos.size()*4));CK(cudaMemcpy(pos,h_pos.data(),h_pos.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&cosb,h_cos.size()*4));CK(cudaMemcpy(cosb,h_cos.data(),h_cos.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMalloc(&sinb,h_sin.size()*4));CK(cudaMemcpy(sinb,h_sin.data(),h_sin.size()*4,cudaMemcpyHostToDevice));
  kadd<<<((size_t)S*H+255)/256,256>>>(hid,pos,(size_t)S*H);
  float *a,*qkv,*att,*proj,*mh; CK(cudaMalloc(&a,(size_t)S*H*4));CK(cudaMalloc(&qkv,(size_t)S*3*H*4));
  CK(cudaMalloc(&att,(size_t)S*H*4));CK(cudaMalloc(&proj,(size_t)S*H*4));CK(cudaMalloc(&mh,(size_t)S*IM*4));
  for(int i=0;i<NB;++i){ char p[8]; sprintf(p,"b%d_",i); std::string P(p);
    float*n1w=load(P+"n1w",H),*n1b=load(P+"n1b",H),*qkvw=load(P+"qkvw",(size_t)3*H*H),*qkvb=load(P+"qkvb",3*H);
    float*prw=load(P+"prw",(size_t)H*H),*prb=load(P+"prb",H),*n2w=load(P+"n2w",H),*n2b=load(P+"n2b",H);
    float*f1w=load(P+"f1w",(size_t)IM*H),*f1b=load(P+"f1b",IM),*f2w=load(P+"f2w",(size_t)H*IM),*f2b=load(P+"f2b",H);
    kln<<<S,256,256*4>>>(a,hid,n1w,n1b,S,H);
    gemm(a,qkvw,qkv,S,3*H,H); kbias<<<((size_t)S*3*H+255)/256,256>>>(qkv,qkvb,S,3*H);
    krope<<<dim3(S,NH),HD>>>(qkv,cosb,sinb,S,NH,HD);
    kattn<<<dim3(S,NH),256,(size_t)S*4>>>(qkv,att,S,NH,HD);
    gemm(att,prw,proj,S,H,H); kbias<<<((size_t)S*H+255)/256,256>>>(proj,prb,S,H);
    kadd<<<((size_t)S*H+255)/256,256>>>(hid,proj,(size_t)S*H);
    kln<<<S,256,256*4>>>(a,hid,n2w,n2b,S,H);
    gemm(a,f1w,mh,S,IM,H); kbias<<<((size_t)S*IM+255)/256,256>>>(mh,f1b,S,IM); kgelu<<<((size_t)S*IM+255)/256,256>>>(mh,(size_t)S*IM);
    gemm(mh,f2w,proj,S,H,IM); kbias<<<((size_t)S*H+255)/256,256>>>(proj,f2b,S,H);
    kadd<<<((size_t)S*H+255)/256,256>>>(hid,proj,(size_t)S*H);
    cudaFree(n1w);cudaFree(n1b);cudaFree(qkvw);cudaFree(qkvb);cudaFree(prw);cudaFree(prb);
    cudaFree(n2w);cudaFree(n2b);cudaFree(f1w);cudaFree(f1b);cudaFree(f2w);cudaFree(f2b);
  }
  // merger: LN(1152) -> reshape(196,4608) -> fc1 -> gelu -> fc2
  float*mnw=load("mg_nw",H),*mnb=load("mg_nb",H),*mf1w=load("mg_f1w",(size_t)MG*MG),*mf1b=load("mg_f1b",MG),*mf2w=load("mg_f2w",(size_t)OUT*MG),*mf2b=load("mg_f2b",OUT);
  kln<<<S,256,256*4>>>(a,hid,mnw,mnb,S,H); // a is (784,1152) = (196,4608) reshaped
  float*mgh; CK(cudaMalloc(&mgh,(size_t)TOK*MG*4)); float*mout; CK(cudaMalloc(&mout,(size_t)TOK*OUT*4));
  gemm(a,mf1w,mgh,TOK,MG,MG); kbias<<<((size_t)TOK*MG+255)/256,256>>>(mgh,mf1b,TOK,MG); kgelu<<<((size_t)TOK*MG+255)/256,256>>>(mgh,(size_t)TOK*MG);
  gemm(mgh,mf2w,mout,TOK,OUT,MG); kbias<<<((size_t)TOK*OUT+255)/256,256>>>(mout,mf2b,TOK,OUT);
  CK(cudaDeviceSynchronize());
  std::vector<float> H_out((size_t)TOK*OUT); CK(cudaMemcpy(H_out.data(),mout,H_out.size()*4,cudaMemcpyDeviceToHost));
  float*ref=load("oracle_merged",(size_t)TOK*OUT); std::vector<float> R((size_t)TOK*OUT); cudaMemcpy(R.data(),ref,R.size()*4,cudaMemcpyDeviceToHost);
  double se=0,sr=0; for(size_t i=0;i<H_out.size();++i){double e=H_out[i]-R[i];se+=e*e;sr+=(double)R[i]*R[i];}
  printf("SELF-CONTAINED CUDA ViT (C host pos/rope) vs oracle: NMSE=%.3e  shape=(%d,%d)\n",se/sr,TOK,OUT);
  printf("  out[0,:4]=%.4f %.4f %.4f %.4f  ref=%.4f %.4f %.4f %.4f\n",H_out[0],H_out[1],H_out[2],H_out[3],R[0],R[1],R[2],R[3]);
  return 0;
}
