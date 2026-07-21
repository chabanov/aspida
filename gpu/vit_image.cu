// Image -> visual tokens: stb_image decode + Qwen3VL smart-resize + patchify + ViT.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
extern "C" void aspida_vit(const float* pv,int gh,int gw,float* out); // from vit_engine.cu
// bilinear resize RGB [H,W,3] -> [oh,ow,3]
static void resize_rgb(const unsigned char*src,int H,int W,std::vector<float>&dst,int oh,int ow){
  dst.resize((size_t)oh*ow*3);
  for(int y=0;y<oh;++y)for(int x=0;x<ow;++x){
    float fy=(y+0.5f)*H/oh-0.5f, fx=(x+0.5f)*W/ow-0.5f; int y0=(int)floorf(fy),x0=(int)floorf(fx);
    float dy=fy-y0,dx=fx-x0; int y1=y0+1<H?y0+1:H-1,x1=x0+1<W?x0+1:W-1; if(y0<0)y0=0; if(x0<0)x0=0;
    for(int c=0;c<3;++c){ float a=src[((size_t)y0*W+x0)*3+c],b=src[((size_t)y0*W+x1)*3+c],cc=src[((size_t)y1*W+x0)*3+c],d=src[((size_t)y1*W+x1)*3+c];
      dst[((size_t)y*ow+x)*3+c]=(a*(1-dx)+b*dx)*(1-dy)+(cc*(1-dx)+d*dx)*dy; }
  }
}
// smart_resize: multiples of factor=32, within [min_px,max_px]
static void smart_resize(int H,int W,int&oh,int&ow){
  const int F=32; const double MINP=256*256, MAXP=4096*4096;
  auto rnd=[&](int v){int r=(int)std::nearbyint((double)v/F)*F; return r<F?F:r;};
  oh=rnd(H); ow=rnd(W); double px=(double)oh*ow;
  if(px>MAXP){double s=std::sqrt(MAXP/((double)H*W)); oh=(int)std::floor(H*s/F)*F; ow=(int)std::floor(W*s/F)*F;}
  else if(px<MINP){double s=std::sqrt(MINP/((double)H*W)); oh=(int)std::ceil(H*s/F)*F; ow=(int)std::ceil(W*s/F)*F;}
  if(oh<F)oh=F; if(ow<F)ow=F;
}
// Decode image bytes -> visual tokens. Returns ntok; writes tokens[ntok*2048], grid.
extern "C" int aspida_vit_from_image(const unsigned char*data,int n,float*out_tokens,int*out_gh,int*out_gw){
  int W,H,ch; unsigned char*img=stbi_load_from_memory(data,n,&W,&H,&ch,3);
  if(!img)return -1;
  int oh,ow; smart_resize(H,W,oh,ow); std::vector<float> rs; resize_rgb(img,H,W,rs,oh,ow); stbi_image_free(img);
  int gh=oh/16, gw=ow/16, S=gh*gw, F=3*2*16*16;
  std::vector<float> pv((size_t)S*F);
  int ghb=gh/2,gwb=gw/2;
  for(int hb=0;hb<ghb;++hb)for(int wb=0;wb<gwb;++wb)for(int mh=0;mh<2;++mh)for(int mw=0;mw<2;++mw){
    int patch=hb*(gwb*4)+wb*4+mh*2+mw;
    for(int c=0;c<3;++c)for(int tp=0;tp<2;++tp)for(int ph=0;ph<16;++ph)for(int pw=0;pw<16;++pw){
      int feat=c*(2*256)+tp*256+ph*16+pw; int Y=(hb*2+mh)*16+ph, X=(wb*2+mw)*16+pw;
      pv[(size_t)patch*F+feat]=(rs[((size_t)Y*ow+X)*3+c]/255.f-0.5f)/0.5f;
    }
  }
  aspida_vit(pv.data(),gh,gw,out_tokens);
  *out_gh=gh; *out_gw=gw; return S/4;
}

// base64 (optionally a data: URI) -> visual tokens. Convenience wrapper for Ada.
static int b64val(char c){ if(c>=65&&c<=90)return c-65; if(c>=97&&c<=122)return c-71; if(c>=48&&c<=57)return c+4; if(c==43)return 62; if(c==47)return 63; return -1; }
extern "C" int aspida_vit_from_b64(const char*b64,float*out_tokens,int*out_gh,int*out_gw){
  const char COMMA=44, EQ=61, NL=10, CR=13, SP=32;
  const char*p=b64; const char*comma=strchr(b64,COMMA); if(strncmp(b64,"data:",5)==0 && comma) p=comma+1;
  std::vector<unsigned char> raw; raw.reserve(strlen(p)*3/4); int acc=0,bits=0;
  for(;*p;++p){ if(*p==EQ||*p==NL||*p==CR||*p==SP)continue; int v=b64val(*p); if(v<0)continue;
    acc=(acc<<6)|v; bits+=6; if(bits>=8){bits-=8; raw.push_back((unsigned char)((acc>>bits)&0xFF));} }
  if(raw.empty())return -1;
  return aspida_vit_from_image(raw.data(),(int)raw.size(),out_tokens,out_gh,out_gw);
}
