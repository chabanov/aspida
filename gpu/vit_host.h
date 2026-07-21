// Host-side ports of the Qwen3-VL vision helpers (bilinear pos-embed, block-major
// position ids, 2D vision-RoPE). Deterministic from grid_thw; validated to match
// the transformers oracle. C translation of full_vit_reference.py host math.
#include <vector>
#include <cmath>
// pos_add[S,H] = bilinear interpolation of pos_embed_W[side*side, H] to the h x w grid,
// reordered block-major over merge x merge spatial blocks.
static void host_pos_add(const float* pe, int side, int H, int t, int h, int w, int ms, std::vector<float>& pos){
  int S=t*h*w; pos.assign((size_t)S*H,0.f);
  std::vector<float> hg(h),wg(w); for(int i=0;i<h;++i)hg[i]=(side-1)*(h>1?(float)i/(h-1):0);
  for(int i=0;i<w;++i)wg[i]=(side-1)*(w>1?(float)i/(w-1):0);
  // reorder index (block-major)
  std::vector<int> reorder(h*w); int q=0;
  for(int hb=0;hb<h/ms;++hb)for(int wb=0;wb<w/ms;++wb)for(int hi=0;hi<ms;++hi)for(int wi=0;wi<ms;++wi)
    reorder[q++]=(hb*ms+hi)*w+(wb*ms+wi);
  for(int r=0;r<h*w;++r){ int src=reorder[r]; int si=src/w, sj=src%w;
    float hgg=hg[si],wgg=wg[sj]; int hf=(int)hgg,wf=(int)wgg; int hc=hf+1<side?hf+1:side-1,wc=wf+1<side?wf+1:side-1;
    float hfr=hgg-hf,wfr=wgg-wf;
    int c00=hf*side+wf,c01=hf*side+wc,c10=hc*side+wf,c11=hc*side+wc;
    float w00=(1-hfr)*(1-wfr),w01=(1-hfr)*wfr,w10=hfr*(1-wfr),w11=hfr*wfr;
    for(int d=0;d<H;++d) pos[(size_t)r*H+d]=w00*pe[(size_t)c00*H+d]+w01*pe[(size_t)c01*H+d]+w10*pe[(size_t)c10*H+d]+w11*pe[(size_t)c11*H+d];
  }
  // repeat for t (t=1 here)
}
// cos/sin[S,HD] from block-major position ids (h,w) + inv_freq (dim=HD/2, step 2 -> HD/4 freqs, doubled)
static void host_rope(int h,int w,int ms,int HD,std::vector<float>&cosb,std::vector<float>&sinb){
  int S=h*w; cosb.assign((size_t)S*HD,0.f); sinb.assign((size_t)S*HD,0.f);
  int nf=HD/4; std::vector<float> inv(nf); for(int i=0;i<nf;++i)inv[i]=1.f/powf(10000.f,(float)(2*i)/(HD/2));
  int q=0;
  for(int hb=0;hb<h/ms;++hb)for(int wb=0;wb<w/ms;++wb)for(int hi=0;hi<ms;++hi)for(int wi=0;wi<ms;++wi){
    int hp=hb*ms+hi, wp=wb*ms+wi;
    // rope = [hp*inv(nf), wp*inv(nf)] -> (HD/2), then emb=cat(rope,rope) -> HD
    float half[128]; for(int i=0;i<nf;++i)half[i]=hp*inv[i]; for(int i=0;i<nf;++i)half[nf+i]=wp*inv[i];
    for(int d=0;d<HD/2;++d){ cosb[(size_t)q*HD+d]=cosf(half[d]); cosb[(size_t)q*HD+HD/2+d]=cosf(half[d]);
                             sinb[(size_t)q*HD+d]=sinf(half[d]); sinb[(size_t)q*HD+HD/2+d]=sinf(half[d]); }
    ++q;
  }
}
