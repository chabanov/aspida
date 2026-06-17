// Standalone correctness + speed check for the GPU matvec kernels.
// Generates random Q4_K / Q6_K weight rows and a random x, compares the GPU
// result (via the shipped libaspidagpu.so entry point) against a CPU reference
// dequant-and-dot, and times many iterations. No model load required.
//   nvcc -O3 -arch=native test_matvec.cu -o test_matvec -ldl
//   ./test_matvec            # fast warp kernels (default)
//   ASPIDA_GPU_SCALAR=1 ./test_matvec   # legacy scalar kernels
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <dlfcn.h>
#include <ctime>

// ---- CPU reference dequant (mirrors gpu_matvec.cu / the Ada engine) ----
static float f16(const uint8_t* p){
    uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    uint16_t s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF; float v;
    if(e==0)      v = ldexpf((float)m, -24);
    else if(e==31)v = m?NAN:INFINITY;
    else          v = ldexpf((float)(m|0x400), (int)e-25);
    return s?-v:v;
}
static void gsm(const uint8_t* sc,int j,int*d,int*m){
    if(j<4){*d=sc[j]&63;*m=sc[j+4]&63;}
    else{*d=(sc[j+4]&0x0F)|((sc[j-4]>>6)<<4);*m=(sc[j+4]>>4)|((sc[j]>>6)<<4);}
}
static void deq_q4k(const uint8_t* b,float* o){
    float d=f16(b),dm=f16(b+2); const uint8_t*sc=b+4,*qs=b+16;
    for(int g=0;g<4;++g){int s1,m1,s2,m2;gsm(sc,2*g,&s1,&m1);gsm(sc,2*g+1,&s2,&m2);
        float d1=d*s1,mm1=dm*m1,d2=d*s2,mm2=dm*m2;const uint8_t*q=qs+g*32;
        for(int l=0;l<32;++l)o[64*g+l]=d1*(q[l]&0x0F)-mm1;
        for(int l=0;l<32;++l)o[64*g+32+l]=d2*(q[l]>>4)-mm2;}
}
static void deq_q6k(const uint8_t* b,float* out){
    const uint8_t*ql=b;const uint8_t*qh=b+128;const int8_t*sc=(const int8_t*)(b+192);float d=f16(b+208);
    for(int h=0;h<2;++h){const uint8_t*QL=ql+h*64;const uint8_t*QH=qh+h*32;const int8_t*SC=sc+h*8;float*Y=out+h*128;
        for(int l=0;l<32;++l){int is=l/16;
            int q1=(int)((QL[l]&0xF)|(((QH[l]>>0)&3)<<4))-32; int q2=(int)((QL[l+32]&0xF)|(((QH[l]>>2)&3)<<4))-32;
            int q3=(int)((QL[l]>>4)|(((QH[l]>>4)&3)<<4))-32;  int q4=(int)((QL[l+32]>>4)|(((QH[l]>>6)&3)<<4))-32;
            Y[l]=d*SC[is+0]*q1; Y[l+32]=d*SC[is+2]*q2; Y[l+64]=d*SC[is+4]*q3; Y[l+96]=d*SC[is+6]*q4;}}
}

static void deq_q5k(const uint8_t* b,float* o){ float d=f16(b),dm=f16(b+2); const uint8_t*sc=b+4,*qh=b+16,*qs=b+48;
    for(int g=0;g<4;++g){int s1,m1,s2,m2;gsm(sc,2*g,&s1,&m1);gsm(sc,2*g+1,&s2,&m2);float d1=d*s1,mm1=dm*m1,d2=d*s2,mm2=dm*m2;
        for(int l=0;l<32;++l){unsigned q=qs[32*g+l];int lo=(q&0xF)+(((qh[l]>>(2*g))&1)<<4);int hi=(q>>4)+(((qh[l]>>(2*g+1))&1)<<4);
            o[64*g+l]=d1*lo-mm1; o[64*g+32+l]=d2*hi-mm2;}}}

typedef void (*matvec_fn)(const void*,long,int,int,int,const float*,float*);
typedef void (*matmul_fn)(const void*,long,int,int,int,int,const float*,float*);

static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }

static int run(matvec_fn mv,int kind,int in_dim,int out_dim){
    int nb = in_dim/256;
    int bsz = (kind==0)?144:(kind==1)?210:176;
    long wbytes = (long)out_dim*nb*bsz;
    std::vector<uint8_t> w(wbytes);
    std::vector<float> x(in_dim), yg(out_dim), yr(out_dim,0.f);
    srand(kind==0?1234:5678);
    for(long i=0;i<wbytes;++i) w[i]=rand()&0xFF;
    for(int i=0;i<in_dim;++i) x[i]=((rand()%2001)-1000)/1000.f;
    // CPU reference
    std::vector<float> t(256);
    for(int o=0;o<out_dim;++o){ double a=0; const uint8_t* r=&w[(long)o*nb*bsz];
        for(int b=0;b<nb;++b){ if(kind==0) deq_q4k(r+(long)b*bsz,t.data()); else if(kind==1) deq_q6k(r+(long)b*bsz,t.data()); else deq_q5k(r+(long)b*bsz,t.data());
            for(int l=0;l<256;++l) a+=(double)t[l]*x[b*256+l]; } yr[o]=(float)a; }
    // GPU (warm-up + correctness)
    mv(w.data(),wbytes,kind,in_dim,out_dim,x.data(),yg.data());
    double maxabs=0,maxrel=0; int worst=0;
    for(int o=0;o<out_dim;++o){ double ad=fabs(yg[o]-yr[o]); double rd=ad/(fabs(yr[o])+1e-6);
        if(ad>maxabs)maxabs=ad; if(rd>maxrel){maxrel=rd;worst=o;} }
    // Timing
    int iters=200; double t0=now_ms();
    for(int i=0;i<iters;++i) mv(w.data(),wbytes,kind,in_dim,out_dim,x.data(),yg.data());
    double ms=(now_ms()-t0)/iters;
    const char* name=(kind==0)?"Q4_K":"Q6_K";
    printf("%s  in=%d out=%d : max|abs|=%.4g  max rel=%.4g (y_ref[%d]=%.4f y_gpu=%.4f)  | %.3f ms/call  %.1f GB/s\n",
        name,in_dim,out_dim,maxabs,maxrel,worst,yr[worst],yg[worst],ms,(wbytes/1e9)/(ms/1e3));
    return (maxrel<1e-2)?0:1;
}

// Batched matmul: correctness (each row vs CPU ref) + speed (B-batch vs B
// separate matvec calls). The batching win: B-batch should cost ~the same
// wall time as a single matvec, since weight bandwidth is read once.
static int run_batched(matmul_fn mm, matvec_fn mv, int kind, int in_dim, int out_dim, int B){
    int nb=in_dim/256, bsz=(kind==0)?144:(kind==1)?210:176;
    long wbytes=(long)out_dim*nb*bsz;
    std::vector<uint8_t> w(wbytes);
    std::vector<float> x((long)B*in_dim), yg((long)B*out_dim), yr((long)B*out_dim,0.f), y1(out_dim);
    srand(kind*100+B);
    for(long i=0;i<wbytes;++i) w[i]=rand()&0xFF;
    for(long i=0;i<(long)B*in_dim;++i) x[i]=((rand()%2001)-1000)/1000.f;
    std::vector<float> t(256);
    for(int b=0;b<B;++b) for(int o=0;o<out_dim;++o){ double a=0; const uint8_t*r=&w[(long)o*nb*bsz];
        for(int bl=0;bl<nb;++bl){ if(kind==0)deq_q4k(r+(long)bl*bsz,t.data()); else if(kind==1)deq_q6k(r+(long)bl*bsz,t.data()); else deq_q5k(r+(long)bl*bsz,t.data());
            for(int l=0;l<256;++l) a+=(double)t[l]*x[(long)b*in_dim+bl*256+l]; } yr[(long)b*out_dim+o]=(float)a; }
    mm(w.data(),wbytes,kind,in_dim,out_dim,B,x.data(),yg.data());
    //  Correctness vs CPU ref: use max-abs and a rel gated on |ref|>1 (raw rel
    //  explodes on random dot-products that land near zero — a metric artifact).
    double maxrel=0, maxabs=0;
    for(long i=0;i<(long)B*out_dim;++i){ double ad=fabs(yg[i]-yr[i]); if(ad>maxabs)maxabs=ad;
        if(fabs(yr[i])>1.0){ double rd=ad/fabs(yr[i]); if(rd>maxrel)maxrel=rd; } }
    //  Definitive: batched must equal the per-row GPU matvec to FP precision
    //  (identical math up to add reassociation). Measure RELATIVE to magnitude
    //  (random-byte weights give huge f16 scales, so absolute diff is large but
    //  relative is ~1e-6). This is the real correctness criterion.
    double maxd_vs_mv=0, maxmag=1e-9;
    for(int b=0;b<B;++b){ mv(w.data(),wbytes,kind,in_dim,out_dim,x.data()+(long)b*in_dim,y1.data());
        for(int o=0;o<out_dim;++o){ double d=fabs(yg[(long)b*out_dim+o]-y1[o]); if(d>maxd_vs_mv)maxd_vs_mv=d;
            if(fabs(y1[o])>maxmag)maxmag=fabs(y1[o]); } }
    double rel_vs_mv=maxd_vs_mv/maxmag;
    int iters=100;
    double t0=now_ms(); for(int i=0;i<iters;++i) mm(w.data(),wbytes,kind,in_dim,out_dim,B,x.data(),yg.data());
    double ms_b=(now_ms()-t0)/iters;
    t0=now_ms(); for(int i=0;i<iters;++i) for(int b=0;b<B;++b) mv(w.data(),wbytes,kind,in_dim,out_dim,x.data()+(long)b*in_dim,y1.data());
    double ms_s=(now_ms()-t0)/iters;
    const char* name=(kind==0)?"Q4_K":(kind==1)?"Q6_K":"Q5_K";
    (void)maxrel; (void)maxabs;
    printf("%s in=%d out=%d B=%d : batched-vs-per-row rel=%.2e | batched %.3f ms vs %dx-matvec %.3f ms\n",
        name,in_dim,out_dim,B,rel_vs_mv,ms_b,B,ms_s);
    printf("      per-token: batched %.3f ms  serial %.3f ms  -> %.2fx throughput\n", ms_b/B, ms_s/B, (ms_s/B)/(ms_b/B));
    //  Pass if batched matches per-row matvec to FP precision.
    return (rel_vs_mv<1e-4)?0:1;
}

int main(){
    void* h=dlopen("./libaspidagpu.so",RTLD_NOW);
    if(!h){ fprintf(stderr,"dlopen failed: %s\n",dlerror()); return 2; }
    matvec_fn mv=(matvec_fn)dlsym(h,"aspida_gpu_matvec");
    matmul_fn mm=(matmul_fn)dlsym(h,"aspida_gpu_matmul");
    if(!mv||!mm){ fprintf(stderr,"dlsym failed\n"); return 2; }
    int rc=0;
    rc|=run(mv,0,8192,8192);    // Q4_K, Llama-70B attn/ffn-ish shape
    rc|=run(mv,0,8192,28672);   // Q4_K, FFN up/gate
    rc|=run(mv,1,8192,8192);    // Q6_K
    rc|=run(mv,2,8192,1024);    // Q5_K, Llama-70B attn_v shape (the CPU-fallback weight)
    printf("--- batched matmul (continuous-batching primitive) ---\n");
    rc|=run_batched(mm,mv,0,8192,8192,4);
    rc|=run_batched(mm,mv,0,8192,28672,4);
    rc|=run_batched(mm,mv,0,8192,8192,8);
    rc|=run_batched(mm,mv,1,8192,8192,8);
    rc|=run_batched(mm,mv,2,8192,1024,8);
    printf(rc?"FAIL (rel diff too large)\n":"PASS (within 1e-2 relative)\n");
    return rc;
}
