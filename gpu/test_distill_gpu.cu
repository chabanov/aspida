// test_distill_gpu.cu — P3: distillation in the GPU engine. The resident Student
// trains against a teacher distribution Q (soft-target loss) instead of hard
// labels; the soft-CE loss must converge toward the teacher entropy H(Q) (its
// floor, reached when the student's softmax == Q). Proves the GPU engine learns
// from real-teacher distributions, unifying distillation with training.
//
//   nvcc -O3 -arch=native test_distill_gpu.cu -L. -laspidastudent -o test_distill_gpu
//   LD_LIBRARY_PATH=. ./test_distill_gpu

#include <cstdio>
#include <cstdlib>
#include <cmath>
extern "C" {
  void* stu_create(int,int,int,int,int,int);
  void  stu_set_distill(void*,const int*,const float*);
  float stu_step(void*,float);
  void  stu_free(void*);
}
int main(){
  const int V=32,D=16,F=32,S=16,L=2,H=2;
  void* M=stu_create(V,D,F,S,L,H);
  int ids[S]; for(int i=0;i<S;i++) ids[i]=(i*5+3)%V;
  float* Q=(float*)malloc((size_t)S*V*4);
  double HQ=0;
  for(int r=0;r<S;r++){
    int c=r%V; for(int k=0;k<V;k++) Q[r*V+k]=(k==c)?0.7f:(0.3f/(V-1));
    for(int k=0;k<V;k++){ double q=Q[r*V+k]; HQ += -q*log(q); }   // teacher entropy
  }
  stu_set_distill(M,ids,Q);
  float l0=stu_step(M,0.005f), lf=l0;
  for(int it=0;it<500;it++) lf=stu_step(M,0.005f);
  printf("distillation: initial loss=%.4f  final=%.4f  H(Q)=%.4f (floor)\n", l0, lf, HQ);
  double rel=fabs(lf-HQ)/HQ;
  printf("  final vs H(Q): rel gap=%.3f\n", rel);
  bool ok = lf < l0 && rel < 0.20;     // converged toward the teacher entropy floor
  printf("RESULT: %s (GPU Student distilled toward teacher distribution)\n", ok?"PASS":"FAIL");
  stu_free(M); return ok?0:1;
}
