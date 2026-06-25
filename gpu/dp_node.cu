// dp_node.cu — Step 7: REAL networked data-parallel node. Two processes each run
// a resident Student; every round they stu_micro on their own data shard, then
// exchange + SUM the gradient accumulators over a TCP socket (the all-reduce),
// and stu_apply. Run two instances (rank 0 = server, rank 1 = client) — they must
// stay bit-identical, proving the networked all-reduce. Validatable at $0 over
// localhost on one box; the same binary across two droplets is the real thing.
//
//   nvcc -O3 -arch=native dp_node.cu -L. -laspidastudent -o dp_node
//   LD_LIBRARY_PATH=. ./dp_node 0 127.0.0.1 5599 50 &   # server (rank 0)
//   LD_LIBRARY_PATH=. ./dp_node 1 127.0.0.1 5599 50     # client (rank 1)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

extern "C" {
  void* stu_create(int,int,int,int,int,int);
  void  stu_set_data(void*,const int*,const int*);
  float stu_micro(void*);
  void  stu_apply(void*,float,int);
  int   stu_nparams(void*);
  void  stu_get_acc(void*,float*);
  void  stu_set_acc(void*,const float*);
  void  stu_free(void*);
}

static void send_all(int fd,const char*p,long n){ while(n>0){ long k=send(fd,p,n,0); if(k<=0){perror("send");exit(1);} p+=k; n-=k; } }
static void recv_all(int fd,char*p,long n){ while(n>0){ long k=recv(fd,p,n,0); if(k<=0){perror("recv");exit(1);} p+=k; n-=k; } }

int main(int argc,char**argv){
  if(argc<5){ printf("usage: dp_node rank host port rounds\n"); return 2; }
  int rank=atoi(argv[1]); const char* host=argv[2]; int port=atoi(argv[3]); int rounds=atoi(argv[4]);
  int fd;
  if(rank==0){                                   // server
    int ls=socket(AF_INET,SOCK_STREAM,0); int opt=1; setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof opt);
    sockaddr_in a{}; a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons(port);
    if(bind(ls,(sockaddr*)&a,sizeof a)){perror("bind");return 1;}
    listen(ls,1); fd=accept(ls,0,0); close(ls);
  } else {                                        // client (retry until server is up)
    fd=socket(AF_INET,SOCK_STREAM,0);
    sockaddr_in a{}; a.sin_family=AF_INET; a.sin_port=htons(port); inet_pton(AF_INET,host,&a.sin_addr);
    for(int i=0;i<200 && connect(fd,(sockaddr*)&a,sizeof a)!=0;i++) usleep(100000);
  }

  void* M=stu_create(32,16,32,6,2,2);
  int N=stu_nparams(M);
  float* loc=(float*)malloc((size_t)N*4); float* rem=(float*)malloc((size_t)N*4);
  const int id0[6]={3,7,3,11,20,7}, t0[6]={5,1,9,0,14,2};
  const int id1[6]={2,9,4,15,8,1},  t1[6]={6,3,2,11,7,0};
  const int* id=rank?id1:id0; const int* tg=rank?t1:t0;

  for(int r=0;r<rounds;r++){
    stu_set_data(M,id,tg); stu_micro(M);
    stu_get_acc(M,loc);
    if(rank==0){ send_all(fd,(char*)loc,(long)N*4); recv_all(fd,(char*)rem,(long)N*4); }
    else       { recv_all(fd,(char*)rem,(long)N*4); send_all(fd,(char*)loc,(long)N*4); }
    for(int i=0;i<N;i++) loc[i]+=rem[i];          // ---- all-reduce SUM ----
    stu_set_acc(M,loc);
    stu_apply(M,0.005f,2);
  }
  stu_set_data(M,id0,t0); float l=stu_micro(M);    // eval both ranks on the same input
  printf("rank %d: params=%d final_loss=%.6e\n", rank, N, l);
  stu_free(M); close(fd); return 0;
}
