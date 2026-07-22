// aspida_imgd — isolated image-generation daemon.
//
// Runs Qwen-Image-Edit-2511 (via libaspida_imggen.so → stable-diffusion.cpp)
// in its OWN process, hence its OWN CUDA context. secure_server (the LLM
// engine) talks to it over a localhost TCP socket and never links sd.cpp,
// so sd.cpp's ggml-CUDA backend can no longer interpose on the LLM's ggml
// backend at runtime — the illegal-memory-access / wedge that killed the
// in-process design (two ggml-CUDA backends in one process) is impossible.
//
// Wire protocol (one request per connection, length-prefixed so the prompt
// may contain any bytes):
//   header line:  "<W> <H> <steps> <seed> <plen> <rlen> <olen>\n"
//   then exactly: <plen> prompt bytes, <rlen> ref-path bytes, <olen> out-path bytes
//   ref-path empty (rlen 0) => text-to-image; non-empty => img2img.
//   reply:        "OK\n" on success, "ERR <rc>\n" otherwise.
//
// The daemon serialises requests (one sd context); concurrency is bounded by
// the single listen backlog, and secure_server also gates callers.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

// libaspida_imggen.so C API (the same wrapper the in-process build used).
extern int aspida_img_init(const char* dit, const char* vae,
                           const char* llm, const char* mmproj);
extern int aspida_img_generate(const char* prompt, const char* ref_path,
                               int W, int H, int steps, float cfg,
                               long long seed, const char* out_path);

static const char* env_or(const char* k, const char* d) {
    const char* v = getenv(k);
    return (v && *v) ? v : d;
}

// Read exactly n bytes into buf; return 0 on success, -1 on EOF/error.
static int read_exact(int fd, char* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, buf + got, n - got);
        if (r <= 0) return -1;
        got += (size_t)r;
    }
    return 0;
}

// Read a '\n'-terminated line (without the newline) into buf, cap len.
static int read_line(int fd, char* buf, size_t cap) {
    size_t i = 0;
    while (i + 1 < cap) {
        char c;
        ssize_t r = read(fd, &c, 1);
        if (r <= 0) return -1;
        if (c == '\n') { buf[i] = 0; return 0; }
        buf[i++] = c;
    }
    buf[i] = 0;
    return 0;
}

static void handle(int cfd) {
    char header[256];
    if (read_line(cfd, header, sizeof header) != 0) return;

    int W = 1024, H = 1024, steps = 20;
    long long seed = -1;
    long plen = 0, rlen = 0, olen = 0;
    if (sscanf(header, "%d %d %d %lld %ld %ld %ld",
               &W, &H, &steps, &seed, &plen, &rlen, &olen) != 7) {
        const char* e = "ERR badheader\n";
        (void)!write(cfd, e, strlen(e));
        return;
    }
    // Sanity caps: a prompt is small; paths are short. Guards a bad peer.
    if (plen < 0 || plen > (1 << 20) || rlen < 0 || rlen > 4096 ||
        olen < 0 || olen > 4096) {
        const char* e = "ERR toolong\n";
        (void)!write(cfd, e, strlen(e));
        return;
    }

    char* prompt = (char*)calloc((size_t)plen + 1, 1);
    char* ref    = (char*)calloc((size_t)rlen + 1, 1);
    char* out    = (char*)calloc((size_t)olen + 1, 1);
    int rc = -99;
    if (prompt && ref && out &&
        read_exact(cfd, prompt, (size_t)plen) == 0 &&
        read_exact(cfd, ref, (size_t)rlen) == 0 &&
        read_exact(cfd, out, (size_t)olen) == 0) {
        const char* refp = (rlen > 0) ? ref : "";
        rc = aspida_img_generate(prompt, refp, W, H, steps, 2.5f, seed, out);
    }

    char reply[64];
    if (rc == 0) snprintf(reply, sizeof reply, "OK\n");
    else         snprintf(reply, sizeof reply, "ERR %d\n", rc);
    (void)!write(cfd, reply, strlen(reply));

    free(prompt); free(ref); free(out);
}

int main(void) {
    const char* dit = env_or("ASPIDA_IMG_DIT",
        "/opt/sdmodels/dit/qwen-image-edit-2511-Q8_0.gguf");
    const char* vae = env_or("ASPIDA_IMG_VAE",
        "/opt/sdmodels/vae/split_files/vae/qwen_image_vae.safetensors");
    const char* llm = env_or("ASPIDA_IMG_LLM",
        "/opt/sdmodels/llm/Qwen2.5-VL-7B-Instruct.Q8_0.gguf");
    const char* mmproj = env_or("ASPIDA_IMG_MMPROJ",
        "/opt/sdmodels/llm/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf");
    int port = atoi(env_or("ASPIDA_IMGD_PORT", "8790"));

    fprintf(stderr, "aspida-imgd: loading model (own CUDA context)...\n");
    int irc = aspida_img_init(dit, vae, llm, mmproj);
    if (irc != 0) {
        fprintf(stderr, "aspida-imgd: init failed rc=%d\n", irc);
        return 1;
    }
    fprintf(stderr, "aspida-imgd: model ready, listening on 127.0.0.1:%d\n", port);

    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(sfd, (struct sockaddr*)&addr, sizeof addr) != 0) {
        perror("bind"); return 1;
    }
    if (listen(sfd, 16) != 0) { perror("listen"); return 1; }

    for (;;) {
        int cfd = accept(sfd, NULL, NULL);
        if (cfd < 0) { if (errno == EINTR) continue; break; }
        handle(cfd);
        close(cfd);
    }
    close(sfd);
    return 0;
}
