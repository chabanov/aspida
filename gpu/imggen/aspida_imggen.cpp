// aspida_imggen.cpp — native image gen/edit for the aspida engine.
// Thin C wrapper over stable-diffusion.cpp (Qwen-Image-Edit-2511 on ggml).
// Built into libaspida_imggen.so with a version script that exports ONLY the
// aspida_img_* symbols — every ggml/sd symbol is local, so it cannot interpose
// with the engine's own ggml (from llama.cpp) when both live in one process.
#include "stable-diffusion.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

static sd_ctx_t* g_ctx = nullptr;

extern "C" int aspida_img_init(const char* dit, const char* vae,
                               const char* llm, const char* mmproj) {
    if (g_ctx) return 0;
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.diffusion_model_path = dit;
    p.vae_path             = vae;
    p.llm_path             = llm;
    if (mmproj && mmproj[0]) p.llm_vision_path = mmproj;
    p.flash_attn           = true;
    p.diffusion_flash_attn = true;
    p.model_args           = "qwen_image_zero_cond_t=true";  // 2511 edit mode
    g_ctx = new_sd_ctx(&p);
    return g_ctx ? 0 : -1;
}

// Generate (ref_path == NULL) or edit (ref_path != NULL); write PNG to out_path.
extern "C" int aspida_img_generate(const char* prompt, const char* ref_path,
                                   int W, int H, int steps, float cfg,
                                   long long seed, const char* out_path) {
    if (!g_ctx) return -1;
    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt                          = prompt;
    g.width                           = W;
    g.height                          = H;
    g.seed                            = seed;
    g.sample_params.sample_method     = EULER_SAMPLE_METHOD;
    g.sample_params.sample_steps      = steps;
    g.sample_params.guidance.txt_cfg  = cfg;
    g.sample_params.flow_shift        = 3.0f;

    sd_image_t ref;
    unsigned char* refdata = nullptr;
    if (ref_path && ref_path[0]) {
        int rw = 0, rh = 0, nc = 0;
        refdata = stbi_load(ref_path, &rw, &rh, &nc, 3);
        if (!refdata) return -2;
        ref.width = rw; ref.height = rh; ref.channel = 3; ref.data = refdata;
        g.ref_images       = &ref;
        g.ref_images_count = 1;
    }

    sd_image_t* out = nullptr;
    int out_count = 0;
    bool gen_ok = generate_image(g_ctx, &g, &out, &out_count);
    if (refdata) stbi_image_free(refdata);
    if (!gen_ok || !out || out_count < 1) return -3;

    int ok = stbi_write_png(out_path, out[0].width, out[0].height,
                            out[0].channel, out[0].data, 0);
    for (int i = 0; i < out_count; ++i) free(out[i].data);
    free(out);
    return ok ? 0 : -4;
}
