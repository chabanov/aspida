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
#include <string>

static sd_ctx_t* g_ctx = nullptr;

// Optional few-step distillation LoRA (Qwen-Image-Edit-2511-Lightning). When
// ASPIDA_IMG_LORA names one, it is applied via the sd.cpp "<lora:name:w>"
// prompt tag and generation switches to the LoRA's regime (cfg 1.0, ~8 steps)
// — ~5x faster AND higher-fidelity than the 20-step base. Empty => base model.
static std::string g_lora_path;   // full path to the .safetensors ("" => base)
static int         g_lora_steps = 8;

static const char* env_or(const char* k, const char* d) {
    const char* v = getenv(k);
    return (v && *v) ? v : d;
}

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

    const char* lname = env_or("ASPIDA_IMG_LORA", "");
    if (lname[0]) {
        std::string dir = env_or("ASPIDA_IMG_LORA_DIR", "/opt/sdmodels/lora");
        g_lora_path  = dir + "/" + lname + ".safetensors";
        g_lora_steps = atoi(env_or("ASPIDA_IMG_LORA_STEPS", "8"));
        if (g_lora_steps < 1) g_lora_steps = 8;
    }
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

    // With the Lightning LoRA active, attach it and switch to the distilled
    // regime: cfg 1.0 (guidance distilled away) + the LoRA's step count.
    // Without it, honour the caller's steps/cfg on the base model.
    int   eff_steps = steps;
    float eff_cfg   = cfg;
    sd_lora_t lora;
    if (!g_lora_path.empty()) {
        lora.is_high_noise = false;
        lora.multiplier    = 1.0f;
        lora.path          = g_lora_path.c_str();
        g.loras            = &lora;
        g.lora_count       = 1;
        eff_steps          = g_lora_steps;
        eff_cfg            = 1.0f;
    }
    g.prompt                          = prompt;
    g.width                           = W;
    g.height                          = H;
    g.seed                            = seed;
    g.sample_params.sample_method     = EULER_SAMPLE_METHOD;
    g.sample_params.sample_steps      = eff_steps;
    g.sample_params.guidance.txt_cfg  = eff_cfg;
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
