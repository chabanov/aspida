/*
 * capi_smoke.c — end-to-end smoke test for libaspida.dylib.
 *
 * Build:
 *   cc tools/capi_smoke.c -Iinclude -Llib/aspida -laspida \
 *       -Wl,-rpath,@loader_path/lib/aspida -o /tmp/capi_smoke
 * Run (loads a multi-GB model; give it RAM + seconds):
 *   /tmp/capi_smoke [/path/to/model.gguf]
 *   # default model: ~/models/ornith/ornith-1.0-9b-Q4_K_M.gguf
 *
 * Verifies: arch detection, arch_supported, load, streaming chat, free.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aspida.h"

static void on_tick(void *ud) { (void)ud; fputc('.', stdout); fflush(stdout); }
static void on_reasoning(const char *p, void *ud) {
    (void)ud; fputs("\n[reasoning] ", stdout); fputs(p, stdout); fflush(stdout);
}
static void on_text(const char *p, void *ud) {
    (void)ud; fputs(p, stdout); fflush(stdout);
}
static void on_tool(const char *id, const char *name, const char *args, void *ud) {
    (void)ud; printf("\n[tool_call] %s  name=%s  args=%s\n", id, name, args); fflush(stdout);
}
static void on_finish(const char *reason, void *ud) {
    (void)ud; printf("\n[finish] %s\n", reason ? reason : "(null)"); fflush(stdout);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const char *model = (argc > 1) ? argv[1]
        : "/Users/ceo/models/ornith/ornith-1.0-9b-Q4_K_M.gguf";

    printf("== aspida C-ABI smoke ==\nmodel: %s\n", model);

    /* 0) attach this C thread to the Ada runtime (sec stack). */
    if (!aspida_init()) { fprintf(stderr, "aspida_init failed: %s\n", aspida_last_error()); return 1; }

    /* 1) detect arch + supported (no weight load, fast). */
    char *arch = aspida_detect_arch(model);
    if (!arch) { fprintf(stderr, "detect_arch failed: %s\n", aspida_last_error()); return 2; }
    int sup = aspida_arch_supported(model);
    printf("arch: \"%s\"  supported: %d\n", arch, sup);
    aspida_free_string(arch);
    if (!sup) { fprintf(stderr, "model arch not supported\n"); return 3; }

    /* 2) load. */
    printf("loading model (may take a few seconds)...\n"); fflush(stdout);
    aspida_engine_t e = aspida_load(model);
    if (!e) { fprintf(stderr, "load failed: %s\n", aspida_last_error()); return 4; }
    printf("loaded. vocab=%d  arch_name=", aspida_vocab_size(e));
    char *an = aspida_arch_name(e);
    printf("\"%s\"\n", an ? an : "(null)");
    aspida_free_string(an);

    /* 3) streaming chat. */
    aspida_message_t msgs[1] = { { ASPIDA_ROLE_USER, "Привіт! Хто ти в одному реченні?"} };
    aspida_params_t p = { .temperature = 0.7f, .top_k = 0, .top_p = 0.9f,
                          .min_p = 0.0f, .repeat_penalty = 1.0f,
                          .repeat_last_n = 64, .seed = 0 };
    aspida_sink_t sink = { .on_tick = on_tick, .on_reasoning = on_reasoning,
                           .on_text = on_text, .on_tool_call = on_tool,
                           .on_finish = on_finish, .user_data = NULL };
    aspida_stats_t stats = {0};
    printf("== chat (max 32 tokens) ==\n"); fflush(stdout);
    char *res = aspida_chat(e, msgs, 1, 32, &p, &sink, &stats);
    if (!res) { fprintf(stderr, "\nchat failed: %s\n", aspida_last_error()); aspida_unload(e); return 5; }
    printf("\n== result JSON ==\n%s\n", res);
    printf("== usage == prompt=%d completion=%d truncated=%d overflow=%d\n",
           stats.prompt_tokens, stats.completion_tokens,
           stats.truncated, stats.overflow);
    aspida_free_string(res);

    /* 4) unload. */
    aspida_unload(e);
    printf("== ok ==\n");
    return 0;
}