/*
 * aspida.h — C ABI over the aspida LLM inference engine (libaspida.dylib).
 *
 * Build the dylib:  make lib        (-> lib/aspida/libaspida.dylib)
 *
 * Memory ownership
 *   - Functions returning char*  (aspida_detect_arch, aspida_discover_models,
 *     aspida_chat, aspida_arch_name) allocate a NUL-terminated string the
 *     CALLER must release with aspida_free_string. They return NULL on error.
 *   - aspida_last_error returns a pointer into a thread-local buffer owned by
 *     the library; valid until the next API call sets a new error. DO NOT free.
 *   - The opaque aspida_engine_t is allocated by aspida_load and freed by
 *     aspida_unload (call exactly once; passing it again is UB).
 *
 * Threading
 *   - aspida_chat is a synchronous, potentially long (seconds–minutes) call
 *     that streams tokens via the sink callbacks. Run it off the UI thread.
 *   - Error reporting (aspida_last_error) is per-thread.
 *
 * Strings passed IN (path, message text, dirs) are NUL-terminated UTF-8; the
 * library copies what it needs, so the caller may free/reuse them after the
 * call returns.
 */
#ifndef ASPIDA_H
#define ASPIDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>  /* size_t */

/* Opaque engine handle (a heap-allocated LLM engine). */
typedef struct aspida_engine_s* aspida_engine_t;

/* Chat role. */
typedef enum {
    ASPIDA_ROLE_SYSTEM    = 0,
    ASPIDA_ROLE_USER      = 1,
    ASPIDA_ROLE_ASSISTANT = 2
} aspida_role_t;

/* One conversation message. text is NUL-terminated UTF-8 (may be NULL -> ""). */
typedef struct {
    aspida_role_t role;
    const char   *text;
} aspida_message_t;

/* Sampling parameters (mirror LLM_Sampler.Params).
 * temperature <= 0.0 => greedy argmax. top_k <= 0 => none. top_p == 1.0 =>
 * none. min_p <= 0.0 => none. repeat_penalty == 1.0 => none. seed == 0 =>
 * fixed default seed. min_tokens > 0 => the stop/EOS token may not be sampled
 * until at least that many tokens have been generated (defeats 0-token answers
 * from models that emit im_end on the first step). */
typedef struct {
    float temperature;
    int   top_k;
    float top_p;
    float min_p;
    float repeat_penalty;
    int   repeat_last_n;
    int   min_tokens;
    long  seed;
} aspida_params_t;

/* Per-generation accounting (out). Booleans are 0/1. */
typedef struct {
    int prompt_tokens;
    int completion_tokens;
    int truncated;
    int overflow;
} aspida_stats_t;

/* Streaming callbacks. piece strings are NUL-terminated UTF-8 valid only for
 * the duration of the callback (do not retain). user_data is the opaque
 * pointer you set in aspida_sink_t.user_data. Any callback may be NULL. */
typedef void (*aspida_on_tick_fn)(void *user_data);
typedef void (*aspida_on_piece_fn)(const char *piece, void *user_data);
typedef void (*aspida_on_tool_call_fn)(const char *id,
                                       const char *name,
                                       const char *arguments_json,
                                       void *user_data);
typedef void (*aspida_on_finish_fn)(const char *reason, void *user_data);

typedef struct {
    aspida_on_tick_fn       on_tick;       /* prefill progress (per prompt token) */
    aspida_on_piece_fn      on_reasoning;  /* reasoning_content piece */
    aspida_on_piece_fn      on_text;       /* answer text piece */
    aspida_on_tool_call_fn  on_tool_call;
    aspida_on_finish_fn     on_finish;     /* fires once: "stop"|"length"|"tool_calls" */
    void                   *user_data;
} aspida_sink_t;

/* ---- API ---- */

/* Attach the calling (foreign) thread to the Ada runtime so it gets an Ada
 * task context + secondary stack. Required before any call below that touches
 * the engine (the wrappers call it themselves, so hosts may ignore it). Safe
 * to call repeatedly. Returns 1 on success, 0 on failure. */
int aspida_init(void);

/* Load a model from a GGUF path. Returns NULL on failure (read the message
 * with aspida_last_error). Release with aspida_unload. */
aspida_engine_t aspida_load(const char *path);

/* Unload and free an engine. No-op if e is NULL. */
void aspida_unload(aspida_engine_t e);

/* Last error message for the calling thread. Never NULL. Do NOT free. */
const char *aspida_last_error(void);

/* Peek a GGUF's general.architecture without loading weights. Returns "" for
 * an unreadable file. Caller frees. */
char *aspida_detect_arch(const char *path);

/* 1 if the GGUF's architecture is supported, else 0. */
int aspida_arch_supported(const char *path);

/* Enumerate GGUF models. dirs is a ':'-joined list of search roots, or NULL
 * to use the default roots (ASPIDA_MODELS_DIR + common dirs). Returns a JSON
 * array string; caller frees:
 *   [{"path","name","arch","quant","params","size":N,"supported":bool,"status"}]
 */
char *aspida_discover_models(const char *dirs);

/* Run a chat turn. messages points to n consecutive aspida_message_t. sink
 * may be NULL (non-streaming). stats may be NULL. Returns a JSON result on
 * success (caller frees); NULL on failure (aspida_last_error):
 *   {"reasoning","answer","finish",
 *    "tool_calls":[{"id","name","arguments"}],
 *    "usage":{"prompt_tokens","completion_tokens","truncated","overflow"}}
 */
char *aspida_chat(aspida_engine_t            e,
                  const aspida_message_t    *messages,
                  int                        n,
                  int                        max_new_tokens,
                  const aspida_params_t      *params,
                  const aspida_sink_t        *sink,
                  aspida_stats_t            *stats);

/* Engine metadata. */
int  aspida_vocab_size(aspida_engine_t e);
char *aspida_arch_name(aspida_engine_t e);  /* caller frees */

/* Release a string returned by the API. No-op on NULL. */
void aspida_free_string(char *s);

#ifdef __cplusplus
}
#endif

#endif /* ASPIDA_H */