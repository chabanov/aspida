# Aspida

**End-to-end encrypted LLM inference, built from scratch in Ada/SPARK.**
Our own GGUF inference engine (Llama / Qwen / Gemma backends) and our own
cryptography — no third-party crypto or ML libraries. A prompt is sealed in the
client and stays sealed across the network; only the inference boundary ever
opens it.

## Layout

```
src/
  crypto/      from-scratch crypto primitives (the core is SPARK-proven)
  session/     secure channel (handshake + AEAD records), session store, at-rest sealing
  secure/      wire protocol + socket transport
  server/      secure_server, secure_client, ws_bridge (WebSocket⇄TCP), OpenAI-compatible proxy
  llm/         the engine: GGUF loader, Llama/Qwen/Gemma backends, sampler, tokenizer, GPU offload
  aspida.ads, main.adb   WIP HTTP-client-generator CLI (separate from the server)

tests/
  crypto/      RFC test-vector suites (KATs), channel + at-rest tests
  secure/      TCP transport + secure-channel integration tests
  *.adb        model-free LLM unit tests (+ a few *_real that need a local GGUF)

tools/
  probes/      dev/debug executables (gguf_probe, llama_probe, gemma_probe, …)
  convert_gpt2.py

gpu/
  gpu_matvec.cu, test_matvec.cu   CUDA kernels (Q4_K/Q5_K/Q6_K) driven by the Ada core
  experiments/  phase0 / phase1 spikes (FFI bring-up, reference generator)

docs/          plans & engineering notes
blog/          articles
web/           browser E2EE demo (served by ws_bridge)
```

GNAT project files live at the repo root (`shared.gpr` carries the common
switches; the rest `with` it): `crypto.gpr`, `session.gpr`, `server.gpr`,
`aspida_cli.gpr`, the probe builders (`probe.gpr`, `gen.gpr`, `fprobe.gpr`,
`genref.gpr`), and the test projects (`crypto_tests.gpr`, `secure_tests.gpr`,
`tests/llm_tests.gpr`). Build flags and strict warnings-as-errors are
centralised in `shared.gpr`.

## Build & run

```sh
make server     # build the encrypted inference server + client (the product)
make serve      # build + launch the server (set ASPIDA_STORE_PASSWORD to persist history)
make chat       # connect an interactive encrypted client (needs `make serve` running)
```

The server picks its backend from the GGUF architecture. Point it at a model:

```sh
QWEN_MODEL_PATH=<any-supported-gguf> ./obj/secure_server [port]   # env var name is historical
```

Bind address is `Any` by default; set `ASPIDA_BIND=127.0.0.1` to restrict it
(e.g. behind the bridge / a reverse proxy).

## Model discovery & selection

At startup the server enumerates every GGUF model on the system (metadata only,
no weights loaded) and logs the catalog. List them standalone with:

```sh
gprbuild -P probe.gpr && ./obj/model_scan      # or: ASPIDA_MODELS_DIR=/a:/b ./obj/model_scan
```

Search roots: `ASPIDA_MODELS_DIR` (`:`-separated, highest precedence) then common
locations (`./models`, `~/.lmstudio/models`, `~/.cache/huggingface`, …). Projector
files (`mmproj-*`) and architectures the engine can't run are flagged, not offered.

The active model is resolved as: `QWEN_MODEL_PATH` env (deployments pin it) →
a persisted runtime selection (`active_model` file) → a built-in default.

A client may list the catalog and pick a model over the encrypted channel
(`Tag_Models` / `Tag_Select`). Because a model can't be hot-swapped in place
(the batch scheduler binds one model and backends don't unload), switching is
**reload-based**: the selection is persisted and the server reloads with it.
Run under the supervisor so switches apply automatically:

```sh
make serve     # sets ASPIDA_AUTORELOAD; reloads the server when a model is selected
```

The web demo shows a model dropdown only when the server advertises it is
switchable (i.e. running under `make serve`); a pinned deployment hides it.

## Test & verify

```sh
make test          # all model-free tests (crypto/E2EE + LLM units)
make test-crypto   # RFC KATs + channel + at-rest + socket/session integration
make test-llm      # model-free LLM unit tests
make prove         # SPARK: AoRTE + functional proof of the crypto core; flow for the rest
make prove-flow    # SPARK flow analysis over the whole crypto library
```

Model-dependent checks (`make test-tokenizer-real`, `test-weights-real`,
`test-qmatvec-real`) need a local GGUF and are excluded from CI.

## Principles

- **Everything is ours.** Engine, cryptography and transport are written from
  scratch in Ada/SPARK; no third-party crypto or ML libraries in the path.
- **The crypto core is machine-checked.** `make prove` discharges absence of
  run-time errors and functional contracts on the crypto root with SPARK.
- **One chat path.** The only way to talk to the model is the encrypted
  channel — there is no plaintext server mode.
