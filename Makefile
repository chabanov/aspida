# Aspida — ADA/SPARK build & development Makefile
# ==================================================

# ── Paths ────────────────────────────────────────────────────────────
# Auto-detect the installed Alire toolchains (version-agnostic). Override by
# passing GNAT_BIN=/path GPRBUILD_BIN=/path on the make command line.
TOOLCHAINS  := $(HOME)/.local/share/alire/toolchains
GNAT_BIN    ?= $(firstword $(wildcard $(TOOLCHAINS)/gnat_native_*/bin))
GPRBUILD_BIN ?= $(firstword $(wildcard $(TOOLCHAINS)/gprbuild_*/bin))
PATH        := $(GNAT_BIN):$(GPRBUILD_BIN):$(PATH)
export PATH

# ── Target OS / SDK ──────────────────────────────────────────────────
# Linux vs macOS selects the linker flags in shared.gpr (-XOS). On macOS we
# also need the SDK root; on Linux xcrun is absent (SDKROOT stays empty and is
# unused because the darwin linker branch is not taken).
UNAME_S     := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  OS_NAME   := linux
else
  OS_NAME   := darwin
endif
SDKROOT     := $(shell xcrun --show-sdk-path 2>/dev/null)
export SDKROOT

# ── Encrypted chat server ───────────────────────────────────────────
PORT        ?= 8765
HOST        ?= 127.0.0.1
SESSION     ?= new
PUBKEY_FILE := server_pub.hex

# ── Project files ────────────────────────────────────────────────────
MAIN_GPR    := legacy/aspida_cli.gpr      # legacy HTTP-client-generator CLI (main.adb)
SECURE_GPR  := server.gpr                 # the product: E2EE inference server + client
CRYPTO_TEST_GPR := crypto_tests.gpr       # RFC KAT suite for the crypto library
SECURE_TEST_GPR := secure_tests.gpr       # TCP transport + secure-channel integration
LLM_TEST_GPR    := tests/llm_tests.gpr    # model-free LLM unit tests
SRC_DIR     := src
OBJ_DIR     := obj
BIN         := $(OBJ_DIR)/main

# ── Toolchain ────────────────────────────────────────────────────────
GPRBUILD    := $(GPRBUILD_BIN)/gprbuild
GPRCLEAN    := $(GPRBUILD_BIN)/gprclean
# gnatprove is installed via `alr install gnatprove` (lands in ~/.alire/bin);
# fall back to the GNAT toolchain dir if a future toolchain bundles it.
GNATPROVE   := $(firstword $(wildcard $(HOME)/.alire/bin/gnatprove) $(GNAT_BIN)/gnatprove)
FORMATTER   := $(GNAT_BIN)/gnatpp

# ── C ABI / libaspida.dylib ──────────────────────────────────────────
# The GNAT driver and the Ada runtime (libgnat-15/libgnarl-15) live in the
# toolchain's adalib dir. The shared-library link is done manually with the
# gcc driver (gprbuild ignores package-Linker switches for library projects
# on this platform, so its default -syslibroot hits the nonexistent
# MacOSX14.sdk and fails to find -lSystem).
GCC         := $(GNAT_BIN)/gcc
ADALIB      := $(firstword $(wildcard $(dir $(GNAT_BIN))lib/gcc/*/1*/adalib))
ASPIDA_LIB  := lib/aspida/libaspida.dylib

# ── Flags ────────────────────────────────────────────────────────────
# ARCH=portable drops -march=native (binaries run on any CPU); see shared.gpr.
ARCH        ?= native
GPR_FLAGS   := -XSDKROOT=$(SDKROOT) -XARCH=$(ARCH) -XOS=$(OS_NAME)
SPARK_FLAGS := -P $(MAIN_GPR) $(GPR_FLAGS) --mode=flow

# ── Default ──────────────────────────────────────────────────────────
#  The product is the encrypted inference server (server.gpr); the legacy
#  HTTP-client-generator CLI (legacy/aspida_cli.gpr) is built via `make build`
#  but is no longer the default.
.PHONY: all
all: server

# ══════════════════════════════════════════════════════════════════════
#  BUILD
# ══════════════════════════════════════════════════════════════════════

.PHONY: build
build: ## Build aspida CLI (debug)
	$(GPRBUILD) -P $(MAIN_GPR) $(GPR_FLAGS)

.PHONY: server
server: ## Build the encrypted chat server + client (the ONLY chat path — no plaintext mode)
	$(GPRBUILD) -P $(SECURE_GPR) $(GPR_FLAGS)

.PHONY: lib
lib: $(ASPIDA_LIB) ## Build libaspida.dylib (C ABI over the engine) for the native UI
$(ASPIDA_LIB): $(wildcard src/llm/*.ad?) src/llm/aspida_capi.ads src/llm/aspida_capi.adb aspida_lib.gpr shared.gpr
	@mkdir -p lib/aspida obj/aspida-lib
	@# 1) Compile all units with gprbuild. The library-project *link* step is
	@#    expected to fail on macOS (gprbuild won't honour -syslibroot here),
	@#    so we capture its log and only fail if compilation produced no .o.
	@log=$$(mktemp); MACOSX_DEPLOYMENT_TARGET=26.5 $(GPRBUILD) -P aspida_lib.gpr $(GPR_FLAGS) > $$log 2>&1; rc=$$?; \
	if [ ! -f obj/aspida-lib/aspida_capi.o ]; then cat $$log; rm -f $$log; \
	    echo "## gprbuild compile failed (rc=$$rc)"; exit 1; fi; rm -f $$log
	@# 2) Link the shared library manually with the real SDK sysroot.
	@MACOSX_DEPLOYMENT_TARGET=26.5 $(GCC) -dynamiclib obj/aspida-lib/*.o -o $(ASPIDA_LIB) \
	    -Wl,-install_name,@rpath/libaspida.dylib \
	    -Wl,-syslibroot,$(SDKROOT) \
	    -L$(ADALIB) -lgnat -lgnarl
	@echo "Built $(ASPIDA_LIB)  (deps: $$(otool -L $(ASPIDA_LIB) | tail -n +2 | tr -s ' ' | cut -d' ' -f1 | tr '\n' ' '))"

.PHONY: smoke
smoke: $(ASPIDA_LIB) ## Build + run the C-ABI smoke test (loads the default model; pass MODEL=...)
	@cc tools/capi_smoke.c -Iinclude -Llib/aspida -laspida \
	    -Wl,-rpath,@loader_path/lib/aspida -Wl,-rpath,$(ADALIB) \
	    -Wl,-syslibroot,$(SDKROOT) -o ./capi_smoke
	@./capi_smoke $(MODEL)
	@rm -f ./capi_smoke

.PHONY: serve
serve: server ## Build + launch the server; model switches auto-reload (Ctrl-C to stop)
	@echo "serving on $(PORT) — runtime model switching enabled; Ctrl-C to stop"
	@ASPIDA_AUTORELOAD=1; export ASPIDA_AUTORELOAD; \
	while true; do \
	  ./$(OBJ_DIR)/secure_server $(PORT); code=$$?; \
	  if [ $$code -eq 75 ]; then echo "[serve] reloading with newly selected model…"; continue; fi; \
	  exit $$code; \
	done

.PHONY: chat
chat: server ## Open an interactive encrypted chat (needs a running `make serve`; SESSION=<id> to resume)
	@test -f $(PUBKEY_FILE) || { \
		echo "No $(PUBKEY_FILE) found — start the server first in another terminal:"; \
		echo "    make serve"; exit 1; }
	@echo "connecting to $(HOST):$(PORT) — pinning $$(cat $(PUBKEY_FILE))"
	./$(OBJ_DIR)/secure_client $(HOST) $(PORT) $$(cat $(PUBKEY_FILE)) $(SESSION)

.PHONY: release
release: ## Build aspida CLI (optimized)
	$(GPRBUILD) -P $(MAIN_GPR) $(GPR_FLAGS) -XBUILD=release

.PHONY: tests
tests: ## Build every test suite (crypto/E2EE + model-free LLM)
	$(GPRBUILD) -P $(CRYPTO_TEST_GPR) $(GPR_FLAGS)
	$(GPRBUILD) -P $(SECURE_TEST_GPR) $(GPR_FLAGS)
	$(GPRBUILD) -P $(LLM_TEST_GPR) $(GPR_FLAGS)

# ══════════════════════════════════════════════════════════════════════
#  TEST
# ══════════════════════════════════════════════════════════════════════

.PHONY: test
test: test-crypto test-json test-llm ## Run all model-free unit tests (crypto/E2EE + JSON + LLM)

.PHONY: test-json
test-json: ## Regression: JSON depth cap (Batch 1.2) — deep nesting raises, normal parses
	$(GPRBUILD) -P tests/test_json.gpr $(GPR_FLAGS)
	./obj/test_json

.PHONY: check
check: build test ## Build CLI + run model-free tests (CI target)

.PHONY: train-test
train-test: ## Build + run the from-scratch training-core self-tests (model-free; grad-check, distill, block, QAT, BPE, ckpt, mha)
	$(GPRBUILD) -P train.gpr $(GPR_FLAGS)
	./obj/test_train
	./obj/test_block
	./obj/test_distill
	./obj/test_distill_train
	./obj/test_gguf
	./obj/test_serve
	./obj/test_mha
	./obj/test_ckpt
	./obj/test_bpe
	./obj/test_bpe_gguf
	./obj/test_qat
	./obj/test_qat_train
	./obj/test_teacher
	./obj/test_multi_teacher
	./obj/test_scale
	./obj/test_platform
	./obj/test_eval
	./obj/test_data_pipeline
	./obj/test_job_store
	./obj/train_resident_probe
	./obj/student_gpu_probe
	./obj/test_turnkey

.PHONY: distill-demos
distill-demos: ## Verifier-driven distillation demos (model-free): student >= teacher
	$(GPRBUILD) -P train.gpr $(GPR_FLAGS)
	./obj/code_distill      # verifier-filtered distillation: filtered student > noisy teacher
	./obj/code_iterate      # verifier-bootstrapped self-improvement (no teacher) -> full coverage
	./obj/exec_distill      # Track 2: a REAL interpreter (python3) verifies real code
	./obj/turnkey_demo     # MVP turnkey loop end-to-end: admit->train->gate->charge
	./obj/turnkey_serve_demo # delivery glue: train -> Delivered -> export GGUF -> engine-load -> endpoint

# ══════════════════════════════════════════════════════════════════════
#  SPARK (formal verification)
# ══════════════════════════════════════════════════════════════════════

.PHONY: test-rmsnorm
test-rmsnorm: ## Run RMSNorm unit test
	$(GPRBUILD) -P tests/llm_tests.gpr $(GPR_FLAGS)
	./obj/test_rmsnorm

.PHONY: test-llm
test-llm: ## Build + run all LLM unit tests
	$(GPRBUILD) -P tests/llm_tests.gpr $(GPR_FLAGS)
	./obj/test_rmsnorm
	./obj/test_moe
	./obj/test_tokenizer
	./obj/test_tensor
	./obj/test_deltanet
	./obj/test_deltanet_blk
	./obj/test_fullattn
	./obj/test_block
	./obj/test_pool
	./obj/test_quant
	./obj/test_kquant_synth
	./obj/test_qmatvec_q4k
	./obj/test_ctx_window
	./obj/test_rope_scale
	./obj/test_mrope_sections
	./obj/test_kv_pool
	./obj/test_sampler
	./obj/test_kv_overflow
	./obj/test_toolcap
	./obj/test_batcher_cfg

.PHONY: test-crypto
test-crypto: ## Build + run the crypto / E2EE test vectors (no model needed)
	$(GPRBUILD) -P crypto_tests.gpr $(GPR_FLAGS)
	$(GPRBUILD) -P secure_tests.gpr $(GPR_FLAGS)
	./obj/test_crypto
	./obj/test_hash
	./obj/test_random
	./obj/test_x25519
	./obj/test_channel
	./obj/test_pbkdf2
	./obj/test_atrest
	./obj/test_socket
	./obj/test_session

.PHONY: test-tokenizer-real
test-tokenizer-real: ## Validate the tokenizer against the real GGUF vocab (needs model)
	$(GPRBUILD) -P tests/llm_tests.gpr $(GPR_FLAGS)
	./obj/test_tokenizer_real

.PHONY: test-weights-real
test-weights-real: ## Validate GGUF load + dequant on real tensors (needs model)
	$(GPRBUILD) -P tests/llm_tests.gpr $(GPR_FLAGS)
	./obj/test_weights_real

.PHONY: test-weight
test-weight: ## H19 weight-streaming: remote-sourced GGUF read byte-identical to local (needs svgdata/student.gguf)
	MACOSX_DEPLOYMENT_TARGET=26.5 $(GPRBUILD) -P tests/weight_tests.gpr $(GPR_FLAGS)
	./obj/test_weight_stream
	./obj/test_weight_cache
	./obj/test_weight_disk
	./obj/test_weight_pin
	./obj/test_weight_prefetch
	./obj/test_weight_parity
	./obj/test_weight_egress
	./obj/test_weight_partial
	./obj/test_weight_concurrent

.PHONY: prove
prove: ## SPARK: full AoRTE+functional proof of crypto root, ChaCha20, SHA-256, HKDF & PBKDF2; flow for the rest
	$(GNATPROVE) -P crypto.gpr $(GPR_FLAGS) \
	  -u crypto.adb -u crypto-chacha20.adb -u crypto-sha256.adb \
	  -u crypto-hkdf.adb -u crypto-pbkdf2.adb \
	  --mode=all --level=2 -j0 --report=all

.PHONY: prove-flow
prove-flow: ## SPARK flow analysis (init/deps/aliasing) over the whole crypto library
	$(GNATPROVE) -P crypto.gpr $(GPR_FLAGS) --mode=flow -j0

.PHONY: prove-egress
prove-egress: ## H19 Phase 6: SPARK flow analysis of the client egress encoder (LLM_Weight_Proto)
	$(GNATPROVE) -P server.gpr $(GPR_FLAGS) -u src/server/llm_weight_proto.adb --mode=flow -j0

# ══════════════════════════════════════════════════════════════════════
#  CLEAN
# ══════════════════════════════════════════════════════════════════════

.PHONY: clean
clean: ## Remove build artifacts
	$(GPRCLEAN) -P $(MAIN_GPR) $(GPR_FLAGS) || true
	$(GPRCLEAN) -P $(SECURE_GPR) $(GPR_FLAGS) || true
	$(GPRCLEAN) -P $(CRYPTO_TEST_GPR) $(GPR_FLAGS) || true
	$(GPRCLEAN) -P $(SECURE_TEST_GPR) $(GPR_FLAGS) || true
	$(GPRCLEAN) -P $(LLM_TEST_GPR) $(GPR_FLAGS) || true
	rm -rf $(OBJ_DIR)/tests

.PHONY: distclean
distclean: clean ## Remove Alire cache too
	rm -rf alire/ .alire/

# ══════════════════════════════════════════════════════════════════════
#  FORMAT
# ══════════════════════════════════════════════════════════════════════

.PHONY: fmt
fmt: ## Format all ADA sources
	$(FORMATTER) -P $(MAIN_GPR) $(GPR_FLAGS) --pipe

.PHONY: fmt-check
fmt-check: ## Check formatting (CI)
	$(FORMATTER) -P $(MAIN_GPR) $(GPR_FLAGS) --pipe --check

# ══════════════════════════════════════════════════════════════════════
#  WATCH (dev loop — requires fswatch)
# ══════════════════════════════════════════════════════════════════════

.PHONY: watch
watch: ## Rebuild + retest on file changes
	@which fswatch >/dev/null 2>&1 || { echo "Install fswatch: brew install fswatch"; exit 1; }
	@echo "Watching $(SRC_DIR)/ tests/ …"
	fswatch -o $(SRC_DIR)/ tests/ | while read _; do \
		clear; \
		$(MAKE) build 2>&1 && $(MAKE) test 2>&1; \
	done

# ══════════════════════════════════════════════════════════════════════
#  HELP
# ══════════════════════════════════════════════════════════════════════

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: test-qmatvec-real
test-qmatvec-real: ## Validate streaming quantized matvec on real tensors (needs model)
	$(GPRBUILD) -P tests/llm_tests.gpr $(GPR_FLAGS)
	./obj/test_qmatvec_real
