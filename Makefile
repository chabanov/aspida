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

# ── SDK / macOS ──────────────────────────────────────────────────────
SDKROOT     := $(shell xcrun --show-sdk-path 2>/dev/null)
export SDKROOT

# ── LLM model defaults (tiny for quick iteration) ───────────────────
DIM         ?= 64
LAYERS      ?= 2

# ── Project files ────────────────────────────────────────────────────
MAIN_GPR    := aspida_cli.gpr
LLM_GPR     := llm.gpr
TEST_GPR    := tests/aspida_tests.gpr
LLM_GPR     := llm.gpr
SRC_DIR     := src
OBJ_DIR     := obj
BIN         := $(OBJ_DIR)/main
TEST_BIN    := $(OBJ_DIR)/test_runner

# ── Toolchain ────────────────────────────────────────────────────────
GPRBUILD    := $(GPRBUILD_BIN)/gprbuild
GPRCLEAN    := $(GPRBUILD_BIN)/gprclean
GNATPROVE   := $(GNAT_BIN)/gnatprove
FORMATTER   := $(GNAT_BIN)/gnatpp

# ── Flags ────────────────────────────────────────────────────────────
GPR_FLAGS   := -XSDKROOT=$(SDKROOT)
SPARK_FLAGS := -P $(MAIN_GPR) $(GPR_FLAGS) --mode=flow

# ── Default ──────────────────────────────────────────────────────────
.PHONY: all
all: build

# ══════════════════════════════════════════════════════════════════════
#  BUILD
# ══════════════════════════════════════════════════════════════════════

.PHONY: build
build: ## Build aspida CLI (debug)
	$(GPRBUILD) -P $(MAIN_GPR) $(GPR_FLAGS)

.PHONY: llm
llm: ## Build LLM chat binary
	$(GPRBUILD) -P $(LLM_GPR) $(GPR_FLAGS)

.PHONY: chat
chat: llm ## Build + launch chat REPL
	./$(OBJ_DIR)/llm_main $(DIM) $(LAYERS)

.PHONY: release
release: ## Build aspida CLI (optimized)
	$(GPRBUILD) -P $(MAIN_GPR) $(GPR_FLAGS) -XBUILD=release

.PHONY: tests
tests: ## Build the test binary
	$(GPRBUILD) -P $(TEST_GPR) $(GPR_FLAGS)

# ══════════════════════════════════════════════════════════════════════
#  TEST
# ══════════════════════════════════════════════════════════════════════

.PHONY: test
test: tests ## Run all unit tests
	./$(TEST_BIN)

.PHONY: check
check: build test ## Build + run tests (CI target)

# ══════════════════════════════════════════════════════════════════════
#  SPARK (formal verification)
# ══════════════════════════════════════════════════════════════════════

.PHONY: test-rmsnorm
test-rmsnorm: ## Run RMSNorm unit test
	$(GPRBUILD) -P tests/llm_tests.gpr $(GPR_FLAGS)
	./obj/test_rmsnorm

.PHONY: test-llm
test-llm: ## Build + run all LLM unit tests (rmsnorm, attention, moe, tokenizer, ssm)
	$(GPRBUILD) -P tests/llm_tests.gpr $(GPR_FLAGS)
	./obj/test_rmsnorm
	./obj/test_attention
	./obj/test_moe
	./obj/test_tokenizer
	./obj/test_ssm
	./obj/test_tensor

.PHONY: prove
prove: ## Run SPARK flow analysis
	$(GNATPROVE) $(SPARK_FLAGS)

.PHONY: prove-report
prove-report: ## Run SPARK + generate HTML report
	$(GNATPROVE) $(SPARK_FLAGS) --report=all

# ══════════════════════════════════════════════════════════════════════
#  CLEAN
# ══════════════════════════════════════════════════════════════════════

.PHONY: clean
clean: ## Remove build artifacts
	$(GPRCLEAN) -P $(MAIN_GPR) $(GPR_FLAGS) || true
	$(GPRCLEAN) -P $(TEST_GPR) $(GPR_FLAGS) || true
	$(GPRCLEAN) -P $(LLM_GPR) $(GPR_FLAGS) || true
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
