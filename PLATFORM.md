# Aspida Training Platform — blueprint

A turnkey **train-your-own-LLM** platform on top of the Aspida engine. An
engineer picks **teacher models** and a **domain**, picks how many **GPU
droplets** to rent, pays a **deposit**, and receives a **personal LLM** that —
**on domains with a sound executable verifier** — **matches or exceeds the
teacher ensemble** on a held-out evaluation, served through Aspida's
end-to-end-encrypted inference (trusted-server topology — see §Security).

> **Scope of the guarantee (revised after engineering review).** A student can
> exceed its teachers ONLY where an **executable verifier** supplies signal the
> teachers' distributions lack: the student trains on verified-correct outputs
> and self-improves (STaR), so its ceiling is the **verifier's** quality. This
> holds where the verifier is sound (resists reward-hacking) and the held-out
> eval is disjoint from training. It does NOT hold for domains without a real
> verifier (SVG/web/multilingual today have none) or for from-scratch students
> lacking base competence. Proven only at TOY scale so far (`code_distill`:
> 100% vs 40% on a 3-token DSL; `code_iterate`: random → full coverage). **The
> commercial promise is gated on a real-domain proof (Step 2) and a rigorous
> delivery gate** (`Platform.Make_Report`: verified domain + held-out N ≥ 50 +
> student ≥ teacher + margin). If the gate fails, the platform charges
> provider-cost only (no margin) — `Platform.Final_Charge`.

> **Teacher selection constraint:** ensembled teachers must share a tokenizer/
> vocabulary (Llama, Qwen, Gemma do NOT) — `Distill.Vocab_Mismatch` enforces it.
> So "pick any teachers" means any with a compatible vocab, or a single teacher.

## Job lifecycle

```
engineer: { teachers[], domain, student tier, persona/copyright, droplets }
   │  Platform.Quote → { provider cost + markup → price → deposit }
   ▼  deposit paid
provision N droplets  ──►  multi-teacher verifier-driven distillation
   │                         (ensemble capture → verify-filter → train → STaR)
   │  quality-gate: held-out eval  →  Student_Pass > Teacher_Pass  (the promise)
   ▼  export GGUF
deliver: student served over Aspida E2EE inference  →  endpoint + keys to engineer
   meter GPU-hours · settle against deposit
```

## Components (have / partial / to-build)

| Layer | Contract | State |
|---|---|---|
| Quality core | multi-teacher ensemble + verifier-filter + STaR | ✅ proven |
| Teacher registry | any LLM as `Distill.Teacher` (adapters) | ✅ Llama/Qwen/Gemma |
| Domain verifiers | `Verifier` interface (code/SVG/web/lang) | 🔶 code ✅, rest to-build |
| Persona / copyright | student identity + system behaviour (SFT layer) | ⛔ |
| GPU training | resident loop (Stage 1) → multi-droplet (2–4) | 🔶 Stage 1 loop ✅ |
| Delivery | export → E2EE serve + endpoint | ✅ proxy/secure_server |
| Quality gate | held-out "beats teachers" eval | 🔶 verifier ✅, harness to-build |
| **Control plane** | `Platform.Job_Spec` → `Quote` → deposit → provision → run → `Job_Report` | ✅ contract+pricing+gate (Step 1) |
| **Verifier sandbox** | microVM/container, non-root, no-net, rlimits — for untrusted code | ⛔ **blocker before multi-tenant** |
| Provisioning / billing | droplet API, authenticated metering, escrow, cap kill-switch, teardown | ⛔ |
| Legal / licensing | teacher-license attestation, IP terms, honest E2EE claims | ⛔ |

## Pricing model
`provider_cost = droplets × hours × provider_rate` → `price = cost × (1+markup)`.
**All arithmetic is exact fixed-point (no Float).** Job carries a hard
`Max_Spend` cap; `deposit = min(price, cap)` (escrow). Outcome charging
(`Platform.Final_Charge`): *Delivered* → metered price capped at `Max_Spend`;
*Failed_Gate* → **provider cost only, no margin**; *Aborted_Cap* → the cap.
`Hours_Per_Drop` is a customer estimate today — to be replaced by a
platform-produced time-to-quality model (see review). See `src/train/platform.ads`.

## Security (mandatory before multi-tenant / rented droplets)
The quality engine **runs model-generated code** (`Exec_Verifier`). In-process
hardening: unique scratch path, direct exec (no shell), output suppressed,
wall-clock `timeout`. **Pluggable isolator (built):** set `ASPIDA_VERIFY_SANDBOX`
to a sandbox command prefix and **every execution is wrapped by it** (mechanism
validated with a benign `/usr/bin/env` wrapper). **Required before any
multi-tenant / rented-droplet use,** set it to a hard isolator, e.g.
```
ASPIDA_VERIFY_SANDBOX='firejail --quiet --net=none --private \
  --rlimit-cpu=10 --rlimit-as=536870912 --seccomp'
# or a container wrapper:
ASPIDA_VERIFY_SANDBOX='docker run --rm -i --network=none --read-only \
  --user 65534 --memory=512m --pids-limit=64 --cap-drop=ALL aspida-verify'
```
providing: non-root, read-only rootfs, fresh tmpfs, **no network + blocked
cloud-metadata 169.254.169.254**, rlimits/cgroups, seccomp. **Built + validated:**
`tools/verify_sandbox.sh` (preinstalled util-linux only — `unshare -n` + `setpriv`
nobody + `timeout -KILL` + ulimits) — verified on Linux to block network egress
incl. metadata, run as nobody, and kill runaways at the timeout; set
`ASPIDA_VERIFY_SANDBOX=tools/verify_sandbox.sh`. One tenant → one
dedicated droplet/GPU while untrusted code runs. Per-tenant at-rest encryption of teacher uploads, data,
checkpoints and the student GGUF; crypto-shred on offboard. Provisioning
credentials (DO token, server static key) must NOT live on the execution host;
issue short-lived, narrowly-scoped tokens. Per-tenant key material for delivery
(not the single shared `ASPIDA_CLIENT_TOKEN`). E2EE claims carry the honest
scope (trusted-server topology; AEAD/MAC not yet field-proved — SECURITY.md).

## Legal / licensing
Distilling third-party teacher models may violate their ToS, and a distilled
student's IP/copyright is unsettled — so "personal LLM with its own copyright"
must be qualified and counsel-reviewed. Require per-teacher **license/provenance
attestation** before a job runs; surface each teacher's use-restrictions.

## Step-by-step build plan (revised by review — MVP-first, gate the promise early)
1. **Control-plane contract + pricing + rigorous gate** (`Platform`): exact money, cap, failed-job policy, `Make_Report` gate — *in-tree, tested*. ✅ **done**
2. **Real-domain "beats teachers" proof + eval harness** — the **go/no-go gate**: HumanEval/MBPP-style real code, held-out tests disjoint from training, *fairly-prompted* teacher baseline. If a properly-baselined teacher isn't beaten at tier-Small on the one domain with a genuine verifier, the premise is falsified before any GPU billing. ◀ **next — highest-risk assumption**
3. **Verifier sandbox** (security blocker before running untrusted code at scale).
4. **Persona/copyright + domain onboarding** (who supplies data/prompt distribution) + **teacher-license attestation**.
5. **GPU training — finish Stage 1, single-node first**: resident FFI (kill per-op round-trips), wire `Train_GPU` into `Student.Step` (resolve FP32/FP64), then SwiGLU/RMSNorm/RoPE/attention → full resident `Student`.
6. **MVP turnkey on 1 (sandboxed) droplet**: provision → run job → gate → deliver served student.
7. **Scale**: single-node multi-GPU (data-parallel + all-reduce) — **multi-node only after a measured all-reduce benchmark proves positive scaling on droplet networking** (no fast interconnect ⇒ likely negative scaling past a few droplets); then sharding; full escrow/provisioning/checkpoint-resume automation.

Each step ships built + validated + documented before the next.

## Review outcome (engineer-agent panel)
A 4-engineer review (ML, distributed-GPU, security, product) did **not** approve
the original plan as written. Incorporated: rescoped/conditional guarantee +
rigorous gate; exact-money + cap + failed-job billing; security/sandbox +
tenancy + legal sections; multi-server gated behind a measured benchmark; and
the **real-domain beats-teachers proof pulled forward as the go/no-go gate**.
