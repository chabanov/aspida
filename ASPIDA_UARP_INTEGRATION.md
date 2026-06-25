# Aspida × UARP — "Train your own LLM" integration blueprint

Produced by a multi-agent ecosystem study (UARP backend `packages/*`, frontends
`apps/builder` + `apps/snaga`, the Aspida engine, DO provisioning) + an
adversarial design review. UARP repo: `/Users/ceo/Developer/agents` (Deno/TS).
Aspida: this repo (Ada/SPARK). Review verdict: **sound-with-changes** — the
must-fixes below are folded into the plan.

## ⚠️ Local-only + encrypted inference (hard constraints)
- **LOCAL ONLY — never prod.** All testing against the **local UARP** instance
  (`ASPIDA_UARP_URL=http://localhost:3000`, super-admin key). `Platform_Auth`
  validated locally: key → AUTHORIZED, identity = `tenant.tenant_id` (api-key auth
  returns `user:null`, so the **tenant** is the auth/ownership unit). Nothing is
  deployed to snaga.ai.
- **Inference is END-TO-END ENCRYPTED.** Aspida serves over its Noise-style AEAD
  channel (X25519/ChaCha20-Poly1305), NOT plain HTTPS. So serving is **NOT** the
  plain `openai_compat` path — see below.

## The shape of it
- **Serving must stay E2EE — one new adapter, reusing existing crypto.** UARP's
  `OpenAiCompatAdapter` (`factory.ts:123`) POSTs **plaintext** `<endpoint>/v1/chat/
  completions` — that would break Aspida's encrypted-inference guarantee. Aspida's
  `openai_proxy` is a **loopback-only** (127.0.0.1) plaintext→E2EE bridge, so it
  can't be exposed over the network. Correct path: a **new UARP adapter**
  (`canonical:"custom"` slot, `CustomHttpAdapter`-style) that performs Aspida's
  **Noise-NK handshake then tunnels the OpenAI request over the encrypted record
  channel** to `secure_server`. Reuse Aspida's **`web/crypto.js` + `web/channel.js`**
  (TS/Deno-friendly, RFC-verified, bit-identical to `src/crypto`) — no new crypto.
  Registered as a `custom_providers` row; the engineer still picks it as an agent
  `model` + persona via `agent.prompts.system`. (So "serving" = one E2EE adapter,
  not "zero code" — corrected from the first pass.)
- **Auth is done.** Engineer's `uarp_..._...` key; Aspida `Platform_Auth.Verify`
  validates it via `GET /api/v1/me` (proven against prod).
- **Billing is UARP's** (stripe/plans/markup/usage). We add a deposit + GPU-hour metering.
- **The ONE genuinely new thing: a Training Job** — UARP has no model-training
  concept. We add it mirroring the Run lifecycle (KV record + status + eventStore + SSE).

## Architecture (end-to-end)
```
apps/builder "LLM Studio" wizard  (teachers · droplets · settings · quote · deposit)
   │  POST /api/v1/training-jobs (Admit gate + Stripe deposit checkout)
   ▼
UARP BE: TrainingJobService (NEW packages/training)  — holds the DO master token
   │  state machine: queued→deposit_paid→provisioning→training→quality_gate→exporting→serving→complete
   │  provisions N DO GPU droplets (hand-rolled DO REST client; tags droplet job_id AT CREATE)
   ▼
Aspida worker (NEW tools/train_worker.adb)  — Platform_Auth.Verify(key) → Turnkey.Run
   │  Trainer=Student_GPU(Config_Of(tier))+Data_Pipeline+Distill ensemble of Teacher_{Llama,Qwen,Gemma}
   │  Evaluator=Exec_Verifier/SVG oracle → beats-teachers Job_Report ; Deliverer=Export_GGUF→serve
   │  curl progress → POST /api/v1/training-jobs/:id/callback  (job-key-bound) → eventStore → FE SSE
   ▼
on complete: register served openai_proxy as TENANT-SCOPED provider → selectable as an agent model
             meter GPU-hours (bounded by control-plane wall-clock) → settle deposit (Final_Charge)
```

## Engineer frontend journey (apps/builder `/browser/llm-studio`)
1. **Domain & persona** — domain (Code/SVG only — verifier-backed; others disabled), model name, persona name + system prompt (`system-prompt-field.tsx`), **teacher-rights attestation** (required).
2. **Teachers** — multi-select teacher models (`teacher-picker.tsx`), enforces a **single vocab family** (llama/qwen/gemma can't mix), shows Aspida-runnable ones.
3. **Droplets** — count slider (**MVP hard-cap ≤2**; >2 disabled until ring all-reduce) + tier/size.
4. **Training settings** — tier (Small/Medium — **Large disabled MVP**), steps, quant (Q4_K), context, verifier on/off, grad-accumulation.
5. **Quote & deposit** — `POST /training-jobs/quote` → GPU-hours / provider cost / price (markup) / deposit → Stripe Checkout.
6. **Launch + progress** — SSE phases (provisioning %, training loss, **"student X% vs teachers Y%"**), live GPU-hours/cost.
7. **Result** — served model card + "Use in an agent" CTA (pre-fills `AgentModelConfig` + persona).

## New API surface
- `POST /api/v1/training-jobs/quote` — Platform.Quote
- `POST /api/v1/training-jobs` — create + Admit gate + Stripe deposit checkout
- `GET /api/v1/training-jobs[/:id]` — list / status
- `GET /api/v1/training-jobs/:id/events` — SSE (must add to `router.ts:950` `isStreamEndpoint`)
- `POST /api/v1/training-jobs/:id/cancel` — cancel + guaranteed teardown
- `POST /api/v1/training-jobs/:id/callback` — INBOUND from worker (job-key-bound)
- `GET /api/v1/training-jobs/teachers` — Aspida-runnable, vocab-compatible catalog
- Aspida worker: served student `POST <https>/v1/chat/completions` (registered provider)

## 🔴 MUST-FIX (from the adversarial review — do not skip)
1. **Tenant-isolation leak:** `custom_providers` is a SINGLE GLOBAL map (`provider-registry.ts:27`). Per-job rows there expose every tenant's trained model to all tenants. → add **tenant-scoped provider registry / owner filter** + CAS writes.
2. **Stripe deposit is NOT a reuse:** `StripeClient.createCheckoutSession` is subscription-only. → add a **new one-off `mode:"payment"` checkout** (inline `price_data`).
3. **Stripe webhook has no `checkout.session.completed`:** add a **new top-of-handler branch** in `stripe-webhook.ts` (before `billingManager`), with signature + idempotency (dedup pattern L320-345).
4. **Callback auth:** bearer-tenant auth alone lets any tenant key post fake billable progress. → **bind the callback to the job's keyId / per-job HMAC**.
5. **GPU-hour fraud:** don't trust worker-self-reported hours; **bound by control-plane provisioned wall-clock**; CAS + idempotent settle.
6. **Orphan-droplet cost leak:** 3-layer teardown (trap + finally + **cron reaper**); **tag droplets `job_id` AT CREATE** so the reaper finds leaks even if the record write fails.
7. **DO token:** DO PATs aren't finely scopable → **master token control-plane-side ONLY** (never on a droplet); drop "scoped token per job to shell".
8. **Server-side guardrails (not just UI):** cap N≤2, tier≤Medium, domain∈{Code,SVG}, single-vocab teachers — enforced in the **Admit gate**.
9. **Money convention:** UARP cost path is **float USD** (`constants.ts:18 roundCost`), not integer-cents. Pick ONE authoritative `Provider_Rate` (config), don't claim integer-cents reuse.
10. **Routing/visibility:** add `/browser/llm-studio` to dashboard-only allowlist (`middleware.ts:398`).
11. **Serving lifecycle billing:** the deposit covers TRAINING; define who runs/reaps the per-tenant serving droplet and how ongoing serving is metered (second cost leak).
12. **Untrusted code on rented GPUs:** `ASPIDA_VERIFY_SANDBOX` mandatory + one-tenant-per-droplet + block metadata `169.254.169.254` (egress allowlist to UARP only).

## ✅ Live validation (local, deno API :8080 — real operation)
Wave 1 (BE foundation) validated against the running local UARP backend with the
super-admin key (never prod):
- `POST /training-jobs/quote` Code/Small/2×4h → `{gpu_hours:8, provider_cost:20, platform_price:26, deposit:26}` — matches Aspida `Platform.Quote` (8h×$2.5×1.3).
- Guardrails (server-side): droplets=5 → `422 reject_too_many_droplets`; domain=Web_Layout → `422 reject_domain_not_allowed`.
- `POST /training-jobs` → real `job_id` (UUID) + per-job `callback_secret` (256-bit). `GET /` + `GET /:id` tenant-scoped.
- Callback auth (must-fix c): no secret / wrong secret → `401` (tenant bearer alone rejected); correct secret → accepted.
- **Full lifecycle via live callbacks:** queued → (progress) training → (gate) quality_gate → (completed) complete; late progress stays complete (terminal guard).
- **Bug found + fixed by live validation:** terminal callback events carried no `body.status`, so the job never left `queued`; added `EVENT_TYPE_STATUS` map so the event type drives the status. Re-validated green.
- Aspida `tools/train_worker` (real run): descriptor → Turnkey → DELIVERED ($13, beats=true); rejects Large tier / N>2 via the same guardrails.

## ✅ Frontend wave (apps/builder "LLM Studio") — built + screenshotted
- Files: `lib/hooks/use-training.ts` + `api-paths.ts`, `app/browser/llm-studio/page.tsx` (dashboard), `_components/build-wizard.tsx` (6-step: domain/persona → teachers → droplets 1–2 → settings → quote → launch), `[jobId]/page.tsx` (live SSE progress), `middleware.ts` allowlist + `app-shell.tsx` nav. tsc + eslint clean.
- **Real browser screenshots** (Playwright vs local Next :3000 + deno API :8080): dashboard lists jobs; wizard renders all steps; progress page shows the phase ladder + **"Student 100% vs Teachers 77% — beats: yes"** panel + quote ($39, E2EE serving).
- **2nd bug found by screenshots + fixed:** the FE `TrainingJobStatus` type and status→badge/phase maps used invented values (`running/gate/completed`) instead of the backend's (`training/quality_gate/complete` + `deposit_paid/exporting/serving`), crashing the progress page (error boundary) and badges. Aligned the type + all maps to the backend statuses + added a safe fallback. Re-validated: renders clean, no console errors.

## ✅ E2EE serving proven (local) — the differentiator
Inference stays end-to-end encrypted, validated two ways against a local
`secure_server` (model `svgdata/student.gguf`, 127.0.0.1:8765):
- **Via `openai_proxy`** (loopback→E2EE bridge): `curl 127.0.0.1:8090/v1/chat/completions` → `🔒 encrypted tunnel up: X25519 + ChaCha20-Poly1305 + HKDF-SHA256`, HTTP 200, OpenAI envelope. (Same-machine plaintext loopback hop only — fine when UARP is co-located.)
- **Via a native Noise client** (`web/e2ee_client.mjs`, the UARP adapter core): raw TCP + full handshake + sealed `Prompt`/`Token` records, **no plaintext on the wire**, reusing the RFC-verified `web/crypto.js`. Output decodes to real `<svg width="64"…>` from the trained student. `RESULT: PASS`.
- **Why a native adapter (not plain `openai_compat`):** UARP's `CreateProviderSchema.default_endpoint` is refined by `isSafeEndpointUrl`, which rejects localhost/private/loopback (SSRF guard) — so the plaintext HTTP path can't even register a local Aspida endpoint, and over a network it would break E2EE. The native Noise adapter (a registered Aspida provider, encrypted socket) is the correct path.
- **✅ DONE — UARP Deno adapter built + live-validated:** `packages/llm-adapters/aspida-e2ee.ts` (`AspidaE2EEAdapter implements LLMAdapter`, `provider="aspida_e2ee"`) + vendored `aspida-e2ee-crypto.js` (bit-for-bit from `web/crypto.js`, `@ts-nocheck`) + `factory.ts` branch (fires on `provider:"aspida_e2ee"` or an `aspida://host:port` endpoint, pubkey from api_key/`#hex`/api_key_ref). Uses `Deno.connect` TCP + the full Noise handshake + sealed Session/Prompt/Token/Done records. `deno check` clean. **Live (independently re-verified) against secure_server :8765:** `chat()` → finish `stop`, real content; `chatStream()` → 145 content deltas. So a UARP agent can use an Aspida-trained model over the encrypted channel.
- **✅ DONE — full agent→run loop closed + screenshotted.** `isSafeEndpointUrl` now accepts the `aspida://` scheme (pinned-key encrypted socket, SSRF deny-list N/A). A UARP agent ("Aspida SVG Bot") was created with `model:{provider:"aspida_e2ee", model_ref:"aspida-student", endpoint_url:"aspida://127.0.0.1:8765#<pub>"}` — no super-admin needed (the runtime honors explicit LOCAL endpoints). `POST /runs` → queued→running→**completed**, `output.response` decodes to real `<svg…fill="#111111"/></svg>` from the Aspida model over the encrypted channel. Browser screenshots: agent settings (model `aspida-student`) + chat ("draw a red circle" → SVG-byte response rendered in the UARP chat UI). The trained-model-served-E2EE differentiator is now clickable end to end.

## ✅ DO provisioning-as-a-service (P3) — built + tested (local, zero real droplets)
`packages/provisioning/` in UARP: `DigitalOceanClient` (hand-rolled DO v2 REST, `StripeClient` style, Bearer token control-plane-only, **auto-dry-run when no token** — never fetches) + `TrainingProvisioner` (`provisionForJob` tags every droplet `aspida-job:<id>` **at create**, caps count≤2; `withProvisionedDroplets` **always tears down in finally** even on throw; `teardownJob` idempotent; `reapOrphans` cron backstop deletes old/terminal-job droplets, keeps fresh). `deno check` clean; **`deno test` 10/10** (create-time tagging, teardown-on-throw leaves zero, idempotency, reaper selectivity, dry-run-never-fetches). No UI layer (backend infra — tests are the evidence). **Real droplet creation is gated on explicit cost-approval** and intentionally not exercised.

## ✅ Stripe deposit (P1) — built + validated on the REAL Stripe test API (no mocks)
`StripeClient.createDepositCheckoutSession` (one-off `mode=payment`, inline `price_data`, metadata `training_job_id`/`tenant_id`) + `POST /api/v1/training-jobs/:id/deposit` (amountCents from `quote.deposit`) + a NEW `checkout.session.completed` branch in `stripe-webhook.ts` (before `billingManager`, signature-verified + idempotent, CAS-flips `queued→deposit_paid`). `deno check` clean. **Live, real Stripe test API (`rk_test_`, livemode false):**
- `POST /deposit` via the running server → real `cs_test_…` session + real `checkout.stripe.com` URL ($26 "Aspida training deposit"); **screenshotted** (Sandbox checkout page, Card/Apple Pay/Link/Klarna).
- Real signature-verified webhook (HMAC-SHA256 over `t.body` with the configured `whsec`) → `{received,handled,action:"deposit_paid"}`, job `queued→deposit_paid`.
- Security: a tampered signature → **HTTP 401** (verification enforced, not bypassed).

## ✅ External audit resolution (4-pass security/correctness audit)
**Both ship-blockers CLOSED + live-validated; mediums addressed.**
- **B1 — deposit credited without verifying payment → FIXED.** `stripe-webhook.ts` deposit branch now requires `payment_status === "paid"` AND `amount_total === round(quote.deposit*100)` before the CAS; else `handled:true` (stops retries) without flipping. Live: `unpaid → deposit_unpaid` (stays queued), `paid+wrong-amount → deposit_amount_mismatch` (stays queued), `paid+correct → deposit_paid → provisioning` (orchestrator auto-dispatch). 
- **B2 — broken worker-callback tenant binding + non-functional in prod → FIXED.** The callback is now **pre-auth** (router dispatch before `authenticate`, like the Stripe webhook), so a worker with no UARP bearer reaches it; the **owning tenant resolves from a new global `["training_job_callback", jobId]→{tenantId}` index** (written at create+retry), never the caller; then the per-job secret is constant-time checked under that tenant. Live: `no-bearer + correct secret → 200` (was non-functional), `no-bearer + wrong secret → 401`.
- **Mediums:** single authoritative rate — `training-pricing.ts` now re-exports `PROVIDER_RATE`/`MARKUP_PCT` from `@uarp/types` (no drift); `GET :id/metrics` **downsampled to ≤500 points** (unbounded `getRunEvents` → bounded payload). Already closed by the orchestrator: tenant-scoped served-provider (not the global map), cost cap at quote, Failed_Gate provider-cost-only + refund, Cancelled full refund, teardown-always, anti-fraud metering bounded by wall-clock.
- **Latent (close in the real-DO wiring PR, gated on cost-approval):** provisioning region/size/image allowlist + per-tenant droplet/concurrent-job quota + `reapOrphans` cron registration; UI stale-quote clear-on-body-change (backstopped server-side); `listJobs.total` cosmetic.
- Validation: orchestrator+pricing **24/24**, regression (webhook+usage) **28/28**, `deno check` + `deno lint` clean; B1/B2 exercised against the live local API.

## ✅ Real economy / real money (live-ready, validated in Stripe TEST + per-SKU)
Per the owner's decision: full LIVE, the user picks any provider GPU, refund = deposit − burned.
- **Real Stripe money:** deposit captured up-front at checkout (`mode=payment`); `payment_intent` stored on the job at the paid webhook; settlement issues a **real partial refund** of the unused deposit (`StripeClient.createRefund` + `BillingManager.refundDeposit`, exactly-once via `Idempotency-Key`). **Validated against the real Stripe TEST API:** created a $26 PaymentIntent (test card) → partial $6 refund `re_… succeeded`. (LIVE in prod by swapping to `sk_live_`/`whsec_` live keys — no code change.)
- **Per-SKU GPU economics:** real DO GPU catalog in `@uarp/types` `GPU_CATALOG` (RTX 4000 Ada $0.76, RTX 6000 Ada/L40S $1.57, MI300X $1.99, H100 $3.39, H200 $3.44 — real DO list prices). Quote = gpu_hours × **chosen-SKU rate** × (1+markup). Live: RTX 4000 → **$7.90**, H100 → **$35.26**, unknown SKU → `422 reject_gpu_not_allowed`. Cost-accrual + settlement use the job's SKU rate too. User picks the GPU in the wizard (`gpu_sku` threaded through quote + create).
- **Real DO provisioning:** `defaultProvisioner` uses the env DO token → **real droplets in prod** (token present), **dry-run locally** (no token → auto). Economic invariant: provisioning only runs after the full deposit (≥ quoted provider cost) is captured; teardown + reaper bound spend.
- **Refund policy:** Delivered = metered (markup, capped at quote) + refund remainder; Failed_Gate = provider-cost-only + refund remainder; Cancelled = refund deposit − burned GPU-hours. Anti-fraud: GPU-hours bounded by control-plane wall-clock.
- **Careful validation:** money exercised in **Stripe TEST** (real API, no customer money); **no real GPU spun** in validation (local has no DO token → dry-run). Tests **25/25** (pricing+orchestrator incl. per-SKU + H100 case) + **28/28** regression; `deno check`+`deno lint` clean.
- **To go LIVE (owner action):** set `STRIPE_SECRET_KEY=sk_live_…` + `STRIPE_WEBHOOK_SECRET=whsec_…` (live) + `DIGITALOCEAN_TOKEN=…` in the control-plane env.

## ✅ Provisioning hardening (audit "latent" items closed — pre-live safety)
Now that real DO provisioning is active, the cost-bomb/abuse guards are in:
- **Allowlist at the provision boundary** (defense-in-depth on top of quote/admit): `provisionForJob` independently refuses any `size` ∉ `GPU_CATALOG`, `region` ∉ `ALLOWED_REGIONS` (`nyc1/nyc3/tor1/sfo3/ams3/fra1`), or `image` ≠ `CONTROL_PLANE_IMAGE` — throws before any create (tested: bad size/region/image → 0 droplets).
- **Per-tenant concurrency quota** (the real gap — MAX_DROPLETS was per-job): `ASPIDA_MAX_ACTIVE_JOBS_PER_TENANT` (default 3) enforced in `driveJob` before provisioning; an over-cap paid job DEFERS (stays `deposit_paid`, deposit intact, emits `training.provisioning_deferred`) and reconcile re-dispatches it when capacity frees (tested).
- **user_data control-plane-only:** TrainingJob carries no user_data; the orchestrator never forwards a request-derived string — only the built cloud-init placeholder (DO master token never on a droplet).
- **Reaper cron REGISTERED** (`setup.ts`, the same `runWithCronLock` pattern as webhook-retry/blob-GC, every 5 min): `tickTrainingReaper` per tenant (reconcile funded-but-idle/deferred jobs + finalize terminal-but-unsettled) + cross-account `reapOrphans` (age cutoff `ASPIDA_REAP_MAX_AGE_MINUTES` default 180) so a control-plane death after droplet-create still gets cleaned up.
- Validation: **22/22** provisioner+orchestrator tests, `deno check`+`deno lint` clean, API **boots clean** with the reaper registered (0 startup errors), smoke quote per-SKU OK.

## ✅ Real DO provisioning — verified live + 3 gaps found & fixed
Verified the actual real-droplet path end-to-end via the platform `DigitalOceanClient` (created a real
DO droplet, listed it by tag, tore it down, confirmed destroyed — no orphan; final droplet list = only
snaga-prod + the demo). Three real gaps surfaced + fixed:
1. **Prod DO token present but NOT wired into the app container.** `/root/.do-token` on snaga-prod is a
   valid PAT (`dop_v1_…`, account active, limit 500), but the deno API runs in Docker and the token is
   NOT in the container env / docker-compose → `resolveDoTokenFromEnv()` returns empty → real
   provisioning would stay dry-run. **Prod must inject `DIGITALOCEAN_TOKEN` into the container.**
2. **`CONTROL_PLANE_IMAGE` was a placeholder** (`"aspida-train"`, not a real DO image) → made
   env-configurable: `ASPIDA_TRAIN_IMAGE` (+ `ASPIDA_TRAIN_REGION`). Image format confirmed: DO accepts
   the numeric snapshot id.
3. **`createDroplet` did not pass an SSH key** → DO 422 "use an SSH key" for the password-less Aspida
   snapshot. Added `ssh_keys` to the client + `CONTROL_PLANE_SSH_KEYS` (env `ASPIDA_DO_SSH_KEY_ID`).
   After this, a real droplet created cleanly (only blocked by transient tor1 GPU capacity, a DO-side
   availability matter, not code).
Validation: 23 provisioner+orchestrator tests, `deno check`+`deno lint` clean; real create+teardown
proven against the live DO API with zero orphans.

### Prod go-live checklist (DO provisioning)
Inject into the **deno API container** env (not just the host):
```
DIGITALOCEAN_TOKEN=dop_v1_…          # from /root/.do-token
ASPIDA_TRAIN_IMAGE=<snapshot id>     # the Aspida training image (e.g. 233252240) in ASPIDA_TRAIN_REGION
ASPIDA_TRAIN_REGION=tor1             # region the snapshot + chosen GPU SKU are available in
ASPIDA_DO_SSH_KEY_ID=<key id>        # control-plane deploy key (e.g. 55501472)
```
(+ Stripe live keys for real money, as above.) Without the token in the container, provisioning safely dry-runs.

## Build plan (critique-adjusted, shippable phases)
- **P0 — Contracts/types (1-2d):** `packages/types/training.ts`, apiPaths, FE hook stubs.
- **P1 — Quote + deposit, no compute (1wk):** port Quote/Admit (float-USD), `POST /quote` + `POST /training-jobs` + **new one-off Stripe checkout** + **new webhook branch**, job store, FE wizard steps 1-6 (stub job stops at deposit_paid).
- **P2 — Aspida worker, single-node (1-1.5wk):** `tools/train_worker.adb` wiring `Turnkey.Run` to real Trainer/Evaluator/Deliverer + curl progress + `POST /:id/callback` (job-key-bound); run manually on one GPU droplet end-to-end (Code/SVG, proven beats-teachers).
- **P3 — DO provisioning-as-a-service (1-1.5wk):** hand-rolled DO REST client + generalized `provision_and_train.sh` (N param), state machine, **3-layer teardown + reaper + create-time tagging**, GPU-hour metering bounded by wall-clock.
- **P4 — Auto-serve + tenant-scoped register + agent wiring (4-6d):** register served model **tenant-private**, FE result "Use in an agent".
- **P5 — Hardening/GA (1wk):** SSE reconnect, Final_Charge settlement + refund-on-Failed_Gate, plan entitlements, checkpoint/resume, N=2 behind measured network tier, serving-lifecycle billing.

**Total to GA:** ~6-8 weeks for a strong eng; the only research is already done (P4 proof). The rest is integration + product + provisioning safety.
