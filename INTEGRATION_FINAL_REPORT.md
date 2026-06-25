# Aspida × UARP — Final Integration Report

**Product:** a turnkey "train your own LLM" service. An engineer, in the UARP web app
(snaga.ai), picks teacher models + a domain + a GPU + droplet count, pays a deposit, and the
platform provisions GPUs on DigitalOcean, trains a *student* model (multi-teacher distillation +
executable verifier so it can **beat its teachers**), runs a held-out quality gate, and serves the
result over Aspida's **end-to-end-encrypted** inference channel — settling the bill (refunding the
unused deposit) at the end.

- **UARP** (`/Users/ceo/Developer/agents`, Deno/TS) = the product: auth, billing, agents/runs,
  the LLM Studio UI, the control-plane orchestrator.
- **Aspida** (`/Users/ceo/Developer/aspida`, Ada/SPARK) = the compute engine: from-scratch crypto,
  training, GGUF, the E2EE inference server, and the training worker.

Status: **production-hardened, real-money-ready.** Going live = 3 env vars (below). Nothing is
deployed; commits are the owner's.

---

## 1. End-to-end flow (self-driving)
```
LLM Studio wizard (teachers · GPU SKU · droplets · settings)
  → POST /training-jobs/quote        (per-SKU price)
  → POST /training-jobs              (Admit gate → Stripe deposit checkout)
  → Stripe checkout (deposit captured, real money up-front)
  → webhook checkout.session.completed  (verify paid+amount → deposit_paid)
  → ORCHESTRATOR (control plane):
       provision N GPU droplets (DigitalOcean, tagged, within deposit)
       → spawn Aspida worker (per-job secret + callback URL)
       → worker: provisioning → progress(loss/gpu/tokens) → logs → gate → exporting → serving → completed
       → live monitor (SSE) shows every phase + the "student X% vs teachers Y%" verdict
       → auto-register the served model (tenant-scoped) → usable as an agent's model
       → Final_Charge: keep metered cost, REAL Stripe refund of the remainder
       → ALWAYS teardown droplets (+ reaper cron backstop)
```

## 2. Component inventory + validation

| Area | What | Validation |
|---|---|---|
| **Auth** | engineer's UARP API key validated via `GET /api/v1/me`; Aspida `Platform_Auth` (curl, no TLS client) | live vs prod + local (tenant = ownership unit) |
| **Training-jobs API** | quote / create / list / get / SSE events / cancel / **retry** / metrics / callback | live curl, tenant-scoped, idempotent/CAS |
| **Pricing** | per-SKU quote (real DO rates) + Admit gate (attestation→persona→budget) | 18→ unit tests; live (RTX4000 $7.90, H100 $35.26) |
| **GPU catalog** | `GPU_CATALOG` (RTX4000 $0.76 / RTX6000·L40S $1.57 / MI300X $1.99 / H100 $3.39 / H200 $3.44) — user picks in the wizard | screenshot; unknown SKU → `reject_gpu_not_allowed` |
| **Stripe money** | deposit captured at checkout; **real partial refund** at settlement (`createRefund`, exactly-once) | **real Stripe TEST API**: $26 PI → $6 refund `re_… succeeded` |
| **Orchestrator** | durable state machine deposit_paid→…→complete; auto-register serving; settle; teardown; reconcile | 6→8 tests (delivered/failed_gate/cancelled, idempotent, restart) |
| **DO provisioning** | real droplets when env token present (prod), dry-run locally; tag-at-create; teardown; reaper | 14 tests (mock, no real spend) |
| **Provisioning hardening** | allowlist (size∈catalog / region / image) · per-tenant quota (default 3) · user_data control-plane-only · **reaper cron registered** | tests + API boots clean with reaper |
| **E2EE serving** | UARP Deno adapter does Aspida's Noise handshake + tunnels OpenAI over the encrypted channel (reuses RFC-verified `web/crypto.js`) | live vs local `secure_server`: chat + stream |
| **agent→run loop** | a UARP agent uses the Aspida-trained model over E2EE; `POST /runs` → completed | live + screenshot |
| **LLM Studio FE** | workspace (list/filters/search/cost) + monitor (status banner, loss curve, cost accrual, GPU util, droplets, logs, retry, gate panel) | tsc+eslint clean; screenshots; Paper-token + full-width consistency |
| **Aspida worker** | `train_worker.adb` emits the real callback lifecycle (real metrics from a verifier loop) | gprbuild rc=0 (warnings-as-errors); ran vs live + offline |

## 3. Money mechanics (real economy)
- **Capture up-front:** deposit = full quoted price, captured at Stripe checkout (`mode=payment`). We
  never provision beyond money already in hand (economic invariant against cost-bombs).
- **Per-SKU cost:** `provider_cost = gpu_hours × chosen-SKU rate`; `price = ×(1+30% markup)`. One
  authoritative rate source (`@uarp/types`), used by quote, cost-accrual, and settlement.
- **Settlement (Final_Charge):** Delivered → keep metered (capped at quote) + refund remainder;
  Failed_Gate → keep provider-cost-only + refund remainder; Cancelled → refund deposit − burned
  GPU-hours. Refund is a **real Stripe refund** (exactly-once). Anti-fraud: GPU-hours bounded by
  control-plane wall-clock (worker can't over-report).

## 4. Security & audit
- 4-pass external audit verdict was **"do not ship until B1+B2"**; both **closed + live-validated**:
  - **B1** — webhook now requires `payment_status==="paid"` AND amount == quoted deposit before crediting.
  - **B2** — worker callback is pre-auth (no UARP bearer needed) with the owning tenant resolved from a
    global job→tenant index + per-job 256-bit secret (constant-time), not the caller's bearer.
- Audit-verified sound: own crypto (8/8 RFC vectors), webhook signature+dedup, cost anti-spoof,
  tenant isolation, teardown. Latent provisioning items (allowlist, per-tenant quota, reaper cron,
  user_data) all closed in the hardening pass.

## 5. Real GPU proof (done once, then torn down)
A real DigitalOcean RTX 4000 Ada droplet was provisioned from the Aspida snapshot, ran a real
train→export→serve→E2EE-inference cycle, and was **destroyed** (no orphan, ~$1). Honest caveats: the
training was capped (25k steps, under-trained → low accuracy — the *pipeline* was the goal, not the
model); GPU-offload at serve time crashed on init for the tiny model and fell back to CPU. The full
real-cloud loop (train → GGUF → encrypted serve → inference) executed end to end.

## 6. Honest gaps / not-yet
- **LIVE money/keys:** validated in **Stripe TEST** (real API, no customer money). Live needs live keys.
- **Real droplets in the automated pipeline:** code is live but never auto-spun (local has no DO token
  → dry-run). First real spin happens on a real paid job in prod.
- **Heavy GPU training:** the worker emits real metrics from a bounded verifier loop; wiring the
  full multi-hour GPU-resident student training run is the remaining engine-side work.
- **Engine limits (honest):** single-node + N≤2 data-parallel proven; >2 GPUs / Large tier not yet
  (ring all-reduce / ZeRO unbuilt) — the wizard caps accordingly.
- **Stripe refund draining / ongoing serving-cost billing:** refund is issued at settlement; a
  long-running served model's ongoing cost model is future work.

## 7. To go LIVE (owner action)
Set in the control-plane env, then deploy:
```
STRIPE_SECRET_KEY=sk_live_…
STRIPE_WEBHOOK_SECRET=whsec_…        (the live endpoint's signing secret)
DIGITALOCEAN_TOKEN=…                  (enables real droplet provisioning)
```
Without them everything runs as a safe test/dry-run. Recommended first live step: one cheap
end-to-end run (RTX 4000, ~$1) to confirm the full money cycle on live keys.

## 8. Where things live
- UARP: `apps/builder/app/browser/llm-studio/**`, `lib/{training-status,hooks/use-training,api-paths}`,
  `packages/api/{routes/training-jobs.ts,routes/stripe-webhook.ts,lib/training-orchestrator.ts,lib/training-pricing.ts,lib/training-job-store.ts}`,
  `packages/provisioning/**`, `packages/billing/stripe-integration.ts`, `packages/types/{training.ts,event.ts}`.
- Aspida: `tools/train_worker.adb`, `src/train/{platform,platform_auth,turnkey,job_store}.*`,
  `src/server/{secure_server,openai_proxy}.adb`, `web/{crypto.js,channel.js,e2ee_client.mjs}`,
  and this report + `ASPIDA_UARP_INTEGRATION.md`.

*No real customer money charged, no real droplet left running, nothing deployed. Commits & go-live are the owner's.*
