# Aspida engine — deployment status (2026-07-23)

## Deployed on prod (H200): P1 instrumentation only
The running `libaspidagpu.so` is the **P1/P1b** build (abort-poisoning
instrumentation: `ASPIDA_OP_TRACE`-gated op-trace + SIGABRT stage attribution +
checked `upload_weight`). `main` reflects this: it builds to the deployed state.

## P2/P3 — committed then REVERTED (present in history, NOT on main HEAD)
- `455386c` P2 (expert-id clamp + scratch-set sync-on-release)
- `e784576` P3 (per-thread thread_local ggml backend+gallocr) + `010fa94` buft fix

Reverted (`7c42c9e`, `f30d9d9`, `ac33b5f`) because a live abort-injection test
proved neither fixes the target crash: the MoE `ggml-moe-prefill` illegal-access
persisted. Root (diagnosed via `CUDA_LAUNCH_BLOCKING=1`, which eliminates it) is
an **async host-vs-in-flight-kernel timing hazard**, likely ggml-internal — not
the concurrency race the shadow reproduced (that race is real but not the prod
trigger). P3 also introduced a `GGML_ASSERT(buft)` regression. To re-explore,
cherry-pick the reverted commits onto a branch.

## The real prod crash IS mitigated (platform-side, no engine change)
The overnight 248K-context retry-loop crashes (37 in the 00:00-05:30 window on
07-22) dropped to **0** on 07-23 after the platform capped context 262K->128K
(`runtime.max_context_tokens`). Remaining: a rare (~1/day) large-single-message
path that bypasses the cap on agent runs; fix = hard-cap the final assembled
prompt in the platform runtime, not the engine.

## Mitigation available for the pathological abort case
`CUDA_LAUNCH_BLOCKING=1` eliminates the abort-injection crash entirely at ~1.5x
latency (measured steady-state). Not deployed — normal traffic is stable without
it; the crash needs pathological abort load or the (now-capped) 248K path.
