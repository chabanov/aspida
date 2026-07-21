# Aspida Engine — Certification Package

Scope: the aspida inference engine (Ada/SPARK + CUDA), covering three
certification dimensions requested 2026-07-21:

1. **Engineering cert-audit** — correctness, memory safety, concurrency,
   resource management, error handling across the Ada orchestration and the
   CUDA compute path. Log: `REMEDIATION-LOG.md`.
2. **SPARK formal proofs** — machine-checked verification of the E2EE crypto
   core; close open VCs, extend SPARK_Mode coverage. Log: `SPARK-PROOFS.md`.
3. **Security / E2EE audit** — threat model, crypto correctness, key
   lifecycle, side-channels, end-to-end E2EE guarantee (Noise secure_server,
   at-rest, session). Log: `SECURITY-E2EE.md`.

Engine areas (LOC): CUDA 11.5k (gpu/*.cu,*.cuh) · Ada crypto 1.6k ·
Ada server/session 6.6k · Ada LLM 18.7k.

Status: IN PROGRESS (opened 2026-07-21).
