# Aspida Engine — Security / E2EE Audit Summary

Full audit of the cryptographic perimeter and end-to-end guarantee.
**Verdict: the cryptographic core is certification-grade.** The two failure
classes flagged as CRITICAL — nonce reuse, and non-constant-time MAC/tag
compare — are both CLEAN. All findings were MEDIUM or below (see
REMEDIATION-LOG.md S1–S6).

## Verified SOUND

- **Nonce uniqueness** (the critical property): per-direction keys
  (K_C2S/K_S2C) with independent 64-bit monotonic counters; nonce 0 is the
  key-confirmation, data starts at 1; counter exhaustion refused before wrap.
  No (key, nonce) pair is ever reused. At-rest: every Save draws a fresh
  random salt → fresh key AND a fresh random nonce, so fixed-key collision is
  structurally impossible.
- **AEAD** (ChaCha20-Poly1305, RFC 8439 exact): Poly key from keystream block
  0, cipher from block 1, tag over AAD|pad|CT|pad|len64|len64. Open verifies
  the tag with Const_Time_Equal and decrypts ONLY after success; no padding
  oracle (stream cipher); uniform error strings.
- **Constant-time**: Const_Time_Equal folds all bytes, single terminal branch,
  used for every secret comparison (tag verify, low-order-point check, token
  auth). X25519 is a branch-free Montgomery ladder with arithmetic selects;
  RFC 7748 clamping correct; low-order/non-contributory points rejected on
  both es and ee.
- **KDF / RNG**: HKDF-Extract salted with the handshake transcript (binds keys
  to the exchange); PBKDF2-HMAC-SHA256 at 600 000 iterations for at-rest and
  session store. RNG = OS CSPRNG (getentropy), hard-fails on error, no weak
  fallback, no manual seeding.
- **Handshake / MITM / replay**: server key is client-pinned; confirmation tag
  over the transcript proves static-key possession (defeats MITM); no suite
  negotiation (no downgrade); fresh ephemerals per handshake (forward secrecy,
  defeats whole-session replay). Receiver derives the nonce from its own
  monotonic counter, so replay/reorder/drop/insert all fail the tag.
- **Length-field / traversal**: frames bounded (≤16 MiB) in the U32 domain
  before any Natural conversion; llm_weight_proto (SPARK-verified) and
  session_store deserialization are fully bounds-checked; session ids validated
  `[A-Za-z0-9_-]{1,64}` against path traversal.
- **Key/secret handling**: static key file 0600 + mlock; session dir 0700,
  temp files 0600 before ciphertext, atomic rename + fsync; env password
  captured then unset. No logging of prompts, keys, tokens, or plaintext
  anywhere — logs carry turn counts and exception names only.

## Remediated / documented (findings S1–S6)

- **S2 (FIXED)**: key-zeroization now backed by a guaranteed System.Machine_Code
  memory barrier (was an anti-DSE idiom that LTO could elide).
- **S1 (DOCUMENTED)**: the :8099 OpenAI-compat proxy is a LOCAL plaintext shim
  (bind hardcoded 127.0.0.1) — genuinely E2EE for the browser/native paths
  (ciphertext-only relay, pinned key); the proxy path is TLS-to-server, NOT the
  same no-CA/no-MITM guarantee, and must never be fronted by a remote-
  terminating reverse proxy. Deployment constraint.
- **S3/S4 (DOCUMENTED)**: mlock gaps on derivation scratch / PBKDF2 key and
  unscrubbed plaintext session turns in Unbounded_String — deployment
  mitigation (swap disabled); full arena-mlock + byte-array session storage
  deferred.
- **S5 (INFO)**: no authenticated end-of-stream marker (truncation = conn
  error); client anonymous by design (Noise-NK + optional constant-time token).

## E2EE guarantee

Browser path (ws_bridge): the bridge relays ciphertext only — genuine E2EE.
Native path (secure_client): pinned server key, Noise handshake — genuine E2EE.
Local proxy path (:8099): plaintext terminates on the server host inside a
loopback shim, then runs its own Noise client to the engine — TLS-to-server
class, documented as such, not advertised as E2EE.
