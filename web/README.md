# Aspida — encrypted-inference web demo

A browser demo that makes Aspida's **end-to-end encrypted inference** visible. The
browser runs the **real** Secure_Channel handshake and seals every record itself,
using our own from-scratch JavaScript crypto (no libraries). A thin Ada bridge
relays the already-encrypted bytes to `secure_server` — it only ever sees
ciphertext.

```
[browser]  ── own JS: X25519 + ChaCha20-Poly1305 + HKDF-SHA256 ──┐  (seals here)
   │  WebSocket (ciphertext frames)                              │
   ▼                                                             │
[ws_bridge]  ── raw TCP (same ciphertext) ──►  [secure_server]  ◄┘  (own Ada engine)
   (dumb byte pipe; never sees plaintext)         model + crypto
```

## Pieces (all our own)
- `crypto.js` — SHA-256/HMAC, HKDF, ChaCha20, Poly1305, AEAD, X25519. Verified
  against the RFC test vectors (`node web/crypto.js` → 8/8). Byte-identical to
  `src/crypto/*`.
- `channel.js` — the client side of `src/session/secure_channel.adb`: Noise-NK
  handshake + 4-byte-length-prefixed ChaCha20-Poly1305 records, with hooks for
  the visualizations.
- `index.html` / `style.css` / `app.js` — the UI: chat + a "what's on the wire"
  panel (handshake ladder, live ciphertext frames, token streaming, and an
  "operator view" toggle that shows exactly what the server/provider sees).
- `src/server/ws_bridge.adb` — serves these files and bridges WebSocket ⟷ TCP
  (single-task selector relay; own SHA-1 + Base64 for the WS handshake).

## Run
```sh
# 1. build (engine + crypto + bridge)
gprbuild -P server.gpr -XSDKROOT=$(xcrun --show-sdk-path)

# 2. start the inference server with a model; note the public key it prints
QWEN_MODEL_PATH=/path/to/model.gguf ./obj/secure_server 8765

# 3. start the bridge, pinning that public key
./obj/ws_bridge 127.0.0.1 8765 <server_pub_hex> 8888 web

# 4. open the demo
open http://127.0.0.1:8888/
```
The bridge binds 127.0.0.1 only and serves `/serverkey` so the page auto-pins the
key. Set `ASPIDA_WSDBG=1` for a verbose relay trace.

Note: on CPU (no GPU) the model serializes generation behind a single lock, so
first-token latency under load can be tens of seconds — that's the engine, not
the channel.
