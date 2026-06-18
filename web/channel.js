// channel.js — the Aspida Secure_Channel, client side, in the browser.
//
// Mirrors src/session/secure_channel.adb byte-for-byte: a Noise-NK-style
// handshake (client ephemeral e, server ephemeral f, ES=DH(e,s), EE=DH(e,f),
// transcript = SHA256(prologue|s|e|f), HKDF -> K_c2s|K_s2c, server key-
// confirmation tag), then 4-byte big-endian length-prefixed ChaCha20-Poly1305
// records with a per-direction 64-bit nonce counter.
//
// Transport is a WebSocket to the Ada ws_bridge, which relays the raw bytes to
// the secure_server's TCP socket. The bridge only ever sees ciphertext: the
// handshake and every record are sealed HERE, in the browser.

import { Aspida as A } from './crypto.js';

const PROLOGUE = 'aspida-secure-channel/1 X25519-ChaCha20Poly1305-HKDF-SHA256';
const INFO = 'keys';

// Application record tags (src/secure/protocol.ads).
export const Tag = {
  Session: 's'.charCodeAt(0),
  Prompt:  'p'.charCodeAt(0),
  Token:   't'.charCodeAt(0),
  Prefill: '.'.charCodeAt(0),
  Done:    '!'.charCodeAt(0),
  Error:   'e'.charCodeAt(0),
  Models:  'm'.charCodeAt(0),   // C->S: list models    S->C: catalog JSON (Resp)
  Select:  'M'.charCodeAt(0),   // C->S: select model   S->C: result JSON (Resp)
  Resp:    'r'.charCodeAt(0),   // S->C: a JSON response
};

const be32 = (n) => new Uint8Array([(n>>>24)&255, (n>>>16)&255, (n>>>8)&255, n&255]);
function nonce96(counter) {            // 12-byte: LE 64-bit counter | 4 zero bytes
  const n = new Uint8Array(12);
  let v = BigInt(counter);
  for (let i = 0; i < 8; i++) { n[i] = Number(v & 0xffn); v >>= 8n; }
  return n;
}

export class SecureChannel {
  // hooks: optional callbacks for visualization (all receive plain objects).
  constructor(serverPubHex, hooks = {}) {
    this.serverPub = A.fromHex(serverPubHex.trim());
    this.hooks = hooks;
    this.ws = null;
    this.rx = new Uint8Array(0);
    this.frameQueue = [];
    this.frameWaiter = null;
    this.ready = false;
    this.kSend = null; this.kRecv = null;
    this.nSend = 0n; this.nRecv = 0n;
    this.binding = null;
    this.onRecord = null;            // set by caller: (tag, body Uint8Array) => void
  }

  _emit(name, data) { if (this.hooks[name]) this.hooks[name](data); }

  // ---- transport / framing --------------------------------------------
  _onBytes(buf) {
    const merged = new Uint8Array(this.rx.length + buf.length);
    merged.set(this.rx); merged.set(buf, this.rx.length);
    this.rx = merged;
    for (;;) {
      if (this.rx.length < 4) break;
      const len = (this.rx[0]<<24) | (this.rx[1]<<16) | (this.rx[2]<<8) | this.rx[3];
      if (this.rx.length < 4 + len) break;
      const frame = this.rx.slice(4, 4 + len);
      this.rx = this.rx.slice(4 + len);
      this._deliverFrame(frame);
    }
  }
  _deliverFrame(frame) {
    if (this.ready) { this._recvRecord(frame); return; }
    if (this.frameWaiter) { const w = this.frameWaiter; this.frameWaiter = null; w(frame); }
    else this.frameQueue.push(frame);
  }
  _readFrame() {
    if (this.frameQueue.length) return Promise.resolve(this.frameQueue.shift());
    return new Promise((res) => { this.frameWaiter = res; });
  }
  _writeFrame(payload) {
    this.ws.send(A.concat(be32(payload.length), payload));
  }

  // ---- handshake (initiator) ------------------------------------------
  async connect(wsUrl) {
    await new Promise((resolve, reject) => {
      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';
      this.ws.onmessage = (ev) => this._onBytes(new Uint8Array(ev.data));
      this.ws.onopen = resolve;
      this.ws.onerror = () => reject(new Error('WebSocket connection failed'));
      this.ws.onclose = () => { if (this.hooks.onClose) this.hooks.onClose(); };
    });

    const ePriv = A.randomBytes(32);
    const ePub = A.x25519Base(ePriv);
    this._emit('onStep', { step: 'ephemeral', label: 'Client ephemeral key generated',
                           pub: A.toHex(ePub) });
    this._writeFrame(ePub);
    this._emit('onFrameOut', { label: 'handshake: client key', bytes: ePub, kind: 'hs' });

    const fPub = await this._readFrame();
    this._emit('onFrameIn', { label: 'handshake: server key', bytes: fPub, kind: 'hs' });
    if (fPub.length !== 32) throw new Error('bad server ephemeral length');

    const es = A.x25519(ePriv, this.serverPub);   // authenticates the server (pinned s)
    const ee = A.x25519(ePriv, fPub);             // forward secrecy
    if (A.ctEqual(es, new Uint8Array(32)) || A.ctEqual(ee, new Uint8Array(32)))
      throw new Error('degenerate (low-order) DH result');
    this._emit('onStep', { step: 'dh', label: 'ECDH shared secrets computed (ES, EE)',
                           es: A.toHex(es), ee: A.toHex(ee) });

    const transcript = A.sha256(A.concat(A.enc(PROLOGUE), this.serverPub, ePub, fPub));
    const prk = A.hkdfExtract(transcript, A.concat(es, ee));
    const k = A.hkdfExpand(prk, A.enc(INFO), 64);
    const kC2S = k.slice(0, 32), kS2C = k.slice(32, 64);
    this.binding = transcript;
    this._emit('onStep', { step: 'derive', label: 'Session keys derived',
                           transcript: A.toHex(transcript),
                           kc2s: A.toHex(kC2S), ks2c: A.toHex(kS2C) });

    const confTag = await this._readFrame();
    this._emit('onFrameIn', { label: 'handshake: server key-confirmation tag', bytes: confTag, kind: 'hs' });
    if (confTag.length !== 16) throw new Error('bad confirmation length');
    const ok = A.aeadOpen(kS2C, nonce96(0), transcript, new Uint8Array(0), confTag);
    if (ok === null) throw new Error('server authentication failed (tag mismatch)');
    this._emit('onStep', { step: 'confirmed',
                           label: 'Server authenticated — no MITM, forward-secret session live',
                           binding: A.toHex(transcript) });

    this.kSend = kC2S; this.kRecv = kS2C;
    this.nSend = 0n; this.nRecv = 1n;     // server consumed recv-nonce 0 for confirmation
    this.ready = true;
    return { binding: A.toHex(transcript), cipher: 'authenticated key exchange + AEAD',
             serverKey: A.toHex(this.serverPub) };
  }

  // ---- records ---------------------------------------------------------
  // Send an application record (tag byte + body bytes), sealed.
  sendRecord(tag, body) {
    const plaintext = A.concat(new Uint8Array([tag]), body || new Uint8Array(0));
    const { ct, tag: mac } = A.aeadSeal(this.kSend, nonce96(this.nSend), new Uint8Array(0), plaintext);
    const frame = A.concat(ct, mac);
    this._writeFrame(frame);
    this._emit('onRecordOut', { tag, plaintext, frame, nonce: this.nSend });
    this.nSend += 1n;
  }
  sendText(tag, text) { this.sendRecord(tag, A.enc(text)); }

  _recvRecord(frame) {
    if (frame.length < 16) return;
    const ct = frame.slice(0, frame.length - 16);
    const mac = frame.slice(frame.length - 16);
    const pt = A.aeadOpen(this.kRecv, nonce96(this.nRecv), new Uint8Array(0), ct, mac);
    if (pt === null) { this._emit('onAuthError', {}); return; }
    this._emit('onRecordIn', { frame, plaintext: pt, nonce: this.nRecv });
    this.nRecv += 1n;
    if (pt.length >= 1 && this.onRecord) this.onRecord(pt[0], pt.slice(1));
  }

  close() { if (this.ws) this.ws.close(); this.ready = false; }
}
