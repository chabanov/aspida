// crypto.js — Aspida's own cryptography, reimplemented in the browser.
//
// NO third-party libraries. These are from-scratch implementations of exactly
// the primitives the Ada engine uses, so the browser can run the real
// end-to-end-encrypted handshake itself:
//   SHA-256 / HMAC (FIPS 180-4, RFC 2104), HKDF (RFC 5869),
//   ChaCha20 + Poly1305 + AEAD (RFC 8439), X25519 (RFC 7748).
//
// Outputs are byte-for-byte identical to src/crypto/*.adb (verified against the
// same RFC test vectors). Field arithmetic for X25519/Poly1305 uses BigInt for
// compactness/clarity; this demo does NOT claim constant-time execution in JS
// (the Ada implementation is the constant-time one). Interop is the goal here.

export const Aspida = (() => {
  'use strict';

  // ---- byte helpers ----------------------------------------------------
  const te = new TextEncoder();
  const enc = (s) => te.encode(s);
  const concat = (...arrs) => {
    let n = 0; for (const a of arrs) n += a.length;
    const out = new Uint8Array(n); let p = 0;
    for (const a of arrs) { out.set(a, p); p += a.length; }
    return out;
  };
  const toHex = (b) => Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
  const fromHex = (h) => {
    const out = new Uint8Array(h.length / 2);
    for (let i = 0; i < out.length; i++) out[i] = parseInt(h.substr(2 * i, 2), 16);
    return out;
  };
  // constant-ish time compare (demo)
  const ctEqual = (a, b) => {
    if (a.length !== b.length) return false;
    let d = 0; for (let i = 0; i < a.length; i++) d |= a[i] ^ b[i];
    return d === 0;
  };

  // ---- SHA-256 (FIPS 180-4) --------------------------------------------
  const K256 = new Uint32Array([
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
  const rotr = (x, n) => (x >>> n) | (x << (32 - n));

  function sha256(msg) {
    const h = new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                               0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);
    const ml = msg.length;
    const withOne = ml + 1;
    const k = (56 - (withOne % 64) + 64) % 64;
    const total = withOne + k + 8;
    const buf = new Uint8Array(total);
    buf.set(msg); buf[ml] = 0x80;
    const bitLenHi = Math.floor((ml * 8) / 0x100000000);
    const bitLenLo = (ml * 8) >>> 0;
    const dv = new DataView(buf.buffer);
    dv.setUint32(total - 8, bitLenHi); dv.setUint32(total - 4, bitLenLo);

    const w = new Uint32Array(64);
    for (let off = 0; off < total; off += 64) {
      for (let i = 0; i < 16; i++) w[i] = dv.getUint32(off + i * 4);
      for (let i = 16; i < 64; i++) {
        const s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >>> 3);
        const s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >>> 10);
        w[i] = (w[i-16] + s0 + w[i-7] + s1) >>> 0;
      }
      let [a,b,c,d,e,f,g,hh] = h;
      for (let i = 0; i < 64; i++) {
        const S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        const ch = (e & f) ^ (~e & g);
        const t1 = (hh + S1 + ch + K256[i] + w[i]) >>> 0;
        const S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        const maj = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (S0 + maj) >>> 0;
        hh=g; g=f; f=e; e=(d+t1)>>>0; d=c; c=b; b=a; a=(t1+t2)>>>0;
      }
      h[0]=(h[0]+a)>>>0; h[1]=(h[1]+b)>>>0; h[2]=(h[2]+c)>>>0; h[3]=(h[3]+d)>>>0;
      h[4]=(h[4]+e)>>>0; h[5]=(h[5]+f)>>>0; h[6]=(h[6]+g)>>>0; h[7]=(h[7]+hh)>>>0;
    }
    const out = new Uint8Array(32);
    const odv = new DataView(out.buffer);
    for (let i = 0; i < 8; i++) odv.setUint32(i * 4, h[i]);
    return out;
  }

  // ---- HMAC-SHA256 (RFC 2104) ------------------------------------------
  function hmacSha256(key, msg) {
    if (key.length > 64) key = sha256(key);
    const k = new Uint8Array(64); k.set(key);
    const ipad = new Uint8Array(64), opad = new Uint8Array(64);
    for (let i = 0; i < 64; i++) { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5c; }
    return sha256(concat(opad, sha256(concat(ipad, msg))));
  }

  // ---- HKDF (RFC 5869) -------------------------------------------------
  function hkdfExtract(salt, ikm) {
    if (!salt || salt.length === 0) salt = new Uint8Array(32);
    return hmacSha256(salt, ikm);
  }
  function hkdfExpand(prk, info, length) {
    const out = new Uint8Array(length);
    let t = new Uint8Array(0), p = 0, counter = 1;
    while (p < length) {
      t = hmacSha256(prk, concat(t, info, new Uint8Array([counter])));
      const n = Math.min(t.length, length - p);
      out.set(t.subarray(0, n), p); p += n; counter++;
    }
    return out;
  }

  // ---- ChaCha20 (RFC 8439 §2.1-2.4) ------------------------------------
  const rol = (x, n) => ((x << n) | (x >>> (32 - n))) >>> 0;
  function chachaBlock(key32, counter, nonce32, out) {
    const s = new Uint32Array(16);
    s[0]=0x61707865; s[1]=0x3320646e; s[2]=0x79622d32; s[3]=0x6b206574;
    for (let i = 0; i < 8; i++) s[4 + i] = key32[i];
    s[12] = counter >>> 0; s[13] = nonce32[0]; s[14] = nonce32[1]; s[15] = nonce32[2];
    const x = s.slice();
    const QR = (a,b,c,d) => {
      x[a]=(x[a]+x[b])>>>0; x[d]=rol(x[d]^x[a],16);
      x[c]=(x[c]+x[d])>>>0; x[b]=rol(x[b]^x[c],12);
      x[a]=(x[a]+x[b])>>>0; x[d]=rol(x[d]^x[a], 8);
      x[c]=(x[c]+x[d])>>>0; x[b]=rol(x[b]^x[c], 7);
    };
    for (let i = 0; i < 10; i++) {
      QR(0,4,8,12); QR(1,5,9,13); QR(2,6,10,14); QR(3,7,11,15);
      QR(0,5,10,15); QR(1,6,11,12); QR(2,7,8,13); QR(3,4,9,14);
    }
    const dv = new DataView(out.buffer, out.byteOffset, 64);
    for (let i = 0; i < 16; i++) dv.setUint32(i * 4, (x[i] + s[i]) >>> 0, true);
  }
  function keyToWords(key) {
    const dv = new DataView(key.buffer, key.byteOffset, 32);
    const w = new Uint32Array(8);
    for (let i = 0; i < 8; i++) w[i] = dv.getUint32(i * 4, true);
    return w;
  }
  function nonceToWords(nonce) {
    const dv = new DataView(nonce.buffer, nonce.byteOffset, 12);
    return [dv.getUint32(0, true), dv.getUint32(4, true), dv.getUint32(8, true)];
  }
  function chacha20Xor(key, nonce, counter, input) {
    const key32 = keyToWords(key), nonce32 = nonceToWords(nonce);
    const out = new Uint8Array(input.length);
    const block = new Uint8Array(64);
    for (let off = 0; off < input.length; off += 64) {
      chachaBlock(key32, counter + (off >> 6), nonce32, block);
      const n = Math.min(64, input.length - off);
      for (let i = 0; i < n; i++) out[off + i] = input[off + i] ^ block[i];
    }
    return out;
  }

  // ---- Poly1305 (RFC 8439 §2.5) ----------------------------------------
  function poly1305(key, msg) {
    const P = (1n << 130n) - 5n;
    let r = 0n, s = 0n;
    for (let i = 15; i >= 0; i--) r = (r << 8n) | BigInt(key[i]);
    for (let i = 31; i >= 16; i--) s = (s << 8n) | BigInt(key[i]);
    r &= 0x0ffffffc0ffffffc0ffffffc0fffffffn;
    let acc = 0n;
    for (let i = 0; i < msg.length; i += 16) {
      const n = Math.min(16, msg.length - i);
      let block = 0n;
      for (let j = n - 1; j >= 0; j--) block = (block << 8n) | BigInt(msg[i + j]);
      block |= (1n << BigInt(8 * n));
      acc = ((acc + block) * r) % P;
    }
    acc = (acc + s) & ((1n << 128n) - 1n);
    const out = new Uint8Array(16);
    for (let i = 0; i < 16; i++) { out[i] = Number(acc & 0xffn); acc >>= 8n; }
    return out;
  }

  // ---- AEAD ChaCha20-Poly1305 (RFC 8439 §2.8) --------------------------
  function le64(n) {
    const out = new Uint8Array(8);
    let v = BigInt(n);
    for (let i = 0; i < 8; i++) { out[i] = Number(v & 0xffn); v >>= 8n; }
    return out;
  }
  function pad16(len) { return new Uint8Array((16 - (len % 16)) % 16); }
  function polyKeyGen(key, nonce) {
    const block = new Uint8Array(64);
    chachaBlock(keyToWords(key), 0, nonceToWords(nonce), block);
    return block.subarray(0, 32);
  }
  function aeadTag(key, nonce, aad, ct) {
    const otk = polyKeyGen(key, nonce);
    const mac = concat(aad, pad16(aad.length), ct, pad16(ct.length),
                       le64(aad.length), le64(ct.length));
    return poly1305(otk, mac);
  }
  function aeadSeal(key, nonce, aad, plaintext) {
    const ct = chacha20Xor(key, nonce, 1, plaintext);
    const tag = aeadTag(key, nonce, aad, ct);
    return { ct, tag };
  }
  function aeadOpen(key, nonce, aad, ct, tag) {
    const expected = aeadTag(key, nonce, aad, ct);
    if (!ctEqual(expected, tag)) return null;
    return chacha20Xor(key, nonce, 1, ct);
  }

  // ---- X25519 (RFC 7748) -----------------------------------------------
  const P25519 = (1n << 255n) - 19n;
  const A24 = 121665n;
  function inv25519(z) { // z^(p-2) mod p
    return modpow(z, P25519 - 2n, P25519);
  }
  function modpow(b, e, m) {
    b %= m; let r = 1n;
    while (e > 0n) { if (e & 1n) r = (r * b) % m; e >>= 1n; b = (b * b) % m; }
    return r;
  }
  function decodeScalar(k) {
    const e = k.slice();
    e[0] &= 248; e[31] &= 127; e[31] |= 64;
    let s = 0n; for (let i = 31; i >= 0; i--) s = (s << 8n) | BigInt(e[i]);
    return s;
  }
  function decodeU(u) {
    const e = u.slice(); e[31] &= 127;
    let s = 0n; for (let i = 31; i >= 0; i--) s = (s << 8n) | BigInt(e[i]);
    return s % P25519;
  }
  function encodeU(x) {
    const out = new Uint8Array(32);
    let v = x % P25519; if (v < 0n) v += P25519;
    for (let i = 0; i < 32; i++) { out[i] = Number(v & 0xffn); v >>= 8n; }
    return out;
  }
  function cswap(swap, a, b) { return swap ? [b, a] : [a, b]; }
  function x25519(scalarBytes, uBytes) {
    const k = decodeScalar(scalarBytes);
    const x1 = decodeU(uBytes);
    let x2 = 1n, z2 = 0n, x3 = x1, z3 = 1n, swap = 0n;
    for (let t = 254; t >= 0; t--) {
      const kt = (k >> BigInt(t)) & 1n;
      swap ^= kt;
      [x2, x3] = cswap(swap, x2, x3);
      [z2, z3] = cswap(swap, z2, z3);
      swap = kt;
      const A = (x2 + z2) % P25519, AA = (A * A) % P25519;
      const B = (x2 - z2 + P25519) % P25519, BB = (B * B) % P25519;
      const E = (AA - BB + P25519) % P25519;
      const C = (x3 + z3) % P25519, D = (x3 - z3 + P25519) % P25519;
      const DA = (D * A) % P25519, CB = (C * B) % P25519;
      x3 = (DA + CB) % P25519; x3 = (x3 * x3) % P25519;
      z3 = (DA - CB + P25519) % P25519; z3 = (z3 * z3) % P25519; z3 = (z3 * x1) % P25519;
      x2 = (AA * BB) % P25519;
      z2 = (E * ((AA + (A24 * E) % P25519) % P25519)) % P25519;
    }
    [x2, x3] = cswap(swap, x2, x3);
    [z2, z3] = cswap(swap, z2, z3);
    const res = (x2 * inv25519(z2)) % P25519;
    return encodeU(res);
  }
  const BASE9 = (() => { const b = new Uint8Array(32); b[0] = 9; return b; })();
  function x25519Base(scalarBytes) { return x25519(scalarBytes, BASE9); }

  function randomBytes(n) {
    const b = new Uint8Array(n);
    globalThis.crypto.getRandomValues(b);  // WebCrypto: browser + node 20+
    return b;
  }

  return {
    enc, concat, toHex, fromHex, ctEqual,
    sha256, hmacSha256, hkdfExtract, hkdfExpand,
    chacha20Xor, poly1305, aeadSeal, aeadOpen, aeadTag,
    x25519, x25519Base, randomBytes,
  };
})();

// Node test harness: `node web/crypto.js` runs RFC vectors.
if (typeof process !== 'undefined' && process.argv && process.argv[1] &&
    process.argv[1].endsWith('crypto.js')) {
  const A = Aspida;
  let pass = 0, fail = 0;
  const eq = (name, got, want) => {
    const g = A.toHex(got), w = want.replace(/\s/g, '').toLowerCase();
    if (g === w) { pass++; } else { fail++; console.log('FAIL', name, '\n got', g, '\n want', w); }
  };
  // SHA-256("abc")
  eq('sha256 abc', A.sha256(A.enc('abc')),
     'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  // HMAC-SHA256 RFC4231 case 1
  eq('hmac', A.hmacSha256(new Uint8Array(20).fill(0x0b), A.enc('Hi There')),
     'b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7');
  // HKDF RFC5869 A.1
  {
    const ikm = new Uint8Array(22).fill(0x0b);
    const salt = A.fromHex('000102030405060708090a0b0c');
    const info = A.fromHex('f0f1f2f3f4f5f6f7f8f9');
    const prk = A.hkdfExtract(salt, ikm);
    eq('hkdf prk', prk, '077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5');
    eq('hkdf okm', A.hkdfExpand(prk, info, 42),
       '3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865');
  }
  // ChaCha20 RFC8439 §2.4.2 keystream of the test
  {
    const key = A.fromHex('000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f');
    const nonce = A.fromHex('000000000000004a00000000');
    const pt = A.enc("Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.");
    const ct = A.chacha20Xor(key, nonce, 1, pt);
    eq('chacha20', ct.subarray(0, 16), '6e2e359a2568f98041ba0728dd0d6981');
  }
  // AEAD RFC8439 §2.8.2
  {
    const key = A.fromHex('808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f');
    const nonce = A.fromHex('070000004041424344454647');
    const aad = A.fromHex('50515253c0c1c2c3c4c5c6c7');
    const pt = A.enc("Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.");
    const { ct, tag } = A.aeadSeal(key, nonce, aad, pt);
    eq('aead tag', tag, '1ae10b594f09e26a7e902ecbd0600691');
    const opened = A.aeadOpen(key, nonce, aad, ct, tag);
    eq('aead open', opened, A.toHex(pt));
  }
  // X25519 RFC7748 §5.2
  {
    const k = A.fromHex('a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4');
    const u = A.fromHex('e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c');
    eq('x25519', A.x25519(k, u),
       'c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552');
  }
  console.log(`\ncrypto self-test: ${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
}
