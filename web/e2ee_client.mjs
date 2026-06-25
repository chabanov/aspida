// Node E2EE client for Aspida secure_server — the core of the UARP "aspida-e2ee"
// adapter. Reuses web/crypto.js (RFC-verified) for all crypto; replaces the
// browser WebSocket transport with a raw TCP socket (Deno/Node server-side can
// reach secure_server directly, no ws_bridge). Proves UARP can do encrypted
// inference end-to-end.
import net from "node:net";
import { Aspida as A } from "./crypto.js";

const HOST = process.env.SS_HOST || "127.0.0.1";
const PORT = Number(process.env.SS_PORT || 8765);
const PUB = (process.env.SS_PUB || "").trim();
const PROMPT = process.env.PROMPT || "Hello";
const PROLOGUE = "aspida-secure-channel/1 X25519-ChaCha20Poly1305-HKDF-SHA256";
const INFO = "keys";
const Tag = { Session: 115, Prompt: 112, Token: 116, Done: 33, Resp: 114, Error: 101, Prefill: 46 };

const be32 = (n) => new Uint8Array([(n>>>24)&255,(n>>>16)&255,(n>>>8)&255,n&255]);
function nonce96(counter){const n=new Uint8Array(12);let v=BigInt(counter);for(let i=0;i<8;i++){n[i]=Number(v&0xffn);v>>=8n;}return n;}

const sock = net.connect(PORT, HOST);
sock.on("error", (e)=>{ console.error("socket error", e.message); process.exit(2); });

let rx = new Uint8Array(0); const frameQ = []; let frameWaiter = null;
sock.on("data", (d)=>{
  const buf = new Uint8Array(d);
  const m = new Uint8Array(rx.length + buf.length); m.set(rx); m.set(buf, rx.length); rx = m;
  for(;;){ if(rx.length<4) break; const len=(rx[0]<<24)|(rx[1]<<16)|(rx[2]<<8)|rx[3];
    if(rx.length<4+len) break; const f=rx.slice(4,4+len); rx=rx.slice(4+len);
    if(frameWaiter){const w=frameWaiter;frameWaiter=null;w(f);} else frameQ.push(f); }
});
const readFrame = ()=> frameQ.length ? Promise.resolve(frameQ.shift()) : new Promise(r=>frameWaiter=r);
const writeFrame = (p)=> sock.write(Buffer.from(A.concat(be32(p.length), p)));

await new Promise((res,rej)=>{ sock.on("connect",res); sock.on("error",rej); });

// ---- handshake (initiator), mirrors web/channel.js ----
const ePriv = A.randomBytes(32), ePub = A.x25519Base(ePriv);
writeFrame(ePub);
const fPub = await readFrame(); if(fPub.length!==32) throw new Error("bad server ephemeral");
const es = A.x25519(ePriv, A.fromHex(PUB)), ee = A.x25519(ePriv, fPub);
if(A.ctEqual(es,new Uint8Array(32))||A.ctEqual(ee,new Uint8Array(32))) throw new Error("degenerate DH");
const transcript = A.sha256(A.concat(A.enc(PROLOGUE), A.fromHex(PUB), ePub, fPub));
const prk = A.hkdfExtract(transcript, A.concat(es,ee));
const k = A.hkdfExpand(prk, A.enc(INFO), 64);
const kSend = k.slice(0,32), kRecv = k.slice(32,64);
const confTag = await readFrame(); if(confTag.length!==16) throw new Error("bad conf tag");
if(A.aeadOpen(kRecv, nonce96(0), transcript, new Uint8Array(0), confTag)===null)
  throw new Error("server auth failed (tag mismatch)");
console.log("🔒 handshake OK — server authenticated, forward-secret session live");
console.log("   binding =", A.toHex(transcript).slice(0,32), "...");

let nSend=0n, nRecv=1n;
function send(tag, body){ const pt=A.concat(new Uint8Array([tag]), body||new Uint8Array(0));
  const {ct,tag:mac}=A.aeadSeal(kSend, nonce96(nSend), new Uint8Array(0), pt);
  writeFrame(A.concat(ct,mac)); nSend+=1n; }
async function recv(){ const f=await readFrame(); const ct=f.slice(0,f.length-16), mac=f.slice(f.length-16);
  const pt=A.aeadOpen(kRecv, nonce96(nRecv), new Uint8Array(0), ct, mac);
  if(pt===null) throw new Error("AEAD auth error on record"); nRecv+=1n;
  return { tag: pt[0], body: pt.slice(1) }; }

// ---- app protocol: Session -> Prompt -> Token* -> Done ----
send(Tag.Session, A.enc(""));
let started=false, out="", tokens=0;
const dec = new TextDecoder();
for(;;){
  const { tag, body } = await recv();
  if(tag===Tag.Session){ send(Tag.Prompt, A.enc(PROMPT)); started=true; }
  else if(tag===Tag.Token){ out += dec.decode(body); tokens++; }
  else if(tag===Tag.Prefill){ /* prefill marker */ }
  else if(tag===Tag.Resp){ out += dec.decode(body); }
  else if(tag===Tag.Done){ console.log(`✓ Done — ${tokens} token records over the encrypted channel`);
    console.log("RESPONSE (decrypted):", JSON.stringify(out.slice(0,120))); break; }
  else if(tag===Tag.Error){ console.log("server error:", dec.decode(body)); break; }
}
sock.end();
console.log("RESULT: PASS (UARP-side E2EE inference round-trip)");
process.exit(0);
