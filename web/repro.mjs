// Reproduce the reported web bug: a long generation in session 1, then a
// 2nd session starts mid-stream — does session 1 break? Uses the REAL
// channel.js/crypto.js over Node's built-in WebSocket, against the bridge.
import { SecureChannel, Tag } from './channel.js';

const PUB = '199e790ce9469a2f72ed38aa22feb9a53f1f0b569c0afbc2ea3db70e8eb3a75b';
const URL = process.argv[2] || 'ws://167.99.177.241:8888/ws';
const dec = new TextDecoder();
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function mkSession(name) {
  const s = { name, tokens: 0, done: false, err: null, closed: false, last: 0, text: '' };
  const ch = new SecureChannel(PUB, { onClose: () => { s.closed = true; } });
  ch.onRecord = (tag, body) => {
    if (tag === Tag.Token)      { s.tokens++; s.last = Date.now(); s.text += dec.decode(body); }
    else if (tag === Tag.Done)  { s.done = true; }
    else if (tag === Tag.Error) { s.err = dec.decode(body); }
  };
  s.ch = ch;
  return s;
}

async function start(s, prompt) {
  await s.ch.connect(URL);
  s.ch.sendText(Tag.Session, '');     // new session
  await sleep(400);
  s.ch.sendText(Tag.Prompt, prompt);
  s.last = Date.now();
}

const run = async () => {
  const s1 = mkSession('S1');
  await start(s1, 'Розкажи дуже детально історію Києва від заснування, мінімум 120 слів.');
  await sleep(1500);                  // S2 starts DURING S1 prefill (most fragile)
  const t1_before = s1.tokens;
  console.log(`[t=1.5s] before S2: S1 tokens=${s1.tokens} (likely still prefilling)`);

  const s2 = mkSession('S2');
  await start(s2, 'Розкажи коротко про каву, 40 слів.');
  console.log(`[t=1.5s] S2 launched during S1 prefill`);

  // Watch until both should be done (or up to ~60s).
  for (let i = 0; i < 20; i++) {
    await sleep(3000);
    console.log(`  +${(i+1)*3}s  S1 tok=${s1.tokens} done=${s1.done} err=${s1.err} closed=${s1.closed} | S2 tok=${s2.tokens} done=${s2.done} err=${s2.err} closed=${s2.closed}`);
    if (s1.done && s2.done) break;
  }
  const s1_progressed = s1.tokens > t1_before;
  console.log(`\nVERDICT: S1 made progress after S2 started: ${s1_progressed} (was ${t1_before}, now ${s1.tokens})`);
  console.log(`  S1 finished cleanly (done): ${s1.done}; broke (closed/err before done): ${(s1.closed || s1.err) && !s1.done}`);
  console.log(`  S2 finished cleanly (done): ${s2.done}`);
  console.log(`  S1 text tail: ...${s1.text.slice(-60)}`);
  console.log(`  S2 text: ${s2.text}`);
  process.exit(0);
};
run().catch(e => { console.error('harness error:', e); process.exit(1); });
