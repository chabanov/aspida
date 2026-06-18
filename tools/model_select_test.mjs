// Headless end-to-end test of model discovery + selection over the real
// encrypted channel. Lists models (Tag 'm'), then selects a different one
// (Tag 'M') and prints the server's reply.
//   node tools/model_select_test.mjs <ws-url> <server-pub-hex>
import { SecureChannel } from '../web/channel.js';

const URL = process.argv[2] || 'ws://127.0.0.1:8899/ws';
const PUB = process.argv[3];
const dec = new TextDecoder();
const Tag = { Models: 'm'.charCodeAt(0), Select: 'M'.charCodeAt(0),
              Resp: 'r'.charCodeAt(0), Session: 's'.charCodeAt(0) };

let resolveResp = null;
const waitResp = () => new Promise(r => { resolveResp = r; });

const ch = new SecureChannel(PUB, { onClose: () => console.log('[socket closed]') });
ch.onRecord = (tag, body) => {
  if (tag === Tag.Resp) { const r = resolveResp; resolveResp = null; r && r(dec.decode(body)); }
  else if (tag === Tag.Session) { /* assigned id */ }
};

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

(async () => {
  await ch.connect(URL);
  ch.sendText(Tag.Session, '');
  await sleep(300);

  // 1) list
  const p = waitResp();
  ch.sendText(Tag.Models, '');
  const listJson = await p;
  const list = JSON.parse(listJson);
  console.log(`switchable: ${list.switchable}`);
  console.log(`models (${list.data.length}):`);
  for (const m of list.data)
    console.log(`  ${m.active ? '*' : ' '} [${m.supported ? 'ok ' : 'arch'}] ${m.name}  (${m.arch} ${m.params} ${m.quant}, ${m.size})`);

  // 2) pick a supported model that is NOT active
  const target = list.data.find(m => m.supported && !m.active);
  if (!target) { console.log('no alternate supported model to select'); process.exit(0); }
  console.log(`\nselecting -> ${target.name}\n  ${target.id}`);
  const p2 = waitResp();
  ch.sendText(Tag.Select, target.id);
  const selJson = await p2;
  console.log('select reply: ' + selJson);
  await sleep(600);   // server may exit(75) to reload
  process.exit(0);
})().catch(e => { console.error('ERR', e.message); process.exit(1); });
