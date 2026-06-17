// test_channel.mjs — drive the real Secure_Channel handshake from Node (acting
// as the browser) against a running ws_bridge + secure_server.
// Usage: node web/test_channel.mjs <ws_url> <server_pub_hex> <prompt>
import { SecureChannel, Tag } from './channel.js';

const wsUrl = process.argv[2] || 'ws://127.0.0.1:8888/ws';
const pubHex = process.argv[3];
const prompt = process.argv[4] || 'The capital of France is';

if (!pubHex) { console.error('need server pub hex'); process.exit(2); }

const hooks = {
  onStep: (s) => console.log('  [hs]', s.label),
};
const ch = new SecureChannel(pubHex, hooks);

let answer = '';
const dec = new TextDecoder();
let sessionDone = false;

ch.onRecord = (tag, body) => {
  if (tag === Tag.Session) {
    console.log('  session id:', dec.decode(body));
    sessionDone = true;
    ch.sendText(Tag.Prompt, prompt);
  } else if (tag === Tag.Prefill) {
    process.stdout.write('.');
  } else if (tag === Tag.Token) {
    answer += dec.decode(body);
    process.stdout.write(dec.decode(body));
  } else if (tag === Tag.Done) {
    console.log('\n  DONE. answer =', JSON.stringify(answer.trim()));
    ch.close();
    process.exit(answer.trim().length > 0 ? 0 : 1);
  } else if (tag === Tag.Error) {
    console.log('\n  server error:', dec.decode(body));
    ch.close(); process.exit(1);
  }
};

const info = await ch.connect(wsUrl);
console.log('  cipher :', info.cipher);
console.log('  binding:', info.binding.slice(0, 32), '...');
ch.sendText(Tag.Session, '');   // new session
setTimeout(() => { console.log('timeout'); process.exit(3); }, 120000);
