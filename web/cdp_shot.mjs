// cdp_shot.mjs — dev tool: drive headless Chrome over the DevTools Protocol to
// load the demo, send a chat message, and screenshot the live result.
// Usage: node web/cdp_shot.mjs <url> <out.png> <prompt>
import { writeFileSync } from 'node:fs';

const url = process.argv[2], out = process.argv[3], prompt = process.argv[4] || 'Привіт! Хто ти?';
const base = 'http://127.0.0.1:9222';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// find a page target
let targets = await (await fetch(`${base}/json`)).json();
let page = targets.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
};
const send = (method, params = {}) => new Promise((res) => {
  const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params }));
});
await new Promise((r) => (ws.onopen = r));
await send('Page.enable');
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride',
           { width: 1440, height: 940, deviceScaleFactor: 2, mobile: false });
await send('Page.navigate', { url });
await sleep(4000);  // load + handshake

// type the prompt and submit
await send('Runtime.evaluate', { expression:
  `(() => { const i=document.getElementById('input'); i.value=${JSON.stringify(prompt)};
    document.getElementById('composer').dispatchEvent(new Event('submit',{cancelable:true})); })()` });
await sleep(Number(process.argv[5] || 22000));  // stream the reply (CPU model)

const { data } = await send('Page.captureScreenshot', { format: 'png' });
writeFileSync(out, Buffer.from(data, 'base64'));
console.log('screenshot ->', out);

// also pull a quick status string for the log
const st = await send('Runtime.evaluate', {
  expression: `JSON.stringify({status:document.getElementById('statusText').textContent,
    recs:document.getElementById('recCount').textContent,
    steps:document.querySelectorAll('#hsSteps li').length,
    msgs:[...document.querySelectorAll('.msg .text')].map(e=>e.textContent)})`,
  returnByValue: true });
console.log('state:', st.result.value);
ws.close();
process.exit(0);
