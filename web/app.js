// app.js — UI for the Aspida encrypted-inference demo.
// Drives the chat and renders the live encryption visualizations from the
// SecureChannel hooks. No third-party code.
import { SecureChannel, Tag } from './channel.js';
import { Aspida as A } from './crypto.js';

const $ = (id) => document.getElementById(id);
const dec = new TextDecoder('utf-8');

const els = {
  status: $('statusText'), dot: $('statusDot'),
  messages: $('messages'), input: $('input'), send: $('send'), composer: $('composer'),
  hsSteps: $('hsSteps'), bindingBox: $('bindingBox'),
  cipher: $('cipher'), srvKey: $('srvKey'), binding: $('binding'),
  frames: $('frames'), recCount: $('recCount'),
  modelPill: $('modelPill'), ratePill: $('ratePill'), modelSelect: $('modelSelect'),
  operatorView: $('operatorView'), chatBadge: $('chatBadge'),
};

// inline SVG icons (no emoji, no third-party set) — inherit currentColor
const ICON = {
  check: '<svg class="ico-sm" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7"/></svg>',
  lock:  '<svg class="ico-sm" viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="10.5" width="14" height="9.5" rx="2"/><path d="M8 10.5V7a4 4 0 0 1 8 0v3.5"/></svg>',
  warn:  '<svg class="ico-sm warn" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.2 21 19H3Z"/><path d="M12 9v4.5M12 16.6h.01"/></svg>',
  spin:  '<svg class="ico-sm spin" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" stroke-opacity=".25"/><path d="M21 12a9 9 0 0 0-9-9"/></svg>',
};

let recCount = 0;
function bumpRec() { recCount++; els.recCount.textContent = recCount + ' records'; }

// ---- pretty helpers ----------------------------------------------------
const tagName = (t) => ({
  [Tag.Session]:'session', [Tag.Prompt]:'prompt', [Tag.Token]:'token',
  [Tag.Prefill]:'prefill', [Tag.Done]:'done', [Tag.Error]:'error',
}[t] || 'rec');

function recordPreview(plaintext) {
  const tag = plaintext[0];
  const body = dec.decode(plaintext.slice(1));
  const name = tagName(tag);
  if (tag === Tag.Prefill) return `[${name}] ·`;
  if (tag === Tag.Done) return `[${name}]`;
  return `[${name}] ${body}`;
}

function hexPreview(bytes, max = 40) {
  const n = Math.min(bytes.length, max);
  let s = '';
  for (let i = 0; i < n; i++) s += bytes[i].toString(16).padStart(2, '0') + (i % 2 ? ' ' : '');
  if (bytes.length > max) s += `<span class="b">… +${bytes.length - max} bytes</span>`;
  return s;
}

function addFrame({ dir, kind, label, plaintext, bytes, nonce }) {
  const el = document.createElement('div');
  el.className = `frame ${dir}${kind === 'hs' ? ' hs' : ''}`;
  const arrow = dir === 'out' ? '→ to server' : '← from server';
  let html = `<div class="frame-head"><span class="frame-dir">${arrow}</span>`;
  html += `<span class="frame-label">${label}</span>`;
  if (nonce !== undefined && nonce !== null) html += `<span class="frame-nonce">nonce ${nonce}</span>`;
  html += `</div>`;
  if (plaintext !== undefined)
    html += `<div class="frame-plain">${escapeHtml(plaintext)}</div>`;
  html += `<div class="frame-hex">${hexPreview(bytes)}</div>`;
  el.innerHTML = html;
  els.frames.appendChild(el);
  els.frames.scrollTop = els.frames.scrollHeight;
  while (els.frames.children.length > 60) els.frames.removeChild(els.frames.firstChild);
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ '&':'&amp;','<':'&lt;','>':'&gt;' }[c]));
}

function addHsStep(label, code) {
  const li = document.createElement('li');
  li.innerHTML = `<span class="tick">${ICON.check}</span> <span>${label}</span>` +
                 (code ? ` <code>${code}</code>` : '');
  els.hsSteps.appendChild(li);
}

// ---- chat rendering ----------------------------------------------------
function addMessage(role, text = '') {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="role">${role === 'you' ? 'you' : 'aspida'}</div>` +
                 `<span class="text"></span>`;
  el.querySelector('.text').textContent = text;
  els.messages.appendChild(el);
  els.messages.scrollTop = els.messages.scrollHeight;
  return el;
}

// ---- streaming state ---------------------------------------------------
let aiEl = null, aiText = '', tokDecoder = null, tokCount = 0, tStart = 0;
let thinkingEl = null;

function beginReply() {
  aiText = ''; tokCount = 0; tStart = 0;
  tokDecoder = new TextDecoder('utf-8');
  aiEl = null;
  thinkingEl = addMessage('ai', '');
  thinkingEl.querySelector('.text').innerHTML = ICON.spin + ' thinking';
  thinkingEl.classList.add('thinking');
}
function onToken(bytes) {
  if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
  if (!aiEl) { aiEl = addMessage('ai', ''); tStart = performance.now(); }
  // on mobile, flag the chat tab if the user is viewing the encryption tab
  if (!document.body.classList.contains('tab-chat')) els.chatBadge.hidden = false;
  aiText += tokDecoder.decode(bytes, { stream: true });
  tokCount++;
  aiEl.querySelector('.text').innerHTML = escapeHtml(aiText) + '<span class="cursor"></span>';
  els.messages.scrollTop = els.messages.scrollHeight;
  const secs = (performance.now() - tStart) / 1000;
  if (secs > 0) els.ratePill.textContent = (tokCount / secs).toFixed(1) + ' tok/s';
}
function endReply() {
  if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
  if (aiEl) aiEl.querySelector('.text').textContent = aiText.trim();
  els.input.disabled = false; els.send.disabled = false; els.input.focus();
}

// ---- wire up the secure channel ---------------------------------------
const hooks = {
  onStep(s) {
    if (s.step === 'ephemeral') addHsStep('Generated ephemeral key', s.pub.slice(0, 16) + '…');
    else if (s.step === 'dh') addHsStep('Computed shared secret');
    else if (s.step === 'derive') addHsStep('Derived session keys');
    else if (s.step === 'confirmed') {
      addHsStep('Server authenticated — no MITM, forward secrecy');
      els.bindingBox.hidden = false;
    }
  },
  onFrameOut(f) { addFrame({ dir: 'out', kind: f.kind, label: f.label, bytes: f.bytes }); bumpRec(); },
  onFrameIn(f)  { addFrame({ dir: 'in',  kind: f.kind, label: f.label, bytes: f.bytes }); bumpRec(); },
  onRecordOut(r) {
    addFrame({ dir: 'out', label: 'record ' + tagName(r.tag),
               plaintext: recordPreview(r.plaintext), bytes: r.frame, nonce: r.nonce.toString() });
    bumpRec();
  },
  onRecordIn(r) {
    addFrame({ dir: 'in', label: 'record ' + tagName(r.plaintext[0]),
               plaintext: recordPreview(r.plaintext), bytes: r.frame, nonce: r.nonce.toString() });
    bumpRec();
  },
  onAuthError() { setStatus('err', 'authentication error'); },
  onClose() {
    if (switching) { scheduleReconnect(); return; }   // model reload in progress
    setStatus('err', 'connection closed'); els.input.disabled = true; els.send.disabled = true;
  },
};

// ---- model picker -------------------------------------------------------
let switching = false;

function populateModels(list) {
  // Only offer a picker when the server can actually switch and has options.
  if (!list || !list.switchable || !Array.isArray(list.data) || list.data.length < 2) {
    els.modelSelect.hidden = true;
    return;
  }
  els.modelSelect.innerHTML = '';
  for (const m of list.data) {
    const o = document.createElement('option');
    o.value = m.id;
    const bits = [m.name, m.params, m.quant, m.size].filter(Boolean).join(' · ');
    o.textContent = bits + (m.supported ? '' : ' (unsupported)');
    o.disabled = !m.supported;
    if (m.active) o.selected = true;
    els.modelSelect.appendChild(o);
  }
  els.modelSelect.hidden = false;
}

function onResp(jsonText) {
  let r; try { r = JSON.parse(jsonText); } catch { return; }
  if (Array.isArray(r.data)) { populateModels(r); return; }        // catalog
  if ('ok' in r) {                                                 // select result
    if (!r.ok) {
      switching = false; setStatus('err', r.message || 'model not available');
      els.input.disabled = false; els.send.disabled = false;
    } else if (!r.reload) {
      switching = false; setStatus('live', r.message || 'model selected');
      els.input.disabled = false; els.send.disabled = false;
    } // r.reload: server is restarting; onClose -> scheduleReconnect
  }
}

function scheduleReconnect() {
  setStatus('', 'loading model — reconnecting…');
  let tries = 0;
  const tick = () => {
    tries += 1;
    start().then(() => { switching = false; })
           .catch(() => { if (tries < 90) setTimeout(tick, 2000);
                          else setStatus('err', 'reconnect failed'); });
  };
  setTimeout(tick, 2500);
}

function setStatus(kind, text) {
  els.status.textContent = text;
  els.dot.className = 'dot' + (kind === 'live' ? ' live' : kind === 'err' ? ' err' : '');
}

let channel = null;

async function start() {
  setStatus('', 'fetching server key…');
  const pubHex = (await (await fetch('/serverkey')).text()).trim();
  channel = new SecureChannel(pubHex, hooks);

  channel.onRecord = (tag, body) => {
    if (tag === Tag.Session) {
      els.input.disabled = false; els.send.disabled = false; els.input.focus();
      setStatus('live', 'secured · session ' + dec.decode(body).slice(0, 8));
      channel.sendText(Tag.Models, '');     // ask what models are available
    } else if (tag === Tag.Resp) {
      onResp(dec.decode(body));
    } else if (tag === Tag.Prefill) {
      // thinking dots already shown by beginReply
    } else if (tag === Tag.Token) {
      onToken(body);
    } else if (tag === Tag.Done) {
      endReply();
    } else if (tag === Tag.Error) {
      endReply();
      addMessage('ai', '').querySelector('.text').innerHTML =
        ICON.warn + ' ' + escapeHtml(dec.decode(body));
    }
  };

  setStatus('', 'handshake…');
  const info = await channel.connect(location.origin.replace(/^http/, 'ws') + '/ws');
  els.cipher.textContent = info.cipher;
  els.srvKey.textContent = info.serverKey;
  els.binding.textContent = info.binding;
  els.modelPill.innerHTML = ICON.lock + ' inference on server';
  channel.sendText(Tag.Session, '');     // request a new session
}

els.composer.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text || els.input.disabled) return;
  addMessage('you', text);
  els.input.value = '';
  els.input.disabled = true; els.send.disabled = true;
  els.ratePill.textContent = '0 tok/s';
  beginReply();
  channel.sendText(Tag.Prompt, text);
});

els.operatorView.addEventListener('change', () => {
  document.body.classList.toggle('operator', els.operatorView.checked);
});

els.modelSelect.addEventListener('change', () => {
  const id = els.modelSelect.value;
  if (!id) return;
  switching = true;
  setStatus('', 'switching model…');
  els.input.disabled = true; els.send.disabled = true;
  channel.sendText(Tag.Select, id);
});

// ---- view (Demo/Docs) + mobile tab switching ----
document.body.classList.add('tab-chat');

function setView(view) {           // 'demo' | 'docs'
  document.body.classList.toggle('view-docs', view === 'docs');
  document.body.classList.toggle('view-demo', view !== 'docs');
  document.querySelectorAll('.topnav button')
    .forEach((x) => x.classList.toggle('active', x.dataset.view === view));
  // keep the mobile tabbar's Docs button in sync
  document.querySelectorAll('.tabbar button').forEach((x) => {
    if (x.dataset.tab === 'docs') x.classList.toggle('active', view === 'docs');
    else if (view === 'docs') x.classList.remove('active');
  });
}

function setTab(tab) {             // 'chat' | 'wire' (demo sub-views)
  document.body.classList.remove('tab-chat', 'tab-wire');
  document.body.classList.add('tab-' + tab);
  document.querySelectorAll('.tabbar button')
    .forEach((x) => x.classList.toggle('active', x.dataset.tab === tab));
  if (tab === 'chat') {
    els.chatBadge.hidden = true;
    if (!els.input.disabled) els.input.focus();
  }
}

document.querySelectorAll('.topnav button').forEach((b) =>
  b.addEventListener('click', () => setView(b.dataset.view)));

document.querySelectorAll('.tabbar button').forEach((b) =>
  b.addEventListener('click', () => {
    if (b.dataset.tab === 'docs') { setView('docs'); }
    else { setView('demo'); setTab(b.dataset.tab); }
  }));

start().catch((err) => { setStatus('err', 'error: ' + err.message); console.error(err); });
