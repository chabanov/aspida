# План повної реалізації Qwen3.5-35B-A3B (MoE + SSM гібрид) на ADA/SPARK

> Ціль: inference гібридної моделі Qwen3.5-MoE+SSM напряму з GGUF (Q5_K_M),
> без зовнішніх бібліотек, на ADA/SPARK. Бінарник `obj/llm_main`, чат-REPL.

## Архітектура моделі (з GGUF metadata)

| Параметр | Значення |
|----------|----------|
| Шарів (block_count) | 40 |
| embedding_length (d_model) | 2048 |
| Attention heads | 16, head_dim = 256 (key/value_length) |
| KV heads (GQA) | 2 |
| Experts | 256, top-8 used, expert_ffn=512 |
| Shared expert | expert_shared_ffn=512 |
| RoPE | mrope, freq_base=1e7, dim=64, sections=[11,11,10,0] |
| RMSNorm epsilon | 1e-6 |
| Context | 262144, vocab = 248320 |
| SSM (Mamba) | conv_kernel=4, state=128, groups=16, dt_rank=32, inner=4096 |
| full_attention_interval | 4 (шари 0,4,8… — full attn; решта — SSM/sliding) |

**Тензори шару `blk.N.`:** `attn_qkv.weight` (об'єднаний QKV), `attn_gate.weight`,
`attn_norm`, `post_attention_norm`, `ffn_gate_inp` (роутер), `ffn_gate_exps`/
`ffn_down_exps`/`ffn_up_exps` (256 експертів), `ffn_*_shexp` (shared), SSM-блок:
`ssm_a`/`ssm_alpha`/`ssm_beta`/`ssm_conv1d`/`ssm_dt`/`ssm_norm`/`ssm_out`.

## Стан коду

| Готово | Частково | Відсутнє |
|--------|----------|----------|
| GGUF parser (header, metadata, tensor info) | `llm_ssm.adb` (буфер, shift) | Q8_K / Q4_K / Q6_K dequant |
| F32/F16/Q5_K dequant | `llm_attention.adb` (наївний) | QKV split + GQA |
| Tensor heap-alloc load (фікс stack overflow) | `llm_mlp.adb` (dense) | MoE forward (роутер+top-k) |
| GPT-2 fallback (працює) | — | Повний SSM selective scan |
| Tokenizer gpt2 BPE (частково) | — | mRoPE, гібридний layer routing |

---

## Етапи

### Етап 1 — Квантизація (фундамент)
**Файл:** `src/llm/llm_dequant.adb` (+.ads)
- [ ] `Dequant_Q8_K` — блоки 256 елем., scale(f16)+qs(int8). Критично — `token_embd` і більшість ваг.
- [ ] `Dequant_Q6_K` — 256-блок, 6-біт + scale.
- [ ] `Dequant_Q4_K` — 256-блок, 4-біт + 2 scale/min рівні (super-block).
- [ ] Юніт-тести dequant на синтетичних блоках (відомий вхід→вихід).
**Критерій:** `Dequantize` не кидає `unsupported type` на жодному тензорі моделі.

### Етап 2 — Конфіг моделі з metadata
**Файли:** `src/llm/llm_qwen.ads/.adb`
- [ ] Розширити `Qwen_Model` полями: N_KV_Heads, Head_Dim, N_Experts, N_Experts_Used,
      Expert_FFN, RoPE_Base, RoPE_Sections, RMS_Eps, SSM_* (conv/state/groups/dt_rank/inner).
- [ ] `Read_Meta_Int`/`Read_Meta_Float` з fallback префіксами, читати всі гіперпараметри.
- [ ] Визначати тип шару: `Is_Full_Attention (Layer) := Layer mod 4 = 0`.
**Критерій:** усі гіперпараметри прочитані, надруковані, збігаються з таблицею.

### Етап 3 — Завантаження ваг шару
**Файл:** `src/llm/llm_qwen.adb`
- [ ] Структура `Layer_Weights`: attn (qkv, gate, norm, post_norm), moe (gate_inp,
      gate/up/down_exps[256], shexp), ssm (a/alpha/beta/conv1d/dt/norm/out).
- [ ] Lazy-load по імені `blk.N.<tensor>`, heap-alloc, з обробкою відсутніх (SSM vs attn шари).
**Критерій:** усі 40 шарів завантажуються без `tensor not found`.

### Етап 4 — RMSNorm + mRoPE
**Файли:** `src/llm/llm_rmsnorm.ad*`, `src/llm/llm_rope.ad*`
- [ ] RMSNorm з epsilon=1e-6.
- [ ] mRoPE з sections=[11,11,10,0], freq_base=1e7, dim=64 (multimodal rotary, для тексту — 1D частина).
**Критерій:** числові тести проти референсних значень.

### Етап 5 — Attention (full + GQA)
**Файл:** `src/llm/llm_attention.adb`
- [ ] QKV split з `attn_qkv` → Q[16×256], K[2×256], V[2×256].
- [ ] GQA: розширення 2 KV-heads на 16 Q-heads (repeat 8×).
- [ ] Scaled dot-product + causal mask + softmax + attn_gate (sigmoid gating).
- [ ] KV-cache для авторегресії.
**Критерій:** full-attention шар видає коректну форму, gating застосовано.

### Етап 6 — SSM / Mamba selective scan
**Файл:** `src/llm/llm_ssm.adb`
- [ ] conv1d (kernel=4, causal) над входом.
- [ ] selective scan: A=−exp(ssm_a), dt=softplus(ssm_dt), B/C з ssm_beta/alpha, state=128.
- [ ] gated output через ssm_norm + ssm_out.
**Критерій:** SSM-шар видає послідовність правильної форми, стан переноситься між токенами.

### Етап 7 — MoE FFN
**Файл:** `src/llm/llm_mlp.adb` (→ `llm_moe.adb`)
- [ ] Роутер: `ffn_gate_inp` → логіти 256 → softmax → top-8 індекси+ваги.
- [ ] Для кожного з 8 експертів: SwiGLU (gate_exps, up_exps, down_exps).
- [ ] Shared expert (завжди активний) + зважена сума.
**Критерій:** MoE видає вектор d_model, активні рівно 8+shared експертів.

### Етап 8 — Складання forward pass
**Файл:** `src/llm/llm_qwen.adb` (`Forward`, `Generate`)
- [ ] Цикл по 40 шарах: norm → (attn|ssm за інтервалом) → residual → post_norm → moe → residual.
- [ ] Final norm → output projection (`output.weight` / tie з token_embd) → логіти.
- [ ] Семплінг: greedy + temperature + top-p.
**Критерій:** `Generate` повертає осмислений токен-стрім (не випадковий шум).

### Етап 9 — Токенізатор
**Файл:** `src/llm/llm_tokenizer.adb`
- [ ] gpt2 BPE merges + vocab з GGUF (`tokenizer.ggml.tokens/merges`).
- [ ] Encode (текст→id) + Decode (id→текст) + спецтокени Qwen (chat template).
**Критерій:** round-trip encode→decode = вихідний текст.

### Етап 10 — Інтеграція + чат
**Файл:** `src/llm/llm_chat.adb`
- [ ] Chat template Qwen (`<|im_start|>`/`<|im_end|>`).
- [ ] REPL: prompt→encode→generate→decode→stream, KV-cache між репліками.
- [ ] Прибрати debug-логи, додати `--verbose`.
**Критерій:** інтерактивний осмислений діалог з моделлю.

### Етап 11 — Продуктивність (опційно)
- [ ] mmap GGUF замість read (24GB модель).
- [ ] SIMD/паралельний matmul (`GNAT` tasks по головах/експертах).
- [ ] Квантований matmul без повного dequant (економія RAM).

### Етап 12 — SPARK-верифікація
- [ ] `make prove` — flow-аналіз чистих модулів (dequant, norm, rope).
- [ ] Контракти Pre/Post на формах тензорів.

---

## Послідовність / залежності
```
Етап1 (dequant) ─┬─> Етап3 (ваги) ─> Етап5 (attn) ─┐
Етап2 (конфіг) ──┘                  ├─> Етап6 (ssm) ─┼─> Етап8 (forward) ─> Етап10 (чат)
Етап4 (norm/rope) ─────────────────┘   Етап7 (moe) ─┘        ▲
Етап9 (tokenizer) ──────────────────────────────────────────┘
```

## Оцінка обсягу
~700-1000 рядків нового ADA. Найскладніше: SSM selective scan (Етап 6),
MoE роутинг (Етап 7), Q4_K/Q6_K dequant (Етап 1).

## Ризики
- **RAM**: 35B у Q5_K ≈ 24GB. Хост має 64GB — ок, але dequant у f32 роздуває. Тримати ваги квантованими.
- **Швидкість**: чистий ADA без SIMD — повільно. Перший токен може бути хвилини. Етап 11 критичний для UX.
- **mRoPE sections**: рідкісна схема, мало референсів — ризик числових багів.
- **Точність dequant Q4_K/Q6_K**: super-block формат складний, легко помилитись на офсетах.
