#!/usr/bin/env python3
"""
convert_gpt2.py — Extract GPT-2 weights and save as raw FP32 binary
for the Aspida ADA/SPARK LLM backend.

Produces:
  models/gpt2_small/
    config.txt        — dim, n_heads, n_layers, vocab_size
    wte.bin           — token embeddings [vocab, dim]
    wpe.bin           — position embeddings [max_seq, dim]
    h_N_qw.bin        — attention Q weight [dim, dim] per layer
    h_N_kw.bin        — attention K weight [dim, dim]
    h_N_vw.bin        — attention V weight [dim, dim]
    h_N_qb.bin        — attention Q bias [dim]
    h_N_kb.bin        — attention K bias [dim]  
    h_N_vb.bin        — attention V bias [dim]
    h_N_ow.bin        — attention out weight [dim, dim]
    h_N_ob.bin        — attention out bias [dim]
    h_N_ln1w.bin      — layer norm 1 weight [dim]
    h_N_ln1b.bin      — layer norm 1 bias [dim]
    h_N_ln2w.bin      — layer norm 2 weight [dim]
    h_N_ln2b.bin      — layer norm 2 bias [dim]
    h_N_fcw.bin       — MLP fc weight [dim, 4*dim]
    h_N_fcb.bin       — MLP fc bias [4*dim]
    h_N_pw.bin        — MLP proj weight [4*dim, dim]
    h_N_pb.bin        — MLP proj bias [dim]
    lnfw.bin          — final layer norm weight [dim]
    lnfb.bin          — final layer norm bias [dim]
    vocab.json         — BPE tokenizer vocabulary
    merges.txt         — BPE merge rules
"""

import numpy as np
import json
import os
import struct
from transformers import GPT2Model, GPT2TokenizerFast

MODEL_NAME = "gpt2"  # GPT-2 small (124M)
OUT_DIR = "models/gpt2_small"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading {MODEL_NAME}...")
model = GPT2Model.from_pretrained(MODEL_NAME)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)

state = model.state_dict()
config = model.config

# Save config
print(f"Config: dim={config.n_embd}, heads={config.n_head}, layers={config.n_layer}, vocab={config.vocab_size}")
with open(f"{OUT_DIR}/config.txt", "w") as f:
    f.write(f"vocab_size={config.vocab_size}\n")
    f.write(f"dim={config.n_embd}\n")
    f.write(f"n_heads={config.n_head}\n")
    f.write(f"n_layers={config.n_layer}\n")
    f.write(f"max_seq_len={config.n_positions}\n")

def save_tensor(name: str, arr: np.ndarray):
    """Save as row-major FP32 binary (ADA compatible)."""
    path = f"{OUT_DIR}/{name}.bin"
    arr_f32 = arr.astype(np.float32)
    with open(path, "wb") as f:
        f.write(arr_f32.tobytes())
    print(f"  {name}: {list(arr.shape)} → {path} ({arr_f32.nbytes} bytes)")

# Token embeddings
save_tensor("wte", state["wte.weight"].numpy())          # [50257, 768]
save_tensor("wpe", state["wpe.weight"].numpy())          # [1024, 768]

# Split QKV from c_attn: [dim, 3*dim] → 3 × [dim, dim]
for layer in range(config.n_layer):
    prefix = f"h.{layer}"
    
    c_attn_w = state[f"{prefix}.attn.c_attn.weight"].numpy()  # [768, 2304]
    c_attn_b = state[f"{prefix}.attn.c_attn.bias"].numpy()    # [2304]
    
    dim = config.n_embd
    q_w, k_w, v_w = c_attn_w[:, :dim], c_attn_w[:, dim:2*dim], c_attn_w[:, 2*dim:]
    q_b, k_b, v_b = c_attn_b[:dim], c_attn_b[dim:2*dim], c_attn_b[2*dim:]
    
    save_tensor(f"h_{layer}_qw", q_w)
    save_tensor(f"h_{layer}_kw", k_w)
    save_tensor(f"h_{layer}_vw", v_w)
    save_tensor(f"h_{layer}_qb", q_b)
    save_tensor(f"h_{layer}_kb", k_b)
    save_tensor(f"h_{layer}_vb", v_b)
    
    # Output projection
    save_tensor(f"h_{layer}_ow", state[f"{prefix}.attn.c_proj.weight"].numpy())
    save_tensor(f"h_{layer}_ob", state[f"{prefix}.attn.c_proj.bias"].numpy())
    
    # Layer norms
    save_tensor(f"h_{layer}_ln1w", state[f"{prefix}.ln_1.weight"].numpy())
    save_tensor(f"h_{layer}_ln1b", state[f"{prefix}.ln_1.bias"].numpy())
    save_tensor(f"h_{layer}_ln2w", state[f"{prefix}.ln_2.weight"].numpy())
    save_tensor(f"h_{layer}_ln2b", state[f"{prefix}.ln_2.bias"].numpy())
    
    # MLP
    save_tensor(f"h_{layer}_fcw", state[f"{prefix}.mlp.c_fc.weight"].numpy())   # [768, 3072]
    save_tensor(f"h_{layer}_fcb", state[f"{prefix}.mlp.c_fc.bias"].numpy())     # [3072]
    save_tensor(f"h_{layer}_pw", state[f"{prefix}.mlp.c_proj.weight"].numpy())   # [3072, 768]
    save_tensor(f"h_{layer}_pb", state[f"{prefix}.mlp.c_proj.bias"].numpy())     # [768]

# Final layer norm
save_tensor("lnfw", state["ln_f.weight"].numpy())
save_tensor("lnfb", state["ln_f.bias"].numpy())

# Save tokenizer vocab + merges
vocab = tokenizer.get_vocab()
with open(f"{OUT_DIR}/vocab.json", "w") as f:
    json.dump(vocab, f)
    
merges = tokenizer._tokenizer.model.merges if hasattr(tokenizer._tokenizer, 'model') else []
with open(f"{OUT_DIR}/merges.txt", "w") as f:
    for m in merges:
        f.write(f"{m[0]} {m[1]}\n")

print(f"\nDone! Weights saved to {OUT_DIR}/")
print(f"Total weight files: {len(os.listdir(OUT_DIR))}")
print(f"Total size: {sum(os.path.getsize(f'{OUT_DIR}/{f}') for f in os.listdir(OUT_DIR)) / 1024 / 1024:.1f} MB")
