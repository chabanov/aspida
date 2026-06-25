#!/usr/bin/env python3
"""svg_icons.py — P4 domain: simple 64x64 icons with a RENDER-based verifier.

A narrow, learnable grammar: one primitive (circle/rect/triangle), one of a few
colors, on a 3x3 position grid, a few sizes. Each spec has a canonical SVG that
renders to a target image; the verifier renders a CANDIDATE SVG and accepts it if
its pixels match the target (MSE below threshold). This is the executable oracle
that lets a verifier-filtered student exceed a noisy teacher (the code_distill
mechanism, on a real renderable domain).

  python3 svg_icons.py selftest        # canonical passes, corrupted fails
  python3 svg_icons.py gen N out_dir   # write N (spec, svg, target.png) examples
"""
import sys, io, json, random

SHAPES = ["circle", "rect", "tri"]
COLORS = {"red": "#ee2233", "green": "#22aa44", "blue": "#2244cc", "black": "#111111"}
POS    = [16, 32, 48]            # cx / cy grid
SIZES  = [8, 12, 16]
CANVAS = 64

def all_specs():
    out = []
    for sh in SHAPES:
        for col in COLORS:
            for cx in POS:
                for cy in POS:
                    for sz in SIZES:
                        out.append((sh, col, cx, cy, sz))
    return out                  # 3*4*3*3*3 = 324 distinct icons

def spec_to_svg(spec):
    sh, col, cx, cy, sz = spec
    c = COLORS[col]
    if sh == "circle":
        body = f'<circle cx="{cx}" cy="{cy}" r="{sz}" fill="{c}"/>'
    elif sh == "rect":
        body = f'<rect x="{cx-sz}" y="{cy-sz}" width="{2*sz}" height="{2*sz}" fill="{c}"/>'
    else:  # triangle
        body = f'<polygon points="{cx},{cy-sz} {cx-sz},{cy+sz} {cx+sz},{cy+sz}" fill="{c}"/>'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS}" height="{CANVAS}">'
            f'{body}</svg>')

SH_CH  = {"circle": "c", "rect": "r", "tri": "t"}
COL_CH = {"red": "r", "green": "g", "blue": "b", "black": "k"}

def prefix(spec):
    """Compact fixed-width spec token string the student is conditioned on."""
    sh, col, cx, cy, sz = spec
    return f"{SH_CH[sh]}{COL_CH[col]} {cx:02d} {cy:02d} {sz:02d}|"

def compact_svg(spec):
    """Same image as spec_to_svg but without xmlns / extra chars (shorter to learn).
    Identical colours so it renders pixel-equal to the canonical target."""
    sh, col, cx, cy, sz = spec
    c = COLORS[col]
    if sh == "circle":
        body = f'<circle cx="{cx}" cy="{cy}" r="{sz}" fill="{c}"/>'
    elif sh == "rect":
        body = f'<rect x="{cx-sz}" y="{cy-sz}" width="{2*sz}" height="{2*sz}" fill="{c}"/>'
    else:
        body = f'<polygon points="{cx},{cy-sz} {cx-sz},{cy+sz} {cx+sz},{cy+sz}" fill="{c}"/>'
    return f'<svg width="64" height="64">{body}</svg>'

def noisy_teacher(spec, rng):
    """A noisy teacher: 60% correct (compact canonical), 40% an error."""
    r = rng.random()
    if r < 0.60:
        return compact_svg(spec)
    elif r < 0.75:                                   # wrong colour
        bad = (spec[0], rng.choice([x for x in COLORS if x != spec[1]]), *spec[2:])
        return compact_svg(bad)
    elif r < 0.90:                                   # wrong position
        bad = (spec[0], spec[1], rng.choice(POS), rng.choice(POS), spec[4])
        return compact_svg(bad)
    else:                                            # malformed
        return compact_svg(spec)[:-6]

def prompt(spec):
    sh, col, cx, cy, sz = spec
    name = {"circle": "circle", "rect": "square", "tri": "triangle"}[sh]
    return (f"Output ONLY a 64x64 SVG of a {col} {name} centered at ({cx},{cy}) "
            f"with size {sz} on a white background. No prose.")

# ---- rendering + verification (cairosvg + numpy) ----
def render(svg_str):
    """Render an SVG string to a 64x64x3 RGB numpy array (0..255). None on failure.
    RGB (not grayscale) so wrong colours are caught."""
    import cairosvg, numpy as np
    from PIL import Image
    try:
        png = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=CANVAS, output_height=CANVAS)
        img = Image.open(io.BytesIO(png)).convert("RGB")
        return np.asarray(img, dtype="float32")
    except Exception:
        return None

def verify(candidate_svg, target, thresh=50.0):
    """True iff candidate renders and its pixels match target (RGB mean-squared error < thresh)."""
    import numpy as np
    a = render(candidate_svg)
    if a is None or a.shape != target.shape:
        return False
    return float(np.mean((a - target) ** 2)) < thresh

# ---- CLI ----
def selftest():
    import numpy as np
    specs = all_specs()
    print(f"grammar: {len(specs)} distinct icons")
    ok = 0; n = 0
    rng = random.Random(7)
    for spec in rng.sample(specs, 12):
        svg = spec_to_svg(spec)
        tgt = render(svg)
        if tgt is None:
            print(f"  render FAIL {spec}"); continue
        n += 1
        # canonical must pass against its own target
        good = verify(svg, tgt)
        # a corrupted candidate (wrong color) must fail
        bad_spec = (spec[0], "red" if spec[1] != "red" else "blue", *spec[2:])
        bad = verify(spec_to_svg(bad_spec), tgt)
        # malformed svg must fail
        malformed = verify("<svg><circle", tgt)
        if good and not bad and not malformed:
            ok += 1
        else:
            print(f"  CASE FAIL {spec}: good={good} bad={bad} malformed={malformed}")
    print(f"selftest: {ok}/{n} icons (canonical passes, wrong-color + malformed fail)")
    print("RESULT:", "PASS" if ok == n and n > 0 else "FAIL")
    return 0 if ok == n and n > 0 else 1

def build_dataset(out_dir):
    """Verifier-filtered training set + held-out eval with the teacher baseline.
    Writes train.txt (prefix+compact_svg lines, verified-correct only), eval.txt
    (one prefix per held-out spec) and targets/*.png; prints teacher pass-rates."""
    import os, numpy as np
    from PIL import Image
    os.makedirs(out_dir + "/targets", exist_ok=True)
    specs = all_specs(); rng = random.Random(2); rng.shuffle(specs)
    split = int(len(specs) * 0.8)
    train_specs, eval_specs = specs[:split], specs[split:]

    # sanity: compact_svg renders pixel-equal to the canonical target
    s0 = train_specs[0]; tgt0 = render(spec_to_svg(s0)); cmp0 = render(compact_svg(s0))
    eq = (cmp0 is not None and float(np.mean((tgt0 - cmp0) ** 2)) < 1.0)
    print("compact==canonical render:", eq)

    # training: verifier-filtered teacher outputs
    tr = open(out_dir + "/train.txt", "w"); n_tr = 0; t_filter_ok = 0
    for sp in train_specs:
        tgt = render(spec_to_svg(sp)); tsvg = noisy_teacher(sp, rng)
        if verify(tsvg, tgt):
            tr.write(prefix(sp) + compact_svg(sp) + "\n"); n_tr += 1; t_filter_ok += 1
    tr.close()

    # held-out eval: teacher pass-rate + write prefixes + targets
    ev = open(out_dir + "/eval.txt", "w"); t_eval_ok = 0
    for i, sp in enumerate(eval_specs):
        tgt = render(spec_to_svg(sp))
        Image.fromarray(tgt.astype("uint8")).save(f"{out_dir}/targets/e{i:04d}.png")
        ev.write(prefix(sp) + "\n")
        if verify(noisy_teacher(sp, rng), tgt):
            t_eval_ok += 1
    ev.close()

    max_len = max(len(prefix(s)) + len(compact_svg(s)) for s in train_specs)
    print(f"train examples (verified): {n_tr}/{len(train_specs)}  "
          f"(teacher train-filter pass {t_filter_ok/len(train_specs):.2%})")
    print(f"TEACHER held-out pass-rate: {t_eval_ok}/{len(eval_specs)} = {t_eval_ok/len(eval_specs):.2%}")
    print(f"max sequence length (prefix+svg): {max_len}")
    return 0

def eval_student(out_dir):
    """Render+verify the student's generated SVGs (student_svgs.txt) against the
    held-out targets; print the student pass-rate (compare to the teacher's)."""
    import numpy as np
    from PIL import Image
    svgs = open(out_dir + "/student_svgs.txt").read().splitlines()
    ok = 0; n = 0
    for i, svg in enumerate(svgs):
        try:
            tgt = np.asarray(Image.open(f"{out_dir}/targets/e{i:04d}.png").convert("RGB"), dtype="float32")
        except Exception:
            continue
        n += 1
        if verify(svg, tgt):
            ok += 1
    print(f"STUDENT held-out pass-rate: {ok}/{n} = {ok/n:.2%}")
    return 0

def gen(n, out_dir):
    import os, numpy as np
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    specs = all_specs(); rng = random.Random(1); rng.shuffle(specs)
    meta = []
    for i, spec in enumerate(specs[:n]):
        svg = spec_to_svg(spec); tgt = render(svg)
        if tgt is None: continue
        Image.fromarray(tgt.astype("uint8")).save(f"{out_dir}/t{i:05d}.png")
        meta.append({"i": i, "spec": spec, "prompt": prompt(spec), "svg": svg})
    json.dump(meta, open(f"{out_dir}/meta.json", "w"))
    print(f"wrote {len(meta)} examples to {out_dir}")
    return 0

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    elif len(sys.argv) >= 3 and sys.argv[1] == "dataset":
        sys.exit(build_dataset(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "eval":
        sys.exit(eval_student(sys.argv[2]))
    elif len(sys.argv) >= 4 and sys.argv[1] == "gen":
        sys.exit(gen(int(sys.argv[2]), sys.argv[3]))
    else:
        print(__doc__); sys.exit(2)
