"""
faz9_1b5_local.py
=================
1.5B için eksiksiz AEK değerlendirmesi — Mac veya AWS.

Adımlar:
  1. AEK compress + model kayıt (model varsa atla)
  2. lm-eval: Original model (HellaSwag, ARC-Easy, ARC-Challenge)
  3. lm-eval: AEK compressed model

Çalıştır:
    python faz9_1b5_local.py
    python faz9_1b5_local.py --eps 0.20
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=float, default=0.26,
                    help='AEK epsilon threshold (default 0.26 for 1.5B)')
parser.add_argument('--fisher_samples', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=8)
args = parser.parse_args()

EPS          = args.eps
EPS_STR      = str(EPS).replace('.', '')
MODEL_ID     = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR  = Path("./results")
MODELS_DIR   = Path("./models")
AEK_SAVE_DIR = str(MODELS_DIR / f"qwen_1b5_aek_eps{EPS_STR}_lmeval")
COMPRESS_JSON = str(RESULTS_DIR / f"faz9_1b5_eps{EPS_STR}_compress.json")
TASKS        = "hellaswag,arc_easy,arc_challenge"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  FAZ9 — 1.5B Eksiksiz AEK Değerlendirmesi")
print(f"  Model: {MODEL_ID}")
print(f"  ε={EPS}")
print("=" * 60)

# ── ADIM 1: AEK compress + model kayıt ─────────────────────────
print(f"\n[1/3] AEK compress → {AEK_SAVE_DIR}")

if os.path.exists(AEK_SAVE_DIR) and any(
    f.endswith(".safetensors") or f.endswith(".bin")
    for f in os.listdir(AEK_SAVE_DIR)
):
    print("  Model zaten mevcut, atlanıyor.")
    if os.path.exists(COMPRESS_JSON):
        with open(COMPRESS_JSON) as f:
            compress_data = json.load(f)
    else:
        compress_data = {}
else:
    cmd = [
        sys.executable, "02b_compress_fixed.py",
        "--eps",            str(EPS),
        "--model",          MODEL_ID,
        "--dtype",          "bfloat16",
        "--fisher_samples", str(args.fisher_samples),
        "--save_model_dir", AEK_SAVE_DIR,
        "--output",         COMPRESS_JSON,
        "--no_plot",
    ]
    print("  CMD:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [!] Compress HATA: {result.returncode}")
        sys.exit(1)
    print("  Compress + kayıt tamamlandı.")
    with open(COMPRESS_JSON) as f:
        compress_data = json.load(f)


# ── ADIM 2: lm-eval helper ─────────────────────────────────────
def run_lmeval(model_path: str, label: str) -> dict:
    out_prefix = str(RESULTS_DIR / f"lmeval_{label}_1b5_eps{EPS_STR}")
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", TASKS,
        "--num_fewshot", "0",
        "--batch_size", str(args.batch_size),
        "--output_path", out_prefix,
    ]
    print(f"\n  lm-eval [{label}] başlıyor...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [!] HATA: returncode={result.returncode}")
        return {"label": label, "error": f"returncode={result.returncode}"}

    # lm_eval creates nested subdirs: {prefix}/{model_name}/results_*.json
    matches = glob.glob(f"{out_prefix}/**/*.json", recursive=True)
    if not matches:
        matches = glob.glob(f"{out_prefix}*.json")  # fallback flat
    if not matches:
        return {"label": label, "error": "output file not found"}

    with open(sorted(matches)[-1]) as f:
        raw = json.load(f)

    r = raw.get("results", {})
    out = {"label": label, "elapsed_min": round(elapsed / 60, 1)}
    for task in ["hellaswag", "arc_easy", "arc_challenge"]:
        t = r.get(task, {})
        acc = t.get("acc_norm,none", t.get("acc,none", None))
        out[task] = round(acc * 100, 2) if acc is not None else None
    print(f"  HellaSwag: {out.get('hellaswag')}%")
    print(f"  ARC-Easy:  {out.get('arc_easy')}%")
    print(f"  ARC-Chall: {out.get('arc_challenge')}%")
    print(f"  Süre: {elapsed/60:.1f} dk")
    return out


# ── ADIM 3: lm-eval çalıştır ────────────────────────────────────
print("\n[2/3] lm-eval: Original model...")
orig_lm = run_lmeval(MODEL_ID, "original")

print("\n[3/3] lm-eval: AEK compressed model...")
aek_lm = run_lmeval(AEK_SAVE_DIR, "aek")

# ── Final sonuç ─────────────────────────────────────────────────
final = {
    "model":        MODEL_ID,
    "eps":          EPS,
    "n_eliminated": compress_data.get("n_eliminated"),
    "E_total":      compress_data.get("E_total"),
    "E_bound":      compress_data.get("E_bound"),
    "theorem_sat":  compress_data.get("theorem_satisfied"),
    "ppl_original": compress_data.get("ppl_baseline"),
    "ppl_aek":      compress_data.get("ppl_compressed"),
    "dppl_pct":     compress_data.get("dppl_pct"),
    "lmeval_original": orig_lm,
    "lmeval_aek":   aek_lm,
}

out_path = RESULTS_DIR / f"faz9_1b5_eps{EPS_STR}_full.json"
with open(out_path, "w") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

ppl_o = final.get("ppl_original") or 0
ppl_a = final.get("ppl_aek")      or 0
dppl  = final.get("dppl_pct")     or 0

print(f"\n{'='*60}")
print(f"  SONUÇ — 1.5B  ε={EPS}")
print(f"{'='*60}")
print(f"  PPL: {ppl_o:.4f} → {ppl_a:.4f}  ({dppl:+.2f}%)")
print(f"  Elim: {final['n_eliminated']}  E_total/E_bound: {final['E_total']:.4f}/{final['E_bound']:.2f}")
print(f"  Teorem: {'✓' if final['theorem_sat'] else '✗'}")
print()
print(f"  {'Metrik':<18} {'Original':>10} {'AEK':>10} {'Δ':>8}")
print(f"  {'-'*48}")
for task in ["hellaswag", "arc_easy", "arc_challenge"]:
    ov = orig_lm.get(task)
    av = aek_lm.get(task)
    d = f"{av-ov:+.2f}%" if isinstance(ov, (int, float)) and isinstance(av, (int, float)) else "N/A"
    print(f"  {task:<18} {str(ov):>10} {str(av):>10} {d:>8}")
print(f"\n  Kayıt: {out_path}")
print(f"{'='*60}")
