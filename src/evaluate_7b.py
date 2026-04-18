"""
faz9_7b_proper.py
=================
7B model için kapsamlı AEK değerlendirmesi.

Teorik analiz (qwen_7b_eps020_v2.json spectral data):
  ε=0.20 →  1 elim  (yetersiz)
  ε=0.23 →  3 elim  (güvenli, gate/up_proj)
  ε=0.24 → 32 hedef (uçurum: tüm o_proj devreye giriyor, E_bound sınırlar)
  ε=0.29 → 60 hedef (çok agresif, q_proj dahil)

E_bound = L·ε = 28·ε

Adımlar:
  1. 02b_compress_fixed.py subprocess (orijinal pipeline)
  2. Kaydedilen AEK modelinde sliding-window WikiText-2 PPL
  3. lm-eval AEK vs original karşılaştırması
  4. Disk boyutu (BF16 + INT8)
  5. Tam sonuç tablosu

Çalıştır (AWS g5.xlarge):
    python faz9_7b_proper.py --eps 0.24
    python faz9_7b_proper.py --eps 0.24 --skip_lmeval  # hızlı test
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--eps',            type=float, default=0.30)
parser.add_argument('--delta',          type=float, default=0.1)
parser.add_argument('--fisher_samples', type=int,   default=4)
parser.add_argument('--skip_lmeval',    action='store_true')
parser.add_argument('--skip_int8',      action='store_true')
args = parser.parse_args()

MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
EPS         = args.eps
RESULTS_DIR = Path('./results')
MODELS_DIR  = Path('./models')
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
AEK_SAVE_DIR = str(MODELS_DIR / f"qwen_7b_aek_eps{str(EPS).replace('.','')}")
COMPRESS_OUT = str(RESULTS_DIR / f"faz9_7b_eps{str(EPS).replace('.','')}_compress.json")
FINAL_OUT    = str(RESULTS_DIR / f"faz9_7b_eps{str(EPS).replace('.','')}_full.json")

TASKS = "hellaswag,arc_easy,arc_challenge"

# Önceki run'dan ölçülen original baseline (ayrıca ölçülmeyecek, kredi tasarrufu)
ORIG_PPL_BASELINE  = None  # 02b çalışınca JSON'dan okunacak
ORIG_LMEVAL = {
    "hellaswag":     80.44,
    "arc_easy":      81.06,
    "arc_challenge": 55.29,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print(f"  FAZ9 — 7B Kapsamlı AEK Değerlendirmesi")
print(f"  Model : {MODEL_ID}")
print(f"  ε={EPS}  δ={args.delta}  fisher={args.fisher_samples}")
print(f"  E_bound (L·ε) = 28·{EPS} = {28*EPS:.4f}")
print(f"  AEK kayıt: {AEK_SAVE_DIR}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 1: 02b_compress_fixed.py subprocess
# ══════════════════════════════════════════════════════════════════

def run_compression():
    """Orijinal pipeline ile compress et, JSON kaydet, model kaydet."""
    print(f"\n[1/4] AEK sıkıştırma — 02b_compress_fixed.py (ε={EPS})...")

    cmd = [
        sys.executable, "02b_compress_fixed.py",
        "--eps",           str(EPS),
        "--delta",         str(args.delta),
        "--model",         MODEL_ID,
        "--dtype",         "bfloat16",
        "--fisher_samples", str(args.fisher_samples),
        "--save_model_dir", AEK_SAVE_DIR,
        "--output",        COMPRESS_OUT,
    ]
    print("  CMD:", " ".join(cmd))
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [!] Compress HATA: returncode={result.returncode}")
        sys.exit(1)

    print(f"  Compress tamamlandı ({elapsed/60:.1f} dk)")

    with open(COMPRESS_OUT) as f:
        compress_data = json.load(f)

    print(f"  n_eliminated    : {compress_data.get('n_eliminated')}")
    print(f"  E_total/E_bound : {compress_data.get('E_total'):.4f} / {compress_data.get('E_bound'):.4f}")
    print(f"  Teorem          : {'✓' if compress_data.get('theorem_satisfied') else '✗'}")
    print(f"  PPL baseline    : {compress_data.get('ppl_baseline'):.4f}")
    print(f"  PPL compressed  : {compress_data.get('ppl_compressed'):.4f}  ({compress_data.get('dppl_pct'):+.2f}%)")

    return compress_data


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 2: WikiText-2 sliding-window PPL (tutarlı ölçüm)
# ══════════════════════════════════════════════════════════════════

def get_corpus(n_samples=100):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x for x in ds["text"] if len(x.strip()) > 100][:n_samples]
    return "\n\n".join(texts)

def sliding_ppl(model, tokenizer, corpus, max_len=512, stride=256):
    """Sliding window PPL — daha güvenilir tek-geçiş'e göre."""
    enc = tokenizer(corpus, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0]
    nlls, total = [], 0
    for begin in range(0, len(ids) - 1, stride):
        end   = min(begin + max_len, len(ids))
        chunk = ids[begin:end].unsqueeze(0).to(device)
        tgt   = chunk.clone()
        tgt[0, :max(0, stride - (end - begin))] = -100
        with torch.no_grad():
            loss = model(input_ids=chunk, labels=tgt).loss
        nlls.append(loss.item() * (end - begin))
        total += (end - begin)
        if end == len(ids):
            break
    return math.exp(sum(nlls) / total)

def measure_proper_ppl():
    """AEK kaydedilen modeli yükle, sliding-window PPL ölç."""
    if not os.path.exists(AEK_SAVE_DIR):
        print(f"  [!] AEK model dizini bulunamadı: {AEK_SAVE_DIR}")
        return None, None, None

    print(f"\n[2/4] Sliding-window PPL ölçümü ({AEK_SAVE_DIR})...")
    corpus = get_corpus()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("  Original model yükleniyor...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
    )
    model_orig.eval()
    ppl_orig = sliding_ppl(model_orig, tokenizer, corpus)
    print(f"  PPL original : {ppl_orig:.4f}")
    del model_orig
    torch.cuda.empty_cache()

    print("  AEK model yükleniyor...")
    model_aek = AutoModelForCausalLM.from_pretrained(
        AEK_SAVE_DIR, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
    )
    model_aek.eval()
    ppl_aek = sliding_ppl(model_aek, tokenizer, corpus)
    dppl = (ppl_aek - ppl_orig) / ppl_orig * 100
    print(f"  PPL AEK      : {ppl_aek:.4f}")
    print(f"  PPL Δ%       : {dppl:+.2f}%")
    del model_aek
    torch.cuda.empty_cache()

    return ppl_orig, ppl_aek, dppl


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 3: Disk boyutu
# ══════════════════════════════════════════════════════════════════

def measure_disk():
    print(f"\n[3/4] Disk boyutu...")

    # BF16 (saved model)
    bf16_gb = sum(
        os.path.getsize(os.path.join(r, f)) / 1e9
        for r, _, files in os.walk(AEK_SAVE_DIR)
        for f in files if f.endswith(('.safetensors', '.bin', '.pt'))
    )
    print(f"  AEK BF16 disk : {bf16_gb:.3f} GB")

    int8_gb = None
    if not args.skip_int8:
        try:
            from torchao.quantization import quantize_, int8_weight_only
            print("  INT8 ölçümü için AEK model yükleniyor...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            model_int8 = AutoModelForCausalLM.from_pretrained(
                AEK_SAVE_DIR, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
            )
            quantize_(model_int8, int8_weight_only())
            int8_dir = AEK_SAVE_DIR + "_int8"
            os.makedirs(int8_dir, exist_ok=True)
            save_path = os.path.join(int8_dir, "model_int8.pt")
            torch.save(model_int8.state_dict(), save_path)
            int8_gb = os.path.getsize(save_path) / 1e9
            print(f"  AEK INT8 disk : {int8_gb:.3f} GB")
            del model_int8
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [!] INT8 hatası: {e}")

    return bf16_gb, int8_gb


# ══════════════════════════════════════════════════════════════════
# BÖLÜM 4: lm-eval
# ══════════════════════════════════════════════════════════════════

def run_lmeval(model_path, label):
    out_prefix = f"./results/lmeval_{label}_7B_eps{str(EPS).replace('.','')}"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args",
        f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", TASKS,
        "--num_fewshot", "0",
        "--batch_size", "4",
        "--output_path", out_prefix,
    ]
    print(f"\n  lm-eval [{label}] başlıyor...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
    elapsed = time.time() - t0

    if result.returncode != 0:
        return {"label": label, "error": f"returncode={result.returncode}"}

    matches = glob.glob(f"{out_prefix}/**/*.json", recursive=True)
    if not matches:
        matches = glob.glob(f"{out_prefix}*.json")  # fallback flat
    if not matches:
        return {"label": label, "error": "output not found"}

    with open(sorted(matches)[-1]) as f:
        raw = json.load(f)

    r = raw.get("results", {})
    out = {"label": label, "elapsed_min": round(elapsed/60, 1)}
    for task in ["hellaswag", "arc_easy", "arc_challenge"]:
        t = r.get(task, {})
        acc = t.get("acc_norm,none", t.get("acc,none", None))
        out[task] = round(acc * 100, 2) if acc else None
    return out


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    results = {"model": MODEL_ID, "eps": EPS}

    # ── 1. Compression ──────────────────────────────────────────
    compress_data = run_compression()
    results["compress"] = {
        "n_eliminated":   compress_data.get("n_eliminated"),
        "E_total":        compress_data.get("E_total"),
        "E_bound":        compress_data.get("E_bound"),
        "theorem_sat":    compress_data.get("theorem_satisfied"),
        "ppl_quick_orig": compress_data.get("ppl_baseline"),
        "ppl_quick_aek":  compress_data.get("ppl_compressed"),
        "dppl_quick_pct": compress_data.get("dppl_pct"),
        "elim_by_type":   compress_data.get("elim_by_type", {}),
    }

    # ── 2. Sliding-window PPL ───────────────────────────────────
    ppl_orig, ppl_aek, dppl = measure_proper_ppl()
    results["ppl_sliding"] = {
        "original":  round(ppl_orig, 4) if ppl_orig is not None else None,
        "aek":       round(ppl_aek, 4)  if ppl_aek  is not None else None,
        "delta_pct": round(dppl, 4)     if dppl      is not None else None,
    }

    # ── 3. Disk boyutu ──────────────────────────────────────────
    bf16_gb, int8_gb = measure_disk()
    results["disk"] = {
        "aek_bf16_gb": round(bf16_gb, 3) if bf16_gb else None,
        "aek_int8_gb": round(int8_gb, 3) if int8_gb else None,
    }
    if bf16_gb and int8_gb:
        # Referans: original BF16 disk (0.5B'de 0.988 GB idi, 7B için tahmini ~14.8 GB)
        # Gerçek değer huggingface cache'den ölçülür — burada self.bf16 ile karşılaştır
        results["disk"]["compression_bf16_to_int8"] = round(bf16_gb / int8_gb, 2)

    # ── 4. lm-eval ──────────────────────────────────────────────
    if not args.skip_lmeval:
        print("\n[4/4] lm-eval benchmark...")
        aek_lm = run_lmeval(AEK_SAVE_DIR, "aek")
        results["lmeval"] = {
            "original": ORIG_LMEVAL,
            "aek":      aek_lm,
        }

        print(f"\n  {'Metrik':<18} {'Original':>10} {'AEK':>10} {'Δ':>8}")
        print(f"  {'-'*50}")
        for task in ["hellaswag", "arc_easy", "arc_challenge"]:
            ov = ORIG_LMEVAL.get(task)
            av = aek_lm.get(task)
            d  = f"{av-ov:+.2f}%" if isinstance(ov, float) and isinstance(av, float) else "N/A"
            print(f"  {task:<18} {str(ov):>10} {str(av):>10} {d:>8}")
    else:
        results["lmeval"] = {"original": ORIG_LMEVAL, "aek": "skipped"}
        print("\n[4/4] lm-eval atlandı.")

    # ── Final kayıt ─────────────────────────────────────────────
    with open(FINAL_OUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Final Tablo ─────────────────────────────────────────────
    c = results["compress"]
    p = results["ppl_sliding"]
    d = results["disk"]

    print(f"\n{'='*65}")
    print(f"  SONUÇ — 7B  ε={EPS}")
    print(f"{'='*65}")
    print(f"  n_eliminated    : {c['n_eliminated']}")
    print(f"  E_total/E_bound : {c['E_total']:.4f} / {c['E_bound']:.4f}")
    print(f"  Teorem          : {'✓ SAĞLANDI' if c['theorem_sat'] else '✗ İHLAL'}")
    print(f"")
    print(f"  PPL (quick)     : {c['ppl_quick_orig']:.4f} → {c['ppl_quick_aek']:.4f}  ({c['dppl_quick_pct']:+.2f}%)")
    if p['original'] is not None and p['aek'] is not None:
        dpct_str = f"{p['delta_pct']:+.2f}%" if p['delta_pct'] is not None else "0.00%"
        print(f"  PPL (sliding)   : {p['original']:.4f} → {p['aek']:.4f}  ({dpct_str})")
    print(f"")
    print(f"  AEK BF16 disk   : {d['aek_bf16_gb']} GB")
    if d['aek_int8_gb']:
        print(f"  AEK INT8 disk   : {d['aek_int8_gb']} GB  ({d.get('compression_bf16_to_int8','?')}x vs BF16)")
    if "lmeval" in results and isinstance(results["lmeval"].get("aek"), dict):
        lm = results["lmeval"]["aek"]
        print(f"")
        print(f"  HellaSwag orig→aek : {ORIG_LMEVAL['hellaswag']}% → {lm.get('hellaswag','?')}%")
        print(f"  ARC-Easy  orig→aek : {ORIG_LMEVAL['arc_easy']}% → {lm.get('arc_easy','?')}%")
        print(f"  ARC-Chall orig→aek : {ORIG_LMEVAL['arc_challenge']}% → {lm.get('arc_challenge','?')}%")
    print(f"\n  Kayıt: {FINAL_OUT}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
