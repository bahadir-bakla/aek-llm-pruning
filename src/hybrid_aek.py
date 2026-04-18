"""
faz9_7b_hybrid_aek.py
=====================
Hybrid AEK: AEK rank-reduction + Kalman-P-guided adaptive INT4/INT8 quantization

Motivasyon:
  Standart AEK+INT8: tüm katmanlar INT8 → uniform sıkıştırma
  Hybrid AEK       : Kalman pressure'a göre adaptif bit-width
    sigma_r1 / threshold > 0.70 → kritik katman → INT8  (precision koru)
    sigma_r1 / threshold ≤ 0.70 → rahat katman   → INT4  (daha fazla sıkıştır)

Karşılaştırma:
  1. AEK+INT8 (uniform)  → faz9_7b_proper.py sonuçları
  2. AEK+Hybrid (adaptive) → bu script

Not: Bu script faz9_7b_proper.py'nin AEK modelini ve compress JSON'unu kullanır.
     Önce faz9_7b_proper.py çalıştırılmalı.

Çalıştır (AWS):
    python faz9_7b_hybrid_aek.py --eps 0.30
    python faz9_7b_hybrid_aek.py --eps 0.30 --skip_lmeval
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

# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--eps',             type=float, default=0.30)
parser.add_argument('--pressure_thresh', type=float, default=0.70,
                    help='sigma_r1/threshold > thresh → INT8, else INT4')
parser.add_argument('--skip_lmeval',     action='store_true')
args = parser.parse_args()

EPS         = args.eps
THRESH      = args.pressure_thresh
EPS_STR     = str(EPS).replace('.', '')
RESULTS_DIR = Path('./results')
MODELS_DIR  = Path('./models')
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MODEL_ID     = "Qwen/Qwen2.5-7B-Instruct"
AEK_DIR      = str(MODELS_DIR / f"qwen_7b_aek_eps{EPS_STR}")
HYBRID_DIR   = str(MODELS_DIR / f"qwen_7b_hybrid_eps{EPS_STR}_t{str(THRESH).replace('.','')}")
COMPRESS_JSON = str(RESULTS_DIR / f"faz9_7b_eps{EPS_STR}_compress.json")
FINAL_OUT    = str(RESULTS_DIR / f"faz9_7b_hybrid_eps{EPS_STR}_full.json")

TASKS = "hellaswag,arc_easy,arc_challenge"
ORIG_LMEVAL = {"hellaswag": 80.44, "arc_easy": 81.06, "arc_challenge": 55.29}
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print(f"  FAZ9 — 7B Hybrid AEK Değerlendirmesi")
print(f"  ε={EPS}  pressure_thresh={THRESH}")
print(f"  AEK kaynak: {AEK_DIR}")
print(f"  Hybrid kayıt: {HYBRID_DIR}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════
# BÖLÜM 1: Pressure Map — compress JSON'dan per-layer analiz
# ══════════════════════════════════════════════════════════════

def build_pressure_map(compress_json_path: str) -> dict:
    """
    compress JSON'daki decisions listesinden per-layer pressure hesapla.
    Pressure = sigma_r1 / threshold  (ne kadar eşiğe yakın)
    > 1  → eliminated (zaten elimine edildi)
    ~0.7 → kritik (INT8 gerekir)
    ~0.2 → rahat (INT4 yeterli)

    Döndürür: {module_fqn: pressure} örn. {"model.layers.0.self_attn.q_proj": 0.85}
    """
    if not os.path.exists(compress_json_path):
        print(f"  [!] Compress JSON bulunamadı: {compress_json_path}")
        print(f"  Önce faz9_7b_proper.py çalıştırın.")
        sys.exit(1)

    with open(compress_json_path) as f:
        data = json.load(f)

    decisions = data.get("decisions", [])
    if not decisions:
        print(f"  [!] decisions listesi boş veya yok. JSON formatı uyumsuz.")
        sys.exit(1)

    ATTN_WEIGHTS = {"q_proj", "k_proj", "v_proj", "o_proj"}
    MLP_WEIGHTS  = {"gate_proj", "up_proj", "down_proj"}

    pressure_map = {}
    for d in decisions:
        k   = d["k"]
        wn  = d["weight_name"]
        thr = d.get("threshold", 0.0)
        s1  = d.get("sigma_r1", 0.0)

        pressure = (s1 / thr) if thr > 1e-9 else 0.0

        if wn in ATTN_WEIGHTS:
            fqn = f"model.layers.{k}.self_attn.{wn}"
        elif wn in MLP_WEIGHTS:
            fqn = f"model.layers.{k}.mlp.{wn}"
        else:
            fqn = f"model.layers.{k}.{wn}"

        pressure_map[fqn] = pressure

    n_int8 = sum(1 for p in pressure_map.values() if p > THRESH)
    n_int4 = sum(1 for p in pressure_map.values() if p <= THRESH)
    print(f"  Pressure map: {len(pressure_map)} katman")
    print(f"    INT8 (pressure > {THRESH}): {n_int8} katman")
    print(f"    INT4 (pressure ≤ {THRESH}): {n_int4} katman")

    return pressure_map


# ══════════════════════════════════════════════════════════════
# BÖLÜM 2: Adaptive Quantization
# ══════════════════════════════════════════════════════════════

def quantize_hybrid_inplace(model, pressure_map: dict, thresh: float) -> tuple:
    """
    Her Linear katmanı pressure'a göre INT8 veya BF16 ile quantize+dequantize eder.
    In-place çalışır: hybrid_state biriktirmez, torch.save yapmaz.
    Disk boyutunu analitik hesaplar.

    Atama kuralı:
      pressure > thresh  → INT8 (yüksek baskı, SVD aktif katman)
      0 < pressure ≤ thresh → INT4 (düşük baskı, SVD aktif ama rahat)
      pressure = 0.0    → INT8 (full-rank, sigma_r1 hesaplanmadı → konservatif)
      pressure < 0      → BF16 (decisions listesinde yok: embed, lm_head vb.)

    NOT: sigma_r1=0.0 olan geniş MLP katmanları full-rank tutulmuş demektir,
         INT4 ile ağırlıkları bozulur — bu yüzden INT8 olarak korunur.

    Döndürür: (disk_gb, stats_dict)
    """
    n_int8 = n_int4 = n_bf16 = 0
    disk_bytes = 0

    for name, param in model.named_parameters():
        if name.endswith(".weight") and param.ndim == 2:
            module_fqn = name[: -len(".weight")]
            pressure   = pressure_map.get(module_fqn, -1.0)

            if pressure > thresh or pressure == 0.0:
                # INT8: yüksek baskı VEYA full-rank (sigma_r1=0, konservatif)
                w     = param.data.cpu().float()
                scale = w.abs().max() / 127.0
                if scale.item() == 0:
                    del w
                    disk_bytes += param.numel() * 2
                    n_bf16 += 1
                    continue
                q     = (w / scale).round().clamp(-128, 127)
                orig_dev = param.data.device
                param.data = (q * scale).to(torch.bfloat16).to(orig_dev)
                del w, q
                disk_bytes += param.numel() * 1
                n_int8 += 1
            elif 0 < pressure <= thresh:
                # INT4: düşük baskı, SVD aktif + rahat (σ_r1 << threshold)
                w     = param.data.cpu().float()
                m     = w.abs().max().item()
                scale = torch.tensor(m / 7.0) if m > 0 else torch.tensor(1.0)
                q     = (w / scale).round().clamp(-8, 7)
                orig_dev = param.data.device
                param.data = (q * scale).to(torch.bfloat16).to(orig_dev)
                del w, q
                disk_bytes += param.numel() * 1
                n_int4 += 1
            else:
                # pressure < 0: decisions dışı (embed, lm_head) → BF16 koru
                disk_bytes += param.numel() * 2
                n_bf16 += 1
        else:
            disk_bytes += param.numel() * 2
            n_bf16 += 1

    print(f"  In-place quantization tamamlandı:")
    print(f"    INT8: {n_int8} weight  INT4: {n_int4} weight  BF16/diğer: {n_bf16}")
    disk_gb = disk_bytes / 1e9
    return disk_gb, {"n_int8": n_int8, "n_int4": n_int4, "n_bf16": n_bf16}


# ══════════════════════════════════════════════════════════════
# BÖLÜM 3: PPL ölçümü
# ══════════════════════════════════════════════════════════════

def get_corpus(n_samples=100):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join([x for x in ds["text"] if len(x.strip()) > 100][:n_samples])


def sliding_ppl(model, tokenizer, corpus, max_len=512, stride=256):
    enc  = tokenizer(corpus, return_tensors="pt", truncation=False)
    ids  = enc["input_ids"][0]
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


# ══════════════════════════════════════════════════════════════
# BÖLÜM 4: lm-eval (AEK dequantized model üzerinde)
# ══════════════════════════════════════════════════════════════

def run_lmeval_on_dir(model_dir: str, label: str) -> dict:
    out_prefix = f"./results/lmeval_{label}_7B_hybrid_eps{EPS_STR}"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},trust_remote_code=True,dtype=bfloat16",
        "--tasks", TASKS,
        "--num_fewshot", "0",
        "--batch_size", "4",
        "--output_path", out_prefix,
    ]
    print(f"  lm-eval [{label}] başlıyor...")
    t0     = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
    elapsed = time.time() - t0

    if result.returncode != 0:
        return {"label": label, "error": f"returncode={result.returncode}"}

    # lm_eval creates nested subdirs: {prefix}/{model_name}/results_*.json
    matches = glob.glob(f"{out_prefix}/**/*.json", recursive=True)
    if not matches:
        matches = glob.glob(f"{out_prefix}*.json")  # fallback flat
    if not matches:
        return {"label": label, "error": "output not found"}

    with open(sorted(matches)[-1]) as f:
        raw = json.load(f)

    r   = raw.get("results", {})
    out = {"label": label, "elapsed_min": round(elapsed / 60, 1)}
    for task in ["hellaswag", "arc_easy", "arc_challenge"]:
        t   = r.get(task, {})
        acc = t.get("acc_norm,none", t.get("acc,none", None))
        out[task] = round(acc * 100, 2) if acc is not None else None
    return out


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    results = {"model": MODEL_ID, "eps": EPS, "pressure_thresh": THRESH}

    # ── 1. Pressure map ─────────────────────────────────────────
    print(f"\n[1/5] Pressure map (compress JSON okunuyor)...")
    pressure_map = build_pressure_map(COMPRESS_JSON)

    # ── 2. AEK modeli yükle ─────────────────────────────────────
    if not os.path.exists(AEK_DIR):
        print(f"\n[!] AEK model dizini yok: {AEK_DIR}")
        print(f"    Önce: python faz9_7b_proper.py --eps {EPS}")
        sys.exit(1)

    print(f"\n[2/5] AEK model yükleniyor ({AEK_DIR})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        AEK_DIR, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.eval()
    print(f"  Model yüklendi.")

    # ── 3. Adaptive quantization (in-place, no torch.save) ──────
    print(f"\n[3/5] Adaptive INT4/INT8 in-place quantization (thresh={THRESH})...")
    hybrid_gb, quant_stats = quantize_hybrid_inplace(model, pressure_map, THRESH)
    # Model is now dequantized BF16 in-place — ready for PPL

    # AEK BF16 disk size (for comparison)
    aek_bf16_gb = sum(
        os.path.getsize(os.path.join(r, fn)) / 1e9
        for r, _, files in os.walk(AEK_DIR)
        for fn in files if fn.endswith(('.safetensors', '.bin', '.pt'))
    )
    print(f"  Hybrid disk (analitik): {hybrid_gb:.3f} GB")
    print(f"  AEK BF16 disk         : {aek_bf16_gb:.3f} GB")
    print(f"  Sıkıştırma (BF16→Hybrid): {aek_bf16_gb/hybrid_gb:.2f}x")

    # ── 4. PPL ölçümü (model already dequantized in-place) ──────
    print(f"\n[4/5] PPL ölçümü (in-place dequantized model üzerinde)...")
    corpus = get_corpus()
    ppl_hybrid = sliding_ppl(model, tokenizer, corpus)
    print(f"  PPL (Hybrid deq.) : {ppl_hybrid:.4f}")

    with open(COMPRESS_JSON) as f:
        cdata = json.load(f)
    ppl_aek_bf16 = cdata.get("ppl_compressed")
    dppl_hybrid  = ((ppl_hybrid - ppl_aek_bf16) / ppl_aek_bf16 * 100) if ppl_aek_bf16 else None
    print(f"  PPL (AEK BF16 ref): {ppl_aek_bf16}")
    if dppl_hybrid is not None:
        print(f"  PPL Δ (Hybrid vs AEK BF16): {dppl_hybrid:+.4f}%")

    del model
    torch.cuda.empty_cache()

    # ── 5. lm-eval (save dequantized model to disk first) ───────
    lmeval_result = {"original": ORIG_LMEVAL, "hybrid": "skipped"}
    if not args.skip_lmeval:
        print(f"\n[5/5] lm-eval için dequantized model yükleniyor ve kaydediliyor...")
        model_deq = AutoModelForCausalLM.from_pretrained(
            AEK_DIR, torch_dtype=torch.bfloat16,
            device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True
        )
        # Re-apply quantize inplace on this CPU model
        quantize_hybrid_inplace(model_deq, pressure_map, THRESH)
        lm_dir = HYBRID_DIR + "_lmeval"
        Path(lm_dir).mkdir(parents=True, exist_ok=True)
        model_deq.save_pretrained(lm_dir, max_shard_size="2GB")
        tokenizer.save_pretrained(lm_dir)
        del model_deq

        hybrid_lm = run_lmeval_on_dir(lm_dir, "hybrid")
        lmeval_result["hybrid"] = hybrid_lm
        print(f"  HellaSwag: {hybrid_lm.get('hellaswag')}%")
        print(f"  ARC-Easy:  {hybrid_lm.get('arc_easy')}%")
        print(f"  ARC-Chall: {hybrid_lm.get('arc_challenge')}%")
    else:
        print(f"\n[5/5] lm-eval atlandı (--skip_lmeval).")

    # ── Kayıt ───────────────────────────────────────────────────
    results["quant_stats"] = quant_stats
    results["disk"] = {
        "aek_bf16_gb":   round(aek_bf16_gb, 3),
        "hybrid_gb":     round(hybrid_gb, 3),
        "compression_x": round(aek_bf16_gb / hybrid_gb, 2),
    }
    results["ppl"] = {
        "aek_bf16":  ppl_aek_bf16,
        "hybrid_deq": round(ppl_hybrid, 4),
        "dppl_pct":   round(dppl_hybrid, 4) if dppl_hybrid is not None else None,
    }
    results["lmeval"] = lmeval_result

    with open(FINAL_OUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Final Tablo ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SONUÇ — 7B Hybrid AEK  ε={EPS}  pressure_thresh={THRESH}")
    print(f"{'='*65}")
    print(f"  Quantization: {quant_stats['n_int8']} INT8 + {quant_stats['n_int4']} INT4 + {quant_stats['n_bf16']} BF16")
    print(f"  {'Metrik':<28} {'AEK BF16':>12} {'Hybrid':>12}")
    print(f"  {'-'*54}")
    aek_int8_ref = aek_bf16_gb * 0.5
    print(f"  {'Disk est. (GB)':<28} {aek_bf16_gb:>12.3f} {hybrid_gb:>12.3f}")
    print(f"  {'PPL (deq.)':<28} {str(round(ppl_aek_bf16,4)) if ppl_aek_bf16 else 'N/A':>12} {ppl_hybrid:>12.4f}")
    if dppl_hybrid is not None:
        print(f"  {'PPL Δ':<28} {'0.00%':>12} {dppl_hybrid:>+11.4f}%")

    print(f"\n  Kayıt: {FINAL_OUT}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
