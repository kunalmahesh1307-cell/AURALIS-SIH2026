import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pystoi import stoi
from pesq import pesq


ROOT = Path(__file__).resolve().parents[2]

MIXED_ROOT = ROOT / "data" / "mixed"
RESULT_ROOT = ROOT / "results" / "benchmark_rnnoise"
EVAL_ROOT = ROOT / "results" / "evaluation_rnnoise"

EVAL_ROOT.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = EVAL_ROOT / "rnnoise_objective_results.json"
OUTPUT_CSV = EVAL_ROOT / "rnnoise_objective_results.csv"


def load_wav(path):
    x, sr = sf.read(path, dtype="float32")

    if x.ndim > 1:
        x = np.mean(x, axis=1)

    return x.astype(np.float32), sr


def rms(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))


def snr_db(clean, noisy):
    noise = noisy - clean
    return 10.0 * math.log10(
        np.sum(clean.astype(np.float64) ** 2)
        / (np.sum(noise.astype(np.float64) ** 2) + 1e-12)
    )


def si_sdr(clean, estimate):
    clean = clean.astype(np.float64)
    estimate = estimate.astype(np.float64)

    clean = clean - np.mean(clean)
    estimate = estimate - np.mean(estimate)

    alpha = np.dot(estimate, clean) / (np.dot(clean, clean) + 1e-12)
    target = alpha * clean
    residual = estimate - target

    return 10.0 * math.log10(
        np.sum(target ** 2) / (np.sum(residual ** 2) + 1e-12)
    )


def align_length(*arrays):
    n = min(len(x) for x in arrays)
    return [x[:n] for x in arrays]


rows = []

print("=" * 70)
print("AURALIS — RNNoise Objective Evaluation")
print("=" * 70)
print(f"Project root : {ROOT}")
print(f"RNNoise root : {RESULT_ROOT}")
print(f"Output root  : {EVAL_ROOT}")
print()

clean_files = sorted((ROOT / "data" / "clean").glob("speech_*.wav"))

clean_map = {p.stem: p for p in clean_files}

case_id = 0

for noise_type in ["stationary", "nonstationary", "impulsive"]:

    input_dir = MIXED_ROOT / noise_type
    output_dir = RESULT_ROOT / noise_type

    input_files = sorted(input_dir.glob("*.wav"))

    for noisy_path in input_files:

        case_id += 1

        stem = noisy_path.stem
        speech_id = stem.split("__")[0]

        clean_path = clean_map.get(speech_id)

        if clean_path is None:
            print(f"[{case_id}] SKIP — clean speech missing: {speech_id}")
            continue

        rnnoise_path = output_dir / f"{stem}__rnnoise.wav"

        try:
            clean, clean_sr = load_wav(clean_path)
            noisy, noisy_sr = load_wav(noisy_path)
            enhanced, enhanced_sr = load_wav(rnnoise_path)

            if clean_sr != 48000:
                clean = resample_poly(clean, 48000, clean_sr).astype(np.float32)
                clean_sr = 48000

            if noisy_sr != 48000:
                noisy = resample_poly(noisy, 48000, noisy_sr).astype(np.float32)
                noisy_sr = 48000

            if enhanced_sr != 48000:
                enhanced = resample_poly(
                    enhanced, 48000, enhanced_sr
                ).astype(np.float32)
                enhanced_sr = 48000

            clean, noisy, enhanced = align_length(clean, noisy, enhanced)

            noisy_snr = snr_db(clean, noisy)
            enhanced_snr = snr_db(clean, enhanced)

            noisy_sisdr = si_sdr(clean, noisy)
            enhanced_sisdr = si_sdr(clean, enhanced)

            clean_16 = resample_poly(clean, 16000, 48000).astype(np.float32)
            noisy_16 = resample_poly(noisy, 16000, 48000).astype(np.float32)
            enhanced_16 = resample_poly(
                enhanced, 16000, 48000
            ).astype(np.float32)

            clean_16, noisy_16, enhanced_16 = align_length(
                clean_16, noisy_16, enhanced_16
            )

            noisy_stoi = stoi(clean_16, noisy_16, 16000, extended=False)
            enhanced_stoi = stoi(
                clean_16, enhanced_16, 16000, extended=False
            )

            noisy_pesq = pesq(16000, clean_16, noisy_16, "wb")
            enhanced_pesq = pesq(
                16000, clean_16, enhanced_16, "wb"
            )

            target_snr = None

            if "__snr_+5dB" in stem:
                target_snr = 5.0
            elif "__snr_+0dB" in stem:
                target_snr = 0.0
            elif "__snr_-5dB" in stem:
                target_snr = -5.0

            row = {
                "case_id": case_id,
                "noise_type": noise_type,
                "speech_id": speech_id,
                "input_file": str(noisy_path.relative_to(ROOT)),
                "enhanced_file": str(rnnoise_path.relative_to(ROOT)),
                "target_snr_db": target_snr,
                "snr_noisy_db": noisy_snr,
                "snr_enhanced_db": enhanced_snr,
                "snr_improvement_db": enhanced_snr - noisy_snr,
                "stoi_noisy": noisy_stoi,
                "stoi_enhanced": enhanced_stoi,
                "stoi_improvement": enhanced_stoi - noisy_stoi,
                "pesq_noisy": noisy_pesq,
                "pesq_enhanced": enhanced_pesq,
                "pesq_improvement": enhanced_pesq - noisy_pesq,
                "sisdr_noisy_db": noisy_sisdr,
                "sisdr_enhanced_db": enhanced_sisdr,
                "sisdr_improvement_db": enhanced_sisdr - noisy_sisdr,
                "status": "success",
            }

            rows.append(row)

            print(
                f"[{case_id:02d}/90] {noise_type:13s} "
                f"{speech_id:10s} "
                f"SNR Δ={row['snr_improvement_db']:+.2f} dB | "
                f"STOI Δ={row['stoi_improvement']:+.4f} | "
                f"PESQ Δ={row['pesq_improvement']:+.4f} | "
                f"SI-SDR Δ={row['sisdr_improvement_db']:+.2f} dB"
            )

        except Exception as exc:
            print(f"[{case_id:02d}/90] FAILED: {exc}")

print()
print("=" * 70)
print("Saving results...")
print("=" * 70)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

import csv

if rows:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

print(f"Cases evaluated : {len(rows)}")
print(f"JSON            : {OUTPUT_JSON}")
print(f"CSV             : {OUTPUT_CSV}")
print("=" * 70)
