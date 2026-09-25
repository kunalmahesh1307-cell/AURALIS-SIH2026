import argparse
import csv
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from df.enhance import enhance, init_df, load_audio

try:
    from pystoi import stoi
except Exception:
    stoi = None

try:
    from pesq import pesq
except Exception:
    pesq = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "auralis_demo"

NOISE_FILES = {
    "stationary": DATA_DIR / "noise" / "stationary" / "demand_pstation_stationary.wav",
    "nonstationary": DATA_DIR / "noise" / "nonstationary" / "demand_straffic_nonstationary.wav",
    "impulsive": DATA_DIR / "noise" / "impulsive" / "gunshot_fold00.wav",
}

INPUT_DEVICE = 1
OUTPUT_DEVICE = 4
MIC_SR = 16000
MODEL_SR = 48000
DEFAULT_SECONDS = 5.0

# Validated controlled-benchmark evidence.
# These values are NOT calculated from the live microphone recording.
BENCHMARK = {
    "stationary": {
        "cases": 30,
        "snr": 10.0151,
        "stoi": 0.1062,
        "pesq": 0.5653,
        "si_sdr": 9.5152,
    },
    "nonstationary": {
        "cases": 30,
        "snr": 12.4021,
        "stoi": 0.0579,
        "pesq": 0.8771,
        "si_sdr": 12.1121,
    },
    "impulsive": {
        "cases": 30,
        "snr": 14.9414,
        "stoi": 0.0395,
        "pesq": 1.3291,
        "si_sdr": 19.5885,
    },
}

AGGREGATE = {
    "cases": 90,
    "snr": 12.4529,
    "stoi": 0.0679,
    "pesq": 0.9238,
    "si_sdr": 13.7386,
    "rtf": 0.0609,
}


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def peak_normalize(x, peak=0.98):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    m = np.max(np.abs(x))
    if m > peak:
        x = x * (peak / m)
    return x


def resample_audio(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return np.asarray(x, dtype=np.float32)

    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(int(src_sr), int(dst_sr))
    up = int(dst_sr // g)
    down = int(src_sr // g)

    return np.asarray(
        resample_poly(np.asarray(x, dtype=np.float32), up, down),
        dtype=np.float32,
    )


def repeat_to_length(x, n):
    x = np.asarray(x, dtype=np.float32).flatten()
    if len(x) == 0:
        raise ValueError("Noise file is empty.")
    if len(x) >= n:
        return x[:n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def mix_at_snr(clean, noise, target_snr):
    clean = np.asarray(clean, dtype=np.float32).flatten()
    noise = repeat_to_length(noise, len(clean))

    cp = np.mean(clean.astype(np.float64) ** 2) + 1e-12
    npow = np.mean(noise.astype(np.float64) ** 2) + 1e-12
    desired_noise_power = cp / (10.0 ** (target_snr / 10.0))

    scale = np.sqrt(desired_noise_power / npow)
    scaled_noise = noise * np.float32(scale)

    return peak_normalize(clean + scaled_noise), scaled_noise


def snr_db(reference, estimate):
    reference, estimate = align(reference, estimate)
    noise = estimate.astype(np.float64) - reference.astype(np.float64)
    reference = reference.astype(np.float64)
    return 10 * np.log10(
        (np.sum(reference ** 2) + 1e-12) /
        (np.sum(noise ** 2) + 1e-12)
    )


def si_sdr(reference, estimate):
    reference, estimate = align(reference, estimate)

    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)

    ref -= np.mean(ref)
    est -= np.mean(est)

    scale = np.sum(est * ref) / (np.sum(ref ** 2) + 1e-12)
    target = scale * ref
    residual = est - target

    return 10 * np.log10(
        (np.sum(target ** 2) + 1e-12) /
        (np.sum(residual ** 2) + 1e-12)
    )


def align(a, b):
    n = min(len(a), len(b))
    return np.asarray(a[:n], dtype=np.float32), np.asarray(b[:n], dtype=np.float32)


def record(seconds):
    frames = int(seconds * MIC_SR)

    print("\n🎙️  RECORDING")
    print("Speak clearly into the Airdopes microphone...")
    print(f"Duration: {seconds:.1f} seconds")

    audio = sd.rec(
        frames,
        samplerate=MIC_SR,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE,
    )
    sd.wait()

    audio = peak_normalize(audio[:, 0])

    print("✓ Recording complete")
    return audio


def play(path, label):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    print(f"\n🔊 {label}")
    sd.play(audio, sr, device=OUTPUT_DEVICE)
    sd.wait()
    print("✓ Playback complete")


def load_noise(noise_type):
    path = NOISE_FILES[noise_type]

    if not path.exists():
        raise FileNotFoundError(f"Noise file not found:\n{path}")

    noise, sr = sf.read(path, dtype="float32")

    if noise.ndim > 1:
        noise = np.mean(noise, axis=1)

    noise = resample_audio(noise, sr, MIC_SR)
    return peak_normalize(noise), path


def calculate_indicative_metrics(reference, noisy, enhanced):
    reference, noisy = align(reference, noisy)
    reference2, enhanced = align(reference, enhanced)

    result = {}

    result["input_snr_db"] = snr_db(reference, noisy)
    result["output_snr_db"] = snr_db(reference2, enhanced)
    result["delta_snr_db"] = result["output_snr_db"] - result["input_snr_db"]

    if stoi is not None:
        try:
            result["input_stoi"] = float(
                stoi(reference, noisy, MIC_SR, extended=False)
            )
            result["output_stoi"] = float(
                stoi(reference2, enhanced, MIC_SR, extended=False)
            )
            result["delta_stoi"] = (
                result["output_stoi"] - result["input_stoi"]
            )
        except Exception:
            result["input_stoi"] = np.nan
            result["output_stoi"] = np.nan
            result["delta_stoi"] = np.nan
    else:
        result["input_stoi"] = np.nan
        result["output_stoi"] = np.nan
        result["delta_stoi"] = np.nan

    if pesq is not None:
        try:
            result["input_pesq"] = float(
                pesq(MIC_SR, reference, noisy, "wb")
            )
            result["output_pesq"] = float(
                pesq(MIC_SR, reference2, enhanced, "wb")
            )
            result["delta_pesq"] = (
                result["output_pesq"] - result["input_pesq"]
            )
        except Exception:
            result["input_pesq"] = np.nan
            result["output_pesq"] = np.nan
            result["delta_pesq"] = np.nan
    else:
        result["input_pesq"] = np.nan
        result["output_pesq"] = np.nan
        result["delta_pesq"] = np.nan

    result["input_si_sdr_db"] = si_sdr(reference, noisy)
    result["output_si_sdr_db"] = si_sdr(reference2, enhanced)
    result["delta_si_sdr_db"] = (
        result["output_si_sdr_db"] - result["input_si_sdr_db"]
    )

    return result


def print_benchmark():
    print("\n" + "=" * 72)
    print("VALIDATED AURALIS BENCHMARK EVIDENCE — 90 CASES")
    print("=" * 72)
    print(
        f"Aggregate ΔSNR     : {AGGREGATE['snr']:+.2f} dB\n"
        f"Aggregate ΔSTOI    : {AGGREGATE['stoi']:+.4f}\n"
        f"Aggregate ΔPESQ    : {AGGREGATE['pesq']:+.4f}\n"
        f"Aggregate ΔSI-SDR  : {AGGREGATE['si_sdr']:+.2f} dB\n"
        f"Offline RTF        : {AGGREGATE['rtf']:.4f}"
    )
    print("-" * 72)

    for name, values in BENCHMARK.items():
        print(
            f"{name:13s} | "
            f"ΔSNR {values['snr']:+6.2f} dB | "
            f"ΔSTOI {values['stoi']:+.4f} | "
            f"ΔPESQ {values['pesq']:+.4f} | "
            f"ΔSI-SDR {values['si_sdr']:+6.2f} dB"
        )

    print("=" * 72)
    print("Source: controlled 90-case AURALIS/DFN2 benchmark.")
    print("These are separate from the live microphone prototype metrics.")
    print("=" * 72)


def print_live_metrics(m):
    print("\n" + "=" * 72)
    print("LIVE PROTOTYPE — INDICATIVE METRICS")
    print("=" * 72)

    print(
        f"ΔSNR     : {m['delta_snr_db']:+.2f} dB\n"
        f"ΔSTOI    : {m['delta_stoi']:+.4f}\n"
        f"ΔPESQ    : {m['delta_pesq']:+.4f}\n"
        f"ΔSI-SDR  : {m['delta_si_sdr_db']:+.2f} dB"
    )

    print("-" * 72)
    print("⚠ These use the original microphone recording as reference.")
    print("⚠ Use the controlled 90-case benchmark for quantitative claims.")
    print("=" * 72)


def save_report(row):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "auralis_demo_report.csv"

    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return path


def choose_noise():
    print("\n" + "=" * 55)
    print("SELECT NOISE CLASS")
    print("=" * 55)
    print("1. Stationary")
    print("2. Non-stationary")
    print("3. Impulsive / Gunshot-like")
    print("4. Exit")
    print("=" * 55)

    while True:
        choice = input("Select [1-4]: ").strip()

        if choice == "1":
            return "stationary"
        if choice == "2":
            return "nonstationary"
        if choice == "3":
            return "impulsive"
        if choice == "4":
            return None

        print("Invalid choice. Enter 1, 2, 3 or 4.")


def choose_snr():
    print("\nSELECT TARGET SNR")
    print("1. +5 dB")
    print("2.  0 dB")
    print("3. -5 dB")

    while True:
        choice = input("Select [1-3]: ").strip()

        if choice == "1":
            return 5.0
        if choice == "2":
            return 0.0
        if choice == "3":
            return -5.0

        print("Invalid choice. Enter 1, 2 or 3.")


def run_demo(noise_type, target_snr, seconds):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("AURALIS — LIVE AI SPEECH ENHANCEMENT DEMO")
    print("=" * 72)
    print(f"Noise       : {noise_type}")
    print(f"Target SNR  : {target_snr:+.0f} dB")
    print(f"Duration    : {seconds:.1f} s")
    print("=" * 72)

    # Record clean-ish microphone speech.
    reference = record(seconds)

    reference_path = RESULTS_DIR / "demo_mic_reference.wav"
    sf.write(reference_path, reference, MIC_SR)

    # Mix selected noise.
    noise, noise_path = load_noise(noise_type)
    noisy, _ = mix_at_snr(reference, noise, target_snr)

    noisy_path = (
        RESULTS_DIR /
        f"noisy_{noise_type}_{target_snr:+.0f}dB.wav"
    )
    sf.write(noisy_path, noisy, MIC_SR)

    print(f"\nNoise source: {noise_path}")
    print(f"Saved noisy:  {noisy_path}")

    # Load DFN2 once for this demo.
    print("\n🧠 Loading DeepFilterNet2...")
    model, df_state, _ = init_df("DeepFilterNet2")

    # DFN2 operates at 48 kHz.
    noisy_48k = resample_audio(noisy, MIC_SR, MODEL_SR)

    input_48k = RESULTS_DIR / "demo_noisy_48k.wav"
    sf.write(input_48k, noisy_48k, MODEL_SR)

    print("⚙ Running AURALIS / DFN2...")

    start = time.perf_counter()

    model_audio, _ = load_audio(
        str(input_48k),
        sr=df_state.sr(),
    )

    enhanced_48k = enhance(
        model,
        df_state,
        model_audio,
    )

    processing_time = time.perf_counter() - start

    enhanced_16k = resample_audio(
        np.asarray(enhanced_48k, dtype=np.float32).flatten(),
        MODEL_SR,
        MIC_SR,
    )

    enhanced_16k, reference = align(enhanced_16k, reference)
    enhanced_16k = peak_normalize(enhanced_16k)

    enhanced_path = (
        RESULTS_DIR /
        f"enhanced_{noise_type}_{target_snr:+.0f}dB.wav"
    )
    sf.write(enhanced_path, enhanced_16k, MIC_SR)

    print(f"✓ Enhancement complete")
    print(f"Processing time: {processing_time:.3f} s")

    # Playback.
    print("\n" + "=" * 72)
    print("AUDIO COMPARISON")
    print("=" * 72)

    input("Press ENTER to play NOISY audio... ")
    play(noisy_path, "NOISY AUDIO")

    input("Press ENTER to play AURALIS ENHANCED audio... ")
    play(enhanced_path, "AURALIS ENHANCED AUDIO")

    # Live indicative metrics.
    metrics = calculate_indicative_metrics(
        reference,
        noisy,
        enhanced_16k,
    )

    print_live_metrics(metrics)

    # Save report.
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "noise_type": noise_type,
        "target_snr_db": target_snr,
        "processing_time_sec": processing_time,
        **metrics,
    }

    report = save_report(row)

    print("\n" + "=" * 72)
    print("DEMO COMPLETE")
    print("=" * 72)
    print(f"Enhanced WAV : {enhanced_path}")
    print(f"Report CSV   : {report}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive AURALIS SIH prototype demo."
    )
    parser.add_argument("--noise", choices=list(NOISE_FILES.keys()))
    parser.add_argument("--snr", type=float, choices=[5.0, 0.0, -5.0])
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        print_benchmark()
        return

    if args.noise is not None and args.snr is not None:
        run_demo(args.noise, args.snr, args.seconds)
        return

    print("\n" + "=" * 72)
    print("        A U R A L I S  —  SIH 2026 DEMO")
    print("=" * 72)
    print("AI-based Adaptive Digital Speech Enhancement")
    print("Stationary • Non-stationary • Impulsive Noise")
    print("=" * 72)

    while True:
        noise_type = choose_noise()

        if noise_type is None:
            print("\nExiting AURALIS demo.")
            return

        target_snr = choose_snr()

        try:
            run_demo(
                noise_type,
                target_snr,
                args.seconds,
            )
        except KeyboardInterrupt:
            print("\n\nDemo interrupted.")
            return
        except Exception as exc:
            print("\n❌ Demo failed:")
            print(exc)
            print("\nCheck microphone/output devices and input files.")

        again = input("\nRun another demo? [y/N]: ").strip().lower()

        if again != "y":
            print("\nAURALIS demo finished.")
            return


if __name__ == "__main__":
    main()
