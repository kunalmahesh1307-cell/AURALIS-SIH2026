from pathlib import Path
import json
import csv
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from pystoi import stoi
from pesq import pesq


# ============================================================
# AURALIS — DeepFilterNet2 Objective Evaluation
# ============================================================
#
# Evaluates the 90 previously generated DFN2 outputs using:
#
#   1. Input SNR
#   2. Output SNR
#   3. SNR Improvement
#   4. Input STOI
#   5. Output STOI
#   6. STOI Improvement
#   7. Input PESQ
#   8. Output PESQ
#   9. PESQ Improvement
#  10. Input SI-SDR
#  11. Output SI-SDR
#  12. SI-SDR Improvement
#
# Important:
# - DFN2 operates at 48 kHz.
# - SNR and SI-SDR are evaluated at 48 kHz.
# - STOI/PESQ are evaluated at 16 kHz.
# - This script DOES NOT run DFN2 again.
# - It evaluates the already-generated DFN2 WAV files.
#
# ============================================================


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MIXED_DIR = PROJECT_ROOT / "data" / "mixed"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

DFN2_OUTPUT_DIR = (
    PROJECT_ROOT /
    "results" /
    "benchmark_dfn2"
)

METADATA_FILE = (
    MIXED_DIR /
    "metadata" /
    "benchmark_mixtures.json"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "evaluation_dfn2"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_JSON = (
    RESULTS_DIR /
    "dfn2_objective_results.json"
)

RESULTS_CSV = (
    RESULTS_DIR /
    "dfn2_objective_results.csv"
)

SUMMARY_CSV = (
    RESULTS_DIR /
    "dfn2_summary_by_class_snr.csv"
)


# ============================================================
# Audio utilities
# ============================================================

def load_audio(path):
    """
    Load WAV as mono float32.
    """

    audio, sr = sf.read(
        str(path),
        dtype="float32"
    )

    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    return (
        audio.astype(np.float32),
        sr
    )


def align_length(a, b):
    """
    Trim two signals to the same length.
    """

    n = min(
        len(a),
        len(b)
    )

    return (
        a[:n],
        b[:n]
    )


def resample_audio(
    audio,
    src_sr,
    dst_sr
):
    """
    Resample audio using scipy resample_poly.
    """

    if src_sr == dst_sr:
        return audio.astype(
            np.float32
        )

    g = math.gcd(
        int(src_sr),
        int(dst_sr)
    )

    up = dst_sr // g
    down = src_sr // g

    return resample_poly(
        audio,
        up,
        down
    ).astype(np.float32)


# ============================================================
# SNR
# ============================================================

def calculate_snr(
    clean,
    estimate
):
    """
    Calculate SNR between clean reference
    and estimated/enhanced signal.

    SNR = 10 log10(
        Pclean / Perr
    )

    where:

        error = estimate - clean
    """

    clean, estimate = align_length(
        clean,
        estimate
    )

    clean = clean.astype(
        np.float64
    )

    estimate = estimate.astype(
        np.float64
    )

    error = (
        estimate -
        clean
    )

    signal_power = np.mean(
        clean ** 2
    )

    error_power = np.mean(
        error ** 2
    )

    if error_power < 1e-15:
        return float("inf")

    return float(
        10.0 *
        np.log10(
            (
                signal_power +
                1e-12
            ) /
            (
                error_power +
                1e-12
            )
        )
    )


# ============================================================
# SI-SDR
# ============================================================

def calculate_si_sdr(
    reference,
    estimate
):
    """
    Calculate Scale-Invariant SDR.
    """

    reference, estimate = align_length(
        reference,
        estimate
    )

    reference = reference.astype(
        np.float64
    )

    estimate = estimate.astype(
        np.float64
    )

    # Remove DC component
    reference = (
        reference -
        np.mean(reference)
    )

    estimate = (
        estimate -
        np.mean(estimate)
    )

    reference_energy = np.sum(
        reference ** 2
    )

    if reference_energy < 1e-12:
        return float("nan")

    projection = (
        np.sum(
            estimate *
            reference
        ) /
        reference_energy
    ) * reference

    noise = (
        estimate -
        projection
    )

    target_energy = np.sum(
        projection ** 2
    )

    noise_energy = np.sum(
        noise ** 2
    )

    if noise_energy < 1e-15:
        return float("inf")

    return float(
        10.0 *
        np.log10(
            (
                target_energy +
                1e-12
            ) /
            (
                noise_energy +
                1e-12
            )
        )
    )


# ============================================================
# STOI
# ============================================================

def safe_stoi(
    clean,
    estimate,
    sr
):
    """
    Calculate STOI safely.
    """

    try:

        clean, estimate = align_length(
            clean,
            estimate
        )

        value = stoi(
            clean,
            estimate,
            sr,
            extended=False
        )

        return float(value)

    except Exception as e:

        print(
            f"    STOI warning: {repr(e)}"
        )

        return float("nan")


# ============================================================
# PESQ
# ============================================================

def safe_pesq(
    clean,
    estimate,
    sr
):
    """
    Calculate wideband PESQ.

    PESQ is evaluated at 16 kHz.
    """

    try:

        if sr != 16000:

            clean = resample_audio(
                clean,
                sr,
                16000
            )

            estimate = resample_audio(
                estimate,
                sr,
                16000
            )

            sr = 16000

        clean, estimate = align_length(
            clean,
            estimate
        )

        # PESQ needs a sufficiently long signal.
        if len(clean) < int(
            0.25 * sr
        ):

            return float("nan")

        value = pesq(
            sr,
            clean,
            estimate,
            "wb"
        )

        return float(value)

    except Exception as e:

        print(
            f"    PESQ warning: {repr(e)}"
        )

        return float("nan")


# ============================================================
# Path handling
# ============================================================

def resolve_mixture_path(
    mixture_value
):
    """
    Resolve mixture path correctly.

    Metadata stores paths such as:

        data\mixed\stationary\file.wav

    Therefore we must NOT do:

        MIXED_DIR / metadata_path

    because that would produce:

        data\mixed\data\mixed\stationary\file.wav
    """

    mixture_path = Path(
        mixture_value
    )

    # Case 1:
    # Absolute path
    if mixture_path.is_absolute():

        return mixture_path

    normalized = str(
        mixture_path
    ).replace(
        "/",
        "\\"
    )

    # Case 2:
    # Metadata path already begins with data\mixed
    if normalized.lower().startswith(
        "data\\mixed\\"
    ):

        return (
            PROJECT_ROOT /
            mixture_path
        )

    # Case 3:
    # Relative path inside data\mixed
    return (
        MIXED_DIR /
        mixture_path
    )


def extract_clean_name(
    entry
):
    """
    Determine the clean speech filename
    associated with a benchmark entry.
    """

    candidates = [
        "clean_file",
        "clean_filename",
        "clean",
        "speech_file",
        "reference_file"
    ]

    for key in candidates:

        if key in entry:

            value = entry[key]

            if value:

                return Path(
                    value
                ).name

    # Fallback:
    #
    # speech_001__stationary__snr_+5dB.wav
    #
    # becomes:
    #
    # speech_001.wav

    mixture_name = Path(
        entry["mixture_file"]
    ).stem

    speech_name = (
        mixture_name.split(
            "__"
        )[0]
    )

    return (
        speech_name +
        ".wav"
    )


def find_clean_file(
    clean_name
):
    """
    Locate clean reference WAV.
    """

    direct_path = (
        CLEAN_DIR /
        clean_name
    )

    if direct_path.exists():
        return direct_path

    matches = list(
        CLEAN_DIR.rglob(
            clean_name
        )
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Could not locate clean "
        f"reference: {clean_name}"
    )


def find_output_file(
    entry
):
    """
    Locate the already-generated DFN2 output.
    """

    mixture_value = Path(
        entry["mixture_file"]
    )

    output_name = (
        mixture_value.stem +
        "_dfn2.wav"
    )

    noise_type = entry.get(
        "noise_type",
        mixture_value.parent.name
    )

    # Normal expected locations
    candidates = [

        (
            DFN2_OUTPUT_DIR /
            noise_type /
            output_name
        ),

        (
            DFN2_OUTPUT_DIR /
            mixture_value.parent.name /
            output_name
        )
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    # Recursive fallback
    matches = list(
        DFN2_OUTPUT_DIR.rglob(
            output_name
        )
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        "DFN2 output not found: "
        f"{output_name}"
    )


# ============================================================
# Numeric utilities
# ============================================================

def finite_values(
    rows,
    key
):
    """
    Return only finite metric values.
    """

    values = []

    for row in rows:

        value = row.get(
            key,
            float("nan")
        )

        try:

            value = float(value)

        except (
            TypeError,
            ValueError
        ):

            continue

        if np.isfinite(value):
            values.append(value)

    return values


def mean_metric(
    rows,
    key
):
    """
    Calculate mean of finite metric values.
    """

    values = finite_values(
        rows,
        key
    )

    if not values:
        return float("nan")

    return float(
        np.mean(values)
    )


# ============================================================
# Load benchmark metadata
# ============================================================

print("=" * 72)
print(
    "AURALIS — DeepFilterNet2 "
    "Objective Evaluation"
)
print("=" * 72)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"Metadata     : {METADATA_FILE}"
)

print(
    f"DFN2 outputs : {DFN2_OUTPUT_DIR}"
)

print(
    f"Results      : {RESULTS_DIR}"
)

print()


if not METADATA_FILE.exists():

    raise FileNotFoundError(
        "Benchmark metadata file does "
        "not exist:\n"
        f"{METADATA_FILE}"
    )


with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


# ============================================================
# Normalize metadata structure
# ============================================================

if isinstance(
    metadata,
    dict
):

    if "mixtures" in metadata:

        entries = metadata[
            "mixtures"
        ]

    elif "cases" in metadata:

        entries = metadata[
            "cases"
        ]

    else:

        entries = list(
            metadata.values()
        )

else:

    entries = metadata


print(
    f"Total evaluation cases: "
    f"{len(entries)}"
)

print()


# ============================================================
# Evaluation
# ============================================================

results = []

failed_cases = []


for idx, entry in enumerate(
    entries,
    start=1
):

    noise_type = entry.get(
        "noise_type",
        Path(
            entry["mixture_file"]
        ).parent.name
    )

    snr_label = entry.get(
        "snr_db",
        entry.get(
            "target_snr_db",
            "unknown"
        )
    )

    print(
        f"[{idx:02d}/{len(entries):02d}] "
        f"{noise_type:14s} | "
        f"SNR={snr_label}"
    )

    try:

        # ----------------------------------------------------
        # Resolve files
        # ----------------------------------------------------

        mixture_path = resolve_mixture_path(
            entry["mixture_file"]
        )

        clean_name = extract_clean_name(
            entry
        )

        clean_path = find_clean_file(
            clean_name
        )

        output_path = find_output_file(
            entry
        )

        # ----------------------------------------------------
        # Verify paths before opening
        # ----------------------------------------------------

        if not mixture_path.exists():

            raise FileNotFoundError(
                "Mixture file does not exist:\n"
                f"{mixture_path}"
            )

        if not clean_path.exists():

            raise FileNotFoundError(
                "Clean reference does not exist:\n"
                f"{clean_path}"
            )

        if not output_path.exists():

            raise FileNotFoundError(
                "DFN2 output does not exist:\n"
                f"{output_path}"
            )

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        clean, clean_sr = load_audio(
            clean_path
        )

        noisy, noisy_sr = load_audio(
            mixture_path
        )

        enhanced, enhanced_sr = load_audio(
            output_path
        )

        # ----------------------------------------------------
        # Convert to 48 kHz
        # ----------------------------------------------------

        target_sr = 48000

        clean_48 = resample_audio(
            clean,
            clean_sr,
            target_sr
        )

        noisy_48 = resample_audio(
            noisy,
            noisy_sr,
            target_sr
        )

        enhanced_48 = resample_audio(
            enhanced,
            enhanced_sr,
            target_sr
        )

        # ----------------------------------------------------
        # Align all three signals
        # ----------------------------------------------------

        n = min(
            len(clean_48),
            len(noisy_48),
            len(enhanced_48)
        )

        clean_48 = clean_48[:n]
        noisy_48 = noisy_48[:n]
        enhanced_48 = enhanced_48[:n]

        # ----------------------------------------------------
        # SNR
        # ----------------------------------------------------

        input_snr = calculate_snr(
            clean_48,
            noisy_48
        )

        output_snr = calculate_snr(
            clean_48,
            enhanced_48
        )

        snr_improvement = (
            output_snr -
            input_snr
        )

        # ----------------------------------------------------
        # SI-SDR
        # ----------------------------------------------------

        input_si_sdr = calculate_si_sdr(
            clean_48,
            noisy_48
        )

        output_si_sdr = calculate_si_sdr(
            clean_48,
            enhanced_48
        )

        if (
            np.isfinite(input_si_sdr)
            and
            np.isfinite(output_si_sdr)
        ):

            si_sdr_improvement = (
                output_si_sdr -
                input_si_sdr
            )

        else:

            si_sdr_improvement = (
                float("nan")
            )

        # ----------------------------------------------------
        # Convert to 16 kHz
        # for STOI and PESQ
        # ----------------------------------------------------

        clean_16 = resample_audio(
            clean_48,
            48000,
            16000
        )

        noisy_16 = resample_audio(
            noisy_48,
            48000,
            16000
        )

        enhanced_16 = resample_audio(
            enhanced_48,
            48000,
            16000
        )

        n16 = min(
            len(clean_16),
            len(noisy_16),
            len(enhanced_16)
        )

        clean_16 = clean_16[:n16]
        noisy_16 = noisy_16[:n16]
        enhanced_16 = enhanced_16[:n16]

        # ----------------------------------------------------
        # STOI
        # ----------------------------------------------------

        input_stoi = safe_stoi(
            clean_16,
            noisy_16,
            16000
        )

        output_stoi = safe_stoi(
            clean_16,
            enhanced_16,
            16000
        )

        if (
            np.isfinite(input_stoi)
            and
            np.isfinite(output_stoi)
        ):

            stoi_improvement = (
                output_stoi -
                input_stoi
            )

        else:

            stoi_improvement = (
                float("nan")
            )

        # ----------------------------------------------------
        # PESQ
        # ----------------------------------------------------

        input_pesq = safe_pesq(
            clean_16,
            noisy_16,
            16000
        )

        output_pesq = safe_pesq(
            clean_16,
            enhanced_16,
            16000
        )

        if (
            np.isfinite(input_pesq)
            and
            np.isfinite(output_pesq)
        ):

            pesq_improvement = (
                output_pesq -
                input_pesq
            )

        else:

            pesq_improvement = (
                float("nan")
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {

            "case":
                idx,

            "noise_type":
                noise_type,

            "target_snr_db":
                snr_label,

            "clean_file":
                str(clean_path),

            "mixture_file":
                str(mixture_path),

            "enhanced_file":
                str(output_path),

            "input_snr_db":
                input_snr,

            "output_snr_db":
                output_snr,

            "snr_improvement_db":
                snr_improvement,

            "input_si_sdr_db":
                input_si_sdr,

            "output_si_sdr_db":
                output_si_sdr,

            "si_sdr_improvement_db":
                si_sdr_improvement,

            "input_stoi":
                input_stoi,

            "output_stoi":
                output_stoi,

            "stoi_improvement":
                stoi_improvement,

            "input_pesq":
                input_pesq,

            "output_pesq":
                output_pesq,

            "pesq_improvement":
                pesq_improvement
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"    SNR    : "
            f"{input_snr:.2f} -> "
            f"{output_snr:.2f} dB "
            f"(Δ {snr_improvement:+.2f})"
        )

        print(
            f"    STOI   : "
            f"{input_stoi:.4f} -> "
            f"{output_stoi:.4f} "
            f"(Δ {stoi_improvement:+.4f})"
        )

        print(
            f"    PESQ   : "
            f"{input_pesq:.4f} -> "
            f"{output_pesq:.4f} "
            f"(Δ {pesq_improvement:+.4f})"
        )

        print(
            f"    SI-SDR : "
            f"{input_si_sdr:.2f} -> "
            f"{output_si_sdr:.2f} dB "
            f"(Δ {si_sdr_improvement:+.2f})"
        )

    except Exception as e:

        failed_cases.append({

            "case":
                idx,

            "noise_type":
                noise_type,

            "target_snr_db":
                snr_label,

            "error":
                repr(e)
        })

        print(
            f"    ERROR: {repr(e)}"
        )

    print()


# ============================================================
# Save detailed JSON
# ============================================================

evaluation_output = {

    "evaluation": {
        "model":
            "DeepFilterNet2",

        "total_cases":
            len(entries),

        "successful_cases":
            len(results),

        "failed_cases":
            len(failed_cases),

        "success_rate_percent":
            (
                100.0 *
                len(results) /
                len(entries)
                if entries
                else 0.0
            )
    },

    "results":
        results,

    "failures":
        failed_cases
}


with open(
    RESULTS_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_output,
        f,
        indent=2
    )


# ============================================================
# Save detailed CSV
# ============================================================

if results:

    fieldnames = list(
        results[0].keys()
    )

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# Group results
# ============================================================

groups = {}


for row in results:

    key = (
        row["noise_type"],
        str(
            row["target_snr_db"]
        )
    )

    if key not in groups:

        groups[key] = []

    groups[key].append(
        row
    )


# ============================================================
# Summary by noise class + SNR
# ============================================================

summary_rows = []


for (
    noise_type,
    snr
), rows in sorted(
    groups.items()
):

    summary_rows.append({

        "noise_type":
            noise_type,

        "target_snr_db":
            snr,

        "cases":
            len(rows),

        "mean_input_snr_db":
            mean_metric(
                rows,
                "input_snr_db"
            ),

        "mean_output_snr_db":
            mean_metric(
                rows,
                "output_snr_db"
            ),

        "mean_snr_improvement_db":
            mean_metric(
                rows,
                "snr_improvement_db"
            ),

        "mean_input_stoi":
            mean_metric(
                rows,
                "input_stoi"
            ),

        "mean_output_stoi":
            mean_metric(
                rows,
                "output_stoi"
            ),

        "mean_stoi_improvement":
            mean_metric(
                rows,
                "stoi_improvement"
            ),

        "mean_input_pesq":
            mean_metric(
                rows,
                "input_pesq"
            ),

        "mean_output_pesq":
            mean_metric(
                rows,
                "output_pesq"
            ),

        "mean_pesq_improvement":
            mean_metric(
                rows,
                "pesq_improvement"
            ),

        "mean_input_si_sdr_db":
            mean_metric(
                rows,
                "input_si_sdr_db"
            ),

        "mean_output_si_sdr_db":
            mean_metric(
                rows,
                "output_si_sdr_db"
            ),

        "mean_si_sdr_improvement_db":
            mean_metric(
                rows,
                "si_sdr_improvement_db"
            )
    })


if summary_rows:

    fieldnames = list(
        summary_rows[0].keys()
    )

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )


# ============================================================
# Final console summary
# ============================================================

print("=" * 72)
print(
    "OBJECTIVE EVALUATION COMPLETE"
)
print("=" * 72)

print(
    f"Successful evaluations : "
    f"{len(results)} / {len(entries)}"
)

print(
    f"Failed evaluations     : "
    f"{len(failed_cases)} / {len(entries)}"
)

if entries:

    print(
        f"Success rate           : "
        f"{100.0 * len(results) / len(entries):.2f}%"
    )


if results:

    print()

    print(
        "Mean SNR improvement : "
        f"{mean_metric(results, 'snr_improvement_db'):+.3f} dB"
    )

    print(
        "Mean STOI improvement: "
        f"{mean_metric(results, 'stoi_improvement'):+.4f}"
    )

    print(
        "Mean PESQ improvement: "
        f"{mean_metric(results, 'pesq_improvement'):+.4f}"
    )

    print(
        "Mean SI-SDR improvement: "
        f"{mean_metric(results, 'si_sdr_improvement_db'):+.3f} dB"
    )


if failed_cases:

    print()
    print(
        "First failure:"
    )

    print(
        f"    Case       : "
        f"{failed_cases[0]['case']}"
    )

    print(
        f"    Noise type : "
        f"{failed_cases[0]['noise_type']}"
    )

    print(
        f"    SNR        : "
        f"{failed_cases[0]['target_snr_db']}"
    )

    print(
        f"    Error      : "
        f"{failed_cases[0]['error']}"
    )


print()

print(
    f"Detailed JSON : "
    f"{RESULTS_JSON}"
)

print(
    f"Detailed CSV  : "
    f"{RESULTS_CSV}"
)

print(
    f"Summary CSV   : "
    f"{SUMMARY_CSV}"
)

print("=" * 72)
