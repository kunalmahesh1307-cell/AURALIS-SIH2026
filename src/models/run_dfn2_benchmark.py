# ============================================================
# AURALIS - DeepFilterNet2 Benchmark Runner
# ============================================================
#
# Benchmark:
#   90 mixtures
#   30 stationary
#   30 nonstationary
#   30 impulsive
#
# Input:
#   data\mixed\metadata\benchmark_mixtures.json
#
# Output:
#   results\benchmark_dfn2\
#
# ============================================================

from pathlib import Path
import json
import time

import numpy as np
import soundfile as sf
import torch

from df.enhance import enhance
from df.enhance import init_df


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MIXED_DIR = PROJECT_ROOT / "data" / "mixed"
METADATA_DIR = MIXED_DIR / "metadata"

METADATA_FILE = (
    METADATA_DIR / "benchmark_mixtures.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "benchmark_dfn2"
)

OUTPUT_METADATA_DIR = (
    OUTPUT_ROOT / "metadata"
)

RESULTS_FILE = (
    OUTPUT_METADATA_DIR
    / "dfn2_inference_results.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "DeepFilterNet2"

EXPECTED_SAMPLE_RATE = 48000

NOISE_TYPES = [
    "stationary",
    "nonstationary",
    "impulsive",
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for noise_type in NOISE_TYPES:

    (
        OUTPUT_ROOT / noise_type
    ).mkdir(
        parents=True,
        exist_ok=True
    )


OUTPUT_METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "\nBenchmark metadata file not found:\n"
            f"{METADATA_FILE}\n"
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# GET CASES
# ============================================================

def get_cases(metadata):

    if isinstance(metadata, list):

        return metadata

    if isinstance(metadata, dict):

        if "cases" in metadata:

            return metadata["cases"]

        if "mixtures" in metadata:

            return metadata["mixtures"]

    raise ValueError(
        "\nCould not find benchmark cases.\n"
        "Expected a list or a dictionary containing "
        "'cases' or 'mixtures'."
    )


# ============================================================
# GET INPUT MIXTURE PATH
# ============================================================

def get_input_path(case):

    """
    Your actual benchmark metadata uses:

        mixture_file

    Example:

        data\\mixed\\stationary\\
        speech_001__stationary__snr_+5dB.wav
    """

    possible_keys = [

        "mixture_file",

        "mixed_path",

        "mixture_path",

        "input_path",

        "noisy_path",

        "wav_path",

        "path",

    ]

    for key in possible_keys:

        if key in case:

            return case[key]

    raise KeyError(
        "\nCould not find mixture WAV path "
        "in metadata entry:\n"
        + str(case)
    )


# ============================================================
# GET NOISE TYPE
# ============================================================

def get_noise_type(case):

    possible_keys = [
        "noise_type",
        "noise_class",
        "type",
        "category",
    ]

    for key in possible_keys:

        if key in case:

            return str(
                case[key]
            )

    input_path = str(
        get_input_path(case)
    ).lower()

    for noise_type in NOISE_TYPES:

        if noise_type in input_path:

            return noise_type

    raise KeyError(
        "\nCould not determine noise type "
        "from metadata entry:\n"
        + str(case)
    )


# ============================================================
# RESOLVE PROJECT PATH
# ============================================================

def resolve_input_path(path_string):

    path_string = str(
        path_string
    )

    path_string = path_string.replace(
        "\\",
        "/"
    )

    candidate = Path(
        path_string
    )

    # Absolute path
    if candidate.is_absolute():

        return candidate

    # Project-relative path
    candidate = (
        PROJECT_ROOT / candidate
    )

    if candidate.exists():

        return candidate

    # Fallback
    clean_path = path_string.lstrip(
        "./"
    )

    return (
        PROJECT_ROOT / clean_path
    )


# ============================================================
# OUTPUT PATH
# ============================================================

def make_output_path(
    input_path,
    noise_type
):

    filename = (
        f"{input_path.stem}_dfn2.wav"
    )

    return (
        OUTPUT_ROOT
        / noise_type
        / filename
    )


# ============================================================
# RTF
# ============================================================

def calculate_rtf(
    elapsed_seconds,
    num_samples,
    sample_rate
):

    audio_duration = (
        num_samples
        / sample_rate
    )

    if audio_duration <= 0:

        return float("inf")

    return (
        elapsed_seconds
        / audio_duration
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "AURALIS - DeepFilterNet2 Benchmark"
    )

    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Model        : {MODEL_NAME}"
    )

    print(
        f"Sample rate  : "
        f"{EXPECTED_SAMPLE_RATE}"
    )

    print()

    # ========================================================
    # STEP 1
    # ========================================================

    print(
        "[1/5] Loading benchmark metadata..."
    )

    metadata = load_metadata()

    cases = get_cases(
        metadata
    )

    total_cases = len(
        cases
    )

    print(
        f"Found {total_cases} benchmark cases."
    )

    if total_cases != 90:

        print(
            "WARNING:"
            f" Expected 90 cases, "
            f"found {total_cases}."
        )

    print()

    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "[2/5] Loading DeepFilterNet2..."
    )

    # --------------------------------------------------------
    # Load model ONCE.
    #
    # The model is reused.
    #
    # The DF state will be freshly initialized
    # for every independent benchmark file.
    # --------------------------------------------------------

    model, _, suffix = init_df(
        MODEL_NAME
    )

    print(
        "DeepFilterNet2 loaded successfully."
    )

    print(
        f"Model suffix: {suffix}"
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    try:

        device = next(
            model.parameters()
        ).device

    except StopIteration:

        device = torch.device(
            "cpu"
        )

    print(
        f"PyTorch device: {device}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    print()

    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "[3/5] Running inference..."
    )

    print()

    results = []

    successful = 0

    failed = 0

    total_processing_time = 0.0

    total_audio_time = 0.0

    # ========================================================
    # PROCESS ALL CASES
    # ========================================================

    for index, case in enumerate(
        cases,
        start=1
    ):

        print(
            "-" * 70
        )

        print(
            f"Case {index}/{total_cases}"
        )

        # ----------------------------------------------------
        # Case ID
        # ----------------------------------------------------

        case_id = case.get(
            "case_id",
            index
        )

        print(
            f"Case ID    : {case_id}"
        )

        # ----------------------------------------------------
        # Noise type
        # ----------------------------------------------------

        noise_type = get_noise_type(
            case
        )

        print(
            f"Noise type : {noise_type}"
        )

        # ----------------------------------------------------
        # Input path
        # ----------------------------------------------------

        input_path = resolve_input_path(
            get_input_path(case)
        )

        print(
            f"Input      : {input_path}"
        )

        # ----------------------------------------------------
        # Check input
        # ----------------------------------------------------

        if not input_path.exists():

            print(
                "ERROR: Input file does not exist."
            )

            failed += 1

            results.append({

                "case_id":
                    case_id,

                "case_index":
                    index,

                "status":
                    "failed",

                "reason":
                    "input_file_not_found",

                "noise_type":
                    noise_type,

                "input_path":
                    str(input_path),

            })

            continue

        # ----------------------------------------------------
        # Read WAV
        # ----------------------------------------------------

        try:

            noisy_audio, sample_rate = (
                sf.read(
                    str(input_path),
                    dtype="float32",
                    always_2d=False
                )
            )

        except Exception as e:

            print(
                "ERROR reading WAV:"
            )

            print(
                repr(e)
            )

            failed += 1

            results.append({

                "case_id":
                    case_id,

                "case_index":
                    index,

                "status":
                    "failed",

                "reason":
                    "wav_read_error",

                "error":
                    repr(e),

                "noise_type":
                    noise_type,

                "input_path":
                    str(input_path),

            })

            continue

        # ----------------------------------------------------
        # Convert stereo -> mono
        # ----------------------------------------------------

        if noisy_audio.ndim > 1:

            noisy_audio = np.mean(
                noisy_audio,
                axis=1
            ).astype(
                np.float32
            )

        # ----------------------------------------------------
        # Sample rate validation
        # ----------------------------------------------------

        print(
            f"Sample rate: {sample_rate}"
        )

        print(
            f"Samples    : "
            f"{len(noisy_audio)}"
        )

        if sample_rate != (
            EXPECTED_SAMPLE_RATE
        ):

            print(
                "ERROR:"
                f" Expected "
                f"{EXPECTED_SAMPLE_RATE} Hz "
                f"but got "
                f"{sample_rate} Hz."
            )

            failed += 1

            results.append({

                "case_id":
                    case_id,

                "case_index":
                    index,

                "status":
                    "failed",

                "reason":
                    "wrong_sample_rate",

                "sample_rate":
                    int(sample_rate),

                "noise_type":
                    noise_type,

                "input_path":
                    str(input_path),

            })

            continue

        # ----------------------------------------------------
        # Clean NaN / Inf
        # ----------------------------------------------------

        noisy_audio = np.nan_to_num(
            noisy_audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(
            np.float32
        )

        # ====================================================
        # IMPORTANT FIX
        # ====================================================
        #
        # DeepFilterNet expects the waveform with an
        # explicit channel dimension.
        #
        # WRONG:
        #
        #   [samples]
        #
        # CORRECT:
        #
        #   [1, samples]
        #
        # This fixes:
        #
        # TypeError:
        # argument 'input':
        # dimensionality mismatch:
        # from=1, to=2
        #
        # ====================================================

        audio_tensor = torch.from_numpy(
            noisy_audio
        ).float()

        audio_tensor = (
            audio_tensor
            .unsqueeze(0)
        )

        audio_tensor = (
            audio_tensor.to(device)
        )

        print(
            f"Tensor shape: "
            f"{tuple(audio_tensor.shape)}"
        )

        # ====================================================
        # FRESH DF STATE FOR EACH CASE
        # ====================================================
        #
        # Every WAV is an independent benchmark.
        #
        # Therefore do not carry the previous file's
        # streaming/DSP state into the next file.
        #
        # ====================================================

        try:

            _, df_state, _ = init_df(
                MODEL_NAME
            )

            # Make sure the state is on the
            # same device when supported.

            try:

                df_state = df_state.to(
                    device
                )

            except AttributeError:

                pass

        except Exception as e:

            print(
                "ERROR initializing "
                "DeepFilterNet state:"
            )

            print(
                repr(e)
            )

            failed += 1

            results.append({

                "case_id":
                    case_id,

                "case_index":
                    index,

                "status":
                    "failed",

                "reason":
                    "df_state_initialization_error",

                "error":
                    repr(e),

                "noise_type":
                    noise_type,

                "input_path":
                    str(input_path),

            })

            continue

        # ====================================================
        # RUN ENHANCEMENT
        # ====================================================

        print(
            "Running DeepFilterNet2..."
        )

        try:

            start_time = (
                time.perf_counter()
            )

            with torch.no_grad():

                enhanced = enhance(
                    model,
                    df_state,
                    audio_tensor,
                    pad=True
                )

            elapsed = (
                time.perf_counter()
                - start_time
            )

        except Exception as e:

            print()

            print(
                "ERROR during "
                "DeepFilterNet2 inference:"
            )

            print(
                repr(e)
            )

            print()

            failed += 1

            results.append({

                "case_id":
                    case_id,

                "case_index":
                    index,

                "status":
                    "failed",

                "reason":
                    "inference_error",

                "error":
                    repr(e),

                "noise_type":
                    noise_type,

                "input_path":
                    str(input_path),

            })

            continue

        # ====================================================
        # CONVERT OUTPUT TO NUMPY
        # ====================================================

        if isinstance(
            enhanced,
            torch.Tensor
        ):

            enhanced = (
                enhanced
                .detach()
                .cpu()
                .numpy()
            )

        enhanced = np.asarray(
            enhanced,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Remove dimensions of size 1.
        #
        # [1, samples] -> [samples]
        # ----------------------------------------------------

        enhanced = np.squeeze(
            enhanced
        )

        # ----------------------------------------------------
        # Validate output
        # ----------------------------------------------------

        if enhanced.ndim != 1:

            print(
                "ERROR:"
                " Enhanced output has "
                f"shape {enhanced.shape}"
            )

            failed += 1

            results.append({

                "case_id":
                    case_id,

                "case_index":
                    index,

                "status":
                    "failed",

                "reason":
                    "invalid_output_shape",

                "output_shape":
                    list(
                        enhanced.shape
                    ),

                "noise_type":
                    noise_type,

                "input_path":
                    str(input_path),

            })

            continue

        # ----------------------------------------------------
        # Clean output
        # ----------------------------------------------------

        enhanced = np.nan_to_num(
            enhanced,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Check length
        # ----------------------------------------------------

        output_length_changed = (
            len(enhanced)
            != len(noisy_audio)
        )

        if output_length_changed:

            print(
                "NOTE:"
                " output length differs "
                "from input."
            )

            print(
                f"Input samples : "
                f"{len(noisy_audio)}"
            )

            print(
                f"Output samples: "
                f"{len(enhanced)}"
            )

        # ----------------------------------------------------
        # Prevent clipping
        # ----------------------------------------------------

        peak = float(
            np.max(
                np.abs(enhanced)
            )
        )

        if peak > 1.0:

            print(
                f"Output peak "
                f"{peak:.6f} > 1.0."
            )

            print(
                "Normalizing output."
            )

            enhanced = (
                enhanced
                / peak
                * 0.999
            )

        # ====================================================
        # SAVE OUTPUT
        # ====================================================

        output_path = make_output_path(
            input_path,
            noise_type
        )

        try:

            sf.write(
                str(output_path),
                enhanced,
                EXPECTED_SAMPLE_RATE,
                subtype="PCM_16"
            )

        except Exception as e:

            print(
                "ERROR writing output WAV:"
            )

            print(
                repr(e)
            )

            failed += 1

            results.append({

                "case_id":
                    case_id,

                "case_index":
                    index,

                "status":
                    "failed",

                "reason":
                    "wav_write_error",

                "error":
                    repr(e),

                "noise_type":
                    noise_type,

                "input_path":
                    str(input_path),

            })

            continue

        # ====================================================
        # RUNTIME
        # ====================================================

        audio_duration = (
            len(noisy_audio)
            / EXPECTED_SAMPLE_RATE
        )

        rtf = calculate_rtf(
            elapsed,
            len(noisy_audio),
            EXPECTED_SAMPLE_RATE
        )

        # ----------------------------------------------------
        # Update aggregate
        # ----------------------------------------------------

        successful += 1

        total_processing_time += (
            elapsed
        )

        total_audio_time += (
            audio_duration
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"Processing time : "
            f"{elapsed:.4f} s"
        )

        print(
            f"Audio duration  : "
            f"{audio_duration:.4f} s"
        )

        print(
            f"RTF             : "
            f"{rtf:.4f}"
        )

        print(
            f"Output          : "
            f"{output_path}"
        )

        # ====================================================
        # RESULT ENTRY
        # ====================================================

        result_entry = {

            "case_id":
                case_id,

            "case_index":
                index,

            "status":
                "success",

            "noise_type":
                noise_type,

            "input_path":
                str(
                    input_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "output_path":
                str(
                    output_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "sample_rate_hz":
                EXPECTED_SAMPLE_RATE,

            "input_samples":
                int(
                    len(noisy_audio)
                ),

            "output_samples":
                int(
                    len(enhanced)
                ),

            "output_length_changed":
                bool(
                    output_length_changed
                ),

            "audio_duration_seconds":
                float(
                    audio_duration
                ),

            "processing_time_seconds":
                float(
                    elapsed
                ),

            "real_time_factor":
                float(
                    rtf
                ),

            "faster_than_realtime":
                bool(
                    rtf < 1.0
                ),

        }

        # ----------------------------------------------------
        # Preserve benchmark metadata
        # ----------------------------------------------------

        metadata_keys = [

            "speech_id",

            "clean_file",

            "noise_source",

            "target_snr_db",

            "actual_snr_db",

            "sample_rate_hz",

            "num_samples",

            "duration_sec",

        ]

        for key in metadata_keys:

            if key in case:

                result_entry[key] = (
                    case[key]
                )

        results.append(
            result_entry
        )

        print()

    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print("=" * 70)

    print(
        "[4/5] Calculating benchmark results..."
    )

    print("=" * 70)

    successful_results = [

        result

        for result in results

        if result.get(
            "status"
        ) == "success"

    ]

    # --------------------------------------------------------
    # RTF statistics
    # --------------------------------------------------------

    rtf_values = [

        result[
            "real_time_factor"
        ]

        for result
        in successful_results

    ]

    processing_times = [

        result[
            "processing_time_seconds"
        ]

        for result
        in successful_results

    ]

    if rtf_values:

        aggregate_rtf = (

            total_processing_time
            / total_audio_time

            if total_audio_time > 0

            else float("inf")

        )

        mean_rtf = float(
            np.mean(
                rtf_values
            )
        )

        median_rtf = float(
            np.median(
                rtf_values
            )
        )

        minimum_rtf = float(
            np.min(
                rtf_values
            )
        )

        maximum_rtf = float(
            np.max(
                rtf_values
            )
        )

        mean_processing_time = float(
            np.mean(
                processing_times
            )
        )

    else:

        aggregate_rtf = None
        mean_rtf = None
        median_rtf = None
        minimum_rtf = None
        maximum_rtf = None
        mean_processing_time = None

    # ========================================================
    # NOISE CLASS SUMMARY
    # ========================================================

    noise_summary = {}

    for noise_type in NOISE_TYPES:

        class_results = [

            result

            for result
            in successful_results

            if result.get(
                "noise_type"
            ) == noise_type

        ]

        class_rtf = [

            result[
                "real_time_factor"
            ]

            for result
            in class_results

        ]

        noise_summary[
            noise_type
        ] = {

            "total_successful":
                len(
                    class_results
                ),

            "mean_rtf": (

                float(
                    np.mean(
                        class_rtf
                    )
                )

                if class_rtf

                else None

            ),

            "median_rtf": (

                float(
                    np.median(
                        class_rtf
                    )
                )

                if class_rtf

                else None

            ),

        }

    # ========================================================
    # FINAL JSON
    # ========================================================

    final_results = {

        "project":
            "AURALIS",

        "benchmark":
            (
                "DeepFilterNet2 benchmark "
                "on stationary, "
                "nonstationary and "
                "impulsive mixtures"
            ),

        "model":
            MODEL_NAME,

        "sample_rate_hz":
            EXPECTED_SAMPLE_RATE,

        "total_cases":
            total_cases,

        "successful_cases":
            successful,

        "failed_cases":
            failed,

        "success_rate_percent": (

            (
                100.0
                * successful
                / total_cases
            )

            if total_cases > 0

            else 0.0

        ),

        "aggregate": {

            "total_audio_seconds":
                float(
                    total_audio_time
                ),

            "total_processing_seconds":
                float(
                    total_processing_time
                ),

            "aggregate_rtf": (

                float(
                    aggregate_rtf
                )

                if aggregate_rtf
                is not None

                else None

            ),

            "mean_rtf":
                mean_rtf,

            "median_rtf":
                median_rtf,

            "minimum_rtf":
                minimum_rtf,

            "maximum_rtf":
                maximum_rtf,

            "mean_processing_time_seconds":
                mean_processing_time,

        },

        "noise_summary":
            noise_summary,

        "cases":
            results,

    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_results,
            f,
            indent=2
        )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_FILE
    )

    # ========================================================
    # STEP 5
    # ========================================================

    print()
    print("=" * 70)

    print(
        "[5/5] FINAL SUMMARY"
    )

    print("=" * 70)

    print(
        f"Total cases       : "
        f"{total_cases}"
    )

    print(
        f"Successful        : "
        f"{successful}"
    )

    print(
        f"Failed            : "
        f"{failed}"
    )

    if total_cases > 0:

        success_rate = (
            100.0
            * successful
            / total_cases
        )

        print(
            f"Success rate      : "
            f"{success_rate:.2f}%"
        )

    if aggregate_rtf is not None:

        print(
            f"Aggregate RTF     : "
            f"{aggregate_rtf:.4f}"
        )

        print(
            f"Mean RTF          : "
            f"{mean_rtf:.4f}"
        )

        print(
            f"Median RTF        : "
            f"{median_rtf:.4f}"
        )

        print(
            f"Minimum RTF       : "
            f"{minimum_rtf:.4f}"
        )

        print(
            f"Maximum RTF       : "
            f"{maximum_rtf:.4f}"
        )

    print()
    print(
        "Noise-class summary:"
    )

    for noise_type in NOISE_TYPES:

        summary = noise_summary[
            noise_type
        ]

        print(
            f"  {noise_type:15s}"
            f" success="
            f"{summary['total_successful']:2d}"
            f"  mean_RTF="
            f"{summary['mean_rtf']}"
        )

    print()
    print("=" * 70)

    print(
        "DeepFilterNet2 benchmark completed."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
