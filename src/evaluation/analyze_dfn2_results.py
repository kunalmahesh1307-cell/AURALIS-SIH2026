"""
AURALIS — DeepFilterNet2 Evaluation Analysis
=============================================

Purpose
-------
Analyze the completed 90-case DeepFilterNet2 objective evaluation.

Input
-----
results/evaluation_dfn2/dfn2_objective_results.csv

Outputs
-------
results/evaluation_dfn2/analysis/
    class_summary.csv
    snr_summary.csv
    class_snr_summary.csv
    overall_statistics.json
    best_cases.csv
    worst_cases.csv
    dfn2_evaluation_report.txt

    snr_improvement_by_class.png
    stoi_improvement_by_class.png
    pesq_improvement_by_class.png
    sisdr_improvement_by_class.png

    snr_improvement_by_snr.png
    stoi_improvement_by_snr.png
    pesq_improvement_by_snr.png
    sisdr_improvement_by_snr.png

Important
---------
The evaluation CSV contains measured SNR values.

For example, nominal +5 dB mixtures may have values such as:
    5.000038
    4.999998
    4.999971

Likewise, impulsive mixtures can have measured whole-signal SNR
values that differ substantially from the nominal target because
the impulsive event is sparse.

Therefore:

    measured SNR  -> preserved exactly for metric reporting
    target SNR    -> separately reconstructed as +5 / 0 / -5 dB

The SNR-wise analysis uses target SNR.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "evaluation_dfn2"
    / "dfn2_objective_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation_dfn2"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPECTED BENCHMARK STRUCTURE
# ============================================================

EXPECTED_CASES = 90

CLASS_ORDER = [
    "stationary",
    "nonstationary",
    "impulsive",
]

TARGET_SNR_ORDER = [
    5,
    0,
    -5,
]


# ============================================================
# METRIC DEFINITIONS
# ============================================================

METRICS = {
    "snr_improvement_db": "SNR Improvement (dB)",
    "stoi_improvement": "STOI Improvement",
    "pesq_improvement": "PESQ Improvement",
    "si_sdr_improvement_db": "SI-SDR Improvement (dB)",
}


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):
    """
    Normalize possible column-name variants.

    The evaluation script may use slightly different names
    depending on its version. This function maps them to the
    canonical names required by this analysis.
    """

    rename_map = {}

    aliases = {
        "noise_type": [
            "noise_type",
            "noise_class",
            "class",
        ],

        "snr_db": [
            "snr_db",
            "input_snr_db",
            "measured_snr_db",
            "target_snr_db",
        ],

        "snr_improvement_db": [
            "snr_improvement_db",
            "snr_improvement",
            "delta_snr_db",
        ],

        "stoi_improvement": [
            "stoi_improvement",
            "stoi_delta",
            "delta_stoi",
        ],

        "pesq_improvement": [
            "pesq_improvement",
            "pesq_delta",
            "delta_pesq",
        ],

        "si_sdr_improvement_db": [
            "si_sdr_improvement_db",
            "sisdr_improvement_db",
            "si_sdr_improvement",
            "sisdr_improvement",
            "delta_si_sdr_db",
        ],
    }

    for canonical_name, candidates in aliases.items():

        for candidate in candidates:

            if candidate in df.columns:

                rename_map[candidate] = canonical_name
                break

    df = df.rename(
        columns=rename_map
    )

    required_columns = [
        "noise_type",
        "snr_db",
        "snr_improvement_db",
        "stoi_improvement",
        "pesq_improvement",
        "si_sdr_improvement_db",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
            + "\n\nAvailable columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in df.columns
            )
        )

    return df


# ============================================================
# TARGET-SNR CLASSIFICATION
# ============================================================

def classify_target_snr(measured_snr):
    """
    Map measured SNR to the nearest nominal benchmark target.

    Nominal benchmark targets:
        +5 dB
         0 dB
        -5 dB

    A tolerance of ±0.5 dB is used.

    Values outside these ranges are retained in the raw
    measured-SNR column but are not assigned to a target-SNR
    group.

    This is particularly relevant to sparse impulsive noise.
    """

    if pd.isna(measured_snr):
        return np.nan

    measured_snr = float(measured_snr)

    if abs(measured_snr - 5.0) <= 0.5:
        return 5

    if abs(measured_snr - 0.0) <= 0.5:
        return 0

    if abs(measured_snr + 5.0) <= 0.5:
        return -5

    return np.nan


# ============================================================
# ROUND NUMERIC DATAFRAME
# ============================================================

def safe_round(df, digits=4):
    """
    Round only numeric columns.
    """

    result = df.copy()

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = (
        result[numeric_columns].round(digits)
    )

    return result


# ============================================================
# SAVE PLOT
# ============================================================

def save_plot(fig, filename):
    """
    Save a Matplotlib figure as a high-resolution PNG.
    """

    path = OUTPUT_DIR / filename

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved plot: {path}"
    )


# ============================================================
# START
# ============================================================

print("=" * 72)
print("AURALIS — DeepFilterNet2 Evaluation Analysis")
print("=" * 72)

print(
    f"\nProject root : {PROJECT_ROOT}"
)

print(
    f"Input CSV    : {INPUT_CSV}"
)

print(
    f"Output dir   : {OUTPUT_DIR}"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        "\nEvaluation CSV not found:\n"
        f"{INPUT_CSV}\n\n"
        "Run evaluate_dfn2.py first."
    )


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print(
    f"\nLoaded evaluation rows: {len(df)}"
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

df = normalize_columns(
    df
)


# ============================================================
# NORMALIZE NOISE CLASS
# ============================================================

df["noise_type"] = (
    df["noise_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

df["snr_db"] = pd.to_numeric(
    df["snr_db"],
    errors="coerce",
)

for metric in METRICS:

    df[metric] = pd.to_numeric(
        df[metric],
        errors="coerce",
    )


# ============================================================
# CREATE TARGET-SNR GROUP
# ============================================================

# IMPORTANT:
# Keep df["snr_db"] untouched.
#
# This remains the measured SNR from the evaluation.
#
# We create a separate column for the nominal benchmark target.

df["target_snr_db"] = (
    df["snr_db"]
    .apply(classify_target_snr)
)


# ============================================================
# REMOVE INCOMPLETE METRIC ROWS
# ============================================================

required_metric_columns = [
    "noise_type",
    "snr_db",
    "snr_improvement_db",
    "stoi_improvement",
    "pesq_improvement",
    "si_sdr_improvement_db",
]

before_count = len(df)

df = df.dropna(
    subset=required_metric_columns
).copy()

removed_count = (
    before_count
    - len(df)
)

if removed_count:

    print(
        f"Removed incomplete rows: "
        f"{removed_count}"
    )

print(
    f"Valid evaluation rows: {len(df)}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n")
print("Validation")
print("-" * 72)


print(
    "Classes:",
    ", ".join(
        sorted(
            df["noise_type"].unique()
        )
    ),
)


# ------------------------------------------------------------
# Measured SNR
# ------------------------------------------------------------

print(
    "Measured SNR range:",
    f"{df['snr_db'].min():.4f}"
    f" to "
    f"{df['snr_db'].max():.4f} dB"
)


# ------------------------------------------------------------
# Target SNR
# ------------------------------------------------------------

target_snr_values = sorted(
    df["target_snr_db"]
    .dropna()
    .unique(),
    reverse=True,
)

print(
    "Target SNR groups:",
    target_snr_values,
)


# ------------------------------------------------------------
# Case-count validation
# ------------------------------------------------------------

if len(df) == EXPECTED_CASES:

    print(
        f"PASS: {EXPECTED_CASES}/"
        f"{EXPECTED_CASES} cases available."
    )

else:

    print(
        f"WARNING: expected "
        f"{EXPECTED_CASES} cases, "
        f"found {len(df)}."
    )


# ============================================================
# TARGET-SNR GROUP VALIDATION
# ============================================================

missing_target_groups = [
    snr
    for snr in TARGET_SNR_ORDER
    if snr not in target_snr_values
]

if missing_target_groups:

    print(
        "\nWARNING:"
    )

    print(
        "Missing expected target-SNR groups:",
        missing_target_groups,
    )

else:

    print(
        "PASS: +5 dB, 0 dB and -5 dB "
        "target groups detected."
    )


# ============================================================
# OVERALL STATISTICS
# ============================================================

overall = {}


for metric in METRICS:

    series = df[metric]

    overall[metric] = {

        "mean": float(
            series.mean()
        ),

        "median": float(
            series.median()
        ),

        "std": float(
            series.std(
                ddof=1
            )
        ),

        "minimum": float(
            series.min()
        ),

        "maximum": float(
            series.max()
        ),

        "q25": float(
            series.quantile(
                0.25
            )
        ),

        "q75": float(
            series.quantile(
                0.75
            )
        ),
    }


overall["evaluation_cases"] = int(
    len(df)
)

overall["successful_cases"] = int(
    len(df)
)

overall["success_rate_percent"] = (
    float(
        len(df)
        / EXPECTED_CASES
        * 100
    )
    if EXPECTED_CASES > 0
    else None
)


# ============================================================
# CLASS-WISE SUMMARY
# ============================================================

class_summary = (
    df.groupby(
        "noise_type",
        sort=False,
    )
    .agg(

        cases=(
            "noise_type",
            "size",
        ),

        snr_improvement_mean=(
            "snr_improvement_db",
            "mean",
        ),

        snr_improvement_median=(
            "snr_improvement_db",
            "median",
        ),

        snr_improvement_std=(
            "snr_improvement_db",
            "std",
        ),

        stoi_improvement_mean=(
            "stoi_improvement",
            "mean",
        ),

        stoi_improvement_median=(
            "stoi_improvement",
            "median",
        ),

        stoi_improvement_std=(
            "stoi_improvement",
            "std",
        ),

        pesq_improvement_mean=(
            "pesq_improvement",
            "mean",
        ),

        pesq_improvement_median=(
            "pesq_improvement",
            "median",
        ),

        pesq_improvement_std=(
            "pesq_improvement",
            "std",
        ),

        sisdr_improvement_mean=(
            "si_sdr_improvement_db",
            "mean",
        ),

        sisdr_improvement_median=(
            "si_sdr_improvement_db",
            "median",
        ),

        sisdr_improvement_std=(
            "si_sdr_improvement_db",
            "std",
        ),
    )
)


# Preserve desired class order.

existing_classes = [
    noise_class
    for noise_class in CLASS_ORDER
    if noise_class in class_summary.index
]

class_summary = (
    class_summary.reindex(
        existing_classes
    )
)


class_summary = safe_round(
    class_summary
)


class_summary.to_csv(
    OUTPUT_DIR
    / "class_summary.csv"
)


# ============================================================
# TARGET-SNR-WISE SUMMARY
# ============================================================

# Only use rows that could be confidently assigned to
# +5 / 0 / -5 dB nominal target groups.

target_snr_df = df.dropna(
    subset=["target_snr_db"]
).copy()


snr_summary = (
    target_snr_df
    .groupby(
        "target_snr_db"
    )
    .agg(

        cases=(
            "target_snr_db",
            "size",
        ),

        snr_improvement_mean=(
            "snr_improvement_db",
            "mean",
        ),

        snr_improvement_median=(
            "snr_improvement_db",
            "median",
        ),

        snr_improvement_std=(
            "snr_improvement_db",
            "std",
        ),

        stoi_improvement_mean=(
            "stoi_improvement",
            "mean",
        ),

        stoi_improvement_median=(
            "stoi_improvement",
            "median",
        ),

        stoi_improvement_std=(
            "stoi_improvement",
            "std",
        ),

        pesq_improvement_mean=(
            "pesq_improvement",
            "mean",
        ),

        pesq_improvement_median=(
            "pesq_improvement",
            "median",
        ),

        pesq_improvement_std=(
            "pesq_improvement",
            "std",
        ),

        sisdr_improvement_mean=(
            "si_sdr_improvement_db",
            "mean",
        ),

        sisdr_improvement_median=(
            "si_sdr_improvement_db",
            "median",
        ),

        sisdr_improvement_std=(
            "si_sdr_improvement_db",
            "std",
        ),
    )
)


snr_summary = (
    snr_summary.reindex(
        [
            snr
            for snr in TARGET_SNR_ORDER
            if snr in snr_summary.index
        ]
    )
)


snr_summary = safe_round(
    snr_summary
)


snr_summary.to_csv(
    OUTPUT_DIR
    / "snr_summary.csv"
)


# ============================================================
# CLASS × TARGET-SNR SUMMARY
# ============================================================

class_snr_summary = (
    target_snr_df
    .groupby(
        [
            "noise_type",
            "target_snr_db",
        ],
        sort=False,
    )
    .agg(

        cases=(
            "noise_type",
            "size",
        ),

        snr_improvement_mean=(
            "snr_improvement_db",
            "mean",
        ),

        snr_improvement_std=(
            "snr_improvement_db",
            "std",
        ),

        stoi_improvement_mean=(
            "stoi_improvement",
            "mean",
        ),

        stoi_improvement_std=(
            "stoi_improvement",
            "std",
        ),

        pesq_improvement_mean=(
            "pesq_improvement",
            "mean",
        ),

        pesq_improvement_std=(
            "pesq_improvement",
            "std",
        ),

        sisdr_improvement_mean=(
            "si_sdr_improvement_db",
            "mean",
        ),

        sisdr_improvement_std=(
            "si_sdr_improvement_db",
            "std",
        ),
    )
)


class_snr_summary = safe_round(
    class_snr_summary
)


class_snr_summary.to_csv(
    OUTPUT_DIR
    / "class_snr_summary.csv"
)


# ============================================================
# BEST CASES
# ============================================================

best_cases = (
    df.sort_values(
        "snr_improvement_db",
        ascending=False,
    )
    .head(10)
    .copy()
)


best_cases.to_csv(
    OUTPUT_DIR
    / "best_cases.csv",
    index=False,
)


# ============================================================
# WORST CASES
# ============================================================

worst_cases = (
    df.sort_values(
        "snr_improvement_db",
        ascending=True,
    )
    .head(10)
    .copy()
)


worst_cases.to_csv(
    OUTPUT_DIR
    / "worst_cases.csv",
    index=False,
)


# ============================================================
# SAVE OVERALL JSON
# ============================================================

overall_json_path = (
    OUTPUT_DIR
    / "overall_statistics.json"
)


with open(
    overall_json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        overall,
        f,
        indent=4,
    )


# ============================================================
# TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "dfn2_evaluation_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "AURALIS — DeepFilterNet2 "
        "Objective Evaluation Report\n"
    )

    f.write(
        "=" * 72
        + "\n\n"
    )

    f.write(
        f"Evaluation cases : "
        f"{len(df)}\n"
    )

    f.write(
        f"Success rate     : "
        f"{overall['success_rate_percent']:.2f}%\n"
    )

    f.write(
        f"Measured SNR min : "
        f"{df['snr_db'].min():.4f} dB\n"
    )

    f.write(
        f"Measured SNR max : "
        f"{df['snr_db'].max():.4f} dB\n"
    )

    f.write(
        "\nTarget SNR groups: "
        f"{target_snr_values}\n\n"
    )


    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    f.write(
        "OVERALL METRICS\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )


    for metric, values in overall.items():

        if not isinstance(
            values,
            dict,
        ):
            continue

        f.write(
            f"\n{metric}\n"
        )

        f.write(
            f"  Mean   : "
            f"{values['mean']:.4f}\n"
        )

        f.write(
            f"  Median : "
            f"{values['median']:.4f}\n"
        )

        f.write(
            f"  Std    : "
            f"{values['std']:.4f}\n"
        )

        f.write(
            f"  Min    : "
            f"{values['minimum']:.4f}\n"
        )

        f.write(
            f"  Max    : "
            f"{values['maximum']:.4f}\n"
        )

        f.write(
            f"  Q25    : "
            f"{values['q25']:.4f}\n"
        )

        f.write(
            f"  Q75    : "
            f"{values['q75']:.4f}\n"
        )


    # --------------------------------------------------------
    # Class summary
    # --------------------------------------------------------

    f.write(
        "\n\nCLASS-WISE SUMMARY\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        class_summary.to_string()
    )


    # --------------------------------------------------------
    # Target SNR summary
    # --------------------------------------------------------

    f.write(
        "\n\n\nTARGET-SNR-WISE SUMMARY\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        snr_summary.to_string()
    )


    # --------------------------------------------------------
    # Class × target SNR
    # --------------------------------------------------------

    f.write(
        "\n\n\nCLASS × TARGET-SNR SUMMARY\n"
    )

    f.write(
        "-" * 72
        + "\n"
    )

    f.write(
        class_snr_summary.to_string()
    )


# ============================================================
# CONSOLE OVERALL SUMMARY
# ============================================================

print("\n")
print("=" * 72)
print("OVERALL RESULTS")
print("=" * 72)


print(
    f"\nCases evaluated : "
    f"{len(df)}"
)


print(
    f"Success rate    : "
    f"{overall['success_rate_percent']:.2f}%"
)


for metric, label in METRICS.items():

    print(
        f"{label:<30}: "
        f"{overall[metric]['mean']:+.4f}"
    )


# ============================================================
# CONSOLE CLASS SUMMARY
# ============================================================

print("\n")
print("=" * 72)
print("CLASS-WISE MEAN IMPROVEMENTS")
print("=" * 72)


class_display_columns = [
    "snr_improvement_mean",
    "stoi_improvement_mean",
    "pesq_improvement_mean",
    "sisdr_improvement_mean",
]


print(
    class_summary[
        class_display_columns
    ].to_string()
)


# ============================================================
# CONSOLE TARGET-SNR SUMMARY
# ============================================================

print("\n")
print("=" * 72)
print("TARGET-SNR-WISE MEAN IMPROVEMENTS")
print("=" * 72)


if len(snr_summary) == 0:

    print(
        "No target-SNR groups available."
    )

else:

    print(
        snr_summary[
            class_display_columns
        ].to_string()
    )


# ============================================================
# CONSOLE CLASS × TARGET-SNR
# ============================================================

print("\n")
print("=" * 72)
print("CLASS × TARGET-SNR MEAN IMPROVEMENTS")
print("=" * 72)


if len(class_snr_summary) == 0:

    print(
        "No class × target-SNR groups available."
    )

else:

    print(
        class_snr_summary[
            [
                "cases",
                "snr_improvement_mean",
                "stoi_improvement_mean",
                "pesq_improvement_mean",
                "sisdr_improvement_mean",
            ]
        ].to_string()
    )


# ============================================================
# CLASS-WISE PLOTS
# ============================================================

class_labels = [
    noise_class
    for noise_class in CLASS_ORDER
    if noise_class in class_summary.index
]


x = np.arange(
    len(class_labels)
)


class_metric_columns = {

    "snr_improvement_db":
        "snr_improvement_mean",

    "stoi_improvement":
        "stoi_improvement_mean",

    "pesq_improvement":
        "pesq_improvement_mean",

    "si_sdr_improvement_db":
        "sisdr_improvement_mean",
}


class_plot_names = {

    "snr_improvement_db":
        "snr_improvement_by_class.png",

    "stoi_improvement":
        "stoi_improvement_by_class.png",

    "pesq_improvement":
        "pesq_improvement_by_class.png",

    "si_sdr_improvement_db":
        "sisdr_improvement_by_class.png",
}


# ============================================================
# GENERATE CLASS-WISE PLOTS
# ============================================================

for metric, ylabel in METRICS.items():

    mean_column = (
        class_metric_columns[metric]
    )

    values = (
        class_summary
        .loc[
            class_labels,
            mean_column,
        ]
        .values
    )


    fig = plt.figure(
        figsize=(8, 5)
    )


    ax = fig.add_subplot(
        111
    )


    bars = ax.bar(
        x,
        values,
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        class_labels
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        f"DFN2 — {ylabel} by Noise Class"
    )


    ax.grid(
        axis="y",
        alpha=0.25,
    )


    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )


    save_plot(
        fig,
        class_plot_names[metric],
    )


# ============================================================
# TARGET-SNR PLOTS
# ============================================================

target_snr_labels = [
    snr
    for snr in TARGET_SNR_ORDER
    if snr in snr_summary.index
]


x = np.arange(
    len(target_snr_labels)
)


snr_plot_names = {

    "snr_improvement_db":
        "snr_improvement_by_snr.png",

    "stoi_improvement":
        "stoi_improvement_by_snr.png",

    "pesq_improvement":
        "pesq_improvement_by_snr.png",

    "si_sdr_improvement_db":
        "sisdr_improvement_by_snr.png",
}


# ============================================================
# GENERATE TARGET-SNR PLOTS
# ============================================================

for metric, ylabel in METRICS.items():

    mean_column = (
        class_metric_columns[metric]
    )


    if len(target_snr_labels) == 0:

        print(
            f"Skipping {ylabel} target-SNR plot: "
            "no target-SNR groups."
        )

        continue


    values = (
        snr_summary
        .loc[
            target_snr_labels,
            mean_column,
        ]
        .values
    )


    fig = plt.figure(
        figsize=(8, 5)
    )


    ax = fig.add_subplot(
        111
    )


    bars = ax.bar(
        x,
        values,
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        [
            f"{int(value):+d} dB"
            for value in target_snr_labels
        ]
    )


    ax.set_xlabel(
        "Nominal Target SNR"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        f"DFN2 — {ylabel} by Target SNR"
    )


    ax.grid(
        axis="y",
        alpha=0.25,
    )


    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )


    save_plot(
        fig,
        snr_plot_names[metric],
    )


# ============================================================
# FINAL FILE LIST
# ============================================================

print("\n")
print("=" * 72)
print("DFN2 ANALYSIS COMPLETE")
print("=" * 72)


print(
    f"\nAnalysis directory:\n"
    f"{OUTPUT_DIR}"
)


print(
    "\nGenerated files:"
)


for path in sorted(
    OUTPUT_DIR.iterdir()
):

    if path.is_file():

        print(
            f"  - {path.name}"
        )


print("\n")
