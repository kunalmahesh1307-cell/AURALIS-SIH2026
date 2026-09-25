# AURALIS

**AI-based adaptive digital speech enhancement for defence-relevant noise**

AURALIS is a lightweight AI/ML-based digital speech-enhancement prototype designed to suppress stationary, non-stationary and impulsive defence-relevant noise while preserving speech intelligibility, with embedded real-time deployment as the engineering target.

> **Current status:** Working laptop-based software prototype + controlled objective validation. Embedded hardware integration is the next engineering stage.

## Why AURALIS

Defence communication can be affected by continuously varying and impulsive acoustic disturbances. AURALIS focuses on a software-first enhancement layer that can later be integrated into an embedded audio path.

## System

`	ext
Microphone
    â†“
Audio Capture
    â†“
STFT / Feature Processing
    â†“
DeepFilterNet2
    â†“
ERB Gain + Deep Filtering
    â†“
ISTFT
    â†“
Enhanced Speech
`

## Validated objective benchmark

**90/90 successful inference cases**

| Metric | Mean improvement |
|---|---:|
| SNR | **+12.4529 dB** |
| STOI | **+0.0679** |
| PESQ | **+0.9238** |
| SI-SDR | **+13.7386 dB** |
| Offline RTF | **â‰ˆ 0.0609** |

Evaluation coverage:
- 10 clean speech files
- Stationary, non-stationary and impulsive noise
- Target SNR: +5, 0 and -5 dB

The benchmark is controlled offline evaluation. It should not be described as proof of guaranteed end-to-end embedded real-time performance.

## Live software prototype

The demonstration supports:
- microphone recording
- selectable noise class
- selectable target SNR
- DeepFilterNet2 enhancement
- noisy/enhanced playback
- indicative live-segment metrics
- benchmark evidence display

## Hardware deployment roadmap

`	ext
MEMS Microphone
       â†“
 Audio Codec / I2S
       â†“
 Embedded Compute
       â†“
  AURALIS DFN2
       â†“
 Codec / DAC
       â†“
 Headset / Radio
`

The first proposed embedded engineering platform is Raspberry Pi Compute Module 5.

## Repository

- src/ â€” core implementation
- demo/ â€” demonstration entry point and small evidence artifacts
- models/ â€” model documentation and model integration notes
- esults/benchmark/ â€” compact benchmark metadata
- hardware/ â€” proposed embedded architecture
- docs/ â€” benchmark and deployment documentation

## Important technical status

AURALIS is currently a **digital AI speech-enhancement system**, not a completed physical acoustic ANC system. The embedded hardware architecture is a deployment roadmap until physically assembled and validated.

## Reproducibility

Do not commit:
- virtual environments
- raw datasets
- secrets
- large generated audio collections

Use the documented dataset/model sources and reproduce the benchmark locally where licensing and access permit.

## SIH 2026

- Problem Statement: **SIH26052**
- Team: **Auralis**
- Category: **Hardware**
- Theme: **Miscellaneous**
