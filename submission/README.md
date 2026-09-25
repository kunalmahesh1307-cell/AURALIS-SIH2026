\# AURALIS — AI/ML Adaptive Speech Enhancement



\*\*Smart India Hackathon 2026\*\*  

\*\*Problem Statement:\*\* SIH26052  

\*\*Theme:\*\* Miscellaneous  

\*\*Category:\*\* Hardware  

\*\*Team:\*\* Auralis



\---



\## 1. Problem Statement



Defence and mission-critical communication systems operate in environments containing highly variable acoustic disturbances, including:



\- Stationary noise

\- Non-stationary noise

\- Impulsive / gunshot-like noise

\- Vehicle and rotor noise

\- Siren-like disturbances



The challenge is to suppress these disturbances while preserving speech intelligibility and maintaining a path toward real-time embedded deployment.



\---



\## 2. Proposed Solution



\*\*AURALIS\*\* is an AI/ML-based digital speech-enhancement system designed to suppress stationary, non-stationary and impulsive defence-relevant noise while preserving speech information.



The current implementation uses \*\*DeepFilterNet2 (DFN2)\*\* as the primary speech-enhancement engine.



\### Core pipeline



```text

Microphone

&#x20;   ↓

Audio Capture

&#x20;   ↓

Resampling / Preprocessing

&#x20;   ↓

STFT

&#x20;   ↓

ERB Feature Processing

&#x20;   ↓

DeepFilterNet2

&#x20;   ↓

ERB Gain + Deep Filtering

&#x20;   ↓

ISTFT

&#x20;   ↓

Enhanced Speech

&#x20;   ↓

Headset / Speaker

3. Current Working Prototype



The present AURALIS prototype is a working laptop-based digital speech-enhancement system.



The prototype supports:



Live microphone recording

Stationary noise testing

Non-stationary noise testing

Impulsive / gunshot-like noise testing

Configurable target SNR

DeepFilterNet2 enhancement

Before/after audio playback

Indicative live metrics

Benchmark result reporting

CSV-based experiment logging



The practical demonstration currently uses a microphone/audio device and headset as external accessories.



Current status



Working: Laptop-based software prototype

Validated: Controlled 90-case objective benchmark

In progress: Deterministic streaming and embedded deployment

Planned: CM5-based engineering hardware and custom carrier







4\. Objective Validation — 90 Cases



AURALIS was evaluated on 90 controlled speech-noise mixtures covering:



3 noise classes

3 target SNR conditions: +5 dB, 0 dB and −5 dB

10 speech samples per noise/SNR combination





Aggregate DFN2 results

| Metric               | Mean Improvement |

| -------------------- | ---------------: |

| SNR                  |  \*\*+12.4529 dB\*\* |

| STOI                 |      \*\*+0.0679\*\* |

| PESQ                 |      \*\*+0.9238\*\* |

| SI-SDR               |  \*\*+13.7386 dB\*\* |

| Successful inference |      \*\*90 / 90\*\* |

| Offline RTF          |       \*\*\~0.061\*\* |





Noise-class SNR improvement

| Noise Class    |            ΔSNR |

| -------------- | --------------: |

| Stationary     | \*\*+10.0151 dB\*\* |

| Non-stationary | \*\*+12.4021 dB\*\* |

| Impulsive      | \*\*+14.9414 dB\*\* |





The benchmark demonstrates the performance of the current DFN2-based processing pipeline under controlled evaluation conditions.



Important: The measured offline RTF is not treated as proof of a validated 10 ms end-to-end streaming deadline. True streaming latency and embedded timing remain engineering-validation targets.





5\. Technical Architecture



AURALIS uses a full-band speech-enhancement pipeline based on DeepFilterNet2.



Processing stages

1. Audio acquisition

2\. Sample-rate management

3\. Short-Time Fourier Transform (STFT)

4\. ERB-band feature extraction

5\. Neural encoder/recurrent processing

6\. ERB gain estimation

7\. Complex deep-filter coefficient estimation

8\. Complex time-frequency filtering

9\. Inverse STFT (ISTFT)

10\. Enhanced speech output







Primary model configuration



Model: DeepFilterNet2

Audio target: 48 kHz

FFT size: 960

Hop size: 480 samples

ERB bands: 32

Deep-filter bins: 96

Deep-filter order: 5

Model parameters: approximately 2.3 million







6\. Hardware Productization Roadmap



The hardware architecture is being developed in stages.



Stage A — Engineering Prototype



Target compute platform:



Raspberry Pi Compute Module 5

Target variant: 4 GB RAM + 32 GB eMMC



Proposed audio architecture



IM73A135 MEMS Microphones

&#x20;         ↓

TLV320AIC3104 Audio Codec

&#x20;         ↓

&#x20;       I2S

&#x20;         ↓

&#x20;  Raspberry Pi CM5

&#x20;         ↓

&#x20;  AURALIS DFN2 Engine

&#x20;         ↓

&#x20;       I2S

&#x20;         ↓

TLV320AIC3104 DAC / Headphone Path

&#x20;         ↓

&#x20;     Wired Headset







Supporting hardware

TLV320AIC3104 — audio ADC/DAC and audio front-end

Infineon IM73A135 — proposed analog MEMS microphones

ICS-43434 — alternative digital I2S microphone option

BQ24075 / BQ24075-Q1 — proposed 1-cell battery charger/power-path

TPS62840 or equivalent — proposed low-power regulated rail

TAS2563-class amplifier — optional when speaker output is required

USB-C — power, service and development interface

Battery — proposed 1-cell Li-ion/Li-polymer architecture



The component selection and signal chain follow the project's hardware roadmap.



Productization path



Current Laptop Prototype

&#x20;       ↓

CM5 Embedded Proof

&#x20;       ↓

Custom Audio/Power Carrier

&#x20;       ↓

Mechanical Prototype

&#x20;       ↓

Streaming Validation

&#x20;       ↓

Optimization

&#x20;       ↓

Beta Hardware

&#x20;       ↓

Verification

&#x20;       ↓

Pilot

&#x20;       ↓

Production Product





The hardware architecture is currently a proposed engineering roadmap, not a claim that the final custom PCB has already been fabricated.



7\. Innovation / Novelty



AURALIS does not claim to have invented DeepFilterNet2 or the underlying neural speech-enhancement architecture.



The system-level contribution focuses on:



Defence-relevant noise-class evaluation

Stationary, non-stationary and impulsive noise coverage

Transient-aware testing

Reproducible 90-case benchmark

Objective speech-quality evaluation

Comparison with an established baseline

Lightweight AI enhancement suitable for embedded engineering

A staged path from validated software to embedded communication hardware



The novelty is therefore positioned at the system integration, evaluation methodology and deployment pathway, rather than claiming novelty in individual commercial ICs or the DFN2 algorithm itself.







8\. Evidence \& Demonstration



The submission package contains:



Prototype

prototype/AURALIS\_DEMO.py

prototype/README.md



Objective validation

metrics/dfn2\_inference\_results.json

metrics/dfn2\_laptop\_benchmark.csv

metrics/rnnoise\_inference\_results.csv

metrics/benchmark.md

metrics/metrics\_dashboard.png



Screenshots

screenshots/01\_live\_demo.png

screenshots/02\_metrics\_dashboard.png

screenshots/03\_benchmark\_evidence.png

screenshots/04\_github\_repository.png



Hardware

hardware/README.md

hardware/deployment\_roadmap.md



Architecture

architecture/project\_overview.md

architecture/current\_prototype.md

architecture/embedded\_architecture.md




9\. Repository Structure



AURALIS-SIH2026/

│

├── demo/

├── docs/

├── hardware/

├── models/

├── results/

├── src/

│

└── submission/

&#x20;   ├── architecture/

&#x20;   ├── hardware/

&#x20;   ├── metrics/

&#x20;   ├── prototype/

&#x20;   ├── screenshots/

&#x20;   └── README.md







10\. Current Status



Completed



Working laptop-based AURALIS prototype

Live microphone capture

DFN2 speech enhancement

Three noise-class demonstration

Controlled 90-case benchmark

Objective evaluation using SNR, STOI, PESQ and SI-SDR

DFN2 benchmark evidence package

RNNoise benchmark comparison

Metrics dashboard

GitHub repository

Hardware productization roadmap





Engineering in progress



Deterministic chunked streaming

End-to-end latency measurement

Streaming state validation

Embedded CM5 deployment

Audio-interface integration

Power and thermal validation







11\. Limitations



The following claims are intentionally not made:



No claim of 100% noise cancellation

No claim of zero latency

No claim of battlefield validation

No claim of completed custom PCB hardware

No claim of production-ready embedded deployment

No claim that offline RTF alone proves 10 ms real-time operation

No claim that AURALIS is currently a physical acoustic ANC headset



The current implementation is a digital AI speech-enhancement system. Physical acoustic ANC would require additional electro-acoustic components such as a secondary path, anti-noise speaker and error/reference microphone architecture.







12\. Future Development



The next engineering stages are:



Complete deterministic streaming pipeline

Validate capture-to-output latency

Deploy the DFN2 engine on CM5

Integrate external I2S/USB audio hardware

Validate continuous live operation

Measure CPU, memory, thermal and dropout behaviour

Develop Rev-A custom carrier

Integrate battery and power management

Develop compact enclosure

Perform reliability and environmental validation

Optimize model/runtime where required

Progress toward beta and production-intent hardware







13\. Team



Team Auralis



Kunal Raisinghani — Team Leader

Akshara Jain

Vivan Biswas

Devashish Santpal



Smart India Hackathon 2026

Problem Statement: SIH26052



Final Position



AURALIS follows a measurable engineering-first approach:



Validate the AI/signal-processing core → benchmark it across defence-relevant noise classes → demonstrate the working prototype → deploy the validated core on embedded hardware → integrate a compact communication product.



The project deliberately separates validated current functionality from the future hardware roadmap, allowing each development stage to be independently measured and verified.

