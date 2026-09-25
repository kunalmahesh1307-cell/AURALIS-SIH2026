AURALIS — Proposed Embedded Architecture



1\. Architecture Status



This document describes the proposed embedded/product architecture for AURALIS.



The architecture is a development roadmap based on the selected components and system strategy. It is not a claim that the final PCB has already been fabricated or fully validated.



The current AURALIS implementation is a laptop-based digital speech-enhancement system. The embedded architecture is the next engineering stage.



2\. Proposed Embedded Platform



Stage A — Engineering Prototype



Primary compute platform:



Raspberry Pi Compute Module 5



Target memory: 4 GB RAM



Target storage: 32 GB eMMC



Custom audio/power carrier board



MEMS microphone interface



Wired headset/audio output



Battery-powered operation as the product direction



Stage B — Product Carrier



After Stage A validation:



Custom production-oriented carrier PCB



Optimized audio and power layout



Rugged enclosure integration



Battery and charging system



Signed software/model updates



Reliability and thermal monitoring



If the CM5 does not meet the final compute, power or thermal targets, an NXP i.MX 8M Plus-class platform may be evaluated as an alternative.



3\. Proposed Primary Audio Architecture



The preferred analog microphone architecture is:



IM73A135 MEMS MICROPHONES

&#x20;       |

&#x20;       v

Analog Input

Filtering / Protection

&#x20;       |

&#x20;       v

TLV320AIC3104

ADC + PGA

&#x20;       |

&#x20;       | I2S

&#x20;       v

Raspberry Pi CM5

&#x20;       |

&#x20;       v

AURALIS DFN2

Streaming Engine

&#x20;       |

&#x20;       | I2S

&#x20;       v

TLV320AIC3104

DAC / Headphone Output

&#x20;       |

&#x20;       v

WIRED HEADSET



The microphone configuration can use a 2-microphone or 4-microphone arrangement depending on the final carrier-board and enclosure design.



4\. Detailed Signal Chain



Input path



IM73A135 MEMS Mic Array

&#x20;       |

&#x20;       v

Analog Input Protection / Filtering

&#x20;       |

&#x20;       v

TLV320AIC3104 ADC + PGA

&#x20;       |

&#x20;       v

I2S Digital Audio

&#x20;       |

&#x20;       v

CM5 Audio Interface



AI processing path



CM5 Audio Buffer

&#x20;       |

&#x20;       v

48 kHz Audio Frames

&#x20;       |

&#x20;       v

AURALIS Streaming Engine

&#x20;       |

&#x20;       v

STFT

&#x20;       |

&#x20;       v

ERB Features

&#x20;       |

&#x20;       v

DeepFilterNet2

&#x20;       |

&#x20;       v

Deep Filtering + Gain Processing

&#x20;       |

&#x20;       v

ISTFT

&#x20;       |

&#x20;       v

Enhanced Audio Frame



Output path



Enhanced Audio

&#x20;       |

&#x20;       v

CM5 Output Buffer

&#x20;       |

&#x20;       | I2S

&#x20;       v

TLV320AIC3104 DAC

&#x20;       |

&#x20;       v

Headphone / Audio Output

&#x20;       |

&#x20;       v

User



5\. Selected Core Components



Function



Proposed Component



Role



Compute



Raspberry Pi Compute Module 5



Embedded AI/audio processing



Audio Codec



TI TLV320AIC3104



ADC, DAC, mic bias/PGA and audio interface



Microphone



Infineon IM73A135



Analog MEMS microphone



Digital Mic Alternative



TDK InvenSense ICS-43434



Direct digital microphone option



Battery Charger / Power Path



TI BQ24075 / BQ24075-Q1



Li-ion/Li-polymer charging and power path



Regulator



TPS62840 or equivalent



Low-power regulated rail



Optional Speaker Amplifier



TAS2563-class device



Speaker output option



Exact component variants and final footprints will be finalized during carrier-board design.



6\. CM5 ↔ Audio Codec Interface



The CM5 and TLV320AIC3104 are intended to communicate through:



Digital audio



I2S BCLK



I2S LRCLK / frame clock



I2S serial data input



I2S serial data output



Control interface



I2C SCL



I2C SDA



GPIO reset/enable control as required



The exact CM5 pin numbers, pull-up values, termination, clocking arrangement and final electrical implementation must be taken from the selected CM5 carrier/reference design and the final codec design.



This roadmap intentionally does not invent fabrication-ready pin assignments.



7\. Alternative Digital Microphone Architecture



An alternative architecture is to use a digital MEMS microphone such as the ICS-43434.



DIGITAL MEMS MIC ARRAY

&#x20;       |

&#x20;       | I2S / Digital Audio

&#x20;       v

CM5 AUDIO INTERFACE

&#x20;       |

&#x20;       v

AURALIS DFN2

&#x20;       |

&#x20;       v

OUTPUT BUFFER

&#x20;       |

&#x20;       v

DAC / USB AUDIO / HEADSET



This option can reduce the analog front-end requirements and may simplify the microphone interface depending on the selected carrier design.



8\. Power Architecture



The proposed portable power architecture is:



USB-C 5V INPUT

&#x20;      |

&#x20;      v

Input Protection

&#x20;      |

&#x20;      v

BQ24075

Charger / Power Path

&#x20;      |

&#x20;      +------------------+

&#x20;      |                  |

&#x20;      v                  v

1-Cell Li-ion       System Power Rail

Battery                  |

&#x20;                         v

&#x20;                Regulated Compute /

&#x20;                Audio Power Rails



The design should provide appropriate separation between compute/power switching noise and sensitive audio circuitry.



A low-noise audio rail using TPS62840 or an equivalent suitable regulator is part of the proposed design direction.



9\. Power Monitoring



The embedded product should monitor:



Battery condition



System current



Temperature



Power faults



Brownout conditions



The software should provide safe handling of power-related faults and prevent corrupted model/firmware updates.



10\. Storage and Software Architecture



The target embedded storage is:



32 GB eMMC



The proposed software architecture contains:



BOOT / SYSTEM

&#x20;    |

&#x20;    v

AUDIO DRIVER LAYER

&#x20;    |

&#x20;    v

AURALIS AUDIO ENGINE

&#x20;    |

&#x20;    +------------------+

&#x20;    |                  |

&#x20;    v                  v

MODEL MANAGER     UPDATE MANAGER

&#x20;    |                  |

&#x20;    v                  v

DFN2 MODEL        SIGNED PACKAGES



A/B model and firmware slots are proposed to support safer updates and rollback.



11\. Target Live Audio Pipeline



The final embedded streaming pipeline is intended to follow:



MIC ARRAY

&#x20;   |

&#x20;   v

ADC / DIGITAL MIC

&#x20;   |

&#x20;   v

DMA BUFFER

&#x20;   |

&#x20;   v

48 kHz AUDIO FRAME

&#x20;   |

&#x20;   v

AURALIS STREAMING ENGINE

&#x20;   |

&#x20;   v

OUTPUT BUFFER

&#x20;   |

&#x20;   v

DAC / HEADSET

&#x20;   |

&#x20;   v

USER



The streaming engine should use bounded buffers and timestamped audio frames.



Fail-safe mute/bypass behavior should be available if the processing pipeline becomes unavailable or misses its required processing boundary.



12\. Embedded Streaming Objective



The engineering target is bounded-latency streaming operation suitable for live audio.



The current offline DFN2 benchmark has an aggregate RTF of approximately 0.061, but this does not establish the final embedded 10 ms frame deadline.



The embedded stage therefore requires direct measurement of:



Frame processing latency



CPU utilization



Memory usage



Buffer occupancy



Audio dropouts



Thermal behavior



Power consumption



End-to-end latency



Optimization will be performed after profiling the actual embedded implementation.



13\. PCB Architecture



Rev A — Engineering Carrier



The initial carrier board is proposed as a 4-layer PCB with separation of:



Compute section



Audio section



Power section



The board should include appropriate:



Decoupling



Audio trace routing



Clock routing



Power filtering



ESD protection



Test points



Debug interfaces



Rev B — Product Carrier



After Rev A validation:



Refined audio layout



Optimized power distribution



Improved EMI/noise control



Production-oriented connectors



Enclosure integration



Manufacturing review



A 4–6 layer PCB may be used depending on final signal-integrity and mechanical requirements.



14\. Proposed Product Architecture



The final product concept is a compact rugged audio-processing unit.



Proposed external interfaces/features include:



Microphone opening or microphone array



USB-C connector



Headset/audio connector



Status LED



Push button



Battery-powered operation



Optional belt/helmet mounting



Exact enclosure dimensions will be finalized after the PCB and battery architecture are frozen.



15\. Reliability and Safety Features



The embedded system should include:



Watchdog monitoring



Thermal monitoring



Brownout handling



Fail-safe audio bypass/mute



Crash-safe logging



Controlled model loading



Signed model/firmware packages



Verified update process



Rollback/recovery mechanism



No hardcoded credentials



These features are part of the proposed productization architecture.



16\. Embedded Development Sequence



CURRENT LAPTOP PROTOTYPE

&#x20;         |

&#x20;         v

STREAMING DFN2 VALIDATION

&#x20;         |

&#x20;         v

CM5 AUDIO INTERFACE

&#x20;         |

&#x20;         v

CM5 + AUDIO/POWER CARRIER

&#x20;         |

&#x20;         v

LIVE EMBEDDED PROCESSING

&#x20;         |

&#x20;         v

POWER + THERMAL VALIDATION

&#x20;         |

&#x20;         v

REV-A PCB

&#x20;         |

&#x20;         v

RUGGED ENCLOSURE

&#x20;         |

&#x20;         v

PRODUCT-LIKE PROTOTYPE

&#x20;         |

&#x20;         v

FINAL OPTIMIZATION



17\. Important Scope Boundary



AURALIS should currently be represented as an AI-based digital speech-enhancement system.



The embedded architecture does not change this scope.



The proposed hardware platform provides the deployment path for the digital enhancement engine. It should not be described as a completed physical acoustic ANC system until an anti-noise speaker, secondary acoustic path and physical error-sensing architecture are actually implemented and validated.



18\. Current vs Proposed System



Aspect



Current AURALIS



Proposed Embedded AURALIS



Compute



Windows laptop



Raspberry Pi CM5



Processing



DFN2



DFN2 streaming engine



Input



Airdopes microphone



MEMS microphone array



Audio Codec



Existing laptop/Bluetooth audio path



TLV320AIC3104



Output



Airdopes headphones



Wired headset / audio output



Power



Laptop



Battery + BQ24075 power path



Storage



Laptop storage



32 GB eMMC target



PCB



None



Custom carrier PCB



Enclosure



None



Rugged product enclosure



Validation



Offline benchmark + live demo



Streaming + embedded validation planned



Physical acoustic ANC



No



No, unless separately engineered and validated



19\. Architecture Position



The proposed embedded architecture converts the existing validated AURALIS software prototype into a practical embedded audio-processing platform.



The immediate engineering focus is:



Streaming validation



CM5 integration



Audio I/O validation



Latency measurement



Power and thermal profiling



Carrier-board development



Product enclosure integration



The architecture is therefore a staged engineering roadmap rather than a claim of an already fabricated or fully validated final product.

