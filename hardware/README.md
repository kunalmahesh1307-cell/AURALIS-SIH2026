# AURALIS â€” Hardware Architecture

## Current validated prototype
The current AURALIS prototype is a laptop-based digital speech-enhancement system:
Microphone â†’ audio capture â†’ DeepFilterNet2 enhancement â†’ headphone playback.

## Proposed embedded architecture
MEMS microphone â†’ audio codec / I2S â†’ embedded compute platform â†’ AURALIS DFN2 â†’ codec/DAC â†’ headset or radio.

### Proposed engineering platform
Raspberry Pi Compute Module 5 (CM5) is the first embedded engineering target.

### Important status
The embedded hardware architecture is a proposed deployment roadmap. It must not be represented as a completed physical prototype until the hardware is actually assembled and validated.
