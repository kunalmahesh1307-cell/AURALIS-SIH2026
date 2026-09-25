# Model Documentation

AURALIS currently uses DeepFilterNet2 as the primary speech-enhancement engine.

Key configuration used in the validated benchmark:
- Sample rate: 48 kHz
- FFT size: 960
- Hop: 480 samples (10 ms)
- ERB bands: 32
- DF bins: 96
- DF order: 5
- DF lookahead: 2
- Model parameters: approximately 2.31 M

The repository should document the exact model source/version and its license/attribution before redistribution of model artifacts.
