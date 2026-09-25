AURALIS — Current Prototype Architecture



1\. Prototype Status



AURALIS currently exists as a working laptop-based AI/ML digital speech-enhancement prototype.



The present prototype performs digital noise suppression on captured speech/audio. It is not a physical acoustic ANC system and does not currently use an anti-noise loudspeaker, secondary-path model, or physical error microphone.



The hardware implementation described separately is a proposed embedded deployment roadmap.



2\. Current Prototype Hardware



Compute Platform



Windows 11 laptop



Intel Core Ultra 7 155U



16 GB RAM



Intel integrated graphics



CPU-based inference



Audio Interface



Airdopes 148 microphone for live input demonstration



Airdopes 148 headphones for enhanced-audio playback



The Bluetooth headset is currently used only as a convenient demonstration audio interface and is not the proposed final product hardware.



3\. Current Signal Flow



LIVE MICROPHONE INPUT

&#x20;       |

&#x20;       v

Audio Capture

&#x20;       |

&#x20;       v

Input Resampling

16 kHz -> 48 kHz

&#x20;       |

&#x20;       v

STFT

&#x20;       |

&#x20;       v

ERB Feature Extraction

&#x20;       |

&#x20;       v

DeepFilterNet2

&#x20;       |

&#x20;       +----------------------+

&#x20;       |                      |

&#x20;       v                      v

&#x20;  ERB Gain             Deep Filtering

&#x20;       |                      |

&#x20;       +----------+-----------+

&#x20;                  |

&#x20;                  v

&#x20;                ISTFT

&#x20;                  |

&#x20;                  v

&#x20;         Enhanced Speech

&#x20;                  |

&#x20;                  v

&#x20;         Audio Playback



4\. AI/ML Processing Pipeline



The current prototype uses DeepFilterNet2 as the primary speech-enhancement model.



Processing stages



Capture microphone audio.



Convert/resample the input to the model operating sample rate.



Perform short-time Fourier transform (STFT).



Extract ERB-band spectral features.



Process the features using the DeepFilterNet2 neural network.



Estimate ERB gains and complex deep-filter coefficients.



Apply the predicted filtering operation in the time-frequency domain.



Reconstruct the enhanced signal using inverse STFT (ISTFT).



Play or save the enhanced speech signal.



5\. Current Model Configuration



The current AURALIS DFN2 configuration uses:



Sample rate: 48 kHz



FFT size: 960



Hop size: 480 samples



Frame duration: 10 ms



ERB bands: 32



Deep-filter frequency bins: 96



Deep-filter order: 5



Convolution lookahead: 2



Deep-filter lookahead: 2



Model parameters: approximately 2.3 million



The architecture combines spectral/ERB processing with recurrent neural-network components and complex deep filtering.



6\. Working Demonstration



The current prototype supports:



Microphone recording



Noise-class selection



Controlled noise mixing



DFN2 enhancement



Enhanced-audio playback



Indicative live metrics



CSV report generation



Offline benchmark execution



The live microphone demonstration is intended to demonstrate the complete input-to-enhancement-to-output workflow.



7\. Controlled Objective Validation



AURALIS has been evaluated using a controlled 90-case benchmark.



Benchmark structure



10 clean speech samples



3 noise classes



3 target SNR conditions



Total: 90 mixtures



Noise classes



Stationary



Non-stationary



Impulsive



Target SNR conditions



+5 dB



0 dB



\-5 dB



Validation result



Successful inference cases: 90/90



Mean ΔSNR: +12.4529 dB



Mean ΔSTOI: +0.0679



Mean ΔPESQ: +0.9238



Mean ΔSI-SDR: +13.7386 dB



Aggregate offline RTF: approximately 0.061



These results represent the controlled offline benchmark and are the primary objective validation evidence for the current prototype.



8\. Noise-Class Validation



Noise Class



ΔSNR



ΔSTOI



ΔPESQ



ΔSI-SDR



Stationary



+10.0151 dB



+0.1062



+0.5653



+9.5152 dB



Non-stationary



+12.4021 dB



+0.0579



+0.8771



+12.1121 dB



Impulsive



+14.9414 dB



+0.0395



+1.3291



+19.5885 dB



Overall



+12.4529 dB



+0.0679



+0.9238



+13.7386 dB



9\. Real-Time Status



The current offline benchmark achieves an aggregate RTF of approximately 0.061.



However, this value does not by itself establish guaranteed end-to-end real-time 10 ms streaming performance.



Streaming integration and incremental frame-deadline validation are still part of the development roadmap.



Therefore, the current prototype should be described as:



AI-based digital speech enhancement with validated offline performance and ongoing streaming/embedded integration.



10\. Current Limitations



The current prototype does not yet represent the final physical product.



Current limitations include:



Laptop-based processing



No custom PCB



No dedicated embedded compute platform



No physical anti-noise speaker



No secondary-path acoustic model



No physical ANC error microphone



True end-to-end 10 ms streaming performance is not yet fully validated



Final enclosure and power system are not yet implemented



11\. Next Development Stage



The next stage is to migrate the validated AURALIS processing pipeline toward an embedded platform.



The proposed first embedded platform is a Raspberry Pi Compute Module 5-based system with a dedicated audio/power carrier.



The embedded architecture is documented separately in:



embedded\_architecture.md



12\. Prototype Position



The current AURALIS prototype establishes the core AI-based digital speech-enhancement pipeline and provides controlled objective evidence across stationary, non-stationary and impulsive noise conditions.



The next engineering objective is to convert this validated software prototype into a bounded-latency embedded audio-processing system.

