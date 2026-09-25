from pathlib import Path
import sys
import wave

import numpy as np
import torch

from df.enhance import init_df, enhance


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "mixed"
OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_wav(path):
    """Load PCM WAV using Python's built-in wave module."""

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()

        raw = wf.readframes(frames)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio /= 32768.0

    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        audio /= 2147483648.0

    else:
        raise RuntimeError(
            f"Unsupported WAV sample width: {sample_width} bytes"
        )

    if channels > 1:
        audio = audio.reshape(-1, channels)
        audio = audio.mean(axis=1)

    audio = torch.from_numpy(audio.copy()).unsqueeze(0)

    return audio, sample_rate


def save_wav(path, audio, sample_rate):
    """Save mono float32 tensor as 16-bit PCM WAV."""

    audio = audio.detach().cpu().squeeze().numpy()

    audio = np.clip(audio, -1.0, 1.0)

    pcm = (audio * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main():

    print("=" * 60)
    print("AURALIS — DFN2 Speech Enhancement")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD MODEL
    # ---------------------------------------------------------

    print("\n[1/4] Loading DeepFilterNet2...")

    model, df_state, _ = init_df("DeepFilterNet2")

    model.eval()

    print(f"Sample rate : {df_state.sr()} Hz")
    print(f"FFT size    : {df_state.fft_size()}")
    print(f"Hop size    : {df_state.hop_size()}")

    # ---------------------------------------------------------
    # 2. FIND INPUT
    # ---------------------------------------------------------

    wav_files = sorted(INPUT_DIR.glob("*.wav"))

    if not wav_files:

        print("\nERROR: No WAV file found.")

        print(
            f"\nPut a WAV file inside:\n"
            f"{INPUT_DIR}"
        )

        sys.exit(1)

    input_file = wav_files[0]

    print("\n[2/4] Input:")
    print(input_file)

    # ---------------------------------------------------------
    # LOAD WAV
    # ---------------------------------------------------------

    audio, sr = load_wav(input_file)

    print(f"Original sample rate : {sr} Hz")
    print(f"Channels             : 1")
    print(f"Samples               : {audio.shape[-1]}")
    print(
        f"Duration              : "
        f"{audio.shape[-1] / sr:.2f} seconds"
    )

    # ---------------------------------------------------------
    # RESAMPLE
    # ---------------------------------------------------------

    target_sr = df_state.sr()

    if sr != target_sr:

        print(
            f"\nResampling "
            f"{sr} Hz -> {target_sr} Hz"
        )

        import torch.nn.functional as F

        old_length = audio.shape[-1]

        new_length = int(
            old_length * target_sr / sr
        )

        audio = F.interpolate(
            audio.unsqueeze(1),
            size=new_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        sr = target_sr

    # ---------------------------------------------------------
    # 3. ENHANCE
    # ---------------------------------------------------------

    print("\n[3/4] Running AURALIS / DFN2...")

    with torch.no_grad():

        enhanced = enhance(
            model,
            df_state,
            audio,
        )

    # ---------------------------------------------------------
    # 4. SAVE
    # ---------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        f"{input_file.stem}_auralis.wav"
    )

    save_wav(
        output_file,
        enhanced,
        sr,
    )

    print("\n[4/4] COMPLETE")
    print("-" * 60)

    print(f"Output : {output_file}")

    print(
        f"Output samples : "
        f"{enhanced.shape[-1]}"
    )

    print(
        f"Output duration : "
        f"{enhanced.shape[-1] / sr:.2f} seconds"
    )

    print("-" * 60)

    print(
        "AURALIS enhancement completed successfully."
    )


if __name__ == "__main__":
    main()
