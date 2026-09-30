import argparse
import csv
import time
import threading
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf

from df.enhance import enhance, init_df, load_audio


# ============================================================
# AURALIS11 — LIVE MICROPHONE SPEECH ENHANCEMENT DEMO
# ============================================================
#
# LIVE MODE:
#
# Microphone
#     ↓
# User-controlled recording
#     ↓
# Original recorded audio
#     ↓
# Resample to 48 kHz
#     ↓
# DeepFilterNet2
#     ↓
# Enhanced audio
#     ↓
# Playback
#
# IMPORTANT:
# - No artificial noise is mixed into the microphone input.
# - Recording continues until the user presses STOP.
# - There is no fixed recording duration.
# - Live objective metrics are NOT calculated because no clean
#   reference signal is available.
# - Controlled 90-case benchmark remains available separately.
#
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results" / "auralis_demo"

MIC_SR = 16000
MODEL_SR = 48000

CHANNELS = 1

# None = use Windows/system default audio device.
#
# This makes the demo more portable between computers.
# If required, these can be changed to numeric device IDs.
INPUT_DEVICE = None
OUTPUT_DEVICE = None


# ============================================================
# VALIDATED CONTROLLED BENCHMARK EVIDENCE
# ============================================================

BENCHMARK = {
    "stationary": {
        "cases": 30,
        "snr": 10.0151,
        "stoi": 0.1062,
        "pesq": 0.5653,
        "si_sdr": 9.5152,
    },
    "nonstationary": {
        "cases": 30,
        "snr": 12.4021,
        "stoi": 0.0579,
        "pesq": 0.8771,
        "si_sdr": 12.1121,
    },
    "impulsive": {
        "cases": 30,
        "snr": 14.9414,
        "stoi": 0.0395,
        "pesq": 1.3291,
        "si_sdr": 19.5885,
    },
}

AGGREGATE = {
    "cases": 90,
    "snr": 12.4529,
    "stoi": 0.0679,
    "pesq": 0.9238,
    "si_sdr": 13.7386,
    "rtf": 0.0609,
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def peak_normalize(x, peak=0.98):
    """
    Prevent clipping while preserving the original waveform
    as much as possible.
    """
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return x

    max_value = np.max(np.abs(x))

    if max_value > peak:
        x = x * (peak / max_value)

    return x


def resample_audio(x, src_sr, dst_sr):
    """
    Resample mono audio using scipy.signal.resample_poly.
    """

    if src_sr == dst_sr:
        return np.asarray(x, dtype=np.float32)

    from math import gcd
    from scipy.signal import resample_poly

    x = np.asarray(x, dtype=np.float32)

    g = gcd(int(src_sr), int(dst_sr))

    up = int(dst_sr // g)
    down = int(src_sr // g)

    return np.asarray(
        resample_poly(x, up, down),
        dtype=np.float32,
    )


def play_audio(path, label):
    """
    Play a WAV file through the selected/default output device.
    """

    audio, sr = sf.read(
        path,
        dtype="float32",
    )

    if audio.ndim > 1:
        audio = audio[:, 0]

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)

    sd.play(
        audio,
        sr,
        device=OUTPUT_DEVICE,
    )

    sd.wait()

    print("✓ Playback complete")


def save_report(row):
    """
    Save demo session information.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = RESULTS_DIR / "auralis_demo_report.csv"

    write_header = not path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)

    return path


# ============================================================
# LIVE RECORDER
# ============================================================

class LiveRecorder:
    """
    Continuous microphone recorder.

    Recording starts when start() is called and continues
    until stop() is called.

    There is intentionally NO fixed duration.
    """

    def __init__(
        self,
        samplerate=MIC_SR,
        channels=CHANNELS,
        device=INPUT_DEVICE,
    ):

        self.samplerate = samplerate
        self.channels = channels
        self.device = device

        self.stream = None

        self.blocks = []

        self.recording = False

        self.start_time = None

        self.lock = threading.Lock()

    def _callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):

        if status:
            print(f"Audio status: {status}")

        if not self.recording:
            return

        block = np.asarray(
            indata[:, 0],
            dtype=np.float32,
        ).copy()

        with self.lock:
            self.blocks.append(block)

    def start(self):

        if self.recording:
            return

        self.blocks = []

        self.recording = True

        self.start_time = time.perf_counter()

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=self._callback,
            blocksize=0,
        )

        self.stream.start()

    def stop(self):

        if not self.recording:
            return np.array([], dtype=np.float32)

        self.recording = False

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

            self.stream = None

        with self.lock:

            if not self.blocks:
                return np.array([], dtype=np.float32)

            audio = np.concatenate(
                self.blocks
            ).astype(np.float32)

        return audio

    def duration(self):

        if self.start_time is None:
            return 0.0

        if not self.recording:
            return 0.0

        return time.perf_counter() - self.start_time


# ============================================================
# MAIN DASHBOARD
# ============================================================

class LiveDashboard:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "AURALIS11 — Live AI Speech Enhancement"
        )

        self.root.geometry(
            "1000x700"
        )

        self.root.minsize(
            900,
            620,
        )

        self.root.configure(
            bg="#0b1220"
        )

        self.closed = False

        self.processing = False

        self.recorder = LiveRecorder()

        self.recording_thread = None

        self.original_path = None

        self.enhanced_path = None

        self.recorded_audio = None

        self._build()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close,
        )

    # --------------------------------------------------------
    # UI HELPERS
    # --------------------------------------------------------

    def _label(
        self,
        parent,
        text,
        size=12,
        bold=False,
        fg="#dbe7ff",
    ):

        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=fg,
            font=(
                "Segoe UI",
                size,
                "bold" if bold else "normal",
            ),
        )

    # --------------------------------------------------------
    # BUILD UI
    # --------------------------------------------------------

    def _build(self):

        # ==============================
        # HEADER
        # ==============================

        top = tk.Frame(
            self.root,
            bg="#111b2e",
            padx=24,
            pady=20,
        )

        top.pack(
            fill="x"
        )

        tk.Label(
            top,
            text="AURALIS11",
            bg="#111b2e",
            fg="#ffffff",
            font=(
                "Segoe UI",
                27,
                "bold",
            ),
        ).pack(
            side="left"
        )

        tk.Label(
            top,
            text="  LIVE AI SPEECH ENHANCEMENT",
            bg="#111b2e",
            fg="#79a7ff",
            font=(
                "Segoe UI",
                15,
                "bold",
            ),
        ).pack(
            side="left",
            pady=7,
        )

        # ==============================
        # MODEL INFO
        # ==============================

        info = tk.Frame(
            self.root,
            bg="#0b1220",
            padx=24,
            pady=18,
        )

        info.pack(
            fill="x"
        )

        self._label(
            info,
            "INPUT: LIVE MICROPHONE",
            12,
            True,
        ).pack(
            side="left"
        )

        self._label(
            info,
            "MODEL: DeepFilterNet2",
            12,
            True,
        ).pack(
            side="left",
            padx=40,
        )

        self._label(
            info,
            "OUTPUT: ENHANCED SPEECH",
            12,
            True,
        ).pack(
            side="left"
        )

        # ==============================
        # RECORDING PANEL
        # ==============================

        panel = tk.Frame(
            self.root,
            bg="#162238",
            padx=25,
            pady=25,
            highlightthickness=1,
            highlightbackground="#2b3b5a",
        )

        panel.pack(
            fill="x",
            padx=24,
            pady=10,
        )

        tk.Label(
            panel,
            text="LIVE MICROPHONE INPUT",
            bg="#162238",
            fg="#79a7ff",
            font=(
                "Segoe UI",
                13,
                "bold",
            ),
        ).pack()

        self.status_var = tk.StringVar(
            value="READY — press START RECORDING"
        )

        tk.Label(
            panel,
            textvariable=self.status_var,
            bg="#162238",
            fg="#7ee2a8",
            font=(
                "Segoe UI",
                14,
                "bold",
            ),
        ).pack(
            pady=(12, 8)
        )

        self.duration_var = tk.StringVar(
            value="Recording time: 0.0 s"
        )

        tk.Label(
            panel,
            textvariable=self.duration_var,
            bg="#162238",
            fg="#c5d3ec",
            font=(
                "Segoe UI",
                11,
            ),
        ).pack()

        buttons = tk.Frame(
            panel,
            bg="#162238",
        )

        buttons.pack(
            pady=(20, 5)
        )

        self.start_button = tk.Button(
            buttons,
            text="●  START RECORDING",
            command=self.start_recording,
            bg="#16a34a",
            fg="white",
            activebackground="#15803d",
            activeforeground="white",
            font=(
                "Segoe UI",
                12,
                "bold",
            ),
            padx=22,
            pady=12,
            relief="flat",
            cursor="hand2",
        )

        self.start_button.pack(
            side="left",
            padx=8,
        )

        self.stop_button = tk.Button(
            buttons,
            text="■  STOP RECORDING",
            command=self.stop_recording,
            bg="#dc2626",
            fg="white",
            activebackground="#b91c1c",
            activeforeground="white",
            font=(
                "Segoe UI",
                12,
                "bold",
            ),
            padx=22,
            pady=12,
            relief="flat",
            cursor="hand2",
            state="disabled",
        )

        self.stop_button.pack(
            side="left",
            padx=8,
        )

        # ==============================
        # PROCESSING
        # ==============================

        process_frame = tk.Frame(
            self.root,
            bg="#0b1220",
            padx=24,
            pady=14,
        )

        process_frame.pack(
            fill="x"
        )

        self.process_button = tk.Button(
            process_frame,
            text="⚙  ENHANCE WITH AURALIS11",
            command=self.start_processing,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            font=(
                "Segoe UI",
                13,
                "bold",
            ),
            padx=28,
            pady=13,
            relief="flat",
            cursor="hand2",
            state="disabled",
        )

        self.process_button.pack(
            pady=5
        )

        # ==============================
        # AUDIO COMPARISON
        # ==============================

        comparison = tk.Frame(
            self.root,
            bg="#0b1220",
            padx=24,
            pady=10,
        )

        comparison.pack(
            fill="x"
        )

        self.play_original_button = tk.Button(
            comparison,
            text="▶  PLAY ORIGINAL",
            command=self.play_original,
            bg="#334155",
            fg="white",
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
            padx=18,
            pady=10,
            relief="flat",
            cursor="hand2",
            state="disabled",
        )

        self.play_original_button.pack(
            side="left",
            expand=True,
            padx=8,
        )

        self.play_enhanced_button = tk.Button(
            comparison,
            text="▶  PLAY ENHANCED",
            command=self.play_enhanced,
            bg="#2563eb",
            fg="white",
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
            padx=18,
            pady=10,
            relief="flat",
            cursor="hand2",
            state="disabled",
        )

        self.play_enhanced_button.pack(
            side="left",
            expand=True,
            padx=8,
        )

        # ==============================
        # INFORMATION
        # ==============================

        detail = tk.Frame(
            self.root,
            bg="#0b1220",
            padx=24,
            pady=14,
        )

        detail.pack(
            fill="x"
        )

        self._label(
            detail,
            "AURALIS11 PROCESSING FLOW",
            12,
            True,
            "#79a7ff",
        ).pack(
            anchor="w"
        )

        tk.Label(
            detail,
            text=(
                "Live microphone → Original audio capture → "
                "48 kHz processing → DeepFilterNet2 → "
                "Enhanced speech → Playback"
            ),
            bg="#0b1220",
            fg="#aebdd5",
            justify="left",
            font=(
                "Segoe UI",
                10,
            ),
        ).pack(
            anchor="w",
            pady=(7, 0),
        )

        # ==============================
        # FOOTER
        # ==============================

        footer = tk.Frame(
            self.root,
            bg="#111b2e",
            padx=24,
            pady=14,
        )

        footer.pack(
            side="bottom",
            fill="x",
        )

        self.processing_var = tk.StringVar(
            value="Processing time: —"
        )

        tk.Label(
            footer,
            textvariable=self.processing_var,
            bg="#111b2e",
            fg="#ffffff",
            font=(
                "Segoe UI",
                10,
                "bold",
            ),
        ).pack(
            side="left"
        )

        tk.Label(
            footer,
            text=(
                "AURALIS11 • AI-based digital speech enhancement"
            ),
            bg="#111b2e",
            fg="#9fb3d9",
            font=(
                "Segoe UI",
                9,
            ),
        ).pack(
            side="right"
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def set_status(self, text):

        if not self.closed:

            self.root.after(
                0,
                lambda: self.status_var.set(text),
            )

    # --------------------------------------------------------
    # RECORDING TIMER
    # --------------------------------------------------------

    def update_recording_timer(self):

        if self.closed:
            return

        if self.recorder.recording:

            duration = self.recorder.duration()

            self.duration_var.set(
                f"Recording time: {duration:.1f} s"
            )

            self.root.after(
                100,
                self.update_recording_timer,
            )

    # --------------------------------------------------------
    # START RECORDING
    # --------------------------------------------------------

    def start_recording(self):

        if self.recorder.recording:
            return

        try:

            self.original_path = None
            self.enhanced_path = None
            self.recorded_audio = None

            self.play_original_button.configure(
                state="disabled"
            )

            self.play_enhanced_button.configure(
                state="disabled"
            )

            self.process_button.configure(
                state="disabled"
            )

            self.start_button.configure(
                state="disabled"
            )

            self.stop_button.configure(
                state="normal"
            )

            self.set_status(
                "● RECORDING — speak now. Press STOP when finished."
            )

            self.duration_var.set(
                "Recording time: 0.0 s"
            )

            self.recorder.start()

            self.update_recording_timer()

            print()
            print("=" * 72)
            print("AURALIS11 — RECORDING STARTED")
            print("=" * 72)
            print(
                "Speak into the microphone."
            )
            print(
                "There is NO fixed duration."
            )
            print(
                "Press STOP when you finish speaking."
            )
            print("=" * 72)

        except Exception as exc:

            self.start_button.configure(
                state="normal"
            )

            self.stop_button.configure(
                state="disabled"
            )

            self.set_status(
                "❌ Microphone error"
            )

            messagebox.showerror(
                "Microphone Error",
                str(exc),
            )

    # --------------------------------------------------------
    # STOP RECORDING
    # --------------------------------------------------------

    def stop_recording(self):

        if not self.recorder.recording:
            return

        try:

            audio = self.recorder.stop()

            if len(audio) == 0:

                self.set_status(
                    "❌ No audio captured"
                )

                self.start_button.configure(
                    state="normal"
                )

                self.stop_button.configure(
                    state="disabled"
                )

                return

            audio = peak_normalize(
                audio
            )

            self.recorded_audio = audio

            RESULTS_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            timestamp = time.strftime(
                "%Y%m%d_%H%M%S"
            )

            self.original_path = (
                RESULTS_DIR /
                f"original_mic_{timestamp}.wav"
            )

            sf.write(
                self.original_path,
                audio,
                MIC_SR,
            )

            duration = len(audio) / MIC_SR

            self.duration_var.set(
                f"Recording time: {duration:.1f} s"
            )

            self.set_status(
                "✓ Recording complete — ready for enhancement"
            )

            self.start_button.configure(
                state="normal"
            )

            self.stop_button.configure(
                state="disabled"
            )

            self.process_button.configure(
                state="normal"
            )

            self.play_original_button.configure(
                state="normal"
            )

            print()
            print("=" * 72)
            print("AURALIS11 — RECORDING COMPLETE")
            print("=" * 72)
            print(
                f"Duration: {duration:.2f} seconds"
            )
            print(
                f"Saved: {self.original_path}"
            )
            print("=" * 72)

        except Exception as exc:

            self.start_button.configure(
                state="normal"
            )

            self.stop_button.configure(
                state="disabled"
            )

            self.set_status(
                "❌ Recording failed"
            )

            messagebox.showerror(
                "Recording Error",
                str(exc),
            )

    # --------------------------------------------------------
    # START ENHANCEMENT
    # --------------------------------------------------------

    def start_processing(self):

        if self.recorded_audio is None:
            return

        if self.processing:
            return

        self.processing = True

        self.process_button.configure(
            state="disabled"
        )

        self.start_button.configure(
            state="disabled"
        )

        self.play_original_button.configure(
            state="disabled"
        )

        self.play_enhanced_button.configure(
            state="disabled"
        )

        self.set_status(
            "⚙ Loading DeepFilterNet2..."
        )

        self.recording_thread = threading.Thread(
            target=self._process_audio,
            daemon=True,
        )

        self.recording_thread.start()

    # --------------------------------------------------------
    # PROCESS AUDIO
    # --------------------------------------------------------

    def _process_audio(self):

        try:

            RESULTS_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            print()
            print("=" * 72)
            print("AURALIS11 — DEEPFILTERNET2 ENHANCEMENT")
            print("=" * 72)

            print(
                "Loading DeepFilterNet2..."
            )

            model, df_state, _ = init_df(
                "DeepFilterNet2"
            )

            model_sr = df_state.sr()

            print(
                f"Model sample rate: {model_sr} Hz"
            )

            # ---------------------------------------------
            # Resample microphone audio to model rate
            # ---------------------------------------------

            input_48k = resample_audio(
                self.recorded_audio,
                MIC_SR,
                MODEL_SR,
            )

            timestamp = time.strftime(
                "%Y%m%d_%H%M%S"
            )

            model_input_path = (
                RESULTS_DIR /
                f"model_input_{timestamp}.wav"
            )

            sf.write(
                model_input_path,
                input_48k,
                MODEL_SR,
            )

            self.set_status(
                "⚙ AURALIS11 / DFN2 processing..."
            )

            print(
                "Running DeepFilterNet2..."
            )

            start = time.perf_counter()

            model_audio, _ = load_audio(
                str(model_input_path),
                sr=df_state.sr(),
            )

            enhanced_48k = enhance(
                model,
                df_state,
                model_audio,
            )

            processing_time = (
                time.perf_counter() - start
            )

            # ---------------------------------------------
            # Convert enhanced audio back to 16 kHz
            # ---------------------------------------------

            enhanced_16k = resample_audio(
                np.asarray(
                    enhanced_48k,
                    dtype=np.float32,
                ).flatten(),
                MODEL_SR,
                MIC_SR,
            )

            enhanced_16k = peak_normalize(
                enhanced_16k
            )

            self.enhanced_path = (
                RESULTS_DIR /
                f"enhanced_mic_{timestamp}.wav"
            )

            sf.write(
                self.enhanced_path,
                enhanced_16k,
                MIC_SR,
            )

            duration = (
                len(self.recorded_audio) /
                MIC_SR
            )

            rtf = (
                processing_time /
                duration
                if duration > 0
                else np.nan
            )

            self.processing = False

            self.root.after(
                0,
                lambda: self._processing_complete(
                    processing_time,
                    rtf,
                ),
            )

            print()
            print("=" * 72)
            print("AURALIS11 — ENHANCEMENT COMPLETE")
            print("=" * 72)
            print(
                f"Audio duration : {duration:.2f} s"
            )
            print(
                f"Processing time: {processing_time:.3f} s"
            )
            print(
                f"Session RTF    : {rtf:.4f}"
            )
            print(
                f"Enhanced file  : {self.enhanced_path}"
            )
            print("=" * 72)

        except Exception as exc:

            self.processing = False

            self.root.after(
                0,
                lambda: self._processing_failed(
                    exc
                ),
            )

    # --------------------------------------------------------
    # PROCESSING COMPLETE
    # --------------------------------------------------------

    def _processing_complete(
        self,
        processing_time,
        rtf,
    ):

        self.processing_var.set(
            f"Processing time: {processing_time:.3f} s   |   Session RTF: {rtf:.4f}"
        )

        self.set_status(
            "✓ Enhancement complete — compare original and enhanced audio"
        )

        self.start_button.configure(
            state="normal"
        )

        self.process_button.configure(
            state="normal"
        )

        self.play_original_button.configure(
            state="normal"
        )

        self.play_enhanced_button.configure(
            state="normal"
        )

    # --------------------------------------------------------
    # PROCESSING FAILED
    # --------------------------------------------------------

    def _processing_failed(
        self,
        exc,
    ):

        self.set_status(
            "❌ Enhancement failed"
        )

        self.start_button.configure(
            state="normal"
        )

        self.process_button.configure(
            state="normal"
        )

        messagebox.showerror(
            "AURALIS11 Processing Error",
            str(exc),
        )

    # --------------------------------------------------------
    # PLAY ORIGINAL
    # --------------------------------------------------------

    def play_original(self):

        if self.original_path is None:
            return

        try:

            self.set_status(
                "▶ Playing original microphone audio..."
            )

            play_audio(
                self.original_path,
                "ORIGINAL MICROPHONE AUDIO",
            )

            self.set_status(
                "✓ Original playback complete"
            )

        except Exception as exc:

            messagebox.showerror(
                "Playback Error",
                str(exc),
            )

    # --------------------------------------------------------
    # PLAY ENHANCED
    # --------------------------------------------------------

    def play_enhanced(self):

        if self.enhanced_path is None:
            return

        try:

            self.set_status(
                "▶ Playing AURALIS11 enhanced audio..."
            )

            play_audio(
                self.enhanced_path,
                "AURALIS11 ENHANCED AUDIO",
            )

            self.set_status(
                "✓ Enhanced playback complete"
            )

        except Exception as exc:

            messagebox.showerror(
                "Playback Error",
                str(exc),
            )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):

        self.closed = True

        try:

            if self.recorder.recording:
                self.recorder.stop()

        except Exception:
            pass

        try:

            sd.stop()

        except Exception:
            pass

        try:

            self.root.destroy()

        except Exception:
            pass

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(self):

        self.root.mainloop()


# ============================================================
# BENCHMARK DISPLAY
# ============================================================

def print_benchmark():

    print()
    print("=" * 72)
    print(
        "VALIDATED AURALIS11 BENCHMARK EVIDENCE — 90 CASES"
    )
    print("=" * 72)

    print(
        f"Aggregate ΔSNR     : "
        f"{AGGREGATE['snr']:+.2f} dB\n"

        f"Aggregate ΔSTOI    : "
        f"{AGGREGATE['stoi']:+.4f}\n"

        f"Aggregate ΔPESQ    : "
        f"{AGGREGATE['pesq']:+.4f}\n"

        f"Aggregate ΔSI-SDR  : "
        f"{AGGREGATE['si_sdr']:+.2f} dB\n"

        f"Offline RTF        : "
        f"{AGGREGATE['rtf']:.4f}"
    )

    print("-" * 72)

    for name, values in BENCHMARK.items():

        print(
            f"{name:13s} | "
            f"ΔSNR {values['snr']:+6.2f} dB | "
            f"ΔSTOI {values['stoi']:+.4f} | "
            f"ΔPESQ {values['pesq']:+.4f} | "
            f"ΔSI-SDR {values['si_sdr']:+6.2f} dB"
        )

    print("=" * 72)

    print(
        "Source: controlled 90-case AURALIS11/DFN2 benchmark."
    )

    print(
        "These values are separate from the live microphone demo."
    )

    print("=" * 72)


# ============================================================
# DEVICE INFORMATION
# ============================================================

def print_audio_devices():

    print()
    print("=" * 72)
    print("AVAILABLE AUDIO DEVICES")
    print("=" * 72)

    try:

        devices = sd.query_devices()

        for index, device in enumerate(devices):

            print(
                f"[{index}] "
                f"{device['name']} | "
                f"inputs={device['max_input_channels']} | "
                f"outputs={device['max_output_channels']}"
            )

    except Exception as exc:

        print(
            "Could not enumerate audio devices:"
        )

        print(exc)

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AURALIS11 live microphone speech enhancement demo."
        )
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Display validated 90-case benchmark evidence."
        ),
    )

    parser.add_argument(
        "--devices",
        action="store_true",
        help=(
            "List available audio input/output devices."
        ),
    )

    args = parser.parse_args()

    if args.benchmark:

        print_benchmark()

        return

    if args.devices:

        print_audio_devices()

        return

    print()
    print("=" * 72)
    print(
        "        A U R A L I S 1 1"
    )
    print(
        "        SIH 2026 LIVE DEMO"
    )
    print("=" * 72)

    print(
        "AI-based Adaptive Digital Speech Enhancement"
    )

    print(
        "Live microphone → DeepFilterNet2 → Enhanced speech"
    )

    print("=" * 72)

    print()
    print(
        "Audio input/output devices:"
    )

    print(
        "Input device :",
        "System default"
        if INPUT_DEVICE is None
        else INPUT_DEVICE,
    )

    print(
        "Output device:",
        "System default"
        if OUTPUT_DEVICE is None
        else OUTPUT_DEVICE,
    )

    print()
    print(
        "Starting AURALIS11 dashboard..."
    )

    dashboard = LiveDashboard()

    dashboard.run()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
