"""Guitar audio DSP pipeline.

Analyzes a recorded acoustic-electric guitar clip (waveform, FFT, spectral
peaks, STFT spectrogram), then pushes it toward a driven electric-style tone
with tanh soft clipping followed by Butterworth EQ, and saves the processed
audio along with before/after comparison plots.

All filtering uses scipy.signal.filtfilt (zero-phase, forward-backward), so
this is an offline, non-causal pipeline rather than a real-time effect.

Usage:
    python signal_processing.py
    python signal_processing.py --input path/to/clip.wav --output-dir results
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks, spectrogram as sp_spectrogram

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "audio" / "source_guitar.wav"
DEFAULT_OUTPUT_AUDIO = REPO_ROOT / "audio" / "processed_guitar.wav"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

DRIVE = 8.0            # tanh soft-clipping drive
HEADROOM = 0.95        # final gain so the exported peak sits below full scale
FFT_VIEW_MAX = 800     # Hz; range where the fundamental chord tones live
PHASE_FLOOR_DB = -30   # only show phase for bins within 30 dB of the peak

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# =============================================================================
# INPUT AND PREPROCESSING
# =============================================================================

def load_audio(path):
    """Load a WAV file as a mono float64 signal scaled to [-1, 1]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input WAV not found: {path}")

    fs, raw = wavfile.read(path)

    # Scale to floating point based on the stored sample format
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        if info.min == 0:                      # unsigned PCM (8-bit WAV)
            mid = (info.max + 1) / 2
            x = (raw.astype(np.float64) - mid) / mid
        else:                                  # signed PCM (16/24/32-bit)
            x = raw.astype(np.float64) / info.max
    else:                                      # already floating point
        x = raw.astype(np.float64)

    # Mix stereo (or multichannel) down to mono by averaging channels
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Float WAVs are not guaranteed to stay within [-1, 1]
    peak = np.max(np.abs(x))
    if peak > 1.0:
        x = x / peak

    return fs, x


def peak_normalize(x):
    """Scale a signal so its largest absolute sample is 1."""
    peak = np.max(np.abs(x))
    return x / peak if peak > 0 else x


# =============================================================================
# SPECTRAL ANALYSIS
# =============================================================================

def compute_spectrum(x, fs):
    """One-sided FFT magnitude (amplitude-scaled) and phase of a real signal."""
    N = len(x)
    X = rfft(x)
    freqs = rfftfreq(N, d=1 / fs)

    mag = np.abs(X) / N
    # Double every bin that has a negative-frequency twin. DC never does, and
    # neither does the Nyquist bin, which only exists when N is even.
    if N % 2 == 0:
        mag[1:-1] *= 2
    else:
        mag[1:] *= 2

    phase = np.angle(X)
    return freqs, mag, phase


def find_dominant_peaks(freqs, mag, f_max=FFT_VIEW_MAX, n_peaks=6):
    """Return (frequency, magnitude) of the strongest spectral peaks below f_max."""
    band = freqs < f_max
    band_mag = mag[band]
    peaks, _ = find_peaks(band_mag, height=band_mag.max() * 0.1, distance=50)
    strongest = peaks[np.argsort(band_mag[peaks])[::-1][:n_peaks]]
    return [(freqs[band][p], band_mag[p]) for p in strongest]


def plot_waveform(t, x, path):
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(t, x, color="steelblue", linewidth=0.6, label="Source signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Time-Domain Waveform of the Source Recording")
    ax.set_xlim([0, t[-1]])
    ax.legend()
    _save(fig, path)
    return fig


def plot_spectrum(freqs, mag, phase, peaks, path):
    """Magnitude spectrum with labelled peaks, plus phase for significant bins."""
    view = freqs < FFT_VIEW_MAX
    view_max = mag[view].max()

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))

    axes[0].plot(freqs, mag, color="darkorange", linewidth=0.8, label="FFT magnitude")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Magnitude Spectrum (FFT) of the Source Recording")
    axes[0].set_xlim([0, FFT_VIEW_MAX])
    label_levels = np.linspace(0.96, 0.58, len(peaks))
    for (peak_f, peak_m), level in zip(sorted(peaks), label_levels):
        axes[0].axvline(peak_f, color="red", linestyle="--", linewidth=1, alpha=0.7)
        axes[0].annotate(
            f"{peak_f:.0f} Hz",
            xy=(peak_f, peak_m),
            xytext=(peak_f + 14, view_max * level),
            color="red",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "red", "linewidth": 0.7, "alpha": 0.7},
        )
    axes[0].legend(loc="upper right")

    # Phase of near-zero-magnitude bins is numerically meaningless noise, so
    # only bins within PHASE_FLOOR_DB of the in-view peak are drawn.
    significant = view & (mag >= view_max * 10 ** (PHASE_FLOOR_DB / 20))
    axes[1].scatter(freqs[significant], phase[significant], s=6, color="mediumpurple")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (radians)")
    axes[1].set_title(f"Phase Spectrum (FFT), bins within {-PHASE_FLOOR_DB} dB of the peak magnitude")
    axes[1].set_xlim([0, FFT_VIEW_MAX])
    axes[1].set_ylim([-np.pi * 1.05, np.pi * 1.05])
    axes[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axes[1].set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

    _save(fig, path)
    return fig


# =============================================================================
# TIME-FREQUENCY ANALYSIS
# =============================================================================

def _draw_spectrogram(ax, x, fs, title, cmap="inferno", f_max=5000):
    f_s, t_s, S = sp_spectrogram(x, fs=fs, nperseg=2048, noverlap=1536, window="hann")
    pcm = ax.pcolormesh(t_s, f_s, 10 * np.log10(S + 1e-12),
                        shading="gouraud", cmap=cmap, vmin=-80, vmax=0)
    plt.colorbar(pcm, ax=ax, label="Power (dB)")
    ax.set_ylim([0, f_max])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)


def plot_spectrogram(x, fs, path):
    fig, ax = plt.subplots(figsize=(13, 4))
    _draw_spectrogram(ax, x, fs, "Spectrogram (STFT) of the Source Recording")
    _save(fig, path)
    return fig


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

def soft_clip(x, drive=DRIVE):
    """Tanh soft clipper: smooth nonlinear saturation for an overdrive-style effect.

    y[n] = tanh(drive * x[n]) / tanh(drive). Unlike the LTI filters below, this
    nonlinearity creates new harmonic and intermodulation components.
    """
    return np.tanh(drive * x) / np.tanh(drive)


def apply_guitar_eq(x, fs):
    """Frequency-selective Butterworth EQ toward an electric-guitar-style tone.

    Each stage uses filtfilt (forward-backward), which gives zero phase
    distortion but needs the whole signal up front, so it is offline only.
    """
    nyq = 0.5 * fs

    # High-pass at 100 Hz: reduce low-frequency acoustic body resonance
    b, a = butter(4, 100 / nyq, btype="high")
    x = filtfilt(b, a, x)

    # Low-pass at 6 kHz: roll off high-frequency acoustic "air"
    b, a = butter(4, 6000 / nyq, btype="low")
    x = filtfilt(b, a, x)

    # Presence boost: add the 2-4 kHz band-passed component back in. Near the
    # band center this roughly doubles the in-band amplitude, about +6 dB.
    b, a = butter(2, [2000 / nyq, 4000 / nyq], btype="band")
    presence = filtfilt(b, a, x)
    x = x + presence

    return peak_normalize(x)


def process_signal(x, fs, drive=DRIVE):
    """Full effects chain: soft clip -> EQ -> headroom."""
    clipped = peak_normalize(soft_clip(x, drive))
    return apply_guitar_eq(clipped, fs) * HEADROOM


# =============================================================================
# RESULT COMPARISON
# =============================================================================

def generate_comparisons(t, original, processed, fs, results_dir):
    """Save waveform, spectrum, and spectrogram before/after comparison plots."""
    results_dir = Path(results_dir)
    colors = ["steelblue", "crimson"]
    labels = ["Source", "Processed (soft clip + EQ)"]

    # Waveform
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for ax, sig, label, color in zip(axes, [original, processed], labels, colors):
        ax.plot(t, sig, color=color, linewidth=0.6, label=label)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
    axes[0].set_title("Time-Domain Comparison: Source vs. Processed")
    axes[1].set_xlabel("Time (s)")
    _save(fig, results_dir / "waveform_comparison.png")

    # Spectrum: chord-tone detail (left) and full EQ range (right)
    freqs, mag_orig, _ = compute_spectrum(original, fs)
    _, mag_proc, _ = compute_spectrum(processed, fs)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (m, label, color) in enumerate(zip([mag_orig, mag_proc], labels, colors)):
        for col, f_max in enumerate([FFT_VIEW_MAX, 8000]):
            ax = axes[row][col]
            ax.plot(freqs, m, color=color, linewidth=0.8, label=label)
            ax.set_xlim([0, f_max])
            ax.set_ylabel("Magnitude")
            if col == 1:
                ax.axvspan(2000, 4000, alpha=0.08, color="green")
                ax.axvline(100, color="gray", linestyle=":", linewidth=1, alpha=0.7)
                ax.axvline(6000, color="gray", linestyle=":", linewidth=1, alpha=0.7)
            ax.legend(loc="upper right")
    axes[0][0].set_title(f"Spectrum Comparison: 0-{FFT_VIEW_MAX} Hz (chord tones)")
    axes[0][1].set_title("Spectrum Comparison: 0-8000 Hz (EQ range, presence band shaded)")
    axes[1][0].set_xlabel("Frequency (Hz)")
    axes[1][1].set_xlabel("Frequency (Hz)")
    _save(fig, results_dir / "spectrum_comparison.png")

    # Spectrogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    _draw_spectrogram(axes[0], original, fs, "Spectrogram: Source", cmap="inferno")
    _draw_spectrogram(axes[1], processed, fs, "Spectrogram: Processed (soft clip + EQ)", cmap="magma")
    _save(fig, results_dir / "spectrogram_comparison.png")


# =============================================================================
# OUTPUT
# =============================================================================

def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


def save_audio(path, x, fs):
    """Write a float signal in [-1, 1] as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(x, -1.0, 1.0) * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(path, fs, pcm)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze and process a guitar WAV recording.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="source WAV file (default: audio/source_guitar.wav)")
    parser.add_argument("--output-audio", type=Path, default=DEFAULT_OUTPUT_AUDIO,
                        help="processed WAV file (default: audio/processed_guitar.wav)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="directory for result plots (default: results/)")
    args = parser.parse_args(argv)

    try:
        fs, x = load_audio(args.input)
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}")

    duration = len(x) / fs
    t = np.arange(len(x)) / fs
    print(f"Input         : {args.input}")
    print(f"Sampling rate : {fs} Hz")
    print(f"Duration      : {duration:.2f} s")
    print(f"Total samples : {len(x):,}")

    freqs, mag, phase = compute_spectrum(x, fs)
    peaks = find_dominant_peaks(freqs, mag)
    print(f"\nDominant spectral peaks below {FFT_VIEW_MAX} Hz:")
    for f, m in peaks:
        print(f"  {f:7.1f} Hz   magnitude {m:.5f}")

    out = args.output_dir
    for fig in (
        plot_waveform(t, x, out / "waveform.png"),
        plot_spectrum(freqs, mag, phase, peaks, out / "fft.png"),
        plot_spectrogram(x, fs, out / "spectrogram.png"),
    ):
        plt.close(fig)

    processed = process_signal(x, fs)
    save_audio(args.output_audio, processed, fs)
    print(f"\nProcessing    : tanh soft clip (drive {DRIVE}), HPF 100 Hz, LPF 6 kHz, "
          f"2-4 kHz presence boost, gain {HEADROOM}")

    generate_comparisons(t, x, processed, fs, out)
    plt.close("all")

    print("\nOutputs:")
    for path in [args.output_audio] + sorted(out.glob("*.png")):
        print(f"  {path}  ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
