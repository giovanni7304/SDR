
#!/usr/bin/env python3
"""
sdr_ptt_pipeline.py
-------------------
End-to-end SDR development pipeline in a single file for simulating PTT (push-to-talk)
signals and processing them: detect → analyze → demodulate → extract artifacts (WAV/JSON).

Usage examples:
    python sdr_ptt_pipeline.py --mode nbfm --snr 15 --fs 48000 --seconds 6 --outdir artifacts/session_0001
    python sdr_ptt_pipeline.py --mode 2fsk --snr 12 --fs 48000 --seconds 6 --Rs 2400 --df 1200 --outdir artifacts/session_0002

Dependencies:
    - numpy, scipy
    - soundfile (optional, else SciPy fallback for WAV writing)
    - matplotlib (optional; only for spectrogram PNG)
"""

import os
import json
import math
import argparse
import tempfile
import shutil
from typing import List, Tuple, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, lfilter, decimate, welch, spectrogram

# ------------------------------
# I/O utilities
# ------------------------------
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def write_json(data: Dict, path: str, pretty: bool = True) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmpjson_", dir=os.path.dirname(path) or None)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, indent=2, sort_keys=True)
            else:
                json.dump(data, f, separators=(",", ":"))
        shutil.move(tmp_path, path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass

def write_wav_pcm16(audio: NDArray[np.floating], fs: int, path: str) -> None:
    """
    Preferred WAV writer using soundfile (if installed). Falls back to SciPy if not.
    """
    ensure_dir(os.path.dirname(path))
    a = np.asarray(audio)
    if a.ndim == 1:
        a = a[:, None]  # mono -> (N,1)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(a))) if a.size else 1.0
    if peak > 1.0:
        a = a / peak
    try:
        import soundfile as sf
        sf.write(path, a, samplerate=fs, subtype="PCM_16")
    except Exception:
        from scipy.io import wavfile
        # SciPy expects (N,C) float in [-1,1] scaled to int16 manually
        a16 = np.clip(a * 32767.0, -32768, 32767).astype(np.int16, copy=False)
        wavfile.write(path, fs, a16)

def write_iq_npy(x: NDArray[np.complexfloating], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, x.astype(np.complex64), allow_pickle=False)

# ------------------------------
# Simulation blocks
# ------------------------------
def fm_mod(audio: NDArray[np.floating], fs: int, fdev: float) -> NDArray[np.complexfloating]:
    """Narrowband FM modulator: audio in [-1,1] -> complex baseband (unit magnitude)."""
    k = (2 * np.pi * fdev) / fs
    phase = np.cumsum(k * audio.astype(np.float64))
    return np.exp(1j * phase).astype(np.complex64)

def fsk2_mod(bits: NDArray[np.integer], fs: int, Rs: float, df: float) -> NDArray[np.complexfloating]:
    """
    Noncoherent 2-FSK modulation (rectangular pulse), tones at ±df (baseband).
    bits: array of 0/1 (any integer dtype). Returns complex64.
    """
    b = (np.asarray(bits).astype(np.uint8).reshape(-1) & 1).astype(np.int8)
    sps = int(round(fs / Rs))
    if sps <= 0:
        raise ValueError("Samples per symbol (sps) must be >= 1; check fs and Rs.")
    N = len(b) * sps
    n = np.arange(N, dtype=np.float64)
    # Repeat bits to symbol rate
    symbols = np.repeat(b * 2 - 1, sps)  # +1 for bit1, -1 for bit0
    phase = 2 * np.pi * df * n / fs * symbols
    x = np.exp(1j * phase)
    return x.astype(np.complex64)

def make_ptt_bursts(x: NDArray[np.complexfloating], fs: int, on_s: float, off_s: float) -> NDArray[np.complexfloating]:
    """
    Gate the signal with on/off durations across its length.
    Keeps original samples during 'on', zeros during 'off'.
    """
    onN = max(1, int(round(on_s * fs)))
    offN = max(0, int(round(off_s * fs)))
    out = np.zeros_like(x)
    i = 0
    while i < len(x):
        ii = slice(i, min(i + onN, len(x)))
        out[ii] = x[ii]
        i += onN + offN
    return out

def add_channel_impairments(x: NDArray[np.complexfloating],
                            fs: int,
                            snr_db: float,
                            cfo_hz: float = 0.0,
                            timing_ppm: float = 0.0,
                            iq_imbalance: float = 0.0,
                            dc_offset: float = 0.0) -> NDArray[np.complexfloating]:
    """
    Apply CFO, sample-rate mismatch (ppm), IQ amplitude imbalance, DC offset, and AWGN.
    """
    x = np.asarray(x).astype(np.complex64, copy=False)
    N = len(x)
    # CFO
    if abs(cfo_hz) > 0:
        n = np.arange(N, dtype=np.float64)
        x = (x * np.exp(1j * 2 * np.pi * cfo_hz * n / fs)).astype(np.complex64, copy=False)
    # Timing ppm: effective sampling period scaled by alpha -> resample at t2 = n/(fs*alpha)
    alpha = 1.0 + timing_ppm * 1e-6
    if abs(alpha - 1.0) > 1e-12:
        t = np.arange(N, dtype=np.float64) / fs
        t2 = np.arange(N, dtype=np.float64) / (fs * alpha)
        real = np.interp(t2, t, x.real, left=0.0, right=0.0)
        imag = np.interp(t2, t, x.imag, left=0.0, right=0.0)
        x = (real + 1j * imag).astype(np.complex64, copy=False)
    # IQ amplitude imbalance (simple amplitude skew)
    if abs(iq_imbalance) > 0:
        x = x.real * (1.0 + iq_imbalance) + 1j * x.imag * (1.0 - iq_imbalance)
        x = x.astype(np.complex64, copy=False)
    # DC offset
    if abs(dc_offset) > 0:
        x = (x + dc_offset).astype(np.complex64, copy=False)
    # AWGN to achieve target SNR (power-based)
    p_sig = float(np.mean(np.abs(x) ** 2)) if x.size else 1.0
    if p_sig == 0.0:
        noise = (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)) / np.sqrt(2.0)
        return (noise * 1e-3).astype(np.complex64)  # tiny noise if zero signal
    n0 = p_sig / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(n0 / 2.0) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    return (x + noise.astype(np.complex64)).astype(np.complex64)

# ------------------------------
# Detection and estimators
# ------------------------------
def detect_bursts(power: NDArray[np.floating], thr_lin: float, min_len: int) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, stop_idx) inclusive ranges where power > thr_lin with min_len samples.
    """
    above = power > thr_lin
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.view(np.int8), [0]])) != 0)  # rising/falling
    starts = edges[0::2]
    stops  = edges[1::2] - 1
    segments = [(int(s), int(e)) for s, e in zip(starts, stops) if (e - s + 1) >= min_len]
    return segments

def occupied_bandwidth_welch(x: NDArray[np.complexfloating], fs: int, pct: float = 0.99) -> Tuple[float, float, float]:
    """
    Welch PSD → cumulative power → return (OBW_Hz, f_lo, f_hi) covering 'pct' of total power.
    """
    nperseg = min(1024, len(x))
    if nperseg < 32:
        return 0.0, -0.0, 0.0
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, return_onesided=False, scaling="spectrum")
    Pxx = Pxx / max(np.sum(Pxx), 1e-20)
    cdf = np.cumsum(Pxx)
    lo_idx = np.searchsorted(cdf, (1 - pct) / 2.0, side="left")
    hi_idx = np.searchsorted(cdf, 1 - (1 - pct) / 2.0, side="left")
    f_lo = float(f[lo_idx])
    f_hi = float(f[min(hi_idx, len(f)-1)])
    return float(f_hi - f_lo), f_lo, f_hi

def estimate_snr_psd(x: NDArray[np.complexfloating], fs: int, band: Tuple[float, float]) -> float:
    """
    Estimate SNR using PSD integration: in-band vs out-of-band normalized power.
    """
    nperseg = min(1024, len(x))
    if nperseg < 32:
        return float("nan")
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, return_onesided=False, scaling="spectrum")
    in_mask = (f >= band[0]) & (f <= band[1])
    Pin = float(np.sum(Pxx[in_mask]))
    Pout = float(np.sum(Pxx[~in_mask]))
    # normalize Pout to same number of bins
    nb_in = int(np.sum(in_mask))
    nb_out = max(1, len(f) - nb_in)
    Pout_norm = Pout * (nb_in / nb_out)
    if Pout_norm <= 0:
        return float("nan")
    return 10.0 * math.log10(max(Pin, 1e-20) / Pout_norm)

def kay_cfo_estimator(x: NDArray[np.complexfloating], fs: int) -> float:
    """
    Kay's frequency estimator (small CFO) on complex baseband.
    """
    if len(x) < 2:
        return 0.0
    phi = np.angle(x[1:] * np.conj(x[:-1]))
    return fs / (2.0 * np.pi) * float(np.mean(phi))

# ------------------------------
# Demodulators
# ------------------------------
def fm_discriminator(x: NDArray[np.complexfloating]) -> NDArray[np.floating]:
    """Angle difference discriminator (rad/sample)."""
    if len(x) < 2:
        return np.zeros_like(x, dtype=np.float32)
    y = np.angle(x[1:] * np.conj(x[:-1]))
    y = np.concatenate([y[:1], y])  # pad to length
    return y.astype(np.float32)

def deemphasis_iir(x: NDArray[np.floating], fs: int, tau: float = 75e-6) -> NDArray[np.floating]:
    """Single-pole IIR de-emphasis (North America default tau=75 us)."""
    a = math.exp(-1.0 / (fs * tau))
    b = np.array([1 - a], dtype=np.float64)
    a_vec = np.array([1.0, -a], dtype=np.float64)
    y = lfilter(b, a_vec, x.astype(np.float64))
    return y.astype(np.float32)

def demod_nbfm_to_audio(xb: NDArray[np.complexfloating], fs: int, audio_fs: int = 8000) -> Tuple[NDArray[np.floating], int]:
    """
    FM demod chain: discriminator -> deemphasis -> audio LPF ~3.4kHz -> decimate to audio_fs.
    """
    freq = fm_discriminator(xb)
    freq = deemphasis_iir(freq, fs, tau=75e-6)
    # Low-pass filter then decimate
    fp = 3400.0
    b, a = butter(5, fp / (fs / 2.0))
    y = lfilter(b, a, freq)
    dec = max(1, int(round(fs / audio_fs)))
    y_ds = decimate(y, dec, ftype='iir', zero_phase=True)
    fs_out = fs // dec
    return y_ds.astype(np.float32), fs_out

def demod_2fsk_noncoherent(xb: NDArray[np.complexfloating], fs: int, Rs: float, df: float) -> NDArray[np.uint8]:
    """
    MVP noncoherent 2-FSK demod (assumes near-integer sps, no timing recovery).
    Returns a vector of 0/1 uint8 bits.
    """
    sps = int(round(fs / Rs))
    if sps <= 0:
        raise ValueError("sps must be >=1")
    N = len(xb)
    n = np.arange(N, dtype=np.float64)
    tone_p = np.exp(1j * 2 * np.pi * df * n / fs)
    tone_n = np.exp(-1j * 2 * np.pi * df * n / fs)
    # Integrate (boxcar) over each symbol
    h = np.ones(sps, dtype=np.float64)
    yp = np.convolve(xb * np.conj(tone_p), h, mode="valid")
    yn = np.convolve(xb * np.conj(tone_n), h, mode="valid")
    # Sample once per symbol
    yp_s = yp[::sps]
    yn_s = yn[::sps]
    bits = (np.abs(yp_s) ** 2 > np.abs(yn_s) ** 2).astype(np.uint8)
    return bits

# ------------------------------
# Helpers
# ------------------------------
def bits_to_hex(bits: NDArray[np.integer]) -> str:
    """Pack 0/1 bits into bytes (MSB-first) and return uppercase hex string."""
    b = (np.asarray(bits).astype(np.uint8).reshape(-1) & 1)
    pad = (-len(b)) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    # Manual weights (big-endian MSB first)
    B = b.reshape(-1, 8)
    weights = (1 << np.arange(7, -1, -1, dtype=np.uint8))  # [128,64,...,1]
    bytes_arr = (B * weights).sum(axis=1).astype(np.uint8, copy=False)
    return "".join(f"{x:02X}" for x in bytes_arr.tolist())

def save_spectrogram_png(x: NDArray[np.complexfloating], fs: int, path: str) -> None:
    """Save a simple spectrogram PNG (optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return  # plotting is optional; skip if not available
    nperseg = 512
    noverlap = int(nperseg * 0.75)
    f, t, Sxx = spectrogram(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, mode="magnitude", scaling="spectrum")
    plt.figure()
    plt.pcolormesh(t, f/1e3, 20*np.log10(Sxx + 1e-12), shading="auto")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [kHz]")
    plt.colorbar(label="Magnitude [dB]")
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ------------------------------
# Main pipeline
# ------------------------------
def run_offline_ptt(mode: str = "nbfm",
                    fs: int = 48000,
                    seconds: float = 6.0,
                    snr_db: float = 15.0,
                    fdev: float = 2500.0,
                    Rs: float = 2400.0,
                    df: float = 1200.0,
                    ptt_on_s: float = 0.8,
                    ptt_off_s: float = 0.4,
                    preemph_tau: float = 75e-6,
                    outdir: str = "artifacts/session_0001") -> None:
    """
    Simulate a PTT session and process bursts. Saves WAV/JSON (and PNG spectrogram) to outdir.
    """
    ensure_dir(outdir)
    N = int(fs * seconds)
    t = np.arange(N, dtype=np.float64) / fs

    # --- Simulate baseband ---
    mode_l = mode.strip().lower()
    if mode_l == "nbfm":
        audio = 0.4 * np.sin(2 * np.pi * 600.0 * t).astype(np.float32)  # placeholder "voice"
        x = fm_mod(audio, fs, fdev)
    elif mode_l == "2fsk":
        payload_bits = np.random.randint(0, 2, size=(256,), dtype=np.uint8)
        preamble = (np.arange(128) % 2).astype(np.uint8)          # 1010...
        sync = np.array([1,1,1,0,0,1,0,0,1,0,1,0], dtype=np.uint8)
        bits = np.concatenate([preamble, sync, payload_bits]).astype(np.uint8)
        x = fsk2_mod(bits, fs, Rs, df)
        if len(x) < N:
            x = np.pad(x, (0, N - len(x)), mode="constant", constant_values=0)
        else:
            x = x[:N]
    else:
        raise ValueError("mode must be 'nbfm' or '2fsk'")

    # PTT gating
    x = make_ptt_bursts(x, fs, ptt_on_s, ptt_off_s)

    # Channel impairments
    x = add_channel_impairments(x, fs, snr_db, cfo_hz=60.0, timing_ppm=3.0, iq_imbalance=0.02, dc_offset=0.01)

    # Save IQ for reference
    write_iq_npy(x, os.path.join(outdir, "iq.npy"))

    # --- Detection ---
    win = max(1, int(0.010 * fs))  # 10 ms
    p = np.convolve(np.abs(x) ** 2, np.ones(win) / win, mode="same")

    # Robust threshold from lower 30% (noise-ish)
    ps = np.sort(p)
    nb = max(32, int(0.30 * len(ps)))
    noise_med = float(np.median(ps[:nb]))
    noise_mad = float(np.median(np.abs(ps[:nb] - noise_med))) + 1e-12
    thr = noise_med + 8.0 * noise_mad
    min_len = int(0.15 * fs)

    segments = detect_bursts(p, thr, min_len)
    if not segments:
        # fallback threshold
        mu = float(np.mean(p))
        sig = float(np.std(p))
        thr = mu + 2.0 * sig
        segments = detect_bursts(p, thr, min_len)

    # Optional spectrogram
    save_spectrogram_png(x, fs, os.path.join(outdir, "spectrogram.png"))

    # --- Process each burst ---
    for k, (s, e) in enumerate(segments, start=1):
        xb = x[s:e+1]
        # CFO estimate & removal
        fCFO = kay_cfo_estimator(xb, fs)
        n = np.arange(len(xb), dtype=np.float64)
        xb = (xb * np.exp(-1j * 2 * np.pi * fCFO * n / fs)).astype(np.complex64, copy=False)

        # Characterization
        obw, f_lo, f_hi = occupied_bandwidth_welch(xb, fs, pct=0.99)
        snr_burst = estimate_snr_psd(xb, fs, (f_lo, f_hi))

        # Summary record
        summary = {
            "file": "simulated",
            "burst_id": k,
            "t_start_s": s / fs,
            "t_stop_s": e / fs,
            "sample_rate_hz": fs,
            "cfo_hz": fCFO,
            "occupied_bw_hz": obw,
            "snr_db": snr_burst,
            "modulation": mode_l.upper(),
            "symbol_rate_baud": None,
            "audio_wav_path": "",
            "bitstream_hex": "",
            "crc_ok": None
        }

        if mode_l == "nbfm":
            audio, fs_audio = demod_nbfm_to_audio(xb, fs, audio_fs=8000)
            wav_path = os.path.join(outdir, f"burst_{k:04d}_audio.wav")
            write_wav_pcm16(audio, fs_audio, wav_path)
            summary["audio_wav_path"] = wav_path
        else:
            bits_hat = demod_2fsk_noncoherent(xb, fs, Rs, df)
            summary["symbol_rate_baud"] = int(Rs)
            summary["bitstream_hex"] = bits_to_hex(bits_hat)

        json_path = os.path.join(outdir, f"burst_{k:04d}.json")
        write_json(summary, json_path)

    # If no bursts, still drop a session-level note
    if not segments:
        write_json({
            "note": "No bursts detected; consider lowering threshold or increasing SNR.",
            "suggested_thr": thr,
            "fs": fs,
            "seconds": seconds,
            "mode": mode_l
        }, os.path.join(outdir, "session_summary.json"))

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Simulate and process PTT SDR signals (NBFM / 2-FSK).")
    ap.add_argument("--mode", default="nbfm", choices=["nbfm", "2fsk"], help="Signal mode to simulate")
    ap.add_argument("--fs", type=int, default=48000, help="Sample rate [Hz]")
    ap.add_argument("--seconds", type=float, default=6.0, help="Total duration [s]")
    ap.add_argument("--snr", type=float, default=15.0, help="SNR in dB")
    ap.add_argument("--fdev", type=float, default=2500.0, help="NBFM frequency deviation [Hz]")
    ap.add_argument("--Rs", type=float, default=2400.0, help="2-FSK symbol rate [baud]")
    ap.add_argument("--df", type=float, default=1200.0, help="2-FSK tone separation [Hz]")
    ap.add_argument("--on", type=float, default=0.8, help="PTT on-time per cycle [s]")
    ap.add_argument("--off", type=float, default=0.4, help="PTT off-time per cycle [s]")
    ap.add_argument("--outdir", type=str, default="artifacts/session_0001", help="Output directory for artifacts")
    args = ap.parse_args()

    run_offline_ptt(mode=args.mode,
                    fs=args.fs,
                    seconds=args.seconds,
                    snr_db=args.snr,
                    fdev=args.fdev,
                    Rs=args.Rs,
                    df=args.df,
                    ptt_on_s=args.on,
                    ptt_off_s=args.off,
                    outdir=args.outdir)

if __name__ == "__main__":
    main()
