"""
sdr_adsb_pipeline.py
--------------------
End-to-end ADS-B (Mode S extended squitter) simulation and processing pipeline.
Simulates bursts with the standard 8 µs preamble and 112-bit PPM payload,
adds channel impairments, detects via preamble correlation, demodulates PPM,
and extracts fields + CRC check. Saves JSON artifacts and optional plots.

Note: This is a self-contained, pedagogical implementation. The CRC-24 used
here is a common CRC-24 (poly=0x864CFB) for consistency in simulation; it is
NOT the official Mode S parity polynomial. Because the same CRC is used for
both generation and check, crc_ok will be True for simulated frames. Swap
in a Mode S parity implementation if you need protocol-accurate parity.

Usage examples:
    python sdr_adsb_pipeline.py --fs 8000000 --snr 15 --bursts 5 --outdir artifacts/adsb_session_0001
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
from scipy.signal import spectrogram

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

def save_spectrogram_png(x: NDArray[np.complexfloating], fs: int, path: str) -> None:
    """Optional spectrogram for debugging/visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    nperseg = 256
    noverlap = int(nperseg * 0.75)
    f, t, Sxx = spectrogram(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                            mode="magnitude", scaling="spectrum")
    plt.figure()
    plt.pcolormesh(t*1e3, f/1e3, 20*np.log10(Sxx + 1e-12), shading="auto")
    plt.xlabel("Time [ms]")
    plt.ylabel("Freq [kHz]")
    plt.title("ADS-B Sim Spectrogram")
    plt.colorbar(label="Mag [dB]")
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ------------------------------
# CRC-24 (simulation-grade, not Mode S parity)
# ------------------------------
CRC24_POLY = 0x864CFB
CRC24_INIT = 0x000000
CRC24_MASK = 0xFFFFFF

def crc24(bits: NDArray[np.integer]) -> int:
    """Compute CRC-24 over a bit array (MSB-first)."""
    reg = CRC24_INIT
    for b in (bits.astype(np.uint8).reshape(-1) & 1):
        msb = (reg >> 23) & 1
        reg = ((reg << 1) & CRC24_MASK) | int(b)
        if msb:
            reg ^= CRC24_POLY
    return reg & CRC24_MASK

# ------------------------------
# ADS-B / Mode S framing
# ------------------------------
def build_adsb_df17_frame(icao: int, data56: NDArray[np.integer]) -> NDArray[np.uint8]:
    """
    Build a DF17 (ADS-B extended squitter) payload: 112 bits total.
    Structure: DF[5]=17, CA[3]=5 (example), ICAO[24], DATA[56], PARITY[24]
    Parity is simulated with CRC-24 above for demonstration.
    Returns a numpy array of 112 bits (uint8).
    """
    DF = np.array([int(b) for b in f"{17:05b}"], dtype=np.uint8)  # 5 bits
    CA = np.array([int(b) for b in f"{5:03b}"], dtype=np.uint8)   # 3 bits (example)
    ICAO = np.array([int(b) for b in f"{icao:024b}"], dtype=np.uint8)
    D56 = (np.asarray(data56).astype(np.uint8).reshape(56) & 1)

    head = np.concatenate([DF, CA, ICAO, D56])  # 5+3+24+56 = 88 bits
    parity = np.array([int(b) for b in f"{crc24(head):024b}"], dtype=np.uint8)
    frame = np.concatenate([head, parity])      # total 112 bits
    return frame

def ppm_symbolize(bits: NDArray[np.integer], fs: int) -> NDArray[np.float32]:
    """
    Convert ADS-B bits (112) into PPM baseband amplitude at sample rate fs.
    - Bit duration Tb = 1 us
    - Pulse width 0.5 us (first or second half of the bit)
    - '1' -> pulse in first half; '0' -> pulse in second half (convention)
    Returns real-valued baseband amplitude.
    """
    sps_us = int(round(fs / 1_000_000))
    if sps_us < 4:
        raise ValueError("Use fs >= 4 Msps for adequate PPM resolution.")
    half = sps_us // 2
    pulse = np.ones(half, dtype=np.float32)  # rectangular pulse (0.5 us)
    out = np.zeros(len(bits) * sps_us, dtype=np.float32)
    for i, b in enumerate((bits.reshape(-1) & 1)):
        start = i * sps_us + (0 if b == 1 else half)
        out[start:start+len(pulse)] = 1.0
    return out

def adsb_preamble(fs: int) -> NDArray[np.float32]:
    """
    Return an 8 us preamble as a baseband amplitude template.
    Standard-like layout with 0.5 us pulses at 0, 0.5, 1.0, 3.5 us.
    """
    sps_us = int(round(fs / 1_000_000))
    half = sps_us // 2
    pre = np.zeros(8 * sps_us, dtype=np.float32)
    # Pulses (0.5 us)
    pulses_us = [0.0, 0.5, 1.0, 3.5]
    for pu in pulses_us:
        idx = int(round(pu * sps_us))
        pre[idx:idx+half] = 1.0
    return pre

def synthesize_adsb_burst(fs: int,
                          icao: int,
                          data56: NDArray[np.integer],
                          amp: float = 1.0) -> NDArray[np.complex64]:
    """
    Create a single ADS-B burst (preamble + 112-bit PPM payload) at baseband.
    Returns complex64 (real-only signal placed on I; Q=0).
    """
    frame = build_adsb_df17_frame(icao, data56)
    pre = adsb_preamble(fs)
    payload = ppm_symbolize(frame, fs)
    bb = np.concatenate([pre, payload]).astype(np.float32) * float(amp)
    return (bb.astype(np.complex64) + 0j)

def add_channel_impairments(x: NDArray[np.complexfloating],
                            fs: int,
                            snr_db: float,
                            cfo_hz: float = 0.0,
                            timing_ppm: float = 0.0,
                            iq_imbalance: float = 0.0,
                            dc_offset: float = 0.0) -> NDArray[np.complex64]:
    """
    Apply CFO, sample-rate mismatch (ppm), IQ amplitude imbalance, DC offset, and AWGN.
    """
    x = np.asarray(x).astype(np.complex64, copy=False)
    N = len(x)
    # CFO
    if abs(cfo_hz) > 0:
        n = np.arange(N, dtype=np.float64)
        x = (x * np.exp(1j * 2 * np.pi * cfo_hz * n / fs)).astype(np.complex64, copy=False)
    # Timing ppm resample (t2 = n/(fs*alpha))
    alpha = 1.0 + timing_ppm * 1e-6
    if abs(alpha - 1.0) > 1e-12:
        t = np.arange(N, dtype=np.float64) / fs
        t2 = np.arange(N, dtype=np.float64) / (fs * alpha)
        real = np.interp(t2, t, x.real, left=0.0, right=0.0)
        imag = np.interp(t2, t, x.imag, left=0.0, right=0.0)
        x = (real + 1j * imag).astype(np.complex64, copy=False)
    # IQ amplitude imbalance
    if abs(iq_imbalance) > 0:
        x = x.real * (1.0 + iq_imbalance) + 1j * x.imag * (1.0 - iq_imbalance)
        x = x.astype(np.complex64, copy=False)
    # DC offset
    if abs(dc_offset) > 0:
        x = (x + dc_offset).astype(np.complex64, copy=False)
    # AWGN to target SNR
    p_sig = float(np.mean(np.abs(x) ** 2)) if x.size else 1.0
    n0 = p_sig / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(n0 / 2.0) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    return (x + noise.astype(np.complex64)).astype(np.complex64)

# ------------------------------
# Detection and demod
# ------------------------------
def correlate_preamble_mag(x: NDArray[np.complexfloating], fs: int) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Correlate |x| with preamble template. Return correlation and threshold trace.
    """
    tpl = adsb_preamble(fs).astype(np.float32)
    mag = np.abs(x).astype(np.float32)
    corr = np.correlate(mag, tpl, mode="same")
    # Robust threshold: median + k*MAD on correlation
    med = float(np.median(corr))
    mad = float(np.median(np.abs(corr - med))) + 1e-9
    thr = med + 6.0 * mad
    return corr.astype(np.float32), np.full_like(corr, thr, dtype=np.float32)

def find_bursts_from_corr(corr: NDArray[np.floating], thr: float, guard: int) -> List[int]:
    """
    Return list of center indices where corr > thr, with 'guard' samples separating peaks.
    """
    peaks = []
    above = corr > thr
    i = 0
    N = len(corr)
    while i < N:
        if above[i]:
            # local max in a window
            j = i
            end = min(N, i + guard)
            k = j + int(np.argmax(corr[j:end]))
            peaks.append(k)
            i = k + guard
        else:
            i += 1
    return peaks

def demod_ppm_payload(x: NDArray[np.complexfloating], fs: int, start_idx: int) -> Optional[NDArray[np.uint8]]:
    """
    After detecting a preamble centered near start_idx, demodulate the following 112 bits.
    - Preamble length = 8 us => 8*fs/1e6 samples
    - Each bit = 1 us => sps_us samples; decide which half has more energy.
    Returns 112 bits or None if out-of-bounds.
    """
    sps_us = int(round(fs / 1_000_000))
    pre_len = 8 * sps_us
    bit_len = sps_us
    total = pre_len + 112 * bit_len
    # Align start at the END of preamble (approximate): center -> shift by +pre_len//2
    start = start_idx + pre_len // 2
    stop = start + 112 * bit_len
    if stop > len(x):
        return None
    mag = np.abs(x[start:stop]).astype(np.float32)
    half = bit_len // 2
    bits = np.zeros(112, dtype=np.uint8)
    for i in range(112):
        seg = mag[i*bit_len:(i+1)*bit_len]
        e0 = float(np.sum(seg[:half]))
        e1 = float(np.sum(seg[half:]))
        bits[i] = 1 if e0 > e1 else 0
    return bits

# ------------------------------
# Field extraction
# ------------------------------
def bits_to_int(bits: NDArray[np.integer]) -> int:
    b = (np.asarray(bits).astype(np.uint8).reshape(-1) & 1)
    out = 0
    for v in b:
        out = (out << 1) | int(v)
    return out

def decode_df17_fields(bits112: NDArray[np.integer]) -> Dict:
    """
    Parse a 112-bit DF17-like frame into fields (DF, CA, ICAO, data56, parity).
    """
    b = (np.asarray(bits112).astype(np.uint8).reshape(112) & 1)
    DF  = bits_to_int(b[0:5])
    CA  = bits_to_int(b[5:8])
    ICAO= bits_to_int(b[8:32])
    D56 = b[32:88].copy()
    PAR = bits_to_int(b[88:112])
    return {
        "DF": DF, "CA": CA,
        "ICAO_hex": f"{ICAO:06X}",
        "DATA_hex": "".join(f"{bits_to_int(D56[i:i+8]):02X}" for i in range(0,56,8)),
        "PAR_hex": f"{PAR:06X}",
        "crc_ok": (crc24(b[0:88]) == PAR)
    }

# ------------------------------
# Session runner
# ------------------------------
def run_adsb_session(fs: int = 8_000_000,
                     snr_db: float = 15.0,
                     bursts: int = 5,
                     gap_us: Tuple[int,int] = (200, 400),
                     outdir: str = "artifacts/adsb_session_0001") -> None:
    """
    Simulate a stream with several ADS-B bursts separated by random gaps.
    - fs: sample rate (recommend >= 8 Msps)
    - snr_db: AWGN SNR
    - bursts: number of frames to insert
    - gap_us: random gap between frames (uniform range)
    Saves per-burst JSON and a spectrogram.
    """
    ensure_dir(outdir)

    sps_us = int(round(fs / 1_000_000))
    pre_len = 8 * sps_us
    bit_len = sps_us
    burst_len = pre_len + 112 * bit_len

    # Build a long buffer with random gaps
    xs = []
    meta_truth = []
    cur_idx = 0
    for k in range(bursts):
        # Random ICAO and random 56-bit data
        icao = np.random.randint(0, 1 << 24)
        data56 = np.random.randint(0, 2, size=(56,), dtype=np.uint8)
        bb = synthesize_adsb_burst(fs, icao, data56, amp=1.0)
        gap = np.random.randint(gap_us[0], gap_us[1]) * sps_us
        xs.append(np.zeros(gap, dtype=np.complex64))
        start = cur_idx + len(xs[-1])
        xs.append(bb.astype(np.complex64))
        cur_idx = start + len(bb)
        meta_truth.append({"k": k+1, "start": start, "icao": f"{icao:06X}"})
    x = np.concatenate(xs) if xs else np.zeros(burst_len, dtype=np.complex64)

    # Channel
    x = add_channel_impairments(x, fs, snr_db, cfo_hz=10e3, timing_ppm=2.0, iq_imbalance=0.02, dc_offset=0.0)

    # Detection
    corr, thr_arr = correlate_preamble_mag(x, fs)
    guard = burst_len // 2
    centers = find_bursts_from_corr(corr, float(thr_arr[0]), guard=guard)

    # Spectrogram
    save_spectrogram_png(x, fs, os.path.join(outdir, "spectrogram.png"))

    # Demod / Decode
    found = 0
    for i, c in enumerate(centers, start=1):
        bits = demod_ppm_payload(x, fs, c)
        if bits is None:
            continue
        fields = decode_df17_fields(bits)
        found += 1
        rec = {
            "burst_id": found,
            "center_index": int(c),
            "t_center_s": c / fs,
            "sample_rate_hz": fs,
            **fields
        }
        write_json(rec, os.path.join(outdir, f"burst_{found:04d}.json"))

    # Session summary
    write_json({
        "fs": fs,
        "snr_db": snr_db,
        "bursts_requested": bursts,
        "bursts_detected": found,
        "note": "CRC uses CRC-24 (0x864CFB) for simulation; not Mode S parity."
    }, os.path.join(outdir, "session_summary.json"))

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="ADS-B simulation & processing pipeline (PPM @ 1 Mbps).")
    ap.add_argument("--fs", type=int, default=8_000_000, help="Sample rate [Hz], recommend >= 8e6")
    ap.add_argument("--snr", type=float, default=15.0, help="SNR in dB")
    ap.add_argument("--bursts", type=int, default=5, help="Number of ADS-B frames to simulate")
    ap.add_argument("--outdir", type=str, default="artifacts/adsb_session_0001", help="Output directory for artifacts")
    args = ap.parse_args()
    run_adsb_session(fs=args.fs, snr_db=args.snr, bursts=args.bursts, outdir=args.outdir)

if __name__ == "__main__":
    main()

"""
path = "/mnt/data/sdr_adsb_pipeline.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(adsb_code)
"""
