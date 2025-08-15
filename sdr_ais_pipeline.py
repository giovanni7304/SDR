"""
sdr_ais_pipeline.py
-------------------
Educational AIS-style (GMSK @ 9.6 kbps) simulation & processing pipeline with an optional RTL-SDR hook.
It simulates bursty AIS-like packets, applies channel impairments, detects bursts, performs GMSK
demodulation with Gardner timing recovery, differentially decodes, frames on a configurable preamble+sync,
and writes per-burst JSON records. It also supports live capture near 161.975 / 162.025 MHz.

This is a pedagogical pipeline, not a protocol-complete AIS decoder per ITU-R M.1371. It uses
a simple, self-consistent frame (preamble + sync + payload + CRC-16-CCITT) so you can validate
the DSP chain end-to-end without relying on proprietary or jurisdiction-specific details.

Usage (simulation):
  python sdr_ais_pipeline.py --mode sim --fs 192000 --snr 18 --bursts 4 --outdir artifacts/ais_sim_0001

Usage (RTL-SDR, 162.025 MHz, 192 kS/s, 5 s capture):
  pip install pyrtlsdr
  python sdr_ais_pipeline.py --mode rtlsdr --fs 192000 --seconds 5 --center 162025000 --gain auto --ppm 0 --outdir artifacts/ais_live_0001
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
from scipy.signal import firwin, lfilter, resample_poly, spectrogram

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
    plt.pcolormesh(t, f/1e3, 20*np.log10(Sxx + 1e-12), shading="auto")
    plt.xlabel("Time [s]")
    plt.ylabel("Freq [kHz]")
    plt.title("AIS Spectrogram (educational)")
    plt.colorbar(label="Mag [dB]")
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ------------------------------
# Gaussian filter for GMSK
# ------------------------------
def gaussian_taps(bt: float, sps: int, span: int) -> NDArray[np.float64]:
    """
    Gaussian filter taps for GMSK (normalized so area ~1).
    bt: bandwidth-time product (AIS typical ~0.4)
    sps: samples per symbol
    span: pulse span in symbols
    """
    t = (np.arange(-span*sps, span*sps+1) / sps).astype(np.float64)  # in symbols
    alpha = np.sqrt(2*np.pi) * bt
    taps = np.exp(-2*(np.pi*bt*t/np.sqrt(np.log(2)))**2)
    taps /= np.sum(taps)
    return taps

# ------------------------------
# CRC-16-CCITT (0x1021, init 0xFFFF, no reflect, no xorout)
# ------------------------------
def crc16_ccitt(bits: NDArray[np.integer]) -> int:
    reg = 0xFFFF
    for b in (bits.astype(np.uint8).reshape(-1) & 1):
        reg ^= (int(b) << 15)
        for _ in range(8):  # one bit at a time emulated with 8 inner shifts is overkill but simple; we treat b as entering MSB
            msb = (reg & 0x8000) != 0
            reg = ((reg << 1) & 0xFFFF)
            if msb:
                reg ^= 0x1021
    return reg & 0xFFFF

# ------------------------------
# AIS-like framing (educational)
# ------------------------------
def build_ais_like_frame(payload_bits: NDArray[np.integer]) -> NDArray[np.uint8]:
    """
    Build a self-consistent AIS-like frame:
      PREAMBLE: 24 bits alternating '10' (0xAAAAAA)
      SYNC:      8 bits fixed 0x7E (01111110) for visibility
      PAYLOAD:   user-provided (e.g., 168 bits)
      CRC16:     16 bits (CCITT)
    """
    pre = np.tile(np.array([1,0], dtype=np.uint8), 12)  # 24 bits '10'
    sync = np.array([0,1,1,1,1,1,1,0], dtype=np.uint8)
    pl = (np.asarray(payload_bits).astype(np.uint8).reshape(-1) & 1)
    crc_val = crc16_ccitt(pl)
    crc_bits = np.array([int(b) for b in f"{crc_val:016b}"], dtype=np.uint8)
    frame = np.concatenate([pre, sync, pl, crc_bits])
    return frame

# ------------------------------
# GMSK Mod / Demod
# ------------------------------
def gmsk_mod(bits: NDArray[np.integer], bt: float, sps: int, span: int = 4, h: float = 0.5) -> NDArray[np.complex64]:
    """
    Simple GMSK mod:
      - Map bits {0,1} -> NRZ {-1,+1}
      - Gaussian filter at sps
      - Integrate to phase with modulation index h=0.5
    """
    b = (np.asarray(bits).astype(np.uint8).reshape(-1) & 1)
    nrz = 2*b - 1
    # Upsample impulses at symbol rate
    u = np.zeros(len(nrz)*sps, dtype=np.float64)
    u[::sps] = nrz.astype(np.float64)
    g = gaussian_taps(bt, sps, span)
    m = np.convolve(u, g, mode="full")  # frequency pulse
    # integrate to phase (scale for h=0.5 — MSK dev = 0.25 cycles/sym -> pi/2 rad per symbol step)
    phase = np.cumsum(m) * (np.pi * h) / sps
    x = np.exp(1j*phase).astype(np.complex64)
    return x

def discrim(x: NDArray[np.complexfloating]) -> NDArray[np.float32]:
    """Frequency discriminator: angle of x[n]*conj(x[n-1])."""
    if len(x) < 2:
        return np.zeros_like(x, dtype=np.float32)
    y = np.angle(x[1:] * np.conj(x[:-1]))
    y = np.concatenate([y[:1], y]).astype(np.float32)
    return y

def gardner_timing_rrc(x: NDArray[np.float32], sps: int, mu: float = 0.05, n_iters: int = 1) -> Tuple[NDArray[np.float32], float]:
    """
    Minimal Gardner timing on a real-valued stream x (e.g., discriminator output).
    Assumes approx constant sps; returns symbol-rate samples and final fractional offset.
    Uses linear interpolation.
    """
    # initialize at half-symbol to use Gardner's mid-sample
    tau = 0.5
    out = []
    i = 0.0
    N = len(x)
    while True:
        n0 = int(i)
        if n0 + sps >= N - 2:
            break
        # linear interp at current symbol center and mid-point
        frac = i - n0
        # center sample
        c0 = x[n0] * (1-frac) + x[n0+1] * frac
        # previous and next mid samples
        mid_prev_idx = i - sps/2.0
        mid_next_idx = i + sps/2.0
        if mid_prev_idx < 1 or mid_next_idx >= N-2:
            break
        mp0 = x[int(mid_prev_idx)] * (1-(mid_prev_idx-int(mid_prev_idx))) + x[int(mid_prev_idx)+1] * ((mid_prev_idx-int(mid_prev_idx)))
        mp1 = x[int(mid_next_idx)] * (1-(mid_next_idx-int(mid_next_idx))) + x[int(mid_next_idx)+1] * ((mid_next_idx-int(mid_next_idx)))
        e = (mp1 - mp0) * c0
        tau = tau + mu * e
        i += sps + tau
        out.append(c0)
    return np.array(out, dtype=np.float32), tau

def gmsk_demod(x: NDArray[np.complexfloating], fs: int, Rs: float, bt: float, sps_hint: Optional[int] = None) -> Tuple[NDArray[np.uint8], Dict]:
    """
    Demodulate GMSK by:
      - frequency discriminator
      - lowpass (Gaussian matched) on discriminator
      - Gardner timing to symbol rate
      - hard slicing to NRZ symbols {-1,+1}
      - differential to bits {0,1}
    """
    if sps_hint is None:
        sps = int(round(fs / Rs))
    else:
        sps = int(sps_hint)
    # Discriminator -> LPF (Gaussian-ish)
    d = discrim(x)
    # For simplicity we use a small FIR lowpass around Rs/2
    nyq = fs / 2
    lp = firwin(numtaps=101, cutoff=Rs*0.6, fs=fs)
    dlp = lfilter(lp, [1.0], d)
    # Timing recovery
    sym, tau = gardner_timing_rrc(dlp, sps, mu=0.01)
    # Slicer on sign
    nrz = np.sign(sym).astype(np.int8)
    nrz[nrz == 0] = 1
    # Differential to bits: b_k = (nrz_k > 0) XOR (nrz_{k-1} > 0)?
    b = ((nrz > 0).astype(np.uint8))
    # NRZI decode (simple): bit = b_k XOR b_{k-1}
    bits = np.bitwise_xor(b[1:], b[:-1]).astype(np.uint8)
    meta = {"sps": sps, "tau": float(tau), "n_syms": int(len(nrz))}
    return bits, meta

# ------------------------------
# Channel impairments
# ------------------------------
def add_channel_impairments(x: NDArray[np.complexfloating],
                            fs: int,
                            snr_db: float,
                            cfo_hz: float = 0.0,
                            timing_ppm: float = 0.0) -> NDArray[np.complex64]:
    x = np.asarray(x).astype(np.complex64, copy=False)
    N = len(x)
    if abs(cfo_hz) > 0:
        n = np.arange(N, dtype=np.float64)
        x = (x * np.exp(1j * 2 * np.pi * cfo_hz * n / fs)).astype(np.complex64, copy=False)
    alpha = 1.0 + timing_ppm * 1e-6
    if abs(alpha - 1.0) > 1e-12:
        t = np.arange(N, dtype=np.float64) / fs
        t2 = np.arange(N, dtype=np.float64) / (fs * alpha)
        real = np.interp(t2, t, x.real, left=0.0, right=0.0)
        imag = np.interp(t2, t, x.imag, left=0.0, right=0.0)
        x = (real + 1j * imag).astype(np.complex64, copy=False)
    p_sig = float(np.mean(np.abs(x)**2)) if x.size else 1.0
    n0 = p_sig / (10.0 ** (snr_db / 10.0))
    w = np.sqrt(n0/2.0) * (np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape))
    return (x + w.astype(np.complex64)).astype(np.complex64)

# ------------------------------
# Detection
# ------------------------------
def detect_bursts_energy(x: NDArray[np.complexfloating], fs: int, win_ms: float = 5.0) -> List[Tuple[int,int]]:
    """
    Simple energy-based detector on |x|^2 with robust threshold.
    Returns a list of (start, stop) indices.
    """
    win = max(1, int(round(win_ms*1e-3*fs)))
    p = np.convolve(np.abs(x)**2, np.ones(win)/win, mode="same")
    ps = np.sort(p)
    nb = max(32, int(0.3*len(ps)))
    noise_med = float(np.median(ps[:nb]))
    noise_mad = float(np.median(np.abs(ps[:nb]-noise_med))) + 1e-12
    thr = noise_med + 6.0*noise_mad
    above = p > thr
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.view(np.int8), [0]])) != 0)
    starts = edges[0::2]; stops = edges[1::2]-1
    segs = [(int(s), int(e)) for s,e in zip(starts, stops) if e > s + fs*0.01]
    return segs

# ------------------------------
# Framing & CRC
# ------------------------------
def find_sync_and_extract(bits: NDArray[np.uint8]) -> Optional[Dict]:
    """
    Look for preamble '10' repeated 24 times followed by sync 0x7E (01111110).
    If found, parse payload and CRC, return dict.
    """
    pat = np.tile(np.array([1,0], dtype=np.uint8), 12).tolist() + [0,1,1,1,1,1,1,0]
    pat = np.array(pat, dtype=np.uint8)
    b = bits.reshape(-1)
    # sliding correlation on bits with exact match (tolerant search could use Hamming distance)
    for i in range(0, len(b) - len(pat) - 16):  # 16 for CRC
        if np.all(b[i:i+len(pat)] == pat):
            start = i + len(pat)
            if start + 16 > len(b):
                break
            # try various payload sizes (example: 168 bits)
            for L in (168, 200, 224):
                end = start + L + 16
                if end <= len(b):
                    payload = b[start:start+L]
                    crc_bits = b[start+L:end]
                    crc_val = int("".join(map(str, crc_bits.tolist())), 2)
                    if crc16_ccitt(payload) == crc_val:
                        return {"offset": i, "payload_bits": payload, "crc_ok": True, "payload_len": L}
    return None

# ------------------------------
# RTL-SDR capture hook
# ------------------------------
def capture_rtlsdr(center_freq_hz: float,
                   fs: int,
                   seconds: float,
                   gain: Optional[str] = "auto",
                   ppm: int = 0,
                   lpf_bw: float = 25e3) -> NDArray[np.complex64]:
    """
    Capture IQ around the AIS channel using pyrtlsdr, apply a simple complex lowpass to ~25 kHz,
    and return basebanded complex64 at the same fs.
    """
    try:
        from rtlsdr import RtlSdr
    except Exception as e:
        raise RuntimeError("pyrtlsdr not installed. Install with `pip install pyrtlsdr`.") from e

    sdr = RtlSdr()
    try:
        sdr.sample_rate = fs
        sdr.center_freq = center_freq_hz
        sdr.freq_correction = int(ppm)
        if isinstance(gain, str) and gain.lower() == "auto":
            sdr.gain = "auto"
        else:
            sdr.gain = float(gain)

        num_samples = int(fs * seconds)
        # Read in chunks
        chunk = max(8192, fs // 10)
        bufs = []
        got = 0
        while got < num_samples:
            n = min(chunk, num_samples - got)
            iq = sdr.read_samples(n).astype(np.complex64, copy=False)
            bufs.append(iq); got += len(iq)
        x = np.concatenate(bufs) if bufs else np.zeros(0, dtype=np.complex64)
        # DC removal
        x = x - np.mean(x)
        # Complex LPF via real FIR applied to I and Q
        numtaps = 129
        taps = firwin(numtaps, lpf_bw, fs=fs)
        xi = lfilter(taps, [1.0], x.real)
        xq = lfilter(taps, [1.0], x.imag)
        x = (xi + 1j*xq).astype(np.complex64)
        return x
    finally:
        sdr.close()

# ------------------------------
# Orchestration
# ------------------------------
def simulate_ais_stream(fs: int = 192000,
                        Rs: float = 9600.0,
                        bt: float = 0.4,
                        bursts: int = 4,
                        gap_ms_range: Tuple[int,int] = (80, 160),
                        snr_db: float = 18.0,
                        outdir: str = "artifacts/ais_sim_0001") -> None:
    """
    Build a stream with several AIS-like GMSK bursts separated by random gaps.
    """
    ensure_dir(outdir)
    sps = int(round(fs / Rs))
    xs = []
    for k in range(bursts):
        payload = np.random.randint(0, 2, size=(168,), dtype=np.uint8)
        frame = build_ais_like_frame(payload)
        bb = gmsk_mod(frame, bt=bt, sps=sps, span=4, h=0.5)
        gap = np.zeros(np.random.randint(gap_ms_range[0], gap_ms_range[1]) * sps, dtype=np.complex64)
        xs.extend([gap.astype(np.complex64), bb.astype(np.complex64)])
    x = np.concatenate(xs) if xs else np.zeros(0, dtype=np.complex64)
    x = add_channel_impairments(x, fs, snr_db, cfo_hz=50.0, timing_ppm=2.0)
    process_ais_stream(x, fs, Rs, bt, outdir)

def process_ais_stream(x: NDArray[np.complexfloating], fs: int, Rs: float, bt: float, outdir: str) -> None:
    ensure_dir(outdir)
    # Detect bursts
    segs = detect_bursts_energy(x, fs, win_ms=5.0)
    # Save spectrogram
    save_spectrogram_png(x, fs, os.path.join(outdir, "spectrogram.png"))
    # Demod & frame each burst
    found = 0
    for (s, e) in segs:
        xb = x[s:e+1]
        bits, meta = gmsk_demod(xb, fs=fs, Rs=Rs, bt=bt)
        rec = find_sync_and_extract(bits)
        found += 1
        out = {
            "burst_id": found,
            "start_idx": int(s),
            "stop_idx": int(e),
            "t_start_s": s/fs,
            "t_stop_s": e/fs,
            "sample_rate_hz": fs,
            "Rs": Rs,
            "bt": bt,
            "demod_meta": meta,
            "framed": rec if rec is not None else None
        }
        write_json(out, os.path.join(outdir, f"burst_{found:04d}.json"))
    write_json({"fs": fs, "Rs": Rs, "bt": bt, "bursts_detected": found}, os.path.join(outdir, "session_summary.json"))

def run_rtlsdr(fs: int = 192000,
               seconds: float = 5.0,
               center_hz: float = 162_025_000.0,
               gain: str = "auto",
               ppm: int = 0,
               Rs: float = 9600.0,
               bt: float = 0.4,
               outdir: str = "artifacts/ais_live_0001") -> None:
    ensure_dir(outdir)
    x = capture_rtlsdr(center_hz, fs, seconds, gain=gain, ppm=ppm, lpf_bw=30e3)
    process_ais_stream(x, fs, Rs, bt, outdir)

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Educational AIS-style (GMSK 9.6 kbps) simulation & processing pipeline.")
    ap.add_argument("--mode", choices=["sim", "rtlsdr"], default="sim", help="Simulation or RTL-SDR capture")
    ap.add_argument("--fs", type=int, default=192000, help="Sample rate [Hz] (192 kS/s typical)")
    ap.add_argument("--Rs", type=float, default=9600.0, help="Symbol rate [baud]")
    ap.add_argument("--bt", type=float, default=0.4, help="Gaussian BT product")
    ap.add_argument("--bursts", type=int, default=4, help="Simulation: number of bursts")
    ap.add_argument("--snr", type=float, default=18.0, help="Simulation: SNR in dB")
    ap.add_argument("--seconds", type=float, default=5.0, help="RTL-SDR: capture duration [s]")
    ap.add_argument("--center", type=float, default=162_025_000.0, help="RTL-SDR: center frequency [Hz] (161.975e6/162.025e6)")
    ap.add_argument("--gain", type=str, default="auto", help="RTL-SDR: gain in dB or 'auto'")
    ap.add_argument("--ppm", type=int, default=0, help="RTL-SDR: frequency correction in ppm")
    ap.add_argument("--outdir", type=str, default="artifacts/ais_session_0001", help="Output directory")
    args = ap.parse_args()

    if args.mode == "sim":
        simulate_ais_stream(fs=args.fs, Rs=args.Rs, bt=args.bt, bursts=args.bursts, snr_db=args.snr, outdir=args.outdir)
    else:
        run_rtlsdr(fs=args.fs, seconds=args.seconds, center_hz=args.center, gain=args.gain, ppm=args.ppm, Rs=args.Rs, bt=args.bt, outdir=args.outdir)

if __name__ == "__main__":
    main()