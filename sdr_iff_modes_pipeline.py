"""
sdr_iff_modes_pipeline.py
-------------------------
Educational SDR pipeline to simulate and process non-cryptographic IFF-like signals:
- Mode A/C replies (pulse-position coding at 1090 MHz baseband envelope)
- Mode S / ADS-B DF17 (reuses PPM pipeline style)
- Secure Mode STUB (non-functional placeholder; no cryptographic or sensitive details)

This file intentionally avoids implementation details of classified/controlled secure IFF
modes (e.g., Mode 4/5 waveforms and cryptography). The "SecureModeStub" provided here is
ONLY a placeholder so you can test the rest of your pipeline plumbing without revealing or
replicating any sensitive signaling.

Usage examples:
  # Mode A reply with squawk 1200 (approximate educational timing)
  python sdr_iff_modes_pipeline.py --mode modeac --fs 8_000_000 --squawk 1200 --outdir artifacts/iff_modeac_1200

  # Mode C reply with altitude (feet); this uses a simple 12-bit payload demo (not Gillham)
  python sdr_iff_modes_pipeline.py --mode modeac --fs 8_000_000 --alt 12000 --outdir artifacts/iff_modec_alt

  # ADS-B DF17 simulation (like previous)
  python sdr_iff_modes_pipeline.py --mode adsb --fs 8_000_000 --bursts 3 --outdir artifacts/iff_adsb_demo

  # Secure Mode placeholder (no actual secure content; just a dummy burst to exercise pipeline)
  python sdr_iff_modes_pipeline.py --mode secure_stub --fs 8_000_000 --outdir artifacts/iff_secure_stub

Notes:
- All timings are approximate and intended for study. Do NOT use for real-world avionics.
- Secure modes (Mode 4/5) are NOT implemented and are intentionally abstracted.
"""

import os
import json
import argparse
from typing import Tuple, List, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import spectrogram

# ------------------------------
# I/O helpers
# ------------------------------
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def write_json(data: Dict, path: str, pretty: bool = True) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, sort_keys=True)
        else:
            json.dump(data, f, separators=(",", ":"))

def save_spectrogram_png(x: NDArray[np.complexfloating], fs: int, path: str) -> None:
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
    plt.pcolormesh(t*1e6, f/1e3, 20*np.log10(Sxx + 1e-12), shading="auto")
    plt.xlabel("Time [µs]")
    plt.ylabel("Freq [kHz]")
    plt.title("IFF/ADS-B Spectrogram (educational)")
    plt.colorbar(label="Mag [dB]")
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# ------------------------------
# ADS-B DF17 pieces (condensed from earlier)
# ------------------------------
CRC24_POLY = 0x864CFB
CRC24_INIT = 0x000000
CRC24_MASK = 0xFFFFFF

def crc24(bits: NDArray[np.integer]) -> int:
    reg = CRC24_INIT
    for b in (bits.astype(np.uint8).reshape(-1) & 1):
        msb = (reg >> 23) & 1
        reg = ((reg << 1) & CRC24_MASK) | int(b)
        if msb:
            reg ^= CRC24_POLY
    return reg & CRC24_MASK

def build_adsb_df17_frame(icao: int, data56: NDArray[np.integer]) -> NDArray[np.uint8]:
    DF = np.array([int(b) for b in f"{17:05b}"], dtype=np.uint8)  # 5 bits
    CA = np.array([int(b) for b in f"{5:03b}"], dtype=np.uint8)   # 3 bits (example)
    ICAO = np.array([int(b) for b in f"{icao:024b}"], dtype=np.uint8)
    D56 = (np.asarray(data56).astype(np.uint8).reshape(56) & 1)
    head = np.concatenate([DF, CA, ICAO, D56])  # 88 bits
    parity = np.array([int(b) for b in f"{crc24(head):024b}"], dtype=np.uint8)
    frame = np.concatenate([head, parity])      # 112 bits
    return frame

def adsb_preamble(fs: int) -> NDArray[np.float32]:
    sps_us = int(round(fs / 1_000_000))
    half = max(1, sps_us // 2)
    pre = np.zeros(8 * sps_us, dtype=np.float32)
    for pu in [0.0, 0.5, 1.0, 3.5]:
        idx = int(round(pu * sps_us))
        pre[idx:idx+half] = 1.0
    return pre

def ppm_symbolize(bits: NDArray[np.integer], fs: int) -> NDArray[np.float32]:
    sps_us = int(round(fs / 1_000_000))
    half = max(1, sps_us // 2)
    out = np.zeros(len(bits) * sps_us, dtype=np.float32)
    for i, b in enumerate((bits.reshape(-1) & 1)):
        start = i * sps_us + (0 if b == 1 else half)
        out[start:start+half] = 1.0
    return out

def synthesize_adsb_burst(fs: int, icao: int, data56: NDArray[np.integer], amp: float = 1.0) -> NDArray[np.complex64]:
    frame = build_adsb_df17_frame(icao, data56)
    pre = adsb_preamble(fs)
    payload = ppm_symbolize(frame, fs)
    bb = np.concatenate([pre, payload]).astype(np.float32) * float(amp)
    return (bb.astype(np.complex64) + 0j)

def demod_adsb_bits(x: NDArray[np.complexfloating], fs: int, center_idx: int) -> Optional[NDArray[np.uint8]]:
    sps_us = int(round(fs / 1_000_000))
    pre_len = 8 * sps_us
    bit_len = sps_us
    start = center_idx + pre_len // 2
    stop = start + 112 * bit_len
    if stop > len(x): return None
    mag = np.abs(x[start:stop]).astype(np.float32)
    half = max(1, bit_len // 2)
    bits = np.zeros(112, dtype=np.uint8)
    for i in range(112):
        seg = mag[i*bit_len:(i+1)*bit_len]
        bits[i] = 1 if float(np.sum(seg[:half])) > float(np.sum(seg[half:])) else 0
    return bits

# ------------------------------
# Mode A/C (educational approximation)
# ------------------------------
def squawk_to_bits(squawk: int) -> NDArray[np.uint8]:
    """
    Convert a 4-digit octal squawk (0000..7777) to 12 bits (Mode A payload positions).
    This returns a simple 12-bit number derived from the octal digits for educational use.
    """
    s = f"{int(squawk):04o}"[-4:]  # ensure octal-style digits
    val = (int(s[0], 8) << 9) | (int(s[1], 8) << 6) | (int(s[2], 8) << 3) | int(s[3], 8)
    b = np.array([int(x) for x in f"{val:012b}"], dtype=np.uint8)
    return b

def altitude_to_bits_feet(alt_ft: int) -> NDArray[np.uint8]:
    """
    Educational 12-bit encoding for 'Mode C' demo (not true Gillham code).
    Maps altitude (ft) / 100 to a 12-bit field.
    """
    code = max(0, min(4095, int(round(alt_ft/100))))
    return np.array([int(x) for x in f"{code:012b}"], dtype=np.uint8)

def modeac_reply_envelope(fs: int,
                          data12: NDArray[np.integer],
                          pulse_us: float = 0.45,
                          slot_us: float = 1.45,
                          total_us: float = 20.3) -> NDArray[np.float32]:
    """
    Build an educational Mode A/C reply baseband envelope:
    - Two framing pulses (F1, F2)
    - 12 data pulses; presence/absence in slots indicates bits
    Approximated timings only.
    """
    sps_us = int(round(fs / 1_000_000))
    total = int(round(total_us * sps_us))
    env = np.zeros(total, dtype=np.float32)
    pw = max(1, int(round(pulse_us * sps_us)))
    # Framing pulses at start and end
    env[:pw] = 1.0
    end_idx = total - pw
    env[end_idx:total] = 1.0
    # 12 slots between: place pulses for bit '1'
    slots = 12
    usable = total - 2*pw
    step = int(round(usable / (slots + 1)))  # rough spacing
    half = step // 2
    for i, b in enumerate((np.asarray(data12).reshape(-1) & 1)):
        if b:
            pos = pw + (i+1)*step - half
            pos = max(pw, min(total-pw, pos))
            env[pos:pos+pw] = 1.0
    return env

def synthesize_modeac_reply(fs: int,
                            squawk: Optional[int] = None,
                            alt_ft: Optional[int] = None,
                            amp: float = 1.0) -> Tuple[str, NDArray[np.complex64]]:
    """
    Create a Mode A or C reply envelope (educational). Returns (type, complex64 baseband).
    """
    if (squawk is None) == (alt_ft is None):
        raise ValueError("Provide exactly one of squawk (Mode A) or alt_ft (Mode C).")
    if squawk is not None:
        b12 = squawk_to_bits(int(squawk))
        label = f"MODE_A_{int(squawk):04o}"
    else:
        b12 = altitude_to_bits_feet(int(alt_ft))
        label = f"MODE_C_{int(alt_ft)}FT"
    env = modeac_reply_envelope(fs, b12)
    return label, (env.astype(np.float32) * float(amp)).astype(np.complex64) + 0j

def detect_modeac(env: NDArray[np.complexfloating], fs: int) -> List[int]:
    """
    Naive detector: threshold on magnitude, look for leading framing pulse edges.
    Returns list of start indices for candidate replies.
    """
    mag = np.abs(env).astype(np.float32)
    thr = float(np.median(mag) + 4*np.median(np.abs(mag - np.median(mag))))
    above = mag > thr
    starts = np.flatnonzero(np.diff(np.concatenate([[0], above.view(np.int8)])) == 1))
    return starts.tolist()

def decode_modeac_bits(env: NDArray[np.complexfloating], fs: int, start_idx: int) -> Optional[NDArray[np.uint8]]:
    """
    Educational decoder matching the envelope builder above.
    Returns 12 bits or None if out of range.
    """
    sps_us = int(round(fs / 1_000_000))
    total = int(round(20.3 * sps_us))
    pw = max(1, int(round(0.45 * sps_us)))
    if start_idx + total > len(env):
        return None
    mag = np.abs(env[start_idx:start_idx+total]).astype(np.float32)
    usable = total - 2*pw
    slots = 12
    step = int(round(usable / (slots + 1)))
    half = step // 2
    bits = np.zeros(slots, dtype=np.uint8)
    for i in range(slots):
        pos = pw + (i+1)*step - half
        pos = max(pw, min(total-pw, pos))
        seg = mag[pos:pos+pw]
        bits[i] = 1 if np.sum(seg) > 0.5 * pw else 0
    return bits

# ------------------------------
# Secure Mode STUB (placeholder only)
# ------------------------------
class SecureModeStub:
    """
    Placeholder for a secure IFF mode (e.g., Mode 4/5 style). This DOES NOT implement
    any real secure waveform or cryptography. It only generates a generic burst so
    you can exercise your detection → decode plumbing with a black-box placeholder.
    """
    def __init__(self, fs: int, duration_us: float = 64.0):
        self.fs = fs
        self.sps_us = int(round(fs / 1_000_000))
        self.N = int(round(duration_us * self.sps_us))

    def synthesize_burst(self, amp: float = 1.0) -> NDArray[np.complex64]:
        t = np.arange(self.N) / self.fs
        # Arbitrary BPSK-like chip sequence (fixed, non-sensitive)
        chips = np.sign(np.sin(2*np.pi*1e6*t)).astype(np.float32)
        env = 0.5*(1.0 + chips)  # 0/1 envelope
        return (env * amp).astype(np.complex64) + 0j

    def detect_and_decode(self, x: NDArray[np.complexfloating]) -> Dict:
        mag = np.abs(x).astype(np.float32)
        energy = float(np.sum(mag**2))
        # Returns a generic record with no fields
        return {"mode": "SECURE_STUB", "energy": energy, "note": "No secure details implemented."}

# ------------------------------
# Orchestration
# ------------------------------
def run_modeac_demo(fs: int, squawk: Optional[int], alt: Optional[int], outdir: str) -> None:
    ensure_dir(outdir)
    label, bb = synthesize_modeac_reply(fs, squawk=squawk, alt_ft=alt, amp=1.0)
    # Add a little gap before/after in a buffer
    pad = np.zeros(int(100e-6 * fs), dtype=np.complex64)
    x = np.concatenate([pad, bb, pad])
    # Detect & decode
    starts = detect_modeac(x, fs)
    decoded = []
    for s in starts:
        bits = decode_modeac_bits(x, fs, s)
        if bits is None: continue
        decoded.append({"start_idx": int(s), "bits": "".join(map(str, bits.tolist()))})
    # Save
    save_spectrogram_png(x, fs, os.path.join(outdir, "spectrogram.png"))
    write_json({
        "mode": label,
        "fs": fs,
        "detections": decoded
    }, os.path.join(outdir, "result.json"))

def run_adsb_demo(fs: int, bursts: int, outdir: str) -> None:
    ensure_dir(outdir)
    sps_us = int(round(fs / 1_000_000))
    pre_len = 8 * sps_us
    bit_len = sps_us
    xs = []
    for _ in range(bursts):
        icao = np.random.randint(0, 1<<24)
        data56 = np.random.randint(0, 2, size=(56,), dtype=np.uint8)
        bb = synthesize_adsb_burst(fs, icao, data56, amp=1.0)
        gap = np.zeros(np.random.randint(150, 300)*sps_us, dtype=np.complex64)
        xs.extend([gap, bb])
    x = np.concatenate(xs)
    # Simple preamble correlation on |x| to find centers
    tpl = adsb_preamble(fs).astype(np.float32)
    corr = np.correlate(np.abs(x).astype(np.float32), tpl, mode="same")
    med = float(np.median(corr)); mad = float(np.median(np.abs(corr-med))) + 1e-9
    thr = med + 6.0*mad
    # find peaks with guard
    guard = pre_len + 112*bit_len
    centers = []
    i = 0
    while i < len(corr):
        if corr[i] > thr:
            j = i; end = min(len(corr), i+guard)
            k = j + int(np.argmax(corr[j:end]))
            centers.append(k); i = k + guard
        else:
            i += 1
    # Decode
    decs = []
    for c in centers:
        bits = demod_adsb_bits(x, fs, c)
        if bits is None: continue
        decs.append({"center_idx": int(c), "bits112_hex": "".join(f"{int(''.join(map(str,bits[i:i+8])),2):02X}" for i in range(0,112,8))})
    save_spectrogram_png(x, fs, os.path.join(outdir, "spectrogram.png"))
    write_json({"mode": "ADS-B_DF17_SIM", "fs": fs, "detections": decs}, os.path.join(outdir, "result.json"))

def run_secure_stub(fs: int, outdir: str) -> None:
    ensure_dir(outdir)
    stub = SecureModeStub(fs)
    bb = stub.synthesize_burst(amp=1.0)
    pad = np.zeros(int(100e-6 * fs), dtype=np.complex64)
    x = np.concatenate([pad, bb, pad])
    rec = stub.detect_and_decode(x)
    save_spectrogram_png(x, fs, os.path.join(outdir, "spectrogram.png"))
    write_json(rec, os.path.join(outdir, "result.json"))

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Educational IFF Modes pipeline (Mode A/C, ADS-B, Secure STUB).")
    ap.add_argument("--mode", choices=["modeac", "adsb", "secure_stub"], default="modeac")
    ap.add_argument("--fs", type=int, default=8_000_000, help="Sample rate [Hz] (recommend >= 8e6)")
    ap.add_argument("--squawk", type=int, default=None, help="Mode A squawk (octal-like integer, e.g., 1200)")
    ap.add_argument("--alt", type=int, default=None, help="Mode C altitude (feet) [educational mapping]")
    ap.add_argument("--bursts", type=int, default=3, help="ADS-B: number of bursts to simulate")
    ap.add_argument("--outdir", type=str, default="artifacts/iff_demo", help="Output directory")
    args = ap.parse_args()

    if args.mode == "modeac":
        run_modeac_demo(args.fs, args.squawk, args.alt, args.outdir)
    elif args.mode == "adsb":
        run_adsb_demo(args.fs, args.bursts, args.outdir)
    else:
        run_secure_stub(args.fs, args.outdir)

if __name__ == "__main__":
    main()
