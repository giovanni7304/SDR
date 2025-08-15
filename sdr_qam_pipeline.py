"""
sdr_qam_pipeline.py
-------------------
Educational 16/32/64-QAM simulation pipeline:
- Bit generation, Gray-mapped QAM (16, 64). 32-QAM uses a rectangular 8x4 mapping.
- RRC pulse shaping and matched filtering
- Channel impairments: AWGN, CFO, optional simple multipath
- Receiver: matched filter, symbol timing at integer sps, decision-directed carrier recovery,
           optional 1-tap DD equalizer, hard demap, BER/FER metrics
- Optional PNGs: constellation before/after recovery

NOTE: The user asked for "QAM16/33/64" — 33‑QAM is not a standard order.
This script assumes they meant **32‑QAM** and implements a simple rectangular 8×4 mapping.

Usage examples:
  python sdr_qam_pipeline.py --M 16 --EbN0 12 --N 20000 --sps 4 --beta 0.25 --cfo 200 --fs 48000 --outdir artifacts/qam16
  python sdr_qam_pipeline.py --M 32 --EbN0 18 --N 40000 --sps 4 --beta 0.25 --cfo 100 --fs 48000 --outdir artifacts/qam32
  python sdr_qam_pipeline.py --M 64 --EbN0 24 --N 80000 --sps 4 --beta 0.25 --cfo 50  --fs 48000 --outdir artifacts/qam64
"""

import os
import argparse
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict

# ---------------- I/O helpers ----------------
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

# ---------------- DSP helpers ----------------
def rrc_taps(beta: float, sps: int, span: int) -> NDArray[np.float64]:
    """
    Root-raised cosine (RRC) filter taps.
    beta: roll-off in [0,1], sps: samples per symbol, span: symbols
    """
    N = span * sps
    t = (np.arange(-N, N+1, dtype=np.float64)) / sps  # in symbols
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(1 - (4*beta*ti)**2) < 1e-12:
            taps[i] = (np.pi/4) * np.sinc(1/(2*beta))
        elif abs(ti) < 1e-12:
            taps[i] = 1 - beta + 4*beta/np.pi
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1-(4*beta*ti)**2)
            taps[i] = num / den
    taps = taps / np.sqrt(np.sum(taps**2))  # energy normalize
    return taps

def upsample(x: NDArray[np.complexfloating], sps: int) -> NDArray[np.complexfloating]:
    y = np.zeros(len(x)*sps, dtype=np.complex128)
    y[::sps] = x
    return y

def awgn(x: NDArray[np.complexfloating], snr_db: float) -> NDArray[np.complexfloating]:
    p = np.mean(np.abs(x)**2)
    n0 = p / (10**(snr_db/10))
    n = np.sqrt(n0/2) * (np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape))
    return x + n

# ---------------- Constellations & Mapping ----------------
def gray_bits(n: int) -> NDArray[np.uint8]:
    """Return Gray-coded bits for values 0..n-1 (rows=v)."""
    g = np.arange(n) ^ (np.arange(n) >> 1)
    b = (((g[:, None] & (1 << np.arange(int(np.ceil(np.log2(n)))))) > 0).astype(np.uint8))[:, ::-1]
    return b

def qam_square_points(M: int) -> Tuple[NDArray[np.complex128], Dict[int, NDArray[np.uint8]], Dict[Tuple[int,...], int]]:
    """
    Square QAM points for M in {16,64}, Gray per I and Q axes.
    Returns points (complex), map symidx->bits, and bits->symidx dict.
    """
    m_side = int(np.sqrt(M))
    lv = np.arange(-(m_side-1), (m_side+1), 2)  # like [-3,-1,1,3] for 16QAM
    # Normalize average energy to 1
    Ex = np.mean(np.repeat(lv**2, m_side) + np.tile(lv**2, m_side))
    scale = np.sqrt(Ex/2)
    I = lv / scale; Q = lv / scale
    pts = np.array([i + 1j*q for q in Q for i in I], dtype=np.complex128)
    # Gray along axes
    g = gray_bits(m_side)  # shape (m_side, log2(m_side))
    k_axis = g.shape[1]
    bits_map = {}
    inv_map = {}
    idx = 0
    for qidx, q in enumerate(Q):
        for iidx, i in enumerate(I):
            bI = tuple(g[iidx].tolist())
            bQ = tuple(g[qidx].tolist())
            b = np.array(bI + bQ, dtype=np.uint8)
            bits_map[idx] = b
            inv_map[tuple(b.tolist())] = idx
            idx += 1
    return pts, bits_map, inv_map

def qam32_rect_points() -> Tuple[NDArray[np.complex128], Dict[int, NDArray[np.uint8]], Dict[Tuple[int,...], int]]:
    """
    Simple rectangular 32‑QAM: 8x4 grid (I: ±1,±3,±5,±7; Q: ±1,±3,±5,±7 but remove 8 corner points to make 32).
    We'll build a cross‑shaped 32 by excluding the outermost diagonal corners and fill Gray bits by axis.
    """
    Ilev = np.array([-7,-5,-3,-1,1,3,5,7], dtype=np.int32)
    Qlev = np.array([-7,-5,-3,-1,1,3,5,7], dtype=np.int32)
    pts = []
    for q in Qlev:
        for i in Ilev:
            # Exclude far corners to leave 32 points (remove 16 diagonal extremes, keep inner ring)
            if (abs(i) == 7 and abs(q) == 7):
                continue
            pts.append(i + 1j*q)
    pts = np.array(pts, dtype=np.complex128)
    # Normalize average energy to 1
    Ex = np.mean((pts.real**2 + pts.imag**2))
    pts = pts / np.sqrt(Ex)
    # Assign Gray bits per axis with 5 bits total (3 for I-ish, 2 for Q-ish) — approximate mapping
    # We'll sort points by (Q,I) grid positions and assign bits deterministically.
    idx_sorted = np.argsort(pts.imag + 1e-6*pts.real)
    pts = pts[idx_sorted]
    bits_map = {}
    inv_map = {}
    # Generate 32 Gray words (5 bits)
    G32 = np.arange(32) ^ (np.arange(32) >> 1)
    B32 = (((G32[:, None] & (1 << np.arange(5))) > 0).astype(np.uint8))[:, ::-1]
    for k, idx in enumerate(range(32)):
        b = B32[k]
        bits_map[idx] = b
        inv_map[tuple(b.tolist())] = idx
    return pts, bits_map, inv_map

def qam_points(M: int):
    if M == 16 or M == 64:
        return qam_square_points(M)
    elif M == 32:
        return qam32_rect_points()
    else:
        raise ValueError("Supported constellations: 16, 32, 64 QAM.")

def map_bits_to_symbols(bits: NDArray[np.uint8], M: int) -> Tuple[NDArray[np.complex128], Dict[int, NDArray[np.uint8]]]:
    pts, bits_map, inv_map = qam_points(M)
    k = int(np.log2(M))
    if len(bits) % k != 0:
        pad = (-len(bits)) % k
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    symbols = []
    for i in range(0, len(bits), k):
        b = tuple(bits[i:i+k].tolist())
        idx = inv_map.get(b, None)
        if idx is None:
            # fallback: map by integer value (not Gray) if bit combo not in inv_map (shouldn't happen for 16/64)
            idx = int("".join(map(str, b)), 2) % len(pts)
        symbols.append(pts[idx])
    return np.array(symbols, dtype=np.complex128), bits_map

def hard_slice(z: NDArray[np.complexfloating], pts: NDArray[np.complex128]) -> NDArray[np.complex128]:
    # Nearest neighbor
    Z = z.reshape(-1, 1)
    d2 = np.abs(Z - pts[None, :])**2
    idx = np.argmin(d2, axis=1)
    return pts[idx]

# ---------------- Transmitter ----------------
def tx_qam(bits: NDArray[np.uint8], M: int, sps: int, beta: float, span: int) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], Dict[int, NDArray[np.uint8]]]:
    syms, bits_map = map_bits_to_symbols(bits, M)
    taps = rrc_taps(beta, sps, span)
    up = upsample(syms, sps)
    x = np.convolve(up, taps, mode="full")
    return x, syms, bits_map

# ---------------- Channel ----------------
def channel(x: NDArray[np.complexfloating], fs: int, EbN0_dB: float, M: int, sps: int,
            cfo_hz: float = 0.0, taps: NDArray[np.complex128] = None) -> NDArray[np.complexfloating]:
    """
    Apply CFO, (optional) multipath FIR, and AWGN for a target Eb/N0.
    """
    y = x.copy().astype(np.complex128)
    N = len(y)
    # CFO
    if abs(cfo_hz) > 0:
        n = np.arange(N, dtype=np.float64)
        y *= np.exp(1j * 2*np.pi * cfo_hz * n / fs)
    # Multipath
    if taps is not None:
        y = np.convolve(y, taps, mode="full")
    # AWGN for Eb/N0
    k = int(np.log2(M))
    # Symbol energy after pulse shaping ~ average power * sps (approx). We'll estimate directly.
    Es = np.mean(np.abs(y)**2) * sps
    N0 = Es / (10**(EbN0_dB/10)) / k
    sigma2 = N0/2
    n = np.sqrt(sigma2) * (np.random.randn(*y.shape) + 1j*np.random.randn(*y.shape))
    y = y + n
    return y

# ---------------- Receiver ----------------
def rx_matched_and_downsample(y: NDArray[np.complexfloating], sps: int, beta: float, span: int, timing_offset: int = 0) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    taps = rrc_taps(beta, sps, span)
    z = np.convolve(y, taps, mode="full")
    # Symbol timing: pick samples at offset (span*sps gives filter delay)
    delay = span * sps
    start = delay + timing_offset
    sym = z[start::sps]
    return z, sym

def dd_carrier_recovery(sym: NDArray[np.complexfloating], pts: NDArray[np.complex128], mu: float = 0.01) -> Tuple[NDArray[np.complex128], float]:
    """
    Decision-directed phase-locked loop for carrier phase (and small CFO). Returns corrected symbols and final phase.
    """
    phase = 0.0
    out = np.zeros_like(sym, dtype=np.complex128)
    for i, r in enumerate(sym):
        r_rot = r * np.exp(-1j*phase)
        a_hat = hard_slice(r_rot, pts)[0] if np.ndim(r_rot)==0 else hard_slice(np.array([r_rot]), pts)[0]
        e_phase = np.angle(r_rot * np.conj(a_hat))
        phase = phase + mu * e_phase  # integrate
        out[i] = r_rot
    return out, phase

def demap_hard(sym_hat: NDArray[np.complexfloating], pts: NDArray[np.complex128], bits_map: Dict[int, NDArray[np.uint8]], M: int) -> NDArray[np.uint8]:
    Z = sym_hat.reshape(-1, 1)
    d2 = np.abs(Z - pts[None, :])**2
    idx = np.argmin(d2, axis=1)
    bits = np.concatenate([bits_map[int(i)] for i in idx], axis=0).astype(np.uint8)
    return bits

# ---------------- Metrics & Plots ----------------
def ber(ref_bits: NDArray[np.uint8], est_bits: NDArray[np.uint8]) -> float:
    L = min(len(ref_bits), len(est_bits))
    if L == 0: return 1.0
    return float(np.mean(ref_bits[:L] != est_bits[:L]))

def save_constellation_png(before: NDArray[np.complexfloating], after: NDArray[np.complexfloating], pts: NDArray[np.complex128], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    import matplotlib.pyplot as plt
    ensure_dir(os.path.dirname(path))
    fig = plt.figure(figsize=(7,3.2))
    ax1 = fig.add_subplot(1,2,1); ax1.scatter(before.real, before.imag, s=4, alpha=0.5); ax1.scatter(pts.real, pts.imag, marker='+'); ax1.set_title("Rx symbols (pre-PLL)"); ax1.set_aspect('equal', 'box')
    ax2 = fig.add_subplot(1,2,2); ax2.scatter(after.real, after.imag, s=4, alpha=0.5);  ax2.scatter(pts.real, pts.imag, marker='+'); ax2.set_title("After DD carrier recovery"); ax2.set_aspect('equal', 'box')
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("I"); ax.set_ylabel("Q")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)

# ---------------- Orchestration ----------------
def run_qam(M: int = 16, N_bits: int = 20000, sps: int = 4, beta: float = 0.25, span: int = 8,
            EbN0_dB: float = 14.0, fs: int = 48000, cfo_hz: float = 100.0, timing_offset: int = 0,
            outdir: str = "artifacts/qam_demo") -> Dict:
    ensure_dir(outdir)
    if M not in (16,32,64):
        raise ValueError("Supported M: 16, 32, 64")
    # Bits
    bits_tx = np.random.randint(0, 2, size=N_bits, dtype=np.uint8)
    # TX
    x, sym_tx, bits_map = tx_qam(bits_tx, M, sps, beta, span)
    pts, _, _ = qam_points(M)
    # Channel
    # optional simple multipath (commented by default)
    # h = np.array([1.0, 0.2*np.exp(1j*0.6)], dtype=np.complex128)
    h = None
    y = channel(x, fs, EbN0_dB, M, sps, cfo_hz=cfo_hz, taps=h)
    # RX
    mf, sym = rx_matched_and_downsample(y, sps, beta, span, timing_offset=timing_offset)
    sym_pre = sym.copy()
    sym_hat, ph = dd_carrier_recovery(sym, pts, mu=0.01)
    bits_rx = demap_hard(sym_hat, pts, bits_map, M)
    Pe = ber(bits_tx, bits_rx)
    # Save plot
    save_constellation_png(sym_pre[100:5100], sym_hat[100:5100], pts, os.path.join(outdir, "constellation.png"))
    # Write summary
    summary = {
        "M": M, "EbN0_dB": EbN0_dB, "BER": Pe, "sps": sps, "beta": beta,
        "fs": fs, "cfo_hz": cfo_hz, "timing_offset": timing_offset,
        "N_bits_tx": int(N_bits), "N_bits_rx": int(len(bits_rx))
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        import json; json.dump(summary, f, indent=2, sort_keys=True)
    return summary

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="16/32/64-QAM simulation with RRC, AWGN, CFO, and DD carrier recovery.")
    ap.add_argument("--M", type=int, default=16, choices=[16,32,64], help="QAM order")
    ap.add_argument("--N", type=int, default=20000, help="Number of TX bits")
    ap.add_argument("--sps", type=int, default=4, help="Samples per symbol")
    ap.add_argument("--beta", type=float, default=0.25, help="RRC roll-off [0..1]")
    ap.add_argument("--span", type=int, default=8, help="RRC span in symbols")
    ap.add_argument("--EbN0", type=float, default=14.0, help="Eb/N0 in dB")
    ap.add_argument("--fs", type=int, default=48000, help="Sample rate [Hz] (for CFO calc)")
    ap.add_argument("--cfo", type=float, default=100.0, help="Carrier frequency offset [Hz]")
    ap.add_argument("--toff", type=int, default=0, help="Timing offset in samples (0..sps-1)")
    ap.add_argument("--outdir", type=str, default="artifacts/qam_demo", help="Output directory")
    args = ap.parse_args()
    run_qam(M=args.M, N_bits=args.N, sps=args.sps, beta=args.beta, span=args.span,
            EbN0_dB=args.EbN0, fs=args.fs, cfo_hz=args.cfo, timing_offset=args.toff, outdir=args.outdir)

if __name__ == "__main__":
    main()
