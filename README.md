####################################################################################################
#
# This is my first draft
# Many more demod coming soon
#
# These will eventually work with the RTL-SDR USB for Educational purposes
####################################################################################################
# SDR PTT Pipeline

An end-to-end SDR development pipeline in Python for simulating push-to-talk (PTT) radio signals,
detecting bursts, analyzing characteristics, demodulating, and extracting artifacts such as
audio WAV files and JSON metadata.

## Features

- **Signal simulation**
  - Narrowband FM (NBFM) voice tone
  - 2-FSK digital packets
  - PTT burst gating (on/off cycles)
  - Channel impairments: CFO, sample-rate mismatch (ppm), IQ imbalance, DC offset, AWGN

- **Processing pipeline**
  - Burst detection via power envelope thresholding
  - Characterization: occupied bandwidth (Welch PSD), SNR estimate, CFO (Kay estimator)
  - Demodulation: NBFM → audio WAV; 2-FSK → bits → hex string
  - Artifact writers: WAV, JSON, IQ NumPy file
  - Optional spectrogram PNG visualization

- **CLI usage**

```bash
python sdr_ptt_pipeline.py --mode nbfm --snr 15 --fs 48000 --seconds 6 --outdir artifacts/session_0001
python sdr_ptt_pipeline.py --mode 2fsk --snr 12 --fs 48000 --seconds 6 --Rs 2400 --df 1200 --outdir artifacts/session_0002
```

## Installation

Clone this repo or download `sdr_ptt_pipeline.py`, then install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- soundfile (for WAV writing; will fall back to SciPy if not installed)
- matplotlib (optional, for spectrogram PNG)

## Output Artifacts

When you run the pipeline, the `--outdir` folder will contain:

- `iq.npy` — simulated complex IQ samples
- `spectrogram.png` — optional spectrogram with time-frequency view
- `burst_xxxx_audio.wav` — demodulated audio for each burst (NBFM mode)
- `burst_xxxx.json` — JSON metadata per burst (timestamps, SNR, OBW, modulation, etc.)
- `session_summary.json` — summary if no bursts were detected

## License

MIT License — do whatever you want, but attribution is appreciated.
