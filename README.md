# SIG-ID — SIH26147 (NTRO)

**Team Vertex** — Smart India Hackathon, SIH26147 (National Technical Research Organisation)

Automated analysis of `.iq` / `.wav` signal recordings: parameter estimation, demodulation, de-interleaving, FEC decoding, and catalog-bounded automatic identification of modulation, FEC, and interleaver — verified by hard ground truth (CRC-16, sync-word correlation, re-encode BER) rather than arbitrary heuristic percentages.

For the detailed module breakdown, mathematical formulation, and architecture analysis, see **[DEEPDIVE.md](DEEPDIVE.md)**.

---

## What It Does

Give it an unknown RF capture and it will:

1. **Estimate physical parameters** directly from raw IQ: bandwidth, SNR, and symbol rate via Oerder & Meyr non-linear spectral peak detection (modulation-agnostic).
2. **Execute Dual-Engine Search**:
   - **Adaptive Candidate Search Engine (Default)**: A 4-stage coarse-to-fine pruning pipeline that uses coarse CFO compensation, higher-order cumulants ($C_{20}, C_{40}, C_{42}$), sync-word gatekeeper filtering, and priority-queue FEC decoding with early exit. Achieves **70%–90% reduction in decode iterations** and a **4×–5× speedup** while guaranteeing bit-identical payload recovery.
   - **Exhaustive Engine**: Brute-force evaluates all 96 candidate hypotheses (6 modulations × 4 FECs × 4 interleavers) as a ground-truth baseline.
3. **Demodulate full candidate catalog**:
   - **PSK / QAM**: BPSK, QPSK, 8-PSK, 16-QAM via Mueller & Müller timing recovery, Costas loop (PSK), and decision-directed tracking (16-QAM).
   - **FSK**: 2-FSK, 4-FSK via continuous-phase frequency discriminator.
4. **De-interleave & Decode all standard configurations**:
   - **Interleavers**: Block, Convolutional, Diagonal, Pseudo-Random.
   - **FEC**: Viterbi (constraint $K=7$), Reed-Solomon ($RS(255, 223)$), Concatenated ($RS + Conv$), and LDPC (Belief Propagation via Tanner graph).
5. **Verify and extract payload bitstream**: Scores candidates by CRC-16 validity, sync-word correlation, and re-encode BER. If verified, extracts the recovered bitstream; if unverified, flags the signal as unknown rather than returning false positives.

Two front ends, one shared core engine:
- **Desktop App** (PyQt6 + pyqtgraph) — `python -m sigid.gui.main_window`
- **Web Dashboard** (FastAPI + Vanilla JS/Chart.js) — `uvicorn sigid.web.app:app --reload` (open `http://127.0.0.1:8000`)

---

## Setup

Requires Python 3.10+.

```bash
# 1. From the sigid/ directory — create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Editable install with full dependencies
pip install -e ".[gui,web,dev]"
```

`.[gui,web,dev]` installs PyQt6/pyqtgraph (desktop GUI), FastAPI/uvicorn/python-multipart (web API), and pytest. Use `.[web,dev]` if running in headless environments without Qt.

### VS Code Note
Open the `sigid/` subfolder directly in VS Code so it detects `pyproject.toml` and activates `.venv` automatically.

---

## Running It

```bash
# Run complete test suite (132 tests pass)
pytest

# Launch desktop GUI
python -m sigid.gui.main_window

# Launch web server (http://127.0.0.1:8000)
uvicorn sigid.web.app:app --reload
```

### Generating Test Signals

Generate single Dataset A samples:
```bash
python gen_sample.py
```

Generate the 4 Dataset B benchmark scenarios with multipath fading and carrier frequency offset:
```bash
python gen_dataset_b_samples.py
```
This produces:
- `data/dataset_b_qpsk_viterbi_block_cfo.{iq,wav,json}` (AWGN + 150 Hz CFO)
- `data/dataset_b_8psk_rs_diagonal_phase.{iq,wav,json}` (Phase offset + 80 Hz CFO)
- `data/dataset_b_16qam_ldpc_pseudo_random.{iq,wav,json}` (Multipath fading + 50 Hz CFO)
- `data/dataset_b_2fsk_concat_conv.{iq,wav,json}` (Continuous phase + 120 Hz CFO)

---

## Architecture & Project Layout

```
src/sigid/
  io/       .iq (raw float32 interleaved) and .wav (stereo I/Q) loaders
  dsp/      FFT, PSD, spectrogram, bandwidth/SNR/symbol-rate estimators, constellation,
            features.py (cumulants C20/C40/C42, envelope variance, kurtosis, coarse CFO)
  synth/    Signal generation, CRC-16, interleavers, modulation, AWGN/multipath/CFO channels,
            dataset_b.py (Dataset B benchmark generator with intermediate stage logging)
  demod/    Mueller & Müller timing, Costas & decision-directed carrier loops, FSK discriminator,
            find_frame_start (shift & rotation sync correlation)
  fec/      Viterbi, Reed-Solomon, Concatenated (RS + Conv), and LDPC encoders & decoders
  engine/   search.py (Exhaustive 96-hypothesis search),
            adaptive_search.py (4-stage coarse-to-fine pruning engine),
            candidate.py (Telemetry and pruning statistics dataclasses)
  gui/      PyQt6 desktop dashboard with live waveform, spectrum, waterfall, constellation,
            hypothesis evidence table, and adaptive telemetry panel
  web/      FastAPI backend (app.py) + retro dashboard (static/dashboard.html, app.js)
tests/      Complete pytest test suite (132 passed tests covering DSP, Demod, FEC,
            Exhaustive Search, Adaptive Search, and Dataset B benchmarks)
data/       Sample .iq/.wav signals with ground-truth .json sidecars
```

---

## Test Verification

```bash
pytest -v
```
- `test_phase1_smoke.py`: Loader and generator round-trips
- `test_phase2_dsp.py`: Spectral parameter estimation against known ground truth
- `test_phase3_demod_fec.py`: Demodulation and FEC error correction across all 24 scheme pairs
- `test_phase4_hypothesis_search.py`: Exhaustive 96-hypothesis blind search
- `test_adaptive_search.py`: Adaptive feature extraction, coarse CFO compensation, modulation pre-classification, sync gatekeeper pruning, priority queue early-exit, and Dataset B scenarios
