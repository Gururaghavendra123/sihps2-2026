# SIG-ID — SIH26147 (NTRO)

Automated analysis of `.iq` / `.wav` signal recordings: parameter extraction, demodulation,
de-interleaving, FEC decoding, and catalog-bounded automatic identification of modulation /
FEC / interleaver, with an evidence panel showing *why* — not a confidence percentage.

Two front ends, one real pipeline: a PyQt6 desktop app and a browser dashboard (FastAPI + vanilla JS).
See [`SIH26147_Final_Plan.md`](../SIH26147_Final_Plan.md) for the full design rationale, and the
web dashboard's landing page (`/`) for a plain-language walkthrough of what's built and what isn't.

## Setup

Requires Python 3.10+.

```bash
# 1. from the sigid/ folder — create and activate a virtual env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. editable install — code edits take effect immediately, no PYTHONPATH needed
pip install -e ".[gui,web,dev]"
```

`.[gui,web,dev]` pulls in PyQt6/pyqtgraph (desktop app), FastAPI/uvicorn/python-multipart (web app),
and pytest. Drop extras you don't need, e.g. `pip install -e ".[web,dev]"` to skip Qt.

### VS Code

Open the `sigid/` folder (not the repo root) directly in VS Code so it picks up `pyproject.toml`
and the `.venv` automatically. Select the venv's interpreter (`Ctrl+Shift+P` → *Python: Select
Interpreter* → `.venv`). The integrated terminal will then have `sigid` importable and `pytest`/
`uvicorn` on PATH.

## Running it

```bash
# tests (106 pass)
pytest

# desktop app
python -m sigid.gui.main_window

# web app — opens on http://127.0.0.1:8000
uvicorn sigid.web.app:app --reload
```

Either front end can load the sample files already in `data/` (via the "pick a sample" dropdown
in the web app, or the file picker in the desktop app), or generate a fresh one:

```bash
python gen_sample.py          # writes data/sample_qpsk_viterbi_block.{iq,wav,json}
```

To make one whose FEC/interleaver combo isn't in the samples already committed (useful for
proving the search engine actually searches, rather than eyeballing a known answer):

```python
from pathlib import Path
from sigid.synth.generator import GenParams, generate_and_save

params = GenParams(n_payload_bits=512, modulation="8psk", fec="ldpc", interleaver="diagonal", snr_db=18)
generate_and_save(params, Path("data"), "my_sample")
```

## Project layout

```
src/sigid/
  io/       .iq / .wav loading
  dsp/      FFT, PSD, spectrogram, bandwidth/SNR/symbol-rate estimation, constellation extraction
  synth/    synthetic signal generator (ground-truth-labeled test signals), CRC-16, interleavers
  demod/    timing recovery (M&M), carrier recovery (Costas / decision-directed), FSK discriminator
  fec/      Viterbi, Reed-Solomon, concatenated, LDPC — encoders + decoders
  engine/   Hypothesis Search Engine — search_hypotheses() (FEC x interleaver) and
            search_all_modulations() (+ modulation), CRC/correlation/BER scoring
  gui/      PyQt6 desktop dashboard
  web/      FastAPI backend + static browser dashboard (app.py, static/)
tests/      pytest suite, one file per build phase
data/       sample .iq/.wav files with ground-truth .json sidecars
```

## What's implemented vs. what's claimed

The web dashboard's landing page (`/`) is the canonical, up-to-date answer to "how much of the PS
does this solve" — PS alignment table, phase-by-phase plan-vs-achieved, and an explicit honest-
limitations section (catalog-bounded scope, structural-signal dependence, a documented 2-FSK/BPSK
identifiability edge case, 8-PSK/16-QAM SNR floor). Read that before this README goes stale.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'sigid'`** — you're not in the venv, or skipped
  `pip install -e .`. Re-activate the venv and re-run the install.
- **`ImportError` on PyQt6** — desktop app only; install with the `gui` extra, or just use the web
  app instead (`web` extra only).
- **Web app CDN assets (Chart.js, fonts) don't load** — the dashboard and landing page pull
  Chart.js and IBM Plex fonts from CDNs; needs internet access. Everything else (DSP, demod, FEC,
  search) runs fully offline.
