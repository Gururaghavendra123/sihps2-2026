# SIG-ID — SIH26147 (NTRO)

**Team Vertex** — Smart India Hackathon, SIH26147 (National Technical Research Organisation)

Automated analysis of `.iq` / `.wav` signal recordings: parameter extraction, demodulation,
de-interleaving, FEC decoding, and catalog-bounded automatic identification of modulation, FEC,
and interleaver — with an evidence panel that shows *why* (CRC-16, sync-word correlation,
re-encode BER), not a confidence percentage pulled from nowhere.

For the full "what's built, how much of the PS is solved, what's left" writeup, see
**[DEEPDIVE.md](DEEPDIVE.md)**.

## What it does

Give it an unknown RF recording and it will:

1. Estimate bandwidth, SNR, and symbol rate directly from the capture
2. **Search all 6 candidate modulations** (BPSK, QPSK, 8PSK, 16-QAM, 2-FSK, 4-FSK) — not
   analyst-picked
3. Demodulate (Costas + M&M timing recovery for PSK/QAM, decision-directed loop for 16-QAM,
   frequency discriminator for FSK)
4. Try all 4 interleavers × all 4 FEC types (Viterbi, Reed-Solomon, concatenated, LDPC) — 96
   modulation × FEC × interleaver hypotheses per capture
5. Score every hypothesis by CRC-16 validity, sync-word correlation, and re-encode BER, and rank
   them
6. Recover the exact original payload bit-stream for the winning hypothesis, with evidence — or
   say plainly that nothing verified, rather than guess

Two front ends, one real pipeline underneath — nothing here is mocked:

- **Desktop app** (PyQt6 + pyqtgraph) — `python -m sigid.gui.main_window`
- **Web dashboard** (FastAPI + vanilla JS/Chart.js) — `uvicorn sigid.web.app:app --reload`, then
  open `http://127.0.0.1:8000`

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

`.[gui,web,dev]` pulls in PyQt6/pyqtgraph (desktop app), FastAPI/uvicorn/python-multipart (web
app), and pytest. Drop extras you don't need, e.g. `pip install -e ".[web,dev]"` to skip Qt.

### VS Code

Open the `sigid/` folder (not its parent) directly in VS Code so it picks up `pyproject.toml` and
the `.venv` automatically. Select the venv's interpreter (`Ctrl+Shift+P` → *Python: Select
Interpreter* → `.venv`). The integrated terminal will then have `sigid` importable and
`pytest`/`uvicorn` on PATH.

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

To make one whose modulation/FEC/interleaver combo isn't already committed in `data/` (useful for
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
  engine/   Hypothesis Search Engine — search_hypotheses() (FEC x interleaver, modulation given)
            and search_all_modulations() (+ modulation search too), CRC/correlation/BER scoring
  gui/      PyQt6 desktop dashboard
  web/      FastAPI backend + static browser dashboard (app.py, static/) — 90s Winamp-styled UI
tests/      pytest suite, one file per build phase (106 tests, all passing)
data/       sample .iq/.wav files with ground-truth .json sidecars
```

Module-by-module detail, the bugs found and fixed while building each one, and the full PS
alignment breakdown are in **[DEEPDIVE.md](DEEPDIVE.md)**.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'sigid'`** — you're not in the venv, or skipped
  `pip install -e .`. Re-activate the venv and re-run the install.
- **`ImportError` on PyQt6** — desktop app only; install with the `gui` extra, or just use the web
  app instead (`web` extra only).
- **Web app fonts/Chart.js don't load** — the dashboard and landing page pull Chart.js and the
  VT323 font from CDNs; needs internet access. Everything else (DSP, demod, FEC, search) runs
  fully offline.
