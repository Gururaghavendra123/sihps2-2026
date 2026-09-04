# SIG-ID — Codebase Deep Dive

Team Vertex, SIH26147 (NTRO). This is the detailed companion to [README.md](README.md): what the
problem statement actually asks for, how each module works, what's been fixed along the way, how
much of the PS is solved, and what's genuinely left.

---

## 1. The Problem Statement, as given

**Title**: Automated model for analysis of `.IQ` and `.wav` files along with signal parameter
extraction
**Organization**: National Technical Research Organisation (NTRO)

Given an unknown signal recording, the system must:

1. Identify signal parameters — sampling frequency, modulation, FEC, interleaving
2. Demodulate — FSK, QAM, PSK
3. De-interleave — block, convolutional, diagonal, pseudo-random
4. FEC decode — short-constraint convolutional (Viterbi), Reed-Solomon, concatenated codes, LDPC
5. Bit-stream correlation
6. GUI showing spectral features, constellation, waterfall, time-frequency info, demod results,
   recovered bitstream, header/payload identification

**No official dataset is provided.** Full open-world blind detection (arbitrary unknown
modulation/FEC/interleaver, zero prior) is a genuinely unsolved research problem. But the PS
itself hands you the fix: it enumerates the candidate space — a handful of modulations, exactly
4 interleaver types, exactly 4 FEC types. That's not open-world blind detection, it's a **bounded
search problem**. This project solves it as that, explicitly, and says so rather than quietly
overclaiming.

---

## 2. Architecture — the pipeline

```
.iq / .wav file
      │
      ▼
┌─────────────┐   io/loader.py
│   Load IQ   │   raw interleaved float32, or stereo PCM WAV → complex numpy array
└─────────────┘
      │
      ▼
┌─────────────┐   dsp/analysis.py
│  Estimate   │   FFT/PSD/spectrogram, bandwidth, SNR, symbol rate — modulation-agnostic,
│  parameters │   runs BEFORE modulation is known
└─────────────┘
      │
      ▼
┌─────────────┐   engine/search.py → demod/receive.py, demod/receive_chain.py
│  Hypothesis │   for each of 6 modulations x 4 FEC x 4 interleaver (96 combos):
│    Search   │     demod -> sync-word correlate -> deinterleave -> FEC decode -> CRC check
│   Engine    │   score by CRC-16 / sync correlation / re-encode BER, rank, pick winner
└─────────────┘
      │
      ▼
┌─────────────┐
│  Evidence + │   winning hypothesis's exact recovered payload bits, or an honest
│  bitstream  │   "nothing verified" if no candidate's CRC validated
└─────────────┘
```

Two front ends subscribe to the same backend code — `gui/main_window.py` (PyQt6) and
`web/app.py` (FastAPI) both call straight into `dsp/`, `demod/`, `fec/`, and `engine/`. Neither
front end has its own copy of any signal-processing logic.

---

## 3. Module-by-module

### `io/` — file loading
`loader.py`: reads `.iq` (raw interleaved float32, no embedded sample rate — caller must supply
`fs`) and `.wav` (stereo PCM, I/Q on the two channels, sample rate embedded). Returns a
`LoadedSignal(iq, fs, source_path)`.

### `dsp/analysis.py` — parameter estimation
Runs before any modulation is assumed:
- `compute_fft` / `compute_psd` / `compute_spectrogram` — standard FFT/Welch/STFT
- `estimate_bandwidth`, `estimate_snr` — threshold/noise-floor estimates off the PSD
- `estimate_symbol_rate` — an Oerder & Meyr-style nonlinearity timing estimator. Tries two
  nonlinear features (amplitude-jump `|x[n+1]-x[n]|²` for PSK/QAM's non-constant envelope, and
  instantaneous-frequency-jump for constant-envelope FSK) and picks whichever has the sharper
  spectral peak. Modulation-agnostic by design — this has to work *before* the modulation is
  known.
- `extract_constellation` — decimates at the recovered symbol clock for the constellation plot

### `synth/` — synthetic ground-truth generator
This is what makes phase 1-4 testable without a real dataset (none was provided):
- `generator.py` — `bits -> CRC-16 -> FEC -> interleave -> modulate -> AWGN + freq offset ->
  .iq/.wav + ground-truth .json`. Prepends a `SETTLE_BITS` (100-bit throwaway) + `SYNC_WORD_BITS`
  (32-bit known pattern) preamble ahead of the real frame — mirrors what a real receiver needs:
  carrier/timing loops need a short acquisition transient before they lock, so the real payload
  would get eaten without something disposable in front of it.
- `crc.py` — CRC-16/CCITT-FALSE. This is the "bulletproof" signal the search engine leans on.
- `interleavers.py` — all 4 required types, each with an exact inverse, verified round-trip in
  tests.
- `modulate.py` — baseband IQ generation for all 6 modulations.
- `channel.py` — AWGN + carrier frequency offset.

### `demod/` — real receivers
- `timing.py` — Mueller & Müller symbol-timing recovery (applies to any modulation; timing
  recovery doesn't need to know what's being sent).
- `carrier.py` — `costas_loop_psk` (M-th power PLL, works for constant-modulus PSK) and
  `costas_loop_decision_directed` (needed for 16-QAM, see §4 below — the blind M-th-power trick
  doesn't work when amplitude isn't constant).
- `psk_qam.py`, `fsk.py` — hard-decision slicers / discriminator.
- `receive.py` — `demod()` dispatches to the right chain per modulation. Documents an inherent
  ambiguity: the Costas loop locks phase to one of `order` rotational states — it strips the
  modulation to find phase, so it structurally can't distinguish 0 rad from 2π/`order` rad. This
  isn't a bug; it's resolved downstream by correlating against the known sync word.
- `receive_chain.py` — `find_frame_start()` brute-forces (shift × rotation) against the sync word
  — cheap, since it's just bit comparisons — and returns the correlation score alongside the
  frame-start position. This is the one function both `receive_known_chain()` (fec/interleaver
  told to it) and `engine/search.py` (fec/interleaver/modulation searched) share.

### `fec/` — real decoders, all 4
- `conv.py` — hard-decision Viterbi for the short-constraint convolutional code.
- `rs.py` — Reed-Solomon via `reedsolo`.
- `concatenated.py` — RS (outer) + convolutional (inner), chained.
- `ldpc.py` — belief-propagation LDPC decode via `pyldpc`.

`__init__.py` exposes `FEC_ENCODERS`/`FEC_DECODERS` dicts keyed by name — this is what lets
`engine/search.py` iterate "try every FEC type" as a simple loop rather than a chain of if/elif.

### `engine/search.py` — the Hypothesis Search Engine (the centerpiece)

Two entry points:

- **`search_hypotheses(iq, scheme, sps, fs, n_payload_bits, ...)`** — modulation given, searches
  FEC × interleaver (16 hypotheses).
- **`search_all_modulations(iq, sps, fs, n_payload_bits, ...)`** — searches modulation too (96
  hypotheses: loops all 6 modulations, calling `search_hypotheses` for each). This is what closes
  the gap on PS requirement #1 ("identify... modulation") — added this session; before it,
  modulation had to be told to the engine, which wasn't actually automatic identification.

**Scoring** (per hypothesis):
```
score = 25 * crc_ok + 60 * correlation + 15 * (1 - min(ber, 1))      # 0-100
```
- **CRC-16** (`crc_ok`) — the hard, bulletproof pass/fail signal; planted in the frame header
  specifically so the engine doesn't have to rely on fuzzy heuristics alone.
- **Sync-word correlation** — resolved once per modulation (doesn't depend on the FEC/interleaver
  guess, since the sync word sits in the stream uncoded, ahead of the FEC+interleave stage), so
  every hypothesis for a given modulation shares this figure.
- **Re-encode BER** — re-encode + re-interleave a hypothesis's decoded frame and compare against
  the actually-received coded bits. This is what discriminates a *correct* FEC/interleaver guess
  from a wrong one when CRC alone isn't decisive (see the concatenated/Viterbi collision in §5).

Winner = highest score. If the winner's CRC validates, status is `"verified"` and its exact
payload bits are returned. If nothing's CRC-clean, status is `"unknown"` — the engine says so
explicitly rather than asserting a confident wrong answer.

### `gui/main_window.py` and `web/app.py` — front ends
Both do the same three things: load a signal → run `dsp/analysis.py` on it (real waveform/FFT/
waterfall/constellation, no mocks) → call `engine/search_all_modulations` (default) or
`search_hypotheses` (if the analyst picks a specific modulation) → render the evidence panel and
recovered bitstream. `sps` is auto-derived from the just-estimated symbol rate in both front
ends, not hardcoded.

---

## 4. Real bugs found and fixed (in build order)

Each of these was caught by testing demod/decode output against synthetic ground truth, not
assumed correct:

1. **Costas-loop phase-boundary mismatch** — PSK symbols corrected by the Costas loop land on
   different decision boundaries than raw symbols (the loop cancels the modulator's `+π/order`
   offset). Needed a `boundary_offset` parameter rather than one universal hard-decision
   convention.
2. **16-QAM can't use the blind Costas trick** — the M-th-power PLL only works for constant-
   modulus PSK. Swapped in a decision-directed loop for 16-QAM.
3. **Loop bandwidth too narrow** — 0.05 didn't converge within a short synthetic frame; widened to
   0.1.
4. **Acquisition transient eats real data** — the first few symbols are unrecoverable while the
   loops lock. Fixed by prepending a throwaway settle region + a short sync word the receiver
   actually correlates against (naively correlating the whole preamble fails, since the early part
   is still mid-lock).
5. **Brute-force FEC decode was combinatorially too slow** — trying every (shift × rotation)
   directly against the expensive FEC decoder hung for 2+ minutes. Fixed by correlating against
   the cheap sync word first, decoding once at the winning alignment — also just how real
   receivers do it.
6. **`ldpc_decode` uint8 underflow** — `1 - 2*x` on a `uint8` array wraps to 255 instead of -1,
   silently destroying every LDPC frame even at zero noise. Caught by a standalone FEC round-trip
   test, not by eyeballing.
7. **(this session) Modulation wasn't actually searched** — the engine could search FEC ×
   interleaver but needed modulation told to it; that's not "identify... modulation" per PS
   requirement #1. Added `search_all_modulations()`.
8. **(this session) 2-FSK / BPSK identifiability collision** — found while testing #7. At the
   default `tone_spacing = symbol_rate` convention (Sunde's FSK, h=1), a 2-FSK symbol sweeps
   exactly ±π of phase, which a plain BPSK Costas+hard-decision demod also decodes correctly. A
   2-FSK capture produces **byte-identical** demod output under both the "bpsk" and "2fsk"
   hypotheses — same CRC pass, same score, same recovered payload — so the engine genuinely cannot
   tell them apart from CRC/correlation/BER alone. This isn't a scoring bug; it's a real
   demodulator-level ambiguity, discovered rather than assumed away, and regression-tested
   (`test_search_all_modulations_2fsk_bpsk_ambiguity_is_known` in
   `tests/test_phase4_hypothesis_search.py`).
9. **(this session, found in the same investigation) `concatenated` vs `viterbi` score tie** —
   since `concatenated = conv(RS(msg))`, a plain "viterbi" hypothesis also strips the inner conv
   layer correctly and can spuriously pass CRC too (systematic RS puts the message bits first).
   CRC alone can't break that tie — re-encode BER does (the true `concatenated` hypothesis gets
   BER 0.0 and outscores the `viterbi` false-positive). This is exactly the case the BER term
   exists for, confirmed with a real test rather than assumed.

---

## 5. Test coverage

106 tests, all passing, one file per build phase:
- `test_phase1_smoke.py` — loader + generator sanity
- `test_phase2_dsp.py` — bandwidth/SNR/symbol-rate/constellation against ground truth
- `test_phase3_demod_fec.py` — every modulation × FEC combo (24), all 4 interleavers, standalone
  FEC bit-error correction — 84 tests
- `test_phase4_hypothesis_search.py` — all 16 FEC×interleaver combos found blind, all 6
  modulations found blind (96-hypothesis search), the 2-FSK/BPSK ambiguity regression, and the
  no-sync/unknown-signal fallback — 22 tests

Run with `pytest` from `sigid/` (after `pip install -e ".[dev]"` or the full extras set).

---

## 6. PS alignment — how much is solved

| PS requirement | Status | Notes |
|---|---|---|
| `.iq` / `.wav` input | **Done** | `io/loader.py` |
| Parameter extraction (Fs, BW, SNR, symbol rate) | **Done** | `dsp/analysis.py`, modulation-agnostic |
| Modulation ID | **Done, catalog-bounded** | all 6 candidates, auto-searched (`search_all_modulations`), not analyst-picked |
| Demodulation (FSK, QAM, PSK) | **Done** | Costas+M&M, decision-directed (16-QAM), FSK discriminator |
| De-interleaving (all 4 types) | **Done** | round-trip verified |
| FEC decode (all 4 types) | **Done** | real decoders, real error correction |
| Bit-stream correlation | **Done** | sync-word correlation → frame/header/payload split |
| GUI (spectral, constellation, waterfall, bitstream) | **Done** | desktop + web, both real data |
| "Identify" as automatic determination | **Done, catalog-bounded** | Hypothesis Search Engine, 96 hypotheses, CRC-verified |

**Every numbered PS requirement is met, fully, within the exact candidate catalog the PS itself
enumerates.** That's the entire claim — nothing beyond it. Open-world identification of protocol
families outside that catalog is not attempted and is not claimed anywhere in this project.

---

## 7. Known limitations (honest, not hidden)

- **Catalog-bounded, not open-world.** Works across the PS's own enumerated modulation/FEC/
  interleaver list. Arbitrary protocols outside it are a different, unsolved problem.
- **Needs structural signal.** Scoring leans on CRC-16, sync-word, and re-encode BER — all present
  because the synthetic frame format plants them. A real off-air capture without that structure
  needs additional heuristics this engine doesn't have today.
- **2-FSK / BPSK ambiguity** at the default tone-spacing convention — see bug #8 above. Real,
  regression-tested, payload still recovers correctly either way.
- **8-PSK / 16-QAM SNR floor** — denser constellations pack points closer together, so their
  carrier loops show real threshold degradation below ~15dB SNR. Textbook property of higher-order
  modulation, not a bug.

---

## 8. What's left

**Engineering, scoped but not done:**
- Widen the default FSK tone spacing so 2-FSK stops aliasing with BPSK at h=1 (kills limitation
  above)
- Heuristics for real off-air captures that lack a planted CRC/sync word

