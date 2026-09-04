"""Browser front end for SIG-ID (phase 5): same real pipeline as the PyQt6
desktop app (gui/main_window.py) — IQ/WAV loading, DSP analysis, and the
Hypothesis Search Engine — exposed over HTTP so the dashboard can run in a
browser instead of (not replacing) the desktop app. No new signal-processing
logic lives here; this is a thin FastAPI wrapper around dsp/analysis.py and
engine/search.py, mirroring main_window.py's on_load_clicked/on_search_clicked
flow.
"""
import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles

from ..io.loader import load
from ..dsp.analysis import (
    compute_fft,
    compute_spectrogram,
    estimate_bandwidth,
    estimate_snr,
    estimate_symbol_rate,
    extract_constellation,
)
from ..engine.search import search_hypotheses, search_all_modulations, confidence_label

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODULATIONS = ["bpsk", "qpsk", "8psk", "16qam", "2fsk", "4fsk"]
DEFAULT_IQ_FS = 200_000.0

app = FastAPI(title="SIG-ID Web")

# In-memory per-session IQ store (single-analyst local tool — no need for
# a database; mirrors main_window.py holding one self.loaded_signal at a time,
# just keyed so multiple browser tabs/loads don't clobber each other).
_SESSIONS: dict[str, dict] = {}


def _downsample(arr: np.ndarray, max_points: int) -> np.ndarray:
    if len(arr) <= max_points:
        return arr
    step = len(arr) // max_points
    return arr[::step][:max_points]


def _analyze(iq: np.ndarray, fs: float) -> dict:
    """Same real DSP calls main_window.py's load_file() makes, arrays made
    JSON-safe and capped in size for the browser."""
    n_preview = min(2000, len(iq))
    freqs, mag_db = compute_fft(iq, fs)
    freqs_ds, mag_ds = _downsample(freqs, 2000), _downsample(mag_db, 2000)

    sxx_freqs, times, sxx_db = compute_spectrogram(iq, fs)
    # cap the heatmap grid so the JSON payload and canvas draw stay cheap
    freq_step = max(1, len(sxx_freqs) // 200)
    time_step = max(1, len(times) // 200)
    sxx_ds = sxx_db[::freq_step, ::time_step]
    freqs_spec_ds = sxx_freqs[::freq_step]
    times_ds = times[::time_step]

    bw = estimate_bandwidth(iq, fs)
    snr = estimate_snr(iq, fs)
    symbol_rate = estimate_symbol_rate(iq, fs)
    points = extract_constellation(iq, fs, symbol_rate, max_symbols=2000)

    return {
        "n_samples": int(len(iq)),
        "fs": fs,
        "waveform": {
            "i": iq.real[:n_preview].tolist(),
            "q": iq.imag[:n_preview].tolist(),
        },
        "fft": {"freqs": freqs_ds.tolist(), "mag_db": mag_ds.tolist()},
        "spectrogram": {
            "freqs": freqs_spec_ds.tolist(),
            "times": times_ds.tolist(),
            "sxx_db": sxx_ds.tolist(),
        },
        "constellation": {"i": points.real.tolist(), "q": points.imag.tolist()},
        "params": {"bandwidth_hz": bw, "snr_db": snr, "symbol_rate_hz": symbol_rate},
    }


def _register_session(iq: np.ndarray, fs: float, defaults: dict) -> dict:
    session_id = uuid.uuid4().hex
    _SESSIONS[session_id] = {"iq": iq, "fs": fs}
    return {"session_id": session_id, "defaults": defaults, **_analyze(iq, fs)}


def _sidecar_defaults(sidecar: Path) -> dict:
    """Modulation/sps/payload-bits only — deliberately never fec/interleaver
    (same rule as main_window.py's load_file): those stay unknown-to-the-tool
    so Hypothesis Search has something to prove, not just echo back."""
    defaults = {"modulation": "qpsk", "sps": 4, "n_payload_bits": 256}
    if sidecar.exists():
        try:
            gt = json.loads(sidecar.read_text())
            gp = gt.get("params", {})
            if gp.get("modulation") in MODULATIONS:
                defaults["modulation"] = gp["modulation"]
            if "sps" in gp:
                defaults["sps"] = int(gp["sps"])
            if "n_payload_bits" in gp:
                defaults["n_payload_bits"] = int(gp["n_payload_bits"])
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            pass
    return defaults


@app.get("/api/samples")
def list_samples():
    if not DATA_DIR.exists():
        return {"samples": []}
    out = []
    for iq_path in sorted(DATA_DIR.glob("*.iq")):
        out.append({"name": iq_path.stem, "defaults": _sidecar_defaults(iq_path.with_suffix(".json"))})
    return {"samples": out}


@app.post("/api/load_sample")
def load_sample(name: str = Form(...)):
    iq_path = DATA_DIR / f"{name}.iq"
    if not iq_path.exists():
        raise HTTPException(404, f"no such sample: {name}")
    defaults = _sidecar_defaults(iq_path.with_suffix(".json"))
    signal = load(iq_path, fs=DEFAULT_IQ_FS)
    return _register_session(signal.iq, signal.fs, defaults)


@app.post("/api/upload")
async def upload(file: UploadFile, fs: float = Form(DEFAULT_IQ_FS)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".iq", ".wav"):
        raise HTTPException(400, f"unsupported file extension: {suffix}")
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        signal = load(tmp_path, fs=fs if suffix == ".iq" else None)
    finally:
        tmp_path.unlink(missing_ok=True)
    return _register_session(signal.iq, signal.fs, {"modulation": "qpsk", "sps": 4, "n_payload_bits": 256})


@app.post("/api/search")
def search(
    session_id: str = Form(...),
    modulation: str = Form(...),  # "auto" searches modulation too, see search_all_modulations
    sps: int = Form(...),
    n_payload_bits: int = Form(...),
):
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session_id — load a signal first")

    iq, fs = session["iq"], session["fs"]
    if modulation == "auto":
        results, summary = search_all_modulations(iq, sps, fs, n_payload_bits)
    else:
        tone_spacing = fs / sps
        results, summary = search_hypotheses(iq, modulation, sps, fs, n_payload_bits, tone_spacing=tone_spacing)
        for r in results:
            r["modulation"] = modulation
        if summary.get("status") == "verified":
            summary["modulation"] = modulation

    json_results = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "payload"}
        entry["crc_ok"] = bool(entry["crc_ok"])
        json_results.append(entry)

    json_summary = dict(summary)
    if summary.get("status") == "verified":
        payload = summary["payload"]
        json_summary["payload_bits"] = "".join(str(int(b)) for b in payload)
        del json_summary["payload"]
    json_summary["confidence"] = confidence_label(summary)

    return {"results": json_results, "summary": json_summary}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
