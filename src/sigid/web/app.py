"""Browser front end for SIG-ID: Full-featured RF Signal Intelligence & Demodulation Cyberdeck.
Exposes DSP analysis, real-time feature extraction, synthetic signal generation,
10-band equalization parameters, candidate search engine (Exhaustive + Adaptive),
and export tools over HTTP.
"""
import io
import json
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..io.loader import load
from ..dsp.analysis import (
    compute_fft,
    compute_spectrogram,
    estimate_bandwidth,
    estimate_snr,
    estimate_symbol_rate,
    extract_constellation,
    extract_eye_diagram,
    apply_dsp_filter,
)
from ..dsp.features import (
    extract_signal_features,
    classify_top_k_modulations,
    estimate_cfo_coarse,
    correct_cfo,
)
from ..engine.search import search_hypotheses, search_all_modulations, confidence_label
from ..engine.adaptive_search import search_adaptive
from ..synth.generator import GenParams, generate, save_iq, save_wav
from ..synth.dataset_b import DatasetBParams, generate_dataset_b, add_multipath

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODULATIONS = ["bpsk", "qpsk", "8psk", "16qam", "2fsk", "4fsk"]
FECS = ["viterbi", "reed_solomon", "concatenated", "ldpc"]
INTERLEAVERS = ["block", "convolutional", "diagonal", "pseudo_random"]
DEFAULT_IQ_FS = 200_000.0

app = FastAPI(title="SIG-ID Cyberdeck API")

_SESSIONS: dict[str, dict] = {}


def _downsample(arr: np.ndarray, max_points: int) -> np.ndarray:
    if len(arr) <= max_points:
        return arr
    step = len(arr) // max_points
    return arr[::step][:max_points]


def _analyze(iq: np.ndarray, fs: float, sps: int = 4) -> dict:
    n_preview = min(2000, len(iq))
    freqs, mag_db = compute_fft(iq, fs)
    freqs_ds, mag_ds = _downsample(freqs, 2000), _downsample(mag_db, 2000)

    sxx_freqs, times, sxx_db = compute_spectrogram(iq, fs)
    freq_step = max(1, len(sxx_freqs) // 200)
    time_step = max(1, len(times) // 200)
    sxx_ds = sxx_db[::freq_step, ::time_step]
    freqs_spec_ds = sxx_freqs[::freq_step]
    times_ds = times[::time_step]

    bw = estimate_bandwidth(iq, fs)
    snr = estimate_snr(iq, fs)
    symbol_rate = estimate_symbol_rate(iq, fs)
    points = extract_constellation(iq, fs, symbol_rate, max_symbols=2000)
    eye = extract_eye_diagram(iq, sps=max(2, sps))

    features = extract_signal_features(iq)
    top_mods = classify_top_k_modulations(features, k=6)
    cfo_est = estimate_cfo_coarse(iq, fs)

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
        "eye_diagram": eye,
        "params": {
            "bandwidth_hz": float(bw),
            "snr_db": float(snr),
            "symbol_rate_hz": float(symbol_rate),
        },
        "features": {
            "envelope_var": float(features["envelope_var"]),
            "C20": float(features["C20"]),
            "C40": float(features["C40"]),
            "C42": float(features["C42"]),
            "freq_kurtosis": float(features["freq_kurtosis"]),
            "cfo_estimated_hz": float(cfo_est),
            "top_modulations": [{"mod": m, "score": float(sc)} for m, sc in top_mods],
        },
    }


def _register_session(iq: np.ndarray, fs: float, defaults: dict, ground_truth: Optional[dict] = None) -> dict:
    session_id = uuid.uuid4().hex
    _SESSIONS[session_id] = {
        "iq": iq.copy(),
        "iq_raw": iq.copy(),
        "fs": fs,
        "defaults": defaults,
        "ground_truth": ground_truth,
        "last_search_results": None,
        "last_summary": None,
    }
    analysis = _analyze(iq, fs, sps=defaults.get("sps", 4))
    return {
        "session_id": session_id,
        "defaults": defaults,
        "ground_truth": ground_truth,
        **analysis,
    }


def _sidecar_defaults(sidecar: Path) -> dict:
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
        defaults = _sidecar_defaults(iq_path.with_suffix(".json"))
        gt = None
        json_path = iq_path.with_suffix(".json")
        if json_path.exists():
            try:
                gt = json.loads(json_path.read_text())
            except Exception:
                pass
        out.append({
            "name": iq_path.stem,
            "defaults": defaults,
            "ground_truth": gt,
        })
    return {"samples": out}


@app.post("/api/load_sample")
def load_sample(name: str = Form(...)):
    iq_path = DATA_DIR / f"{name}.iq"
    if not iq_path.exists():
        raise HTTPException(404, f"no such sample: {name}")
    defaults = _sidecar_defaults(iq_path.with_suffix(".json"))
    gt = None
    json_path = iq_path.with_suffix(".json")
    if json_path.exists():
        try:
            gt = json.loads(json_path.read_text())
        except Exception:
            pass
    signal = load(iq_path, fs=DEFAULT_IQ_FS)
    return _register_session(signal.iq, signal.fs, defaults, ground_truth=gt)


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


@app.post("/api/synthesize")
def synthesize(
    modulation: str = Form("qpsk"),
    fec: str = Form("viterbi"),
    interleaver: str = Form("block"),
    sps: int = Form(4),
    fs: float = Form(DEFAULT_IQ_FS),
    snr_db: float = Form(20.0),
    freq_offset_hz: float = Form(0.0),
    multipath_delay: int = Form(0),
    multipath_attenuation: float = Form(0.0),
    n_payload_bits: int = Form(512),
    custom_text: Optional[str] = Form(None),
    seed: int = Form(42),
):
    if modulation not in MODULATIONS:
        raise HTTPException(400, f"invalid modulation '{modulation}'")
    if fec not in FECS:
        raise HTTPException(400, f"invalid FEC '{fec}'")
    if interleaver not in INTERLEAVERS:
        raise HTTPException(400, f"invalid interleaver '{interleaver}'")

    if multipath_delay > 0 and multipath_attenuation > 0:
        params = DatasetBParams(
            n_payload_bits=n_payload_bits,
            modulation=modulation,
            fec=fec,
            interleaver=interleaver,
            sps=sps,
            fs=fs,
            snr_db=snr_db,
            freq_offset_hz=freq_offset_hz,
            multipath_delay_samples=multipath_delay,
            multipath_attenuation=multipath_attenuation,
            seed=seed,
            scenario_name=f"synth_{modulation}_{fec}_{interleaver}",
        )
        iq, gt = generate_dataset_b(params, custom_payload=custom_text)
    else:
        params = GenParams(
            n_payload_bits=n_payload_bits,
            modulation=modulation,
            fec=fec,
            interleaver=interleaver,
            sps=sps,
            fs=fs,
            snr_db=snr_db,
            freq_offset_hz=freq_offset_hz,
            seed=seed,
        )
        iq, gt = generate(params, custom_payload=custom_text)

    defaults = {
        "modulation": modulation,
        "sps": sps,
        "n_payload_bits": n_payload_bits,
    }
    return _register_session(iq, fs, defaults, ground_truth=gt)


@app.post("/api/filter")
def filter_signal(
    session_id: str = Form(...),
    filter_type: str = Form("bypass"),
    cutoff_hz: float = Form(20000.0),
    cutoff_high_hz: float = Form(60000.0),
    order: int = Form(4),
    eq_gains: Optional[str] = Form(None),
):
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session_id — load a signal first")

    iq_raw = session.get("iq_raw", session["iq"])
    fs = session["fs"]
    sps = session["defaults"].get("sps", 4)

    gains_list = None
    if eq_gains:
        try:
            gains_list = [float(x.strip()) for x in eq_gains.split(",") if x.strip()]
        except Exception:
            gains_list = None

    filtered_iq = apply_dsp_filter(
        iq=iq_raw,
        fs=fs,
        filter_type=filter_type,
        cutoff_hz=cutoff_hz,
        cutoff_high_hz=cutoff_high_hz,
        order=order,
        eq_gains=gains_list,
    )
    session["iq"] = filtered_iq
    analysis = _analyze(filtered_iq, fs, sps=sps)
    return {
        "session_id": session_id,
        "filter_type": filter_type,
        **analysis,
    }


@app.post("/api/reset_filter")
def reset_filter(session_id: str = Form(...)):
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session_id — load a signal first")
    if "iq_raw" in session:
        session["iq"] = session["iq_raw"].copy()
    fs = session["fs"]
    sps = session["defaults"].get("sps", 4)
    analysis = _analyze(session["iq"], fs, sps=sps)
    return {
        "session_id": session_id,
        "message": "Signal restored to original raw capture",
        **analysis,
    }


@app.post("/api/cfo_compensate")
def cfo_compensate(
    session_id: str = Form(...),
    cfo_hz: float = Form(...),
):
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session_id — load a signal first")

    iq_orig = session["iq"]
    fs = session["fs"]
    sps = session["defaults"].get("sps", 4)
    compensated_iq = correct_cfo(iq_orig, cfo_hz, fs)
    session["iq"] = compensated_iq

    analysis = _analyze(compensated_iq, fs, sps=sps)
    return {
        "session_id": session_id,
        "cfo_applied_hz": cfo_hz,
        **analysis,
    }


@app.post("/api/features")
def features_endpoint(session_id: str = Form(...)):
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session_id — load a signal first")
    iq, fs = session["iq"], session["fs"]
    feat = extract_signal_features(iq)
    top_mods = classify_top_k_modulations(feat, k=6)
    cfo = estimate_cfo_coarse(iq, fs)
    return {
        "features": {k: float(v) for k, v in feat.items()},
        "coarse_cfo_hz": float(cfo),
        "modulations_ranked": [{"mod": m, "score": float(sc)} for m, sc in top_mods],
    }


@app.post("/api/search")
def search(
    session_id: str = Form(...),
    modulation: str = Form(...),
    sps: int = Form(...),
    n_payload_bits: int = Form(...),
    mode: str = Form("exhaustive"),
):
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session_id — load a signal first")

    iq, fs = session["iq"], session["fs"]
    telemetry_json = None
    adaptive_summary_json = None

    if mode == "adaptive":
        results, summary, telemetry_list, adaptive_summary = search_adaptive(
            iq, sps, fs, n_payload_bits,
        )
        telemetry_json = [t.to_dict() for t in telemetry_list]
        adaptive_summary_json = adaptive_summary.to_dict()
    elif modulation == "auto":
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

    session["last_search_results"] = json_results
    session["last_summary"] = json_summary

    resp = {"results": json_results, "summary": json_summary}
    if telemetry_json is not None:
        resp["telemetry"] = telemetry_json
        resp["adaptive_summary"] = adaptive_summary_json
    return resp


@app.post("/api/batch_search")
def batch_search(mode: str = Form("adaptive")):
    if not DATA_DIR.exists():
        return {"results": []}

    records = []
    for iq_path in sorted(DATA_DIR.glob("*.iq")):
        stem = iq_path.stem
        defaults = _sidecar_defaults(iq_path.with_suffix(".json"))
        signal = load(iq_path, fs=DEFAULT_IQ_FS)
        sps = defaults.get("sps", 4)
        n_payload_bits = defaults.get("n_payload_bits", 256)

        if mode == "adaptive":
            results, summary, tel, ad_sum = search_adaptive(signal.iq, sps, signal.fs, n_payload_bits)
            red_pct = ad_sum.search_space_reduction_pct
            decodes = ad_sum.total_decodes_run
        else:
            results, summary = search_all_modulations(signal.iq, sps, signal.fs, n_payload_bits)
            red_pct = 0
            decodes = 96

        records.append({
            "sample": stem,
            "status": summary.get("status"),
            "confidence": confidence_label(summary),
            "winner_mod": summary.get("modulation"),
            "winner_fec": summary.get("fec"),
            "winner_interleaver": summary.get("interleaver"),
            "score": summary.get("score"),
            "decodes_run": decodes,
            "reduction_pct": red_pct,
        })
    return {"batch_results": records}


def _generate_html_dossier(session_id, fs, n_samples, analysis, summary, results, gt, hex_dump, ascii_preview):
    verdict = summary.get("status", "UNANALYZED").upper()
    mod = summary.get("modulation", "UNKNOWN").upper()
    fec = summary.get("fec", "UNKNOWN").upper()
    intl = summary.get("interleaver", "UNKNOWN").upper()
    score = summary.get("score", 0.0)
    confidence = summary.get("confidence", "N/A")
    params = analysis.get("params", {})
    features = analysis.get("features", {})
    top_mods = features.get("top_modulations", [])

    candidates_html = ""
    for idx, r in enumerate(results):
        crc_cls = "green" if r.get("crc_ok") else "dim"
        crc_text = "PASS (OK)" if r.get("crc_ok") else "FAIL"
        candidates_html += f"""
        <tr>
          <td>#{idx+1}</td>
          <td><b>{r.get('modulation','').upper()}</b></td>
          <td>{r.get('fec','').upper()}</td>
          <td>{r.get('interleaver','').upper()}</td>
          <td class="{crc_cls}">{crc_text}</td>
          <td>{r.get('correlation', 0.0):.4f}</td>
          <td>{r.get('ber', 1.0):.4f}</td>
          <td><b>{r.get('score', 0.0):.2f}</b></td>
        </tr>"""

    top_mods_html = "".join(f"<li><b>{m['mod'].upper()}</b>: Likelihood Score <code>{m['score']:.3f}</code></li>" for m in top_mods)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SIG-ID INTELLIGENCE DOSSIER // {session_id[:8]}</title>
<style>
  body {{
    background: #080c14;
    color: #cdd6e8;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    margin: 0;
    padding: 32px;
    line-height: 1.5;
  }}
  .dossier-card {{
    max-width: 900px;
    margin: 0 auto;
    background: #0e1626;
    border: 1px solid #1f3354;
    border-radius: 8px;
    padding: 32px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2px solid #38e1ff;
    padding-bottom: 16px;
    margin-bottom: 24px;
  }}
  .brand {{
    font-family: monospace;
    font-size: 24px;
    font-weight: 900;
    color: #38e1ff;
    letter-spacing: 2px;
  }}
  .badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1px;
    text-transform: uppercase;
  }}
  .badge-verified {{ background: rgba(0,255,136,0.2); color: #00ff88; border: 1px solid #00ff88; }}
  .badge-unverified {{ background: rgba(255,75,75,0.2); color: #ff4b4b; border: 1px solid #ff4b4b; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .panel {{
    background: #09101c;
    border: 1px solid #182842;
    border-radius: 6px;
    padding: 16px;
  }}
  .panel-title {{
    font-size: 13px;
    font-weight: 700;
    color: #ff9f1c;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 12px;
    border-bottom: 1px solid #182842;
    padding-bottom: 6px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #142236; }}
  th {{ color: #7f93b5; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }}
  .green {{ color: #00ff88; font-weight: bold; }}
  .dim {{ color: #5f7396; }}
  pre {{
    background: #05080e;
    border: 1px solid #182842;
    border-radius: 4px;
    padding: 12px;
    color: #38e1ff;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    overflow-x: auto;
    white-space: pre;
  }}
  .print-btn {{
    background: #38e1ff;
    color: #080c14;
    border: none;
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
  }}
  @media print {{
    body {{ background: #fff; color: #000; padding: 0; }}
    .dossier-card {{ border: none; box-shadow: none; max-width: 100%; padding: 0; }}
    .panel, pre {{ border-color: #ccc; background: #fafafa; color: #000; }}
    .header {{ border-bottom-color: #000; }}
    .print-btn {{ display: none; }}
  }}
</style>
</head>
<body>

<div class="dossier-card">
  <div class="header">
    <div>
      <div class="brand">SIG-ID // SIGNAL INTELLIGENCE DOSSIER</div>
      <div style="font-size:12px; color:#7f93b5; margin-top:4px;">NTRO SIH26147 &bull; TEAM VERTEX &bull; AUTOMATED RF IDENTIFICATION &amp; EXTRACTION REPORT</div>
    </div>
    <div style="text-align:right;">
      <button class="print-btn" onclick="window.print()">PRINT / PDF</button>
      <div style="margin-top:8px;">
        <span class="badge { 'badge-verified' if verdict == 'VERIFIED' else 'badge-unverified' }">{verdict} &bull; {confidence}</span>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="panel">
      <div class="panel-title">Physical RF Intercept Parameters</div>
      <table>
        <tr><th>Session Hash</th><td><code>{session_id[:16]}</code></td></tr>
        <tr><th>IQ Samples</th><td>{n_samples:,} samples</td></tr>
        <tr><th>Sampling Rate (Fs)</th><td>{fs:,.0f} Hz</td></tr>
        <tr><th>Est. Bandwidth</th><td>{params.get('bandwidth_hz',0)/1000:.2f} kHz</td></tr>
        <tr><th>Est. SNR</th><td><b>{params.get('snr_db',0):.1f} dB</b></td></tr>
        <tr><th>Est. Symbol Clock</th><td>{params.get('symbol_rate_hz',0)/1000:.2f} kSym/s</td></tr>
        <tr><th>Coarse Carrier CFO</th><td>{features.get('cfo_estimated_hz',0):+.1f} Hz</td></tr>
      </table>
    </div>

    <div class="panel">
      <div class="panel-title">Modulation &amp; Cumulant Fingerprint</div>
      <table>
        <tr><th>Cumulant C20</th><td><code>{features.get('C20',0):.5f}</code></td></tr>
        <tr><th>Cumulant C40</th><td><code>{features.get('C40',0):.5f}</code></td></tr>
        <tr><th>Cumulant C42</th><td><code>{features.get('C42',0):.5f}</code></td></tr>
        <tr><th>Envelope Variance</th><td><code>{features.get('envelope_var',0):.5f}</code></td></tr>
        <tr><th>Spectral Kurtosis</th><td><code>{features.get('freq_kurtosis',0):.5f}</code></td></tr>
      </table>
      <div style="margin-top:10px; font-size:12px; color:#7f93b5;">
        <b>Classified Candidates:</b>
        <ul style="margin:4px 0 0 16px; padding:0;">{top_mods_html}</ul>
      </div>
    </div>
  </div>

  <div class="panel" style="margin-bottom:24px;">
    <div class="panel-title">Identified Signal Hypothesis Verdict</div>
    <div style="display:flex; justify-content:space-around; text-align:center; padding:12px 0;">
      <div><div style="font-size:11px; color:#7f93b5;">MODULATION</div><div style="font-size:20px; font-weight:bold; color:#38e1ff;">{mod}</div></div>
      <div><div style="font-size:11px; color:#7f93b5;">FEC CODEC</div><div style="font-size:20px; font-weight:bold; color:#ff9f1c;">{fec}</div></div>
      <div><div style="font-size:11px; color:#7f93b5;">INTERLEAVER</div><div style="font-size:20px; font-weight:bold; color:#a370f7;">{intl}</div></div>
      <div><div style="font-size:11px; color:#7f93b5;">SCORE / VERDICT</div><div style="font-size:20px; font-weight:bold; color:{ '#00ff88' if verdict == 'VERIFIED' else '#ff4b4b' };">{score:.1f} / {verdict}</div></div>
    </div>
  </div>

  <div class="panel" style="margin-bottom:24px;">
    <div class="panel-title">Candidate Hypothesis Ground-Truth Evidence Table</div>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Modulation</th>
          <th>FEC</th>
          <th>Interleaver</th>
          <th>CRC-16</th>
          <th>Sync Corr</th>
          <th>Re-encode BER</th>
          <th>Composite Score</th>
        </tr>
      </thead>
      <tbody>
        {candidates_html if candidates_html else '<tr><td colspan="8" style="text-align:center;">No search executed yet</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="panel">
    <div class="panel-title">Extracted Bitstream // Hex Dump &amp; ASCII Payload</div>
    <pre>{hex_dump}</pre>
    <div style="margin-top:8px; font-size:12px;"><b>ASCII Text Preview:</b> <code>{ascii_preview}</code></div>
  </div>

  <div style="margin-top:20px; text-align:center; font-size:11px; color:#4a5d78;">
    SIG-ID AUTOMATED RF EXPLOITATION ENGINE &bull; GENERATED {time.strftime("%Y-%m-%d %H:%M:%S UTC")}
  </div>
</div>

</body>
</html>"""


@app.get("/api/export/{session_id}/{fmt}")
def export_file(session_id: str, fmt: str):
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")

    iq = session["iq"]
    fs = session["fs"]

    if fmt == "iq":
        interleaved = np.empty(2 * len(iq), dtype=np.float32)
        interleaved[0::2] = iq.real.astype(np.float32)
        interleaved[1::2] = iq.imag.astype(np.float32)
        return StreamingResponse(
            io.BytesIO(interleaved.tobytes()),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=signal_{session_id[:8]}.iq"},
        )
    elif fmt == "wav":
        buf = io.BytesIO()
        import wave
        peak = np.max(np.abs(iq)) or 1.0
        i16_i = (iq.real / peak * 32767).astype(np.int16)
        i16_q = (iq.imag / peak * 32767).astype(np.int16)
        stereo = np.empty(2 * len(iq), dtype=np.int16)
        stereo[0::2] = i16_i
        stereo[1::2] = i16_q
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(int(fs))
            wf.writeframes(stereo.tobytes())
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=signal_{session_id[:8]}.wav"},
        )
    elif fmt == "json":
        export_data = {
            "session_id": session_id,
            "fs": fs,
            "n_samples": len(iq),
            "defaults": session.get("defaults"),
            "ground_truth": session.get("ground_truth"),
            "summary": session.get("last_summary"),
            "top_candidates": (session.get("last_search_results") or [])[:10],
        }
        return JSONResponse(content=export_data)
    elif fmt == "csv":
        results = session.get("last_search_results") or []
        lines = ["rank,modulation,fec,interleaver,crc_ok,correlation,ber,score"]
        for idx, r in enumerate(results):
            lines.append(f"{idx+1},{r.get('modulation','')},{r.get('fec','')},{r.get('interleaver','')},{r.get('crc_ok',False)},{r.get('correlation',0.0):.3f},{r.get('ber',1.0):.3f},{r.get('score',0.0):.1f}")
        csv_str = "\n".join(lines)
        return StreamingResponse(
            io.StringIO(csv_str),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=report_{session_id[:8]}.csv"},
        )
    elif fmt in ("dossier", "html"):
        summary = session.get("last_summary") or {}
        results = session.get("last_search_results") or []
        gt = session.get("ground_truth") or {}
        analysis = _analyze(iq, fs, sps=session["defaults"].get("sps", 4))
        
        payload_bits_str = summary.get("payload_bits", "")
        hex_dump_lines = []
        ascii_str = ""
        if payload_bits_str:
            n_bytes = len(payload_bits_str) // 8
            byte_vals = [int(payload_bits_str[i*8:(i+1)*8], 2) for i in range(n_bytes)]
            for offset in range(0, len(byte_vals), 16):
                chunk = byte_vals[offset:offset+16]
                hex_part = " ".join(f"{b:02X}" for b in chunk).ljust(48)
                ascii_part = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
                hex_dump_lines.append(f"{offset:04X}  {hex_part}  |{ascii_part}|")
                ascii_str += ascii_part

        html_content = _generate_html_dossier(
            session_id=session_id,
            fs=fs,
            n_samples=len(iq),
            analysis=analysis,
            summary=summary,
            results=results[:12],
            gt=gt,
            hex_dump="\n".join(hex_dump_lines) if hex_dump_lines else "NO RECOVERED PAYLOAD DATA",
            ascii_preview=ascii_str or "N/A",
        )
        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f"inline; filename=dossier_{session_id[:8]}.html"},
        )
    else:
        raise HTTPException(400, f"unsupported export format '{fmt}'")


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
