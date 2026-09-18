"""Analyst dashboard shell (Final Plan sec 4). Waveform, FFT, waterfall,
constellation, estimated params, evidence panel, and recovered bitstream
are all wired to real pipeline output — demod (phase 3) and the
Hypothesis Search Engine (phase 4) fill in what used to be mock text."""
import json
import sys
from pathlib import Path

import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sigid.io.loader import load
from sigid.dsp.analysis import (
    compute_fft,
    compute_spectrogram,
    estimate_bandwidth,
    estimate_snr,
    estimate_symbol_rate,
    extract_constellation,
)
from sigid.engine.search import search_hypotheses, search_all_modulations, confidence_label
from sigid.engine.adaptive_search import search_adaptive

MODULATIONS = ["bpsk", "qpsk", "8psk", "16qam", "2fsk", "4fsk"]
AUTO_MODULATION = "auto-detect (all 6)"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIG-ID — SIH26147 Analyst Dashboard")
        self.resize(1300, 900)
        self.loaded_signal = None
        self.fs = 200_000.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        toolbar = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load .iq / .wav")
        self.load_btn.clicked.connect(self.on_load_clicked)
        self.status_label = QtWidgets.QLabel("No file loaded")
        self.params_label = QtWidgets.QLabel("BW: -- | SNR: -- | Symbol rate: --")
        toolbar.addWidget(self.load_btn)
        toolbar.addWidget(self.status_label)
        toolbar.addStretch()
        toolbar.addWidget(self.params_label)
        root.addLayout(toolbar)

        search_bar = QtWidgets.QHBoxLayout()
        search_bar.addWidget(QtWidgets.QLabel("Engine:"))
        self.engine_combo = QtWidgets.QComboBox()
        self.engine_combo.addItems(["exhaustive", "adaptive"])
        self.engine_combo.setCurrentText("exhaustive")
        search_bar.addWidget(self.engine_combo)

        search_bar.addWidget(QtWidgets.QLabel("Modulation:"))
        self.mod_combo = QtWidgets.QComboBox()
        self.mod_combo.addItems([AUTO_MODULATION] + MODULATIONS)
        self.mod_combo.setCurrentText(AUTO_MODULATION)
        search_bar.addWidget(self.mod_combo)

        search_bar.addWidget(QtWidgets.QLabel("Samples/symbol:"))
        self.sps_spin = QtWidgets.QSpinBox()
        self.sps_spin.setRange(1, 64)
        self.sps_spin.setValue(4)
        search_bar.addWidget(self.sps_spin)

        search_bar.addWidget(QtWidgets.QLabel("Payload bits:"))
        self.payload_spin = QtWidgets.QSpinBox()
        self.payload_spin.setRange(8, 1_000_000)
        self.payload_spin.setValue(256)
        search_bar.addWidget(self.payload_spin)

        self.search_btn = QtWidgets.QPushButton("Run Hypothesis Search")
        self.search_btn.clicked.connect(self.on_search_clicked)
        search_bar.addWidget(self.search_btn)
        search_bar.addStretch()
        root.addLayout(search_bar)

        grid = QtWidgets.QGridLayout()
        root.addLayout(grid)

        self.waveform_plot = pg.PlotWidget(title="Waveform (I/Q, real DATA)")
        self.waveform_plot.setLabel("bottom", "Sample")
        self.fft_plot = pg.PlotWidget(title="Spectrum / FFT (real DATA)")
        self.fft_plot.setLabel("bottom", "Frequency", units="Hz")
        self.fft_plot.setLabel("left", "Magnitude", units="dB")

        self.waterfall_view = pg.PlotWidget(title="Waterfall / Spectrogram (real DATA)")
        self.waterfall_view.setLabel("bottom", "Time", units="s")
        self.waterfall_view.setLabel("left", "Frequency", units="Hz")
        self.waterfall_img = pg.ImageItem()
        self.waterfall_view.addItem(self.waterfall_img)
        self.waterfall_cmap = pg.colormap.get("viridis")
        self.waterfall_img.setColorMap(self.waterfall_cmap)

        self.constellation_plot = pg.PlotWidget(title="Constellation (real DATA, symbol-clock recovered)")
        self.constellation_plot.setAspectLocked(True)
        self.constellation_plot.setLabel("bottom", "I")
        self.constellation_plot.setLabel("left", "Q")
        self.constellation_scatter = pg.ScatterPlotItem(size=4, brush="c", pen=None)
        self.constellation_plot.addItem(self.constellation_scatter)

        self.evidence_box = QtWidgets.QTextEdit()
        self.evidence_box.setReadOnly(True)
        self.evidence_box.setPlainText(
            "EVIDENCE PANEL\n\n"
            "Load a signal, set modulation / samples-per-symbol / payload bits,\n"
            "then Run Hypothesis Search to score all 16 FEC x interleaver combos."
        )

        self.bitstream_box = QtWidgets.QTextEdit()
        self.bitstream_box.setReadOnly(True)
        self.bitstream_box.setPlainText("RECOVERED BITSTREAM — run a search to populate")

        grid.addWidget(self.waveform_plot, 0, 0)
        grid.addWidget(self.fft_plot, 0, 1)
        grid.addWidget(self.waterfall_view, 1, 0)
        grid.addWidget(self.constellation_plot, 1, 1)
        grid.addWidget(self.evidence_box, 2, 0)
        grid.addWidget(self.bitstream_box, 2, 1)

    def on_load_clicked(self):
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open signal file", str(Path(__file__).resolve().parents[3] / "data"),
            "Signal files (*.iq *.wav)"
        )
        if not path_str:
            return
        self.load_file(Path(path_str))

    def load_file(self, path: Path):
        fs = self.fs if path.suffix.lower() == ".iq" else None
        signal = load(path, fs=fs)
        self.loaded_signal = signal
        self.fs = signal.fs

        n_preview = min(2000, len(signal.iq))
        self.waveform_plot.clear()
        self.waveform_plot.plot(signal.iq.real[:n_preview], pen="c", name="I")
        self.waveform_plot.plot(signal.iq.imag[:n_preview], pen="m", name="Q")

        freqs, mag_db = compute_fft(signal.iq, signal.fs)
        self.fft_plot.clear()
        self.fft_plot.plot(freqs, mag_db, pen="y")

        sxx_freqs, times, sxx_db = compute_spectrogram(signal.iq, signal.fs)
        self.waterfall_img.setImage(sxx_db.T, autoLevels=True)
        if len(times) > 1 and len(sxx_freqs) > 1:
            self.waterfall_img.setRect(
                QtCore.QRectF(times[0], sxx_freqs[0], times[-1] - times[0], sxx_freqs[-1] - sxx_freqs[0])
            )

        bw = estimate_bandwidth(signal.iq, signal.fs)
        snr = estimate_snr(signal.iq, signal.fs)
        symbol_rate = estimate_symbol_rate(signal.iq, signal.fs)

        points = extract_constellation(signal.iq, signal.fs, symbol_rate, max_symbols=3000)
        self.constellation_scatter.setData(x=points.real.tolist(), y=points.imag.tolist())

        self.params_label.setText(
            f"BW: {bw / 1e3:.1f} kHz | SNR: {snr:.1f} dB | Symbol rate: {symbol_rate / 1e3:.2f} kSym/s"
        )
        self.status_label.setText(f"Loaded {path.name} — {len(signal.iq)} samples @ {signal.fs:.0f} Hz")

        # sps is derived from the just-estimated symbol rate (real DSP
        # output, above) — the same way a real receiver would set it.
        # Modulation is deliberately left on auto-detect, not prefilled
        # from the sidecar: that's the point of the search engine.
        # Payload-bits comes from the sidecar when present (frame length
        # is a legitimately "already known" protocol parameter). fec/
        # interleaver are never read from the sidecar — those stay
        # unknown-to-the-tool so Hypothesis Search has something to prove.
        if symbol_rate > 0:
            self.sps_spin.setValue(max(1, round(signal.fs / symbol_rate)))
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            try:
                gt = json.loads(sidecar.read_text())
                gp = gt.get("params", {})
                if "n_payload_bits" in gp:
                    self.payload_spin.setValue(int(gp["n_payload_bits"]))
            except (json.JSONDecodeError, OSError, KeyError, ValueError):
                pass

    def on_search_clicked(self):
        if self.loaded_signal is None:
            self.evidence_box.setPlainText("Load a signal first.")
            return

        scheme = self.mod_combo.currentText()
        auto = scheme == AUTO_MODULATION
        engine_mode = self.engine_combo.currentText()
        sps = self.sps_spin.value()
        n_payload_bits = self.payload_spin.value()
        fs = self.loaded_signal.fs

        self.search_btn.setEnabled(False)
        mode_label = engine_mode.upper()
        if engine_mode == "adaptive":
            self.status_label.setText(f"[{mode_label}] Searching with adaptive pruning...")
        elif auto:
            self.status_label.setText(f"[{mode_label}] Searching 6 modulations x 4 FEC x 4 interleavers (96 hypotheses)...")
        else:
            self.status_label.setText(f"[{mode_label}] Searching {scheme} x 4 FEC x 4 interleavers...")
        QtWidgets.QApplication.processEvents()

        telemetry_list = None
        adaptive_summary = None
        try:
            if engine_mode == "adaptive":
                results, summary, telemetry_list, adaptive_summary = search_adaptive(
                    self.loaded_signal.iq, sps, fs, n_payload_bits,
                )
            elif auto:
                results, summary = search_all_modulations(self.loaded_signal.iq, sps, fs, n_payload_bits)
            else:
                tone_spacing = fs / sps
                results, summary = search_hypotheses(
                    self.loaded_signal.iq, scheme, sps, fs, n_payload_bits, tone_spacing=tone_spacing,
                )
                for r in results:
                    r["modulation"] = scheme
                if summary.get("status") == "verified":
                    summary["modulation"] = scheme
        finally:
            self.search_btn.setEnabled(True)

        self._render_evidence(results, summary)
        if telemetry_list is not None and adaptive_summary is not None:
            self._append_telemetry(telemetry_list, adaptive_summary)
        self._render_bitstream(summary)
        self.status_label.setText(f"[{mode_label}] Search complete — {confidence_label(summary)}")

    def _render_evidence(self, results: list, summary: dict) -> None:
        label = confidence_label(summary)
        lines = [f"HYPOTHESIS SEARCH — {label}", ""]

        if not results:
            lines.append(summary.get("message", "sync word not found"))
            self.evidence_box.setPlainText("\n".join(lines))
            return

        for rank, r in enumerate(results[:5]):
            crc_mark = "✓" if r["crc_ok"] else "✗"
            corr_mark = "✓" if r["correlation"] >= 0.8 else "✗"
            ber_mark = "✓" if r["ber"] < 0.05 else "✗"
            tag = "  <- winner" if rank == 0 and label == "VERIFIED" else ""
            mod_tag = f"{r['modulation'].upper()} / " if r.get("modulation") else ""
            lines.append(
                f"{mod_tag}{r['interleaver'].upper()} INTERLEAVER + {r['fec'].upper():<14} SCORE {r['score']:.0f}{tag}"
            )
            lines.append(f"  {crc_mark} CRC-16: {'valid' if r['crc_ok'] else 'invalid'}")
            lines.append(f"  {corr_mark} Sync-word correlation: {r['correlation']:.2f}")
            lines.append(f"  {ber_mark} Re-encode BER: {r['ber']:.3f}")
            if r.get("error"):
                lines.append(f"  ! decode error: {r['error']}")
            lines.append("")

        self.evidence_box.setPlainText("\n".join(lines))

    def _append_telemetry(self, telemetry_list, adaptive_summary) -> None:
        """Append adaptive search telemetry to the evidence panel."""
        lines = ["\n─── ADAPTIVE TELEMETRY ───", ""]
        lines.append(f"Decodes run: {adaptive_summary.total_decodes_run} / {adaptive_summary.total_decodes_possible}")
        reduction = adaptive_summary.search_space_reduction_pct
        lines.append(f"Search space reduction: {reduction:.1f}%")
        lines.append(f"Early exit: {'YES ✔' if adaptive_summary.early_exit_triggered else 'no'}")
        if adaptive_summary.modulations_pruned_by_classifier:
            lines.append(f"Pruned by classifier: {', '.join(adaptive_summary.modulations_pruned_by_classifier)}")
        if adaptive_summary.modulations_pruned_by_sync:
            lines.append(f"Pruned by sync gate: {', '.join(adaptive_summary.modulations_pruned_by_sync)}")
        lines.append("")
        for t in telemetry_list:
            status = "✗ PRUNED" if t.sync_pruned else (
                "✓ WINNER (early exit)" if t.early_exit else f"✓ tested ({t.decodes_run} decodes)"
            )
            lines.append(f"{t.modulation.upper()} [rank {t.mod_rank}] {status}")
        current = self.evidence_box.toPlainText()
        self.evidence_box.setPlainText(current + "\n".join(lines))

    def _render_bitstream(self, summary: dict) -> None:
        if summary.get("status") != "verified":
            self.bitstream_box.setPlainText(
                f"NO VERIFIED CANDIDATE — {summary.get('message', summary.get('status', 'unknown'))}"
            )
            return

        payload = summary["payload"]
        preview_len = min(512, len(payload))
        bits_str = "".join(str(int(b)) for b in payload[:preview_len])
        suffix = "..." if len(payload) > preview_len else ""
        mod_line = f"Modulation: {summary['modulation']}   " if summary.get("modulation") else ""
        self.bitstream_box.setPlainText(
            f"RECOVERED PAYLOAD ({len(payload)} bits) — CRC VERIFIED\n"
            f"{mod_line}FEC: {summary['fec']}   Interleaver: {summary['interleaver']}   Score: {summary['score']:.0f}\n\n"
            f"{bits_str}{suffix}"
        )


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
