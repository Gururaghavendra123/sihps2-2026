"""Headless verification: build the real GUI, load the real generated sample
file through it, render to PNG. Not a substitute for you running it yourself
(instructions at the bottom) — just proof the window builds and plots real data
without crashing."""
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from PyQt6 import QtWidgets
from sigid.gui.main_window import MainWindow

app = QtWidgets.QApplication(sys.argv)
win = MainWindow()
win.resize(1200, 800)
sample = Path(__file__).resolve().parent / "data" / "sample_qpsk_viterbi_block.iq"
win.fs = 200_000.0
win.load_file(sample)
win.show()
app.processEvents()

pixmap = win.grab()
out_path = Path(__file__).resolve().parent / "data" / "gui_phase1_preview.png"
pixmap.save(str(out_path))
print(f"saved {out_path}")
