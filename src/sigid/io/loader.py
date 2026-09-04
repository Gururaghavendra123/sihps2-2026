"""Load .iq (raw interleaved float32 I/Q) and .wav (stereo I/Q) files back
into a complex numpy array, mirroring the format `synth/generator.py` writes."""
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LoadedSignal:
    iq: np.ndarray
    fs: float
    source_path: str


def load_iq(path: Path, fs: float) -> LoadedSignal:
    raw = np.fromfile(path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]
    return LoadedSignal(iq=iq, fs=fs, source_path=str(path))


def load_wav(path: Path) -> LoadedSignal:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        fs = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"only 16-bit PCM wav supported, got {sampwidth * 8}-bit")

    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

    if n_channels == 2:
        iq = data[0::2] + 1j * data[1::2]
    elif n_channels == 1:
        iq = data.astype(np.complex64)
    else:
        raise ValueError(f"unsupported channel count: {n_channels}")

    return LoadedSignal(iq=iq, fs=float(fs), source_path=str(path))


def load(path: Path, fs: float | None = None) -> LoadedSignal:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return load_wav(path)
    if path.suffix.lower() == ".iq":
        if fs is None:
            raise ValueError(".iq files carry no sample-rate metadata — pass fs explicitly")
        return load_iq(path, fs)
    raise ValueError(f"unsupported file extension: {path.suffix}")
