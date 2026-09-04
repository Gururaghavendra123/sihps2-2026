"""Concatenated code: RS (outer) + convolutional (inner) — pure composition
of the two decoders already implemented, per Final Plan sec 6.
Encode order: RS first, then conv wraps it. Decode reverses: conv first
(strips the inner code), then RS (corrects remaining byte errors)."""
import numpy as np
from .rs import rs_encode, rs_decode
from .conv import conv_encode, conv_decode


def concat_encode(bits: np.ndarray) -> np.ndarray:
    return conv_encode(rs_encode(bits))


def concat_decode(coded_bits: np.ndarray, n_message_bits: int, ecc_bytes: int = 32) -> np.ndarray:
    n_payload_bytes = -(-n_message_bits // 8)  # ceil to bytes
    n_rs_bits = (n_payload_bytes + ecc_bytes) * 8
    rs_bits = conv_decode(coded_bits)[:n_rs_bits]
    return rs_decode(rs_bits, n_message_bits, ecc_bytes)
