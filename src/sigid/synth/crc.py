"""CRC-16/CCITT-FALSE — the hard pass/fail signal the Hypothesis Search Engine
scores against (see Final Plan sec 5)."""
import numpy as np


def bits_to_bytes(bits: np.ndarray) -> bytes:
    n = len(bits) - (len(bits) % 8)
    bits = bits[:n]
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def append_crc16(payload_bits: np.ndarray) -> np.ndarray:
    payload_bytes = bits_to_bytes(payload_bits)
    crc = crc16_ccitt(payload_bytes)
    crc_bits = bytes_to_bits(crc.to_bytes(2, "big"))
    return np.concatenate([payload_bits, crc_bits])


def verify_crc16(frame_bits: np.ndarray) -> bool:
    """frame_bits = payload bits + trailing 16 CRC bits. Returns True if valid."""
    if len(frame_bits) < 16:
        return False
    n_payload = len(frame_bits) - 16
    payload_bits = frame_bits[:n_payload]
    crc_bits = frame_bits[n_payload:]
    payload_bytes = bits_to_bytes(payload_bits)
    expected = crc16_ccitt(payload_bytes)
    received_bytes = bits_to_bytes(crc_bits)
    if len(received_bytes) < 2:
        return False
    received = int.from_bytes(received_bytes[:2], "big")
    return expected == received
