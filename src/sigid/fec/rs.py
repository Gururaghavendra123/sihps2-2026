"""Reed-Solomon via `reedsolo` (RS(255,223)-style, byte-oriented)."""
import numpy as np
import reedsolo

ECC_BYTES = 32  # RS(255,223): corrects up to 16 byte errors


def rs_encode(bits: np.ndarray, ecc_bytes: int = ECC_BYTES) -> np.ndarray:
    pad = (-len(bits)) % 8
    padded = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    payload = np.packbits(padded.astype(np.uint8)).tobytes()
    codec = reedsolo.RSCodec(ecc_bytes)
    encoded = codec.encode(payload)
    return np.unpackbits(np.frombuffer(bytes(encoded), dtype=np.uint8))


def rs_decode(coded_bits: np.ndarray, n_message_bits: int, ecc_bytes: int = ECC_BYTES) -> np.ndarray:
    """Correct byte errors and recover the original message bits.
    `n_message_bits` is the pre-padding length `rs_encode` was given —
    known here because this is a catalog-bounded search (fixed frame sizes),
    not blind decoding of an arbitrary unknown length."""
    coded_bytes = np.packbits(coded_bits.astype(np.uint8)).tobytes()
    codec = reedsolo.RSCodec(ecc_bytes)
    decoded_bytes, _, _ = codec.decode(coded_bytes)
    bits = np.unpackbits(np.frombuffer(bytes(decoded_bytes), dtype=np.uint8))
    return bits[:n_message_bits]
