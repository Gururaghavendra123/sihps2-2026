"""LDPC via `pyldpc`. Regular (96,50) code — d_v=3, d_c=6, rate ~0.52.
Fixed system matrix (module-level, seeded) so encode/decode share the same
H/G across calls."""
import numpy as np
import pyldpc

N_CODE = 96
D_V = 3
D_C = 6
SEED = 0

_H, _G = pyldpc.make_ldpc(N_CODE, D_V, D_C, systematic=True, sparse=False, seed=SEED)
K_MESSAGE = _G.shape[1]  # 50


def ldpc_encode(bits: np.ndarray) -> np.ndarray:
    pad = (-len(bits)) % K_MESSAGE
    padded = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    blocks = padded.reshape(-1, K_MESSAGE)
    codewords = [pyldpc.utils.binaryproduct(_G, block) for block in blocks]
    return np.concatenate(codewords).astype(np.uint8)


def ldpc_decode(coded_bits: np.ndarray, n_message_bits: int, snr_db: float = 5.0, maxiter: int = 50) -> np.ndarray:
    """Belief-propagation decode. Input is hard 0/1 bits (post-demod hard
    decision) mapped to bipolar +-1 — `snr_db` controls how strongly the BP
    solver trusts that input while resolving parity-check conflicts; it is
    a decode-time tuning knob, not a measurement of the real channel SNR."""
    n_blocks = -(-len(coded_bits) // N_CODE)
    pad = n_blocks * N_CODE - len(coded_bits)
    padded = np.concatenate([coded_bits, np.zeros(pad, dtype=coded_bits.dtype)])
    blocks = padded.reshape(-1, N_CODE)

    messages = []
    for block in blocks:
        # block is uint8 — must widen before `1 - 2*block` or bit=1 wraps
        # to 255 instead of -1 (unsigned integer underflow)
        y = 1.0 - 2.0 * block.astype(np.float64)
        x_hat = pyldpc.decode(_H, y, snr=snr_db, maxiter=maxiter)
        msg = pyldpc.get_message(_G, x_hat)
        messages.append(np.asarray(msg).ravel())

    bits = np.concatenate(messages).astype(np.uint8)
    return bits[:n_message_bits]
