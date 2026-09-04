from .conv import conv_encode, conv_decode
from .rs import rs_encode, rs_decode
from .concatenated import concat_encode, concat_decode
from .ldpc import ldpc_encode, ldpc_decode

FEC_ENCODERS = {
    "viterbi": conv_encode,
    "reed_solomon": rs_encode,
    "concatenated": concat_encode,
    "ldpc": ldpc_encode,
}

# Decoders take (coded_bits, n_message_bits, ...) -> recovered message bits.
# n_message_bits is known here (catalog-bounded fixed frame sizes, not
# blind decoding of an arbitrary unknown length) — see Final Plan sec 2.
FEC_DECODERS = {
    "viterbi": lambda bits, n: conv_decode(bits)[:n],
    "reed_solomon": rs_decode,
    "concatenated": concat_decode,
    "ldpc": ldpc_decode,
}
