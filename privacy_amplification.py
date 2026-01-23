import random


def toeplitz_hash(bits, out_len, seed=None):

    n = len(bits)
    if out_len < 0:
        raise ValueError("out_len must be non-negative")
    if out_len == 0 or n == 0:
        return []
    
    rng = random.Random(seed)
    length = n + out_len - 1
    
    seed_bits = [rng.randint(0, 1) for _ in range(length)]
    out = []

    for i in range(out_len):
        acc = 0
        for j in range(n):
            acc ^= bits[j] & seed_bits[n - 1 - j + i]
        out.append(acc)

    return out
