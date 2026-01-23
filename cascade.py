import random


def _parity(bits, indices):
    return sum(bits[i] for i in indices) % 2


def _binary_search_error(bits_a, bits_b, indices, stats):
    if len(indices) == 1:
        return indices[0]

    mid = len(indices) // 2
    left = indices[:mid]
    right = indices[mid:]

    stats["parity_checks"] += 1
    if _parity(bits_a, left) != _parity(bits_b, left):
        return _binary_search_error(bits_a, bits_b, left, stats)

    stats["parity_checks"] += 1
    return _binary_search_error(bits_a, bits_b, right, stats)


def cascade_reconcile(
    alice_key,
    bob_key,
    block_size=16,
    rounds=4,
    seed=None,
    final_single_bit_pass=False,
):

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    rng = random.Random(seed)

    n = min(len(alice_key), len(bob_key))
    a_key = list(alice_key[:n])
    b_key = list(bob_key[:n])

    stats = {
        "parity_checks": 0,
        "corrections": 0,
        "rounds": rounds,
        "initial_block_size": block_size,
    }

    for r in range(rounds):
        size = block_size * (2 ** r)
        indices = list(range(n))
        rng.shuffle(indices)

        for start in range(0, n, size):
            block = indices[start:start + size]
            if not block:
                continue

            stats["parity_checks"] += 1
            if _parity(a_key, block) != _parity(b_key, block):
                error_idx = _binary_search_error(a_key, b_key, block, stats)
                b_key[error_idx] ^= 1
                stats["corrections"] += 1

    if final_single_bit_pass:
        for i in range(n):
            stats["parity_checks"] += 1
            if a_key[i] != b_key[i]:
                b_key[i] ^= 1
                stats["corrections"] += 1

    return b_key, stats
