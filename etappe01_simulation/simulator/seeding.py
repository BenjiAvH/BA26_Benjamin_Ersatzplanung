from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, *parts: str) -> int:
    """
    Deterministische Seed-Ableitung (Reproduzierbarkeit).
    Rückgabe ist ein 32-bit Seed für random.Random.
    """

    payload = f"{base_seed}|" + "|".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)

