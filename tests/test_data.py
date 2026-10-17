"""The data files in data/ are the published ones.

data/SHA256SUMS lists every file the notebooks read with its SHA-256 hash
(`shasum -a 256 -c SHA256SUMS` inside data/ checks the same). Skipped when
the data have not been downloaded.
"""

from __future__ import annotations

import hashlib

import pytest

from qising_networks import DATA_DIR

MANIFEST = DATA_DIR / "SHA256SUMS"


def manifest() -> list[tuple[str, str]]:
    if not MANIFEST.exists():
        return []
    return [
        tuple(line.split())
        for line in MANIFEST.read_text().splitlines()
        if line.strip()
    ]


@pytest.mark.skipif(not MANIFEST.exists(), reason="data/ not downloaded")
@pytest.mark.parametrize("digest, name", manifest(), ids=lambda v: v[:16])
def test_data_file_matches_manifest(digest: str, name: str) -> None:
    h = hashlib.sha256()
    with open(DATA_DIR / name, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == digest, name
