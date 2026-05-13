"""
Build a single-file release archive of the assets/ directory for Google Drive upload.

Output:
  dist/skill-aligned-eval-assets.tar.gz
  dist/SHA256SUMS         (sha256 of the tarball)

The archive excludes assets/_annotators_map.local.json so the cleartext
annotator mapping never leaves your machine.

Usage:
    python scripts/build_release_archive.py
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
DIST_DIR = REPO_ROOT / "dist"
ARCHIVE_NAME = "skill-aligned-eval-assets.tar.gz"
ARCHIVE_PATH = DIST_DIR / ARCHIVE_NAME
SUMS_PATH = DIST_DIR / "SHA256SUMS"

EXCLUDE_NAMES = {
    "_annotators_map.local.json",
    "__pycache__",
}


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_NAMES for part in path.parts)


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not ASSETS_DIR.exists():
        sys.exit(f"missing {ASSETS_DIR}")

    DIST_DIR.mkdir(exist_ok=True)
    print(f"writing {ARCHIVE_PATH.relative_to(REPO_ROOT)} ...")

    with tarfile.open(ARCHIVE_PATH, "w:gz", compresslevel=6) as tar:
        for path in sorted(ASSETS_DIR.rglob("*")):
            if should_skip(path):
                continue
            arcname = "assets/" + str(path.relative_to(ASSETS_DIR)).replace("\\", "/")
            tar.add(path, arcname=arcname, recursive=False)

    size_mb = ARCHIVE_PATH.stat().st_size / (1024 * 1024)
    digest = sha256_of(ARCHIVE_PATH)
    SUMS_PATH.write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="utf-8")

    print(f"  size:   {size_mb:.1f} MB")
    print(f"  sha256: {digest}")
    print(f"  sums:   {SUMS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
