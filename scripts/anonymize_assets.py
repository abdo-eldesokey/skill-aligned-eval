"""
Anonymize annotator names in assets/ in place.

Two operations:
  1. JSON top-level keys under assets/annotations/ and assets/ai_answers/:
       Human annotators -> annotator_01..annotator_NN (alphabetical sort)
       chatgpt:<azure-deployment-id> -> llm_judge:<model-name>
       hf:<model> kept as-is (public model names)
  2. PNG brush-mask filenames named '<prompt_id>_<annotator>.png' under
     assets/annotations/<task>/artifact_mask/<model>/: the annotator suffix is
     remapped to the anonymized id. Files with non-annotator suffixes
     (e.g. '<prompt_id>_<model>.png' for LLM-generated masks) are left alone.

The cleartext mapping is written to assets/_annotators_map.local.json
(gitignored). Re-running on already-anonymized files is a no-op.

Usage:
    python scripts/anonymize_assets.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
ANNOTATIONS_DIR = ASSETS_DIR / "annotations"
AI_ANSWERS_DIR = ASSETS_DIR / "ai_answers"
ANNOTATORS_FILE = ASSETS_DIR / "annotators.json"
MAP_FILE = ASSETS_DIR / "_annotators_map.local.json"
LEGACY_MAP_FILE = REPO_ROOT / "release" / "_anonymization_map.json"

ANON_PREFIX = "annotator_"
LLM_PREFIX = "llm_judge:"

AZURE_DEPLOYMENT_RE = re.compile(
    r"^chatgpt:.*?-(gpt5mini|gpt5|gpt4\.1|gpt4omini|gpt4o)-\d{8}$"
)
DEPLOYMENT_TO_MODEL = {
    "gpt5": "gpt-5",
    "gpt5mini": "gpt-5-mini",
    "gpt4.1": "gpt-4.1",
    "gpt4omini": "gpt-4o-mini",
    "gpt4o": "gpt-4o",
}


def is_already_anon(name: str) -> bool:
    return (
        name.startswith(ANON_PREFIX)
        or name.startswith(LLM_PREFIX)
        or name.startswith("hf:")
    )


def llm_key_for(raw: str) -> str | None:
    """Map a chatgpt:<deployment> key to a llm_judge:<model> key, or None if unmappable."""
    m = AZURE_DEPLOYMENT_RE.match(raw)
    if not m:
        return None
    return LLM_PREFIX + DEPLOYMENT_TO_MODEL[m.group(1)]


def load_or_build_mapping() -> dict[str, str]:
    """Build the name->anon mapping. Reuse legacy release map if present."""
    if MAP_FILE.exists():
        return json.loads(MAP_FILE.read_text(encoding="utf-8"))

    mapping: dict[str, str] = {}
    if LEGACY_MAP_FILE.exists():
        legacy = json.loads(LEGACY_MAP_FILE.read_text(encoding="utf-8"))
        for k, v in legacy.items():
            if k == "chatgpt":
                continue
            mapping[k] = v
    else:
        if not ANNOTATORS_FILE.exists():
            sys.exit(f"missing {ANNOTATORS_FILE}")
        names = json.loads(ANNOTATORS_FILE.read_text(encoding="utf-8"))["annotators"]
        humans = sorted(n for n in names if n != "chatgpt")
        for i, name in enumerate(humans, start=1):
            mapping[name] = f"{ANON_PREFIX}{i:02d}"

    return mapping


def remap_keys(obj: dict, mapping: dict[str, str]) -> tuple[dict, int]:
    """Return a new dict with top-level keys remapped. Count keys actually changed."""
    out: dict = {}
    changed = 0
    for k, v in obj.items():
        if k in mapping:
            new_k = mapping[k]
            if new_k != k:
                changed += 1
            out[new_k] = v
        elif k.startswith("chatgpt:"):
            new_k = llm_key_for(k)
            if new_k is None:
                out[k] = v
                continue
            mapping[k] = new_k
            if new_k != k:
                changed += 1
            out[new_k] = v
        else:
            out[k] = v
    return out, changed


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def process_annotators_file(mapping: dict[str, str], dry_run: bool) -> bool:
    """Rewrite assets/annotators.json to use anon IDs. Returns True if changed."""
    data = json.loads(ANNOTATORS_FILE.read_text(encoding="utf-8"))
    names = data.get("annotators", [])
    new_names: list[str] = []
    for n in names:
        if is_already_anon(n):
            new_names.append(n)
        elif n == "chatgpt":
            new_names.append("llm_judge")
        elif n in mapping:
            new_names.append(mapping[n])
        else:
            new_names.append(n)
    if new_names == names:
        return False
    if not dry_run:
        write_atomic(
            ANNOTATORS_FILE,
            json.dumps({"annotators": new_names}, indent=2) + "\n",
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mapping = load_or_build_mapping()

    files_scanned = 0
    files_modified = 0
    keys_changed_total = 0

    for root in (ANNOTATIONS_DIR, AI_ANSWERS_DIR):
        if not root.exists():
            continue
        for json_path in root.rglob("*.json"):
            files_scanned += 1
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"  SKIP (invalid json): {json_path}: {e}", file=sys.stderr)
                continue
            if not isinstance(data, dict):
                continue
            new_data, changed = remap_keys(data, mapping)
            if changed > 0:
                files_modified += 1
                keys_changed_total += changed
                if not args.dry_run:
                    write_atomic(
                        json_path, json.dumps(new_data, indent=2) + "\n"
                    )

    annotators_changed = process_annotators_file(mapping, args.dry_run)

    pngs_renamed = rename_mask_files(mapping, args.dry_run)

    if not args.dry_run:
        write_atomic(
            MAP_FILE, json.dumps(mapping, indent=2, sort_keys=True) + "\n"
        )

    print(f"scanned:           {files_scanned}")
    print(f"modified:          {files_modified}")
    print(f"keys remapped:     {keys_changed_total}")
    print(f"annotators.json:   {'changed' if annotators_changed else 'unchanged'}")
    print(f"masks renamed:     {pngs_renamed}")
    print(f"mapping size:      {len(mapping)}")
    print(f"mapping written:   {MAP_FILE.relative_to(REPO_ROOT)}")
    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


PNG_NAME_RE = re.compile(r"^(\d+)_(.+)\.png$")


def rename_mask_files(mapping: dict[str, str], dry_run: bool) -> int:
    """Rename '<prompt_id>_<annotator>.png' brush masks to use anon ids.

    Only processes files whose suffix matches a known mapping key (i.e. a real
    human annotator name). Files with suffixes that aren't in the mapping
    (LLM-mask names like '<prompt_id>_flux2-dev.png') are left alone.
    """
    renamed = 0
    for png_path in ANNOTATIONS_DIR.rglob("*.png"):
        m = PNG_NAME_RE.match(png_path.name)
        if not m:
            continue
        prompt_id, suffix = m.group(1), m.group(2)
        if suffix not in mapping:
            continue
        new_name = f"{prompt_id}_{mapping[suffix]}.png"
        new_path = png_path.with_name(new_name)
        if new_path.exists():
            # Source carries the original cleartext annotator name and the
            # actual annotation work; target is most likely a blank auto-saved
            # by the app after the rename. Prefer the source.
            print(
                f"  OVERWRITE (target existed): {new_path}",
                file=sys.stderr,
            )
        if not dry_run:
            os.replace(png_path, new_path)
        renamed += 1
    return renamed


if __name__ == "__main__":
    raise SystemExit(main())
