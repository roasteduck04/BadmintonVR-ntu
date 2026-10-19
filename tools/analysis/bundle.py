"""One file per playback: fold an analysis doc's Unity sidecars into the doc itself and move the rest out of the
Unity folder, so StreamingAssets/analysis holds exactly one file per clip.

    PYTHONPATH=tools python -m analysis.bundle                         # every *.analysis.json in the Unity folder
    PYTHONPATH=tools python -m analysis.bundle --clip singles_05x_pose24 --dry-run
    PYTHONPATH=tools python -m analysis.bundle --unbundle --clip singles_05x_pose24   # put the sidecars back

- `<file>.cvdebug.json` (cvdebug/1, the only sidecar a pose-only playback reads) is embedded as the doc's top-level
  "cvdebug" object when it pairs with the doc (same build id / clip / frame count); the viewer reads the embedded block
  first and still falls back to a sidecar file for old docs.
- `<clip>.persons.json` (Python only: cvdebug --persons, people eval) and the moved cvdebug sidecar go to
  --sidecars-dir/<doc name>/ (default data/analysis_sidecars), so nothing is lost and --unbundle restores them.
- Racket / shuttle sidecars are left where they are unless --all-sidecars (they are not part of the pose-only
  playback; moving them hides those rows in the viewer).
Every doc is rewritten atomically and records what was done in meta.bundle."""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time

from analysis.pose24 import doc as docmod

DEFAULT_DIR = os.path.join("Assets", "StreamingAssets", "analysis")
DEFAULT_SIDECARS = os.path.join("data", "analysis_sidecars")
SUFFIX = ".analysis.json"


def _load(p):
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def base_clip(name):
    """singles_05x_pose24 -> singles_05x (the persons file is named after the build's out clip without the variant)."""
    for v in ("_pose24", "_posefit", "_posev2", "_smplx"):
        if name.endswith(v):
            return name[: -len(v)]
    return name


def sidecars_of(doc_path, all_sidecars=False) -> dict:
    d, name = os.path.dirname(doc_path), os.path.basename(doc_path)[: -len(SUFFIX)]
    out = {"cvdebug": os.path.join(d, f"{name}.cvdebug.json"),
           "persons": os.path.join(d, f"{base_clip(name)}.persons.json")}
    if all_sidecars:
        for p in sorted(glob.glob(os.path.join(d, f"{name}.racket_*.json")) +
                        glob.glob(os.path.join(d, f"{base_clip(name)}.shuttle_*.json"))):
            out[os.path.basename(p)] = p
    return {k: p for k, p in out.items() if os.path.exists(p)}


def cvdebug_pairs(doc, cv) -> str | None:
    """None when the cvdebug sidecar belongs to this doc, else why not."""
    if cv.get("format") != "cvdebug/1":
        return f"format {cv.get('format')}"
    if cv.get("frame_count") != doc["meta"].get("frame_count"):
        return f"frame_count {cv.get('frame_count')} != {doc['meta'].get('frame_count')}"
    if cv.get("clip_id") and doc.get("clip_id") and cv["clip_id"] != doc["clip_id"]:
        return f"clip {cv['clip_id']} != {doc['clip_id']}"
    bid = (doc["meta"].get("pose24") or {}).get("build_id")
    if bid and cv.get("source_build_id") and cv["source_build_id"] != bid:
        return f"build {cv['source_build_id']} != {bid}"
    return None


def bundle(doc_path, sidecars_dir=DEFAULT_SIDECARS, all_sidecars=False, dry_run=False, log=print) -> dict:
    name = os.path.basename(doc_path)[: -len(SUFFIX)]
    doc = _load(doc_path)
    found = sidecars_of(doc_path, all_sidecars)
    res = {"doc": doc_path, "embedded": [], "moved": [], "skipped": {}}
    if "cvdebug" in found:
        cv = _load(found["cvdebug"])
        why = cvdebug_pairs(doc, cv)
        if why is None:
            doc["cvdebug"] = cv
            res["embedded"].append("cvdebug")
        else:
            res["skipped"]["cvdebug"] = why
    dest = os.path.join(sidecars_dir, name)
    moves = [(k, p) for k, p in found.items() if k != "cvdebug" or "cvdebug" in res["embedded"]]
    if not res["embedded"] and not moves:
        log(f"  {name}: nothing to bundle")
        return res
    prev = doc["meta"].get("bundle") or {}
    doc["meta"]["bundle"] = {"embedded": sorted(set(prev.get("embedded", [])) | set(res["embedded"])),
                             "sidecars_dir": dest.replace(os.sep, "/"),
                             "moved": sorted(set(prev.get("moved", [])) | {os.path.basename(p) for _, p in moves}),
                             "when": time.strftime("%Y-%m-%d %H:%M")}
    res["moved"] = [os.path.basename(p) for _, p in moves]
    log(f"  {name}: embed {res['embedded'] or '-'}, move {res['moved'] or '-'} -> {dest}"
        + (f", skipped {res['skipped']}" if res["skipped"] else "") + (" (dry run)" if dry_run else ""))
    if dry_run:
        return res
    docmod.write_json(doc_path, docmod.json_safe(doc))
    os.makedirs(dest, exist_ok=True)
    for _, p in moves:
        shutil.move(p, os.path.join(dest, os.path.basename(p)))
        meta = p + ".meta"                                   # Unity's import record of the moved file
        if os.path.exists(meta):
            shutil.move(meta, os.path.join(dest, os.path.basename(meta)))
    return res


def unbundle(doc_path, sidecars_dir=DEFAULT_SIDECARS, log=print) -> dict:
    name = os.path.basename(doc_path)[: -len(SUFFIX)]
    doc = _load(doc_path)
    src = os.path.join(sidecars_dir, name)
    back = []
    if os.path.isdir(src):
        for f in sorted(os.listdir(src)):
            if f.endswith(".meta"):
                continue
            shutil.move(os.path.join(src, f), os.path.join(os.path.dirname(doc_path), f))
            back.append(f)
    doc.pop("cvdebug", None)
    doc["meta"].pop("bundle", None)
    docmod.write_json(doc_path, docmod.json_safe(doc))
    log(f"  {name}: restored {back or '-'}")
    return {"doc": doc_path, "restored": back}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis.bundle", description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=DEFAULT_DIR, help="folder with the analysis docs (default: Unity's)")
    ap.add_argument("--clip", action="append", help="doc name without .analysis.json (repeatable; default: all)")
    ap.add_argument("--sidecars-dir", default=DEFAULT_SIDECARS)
    ap.add_argument("--all-sidecars", action="store_true", help="also move racket / shuttle sidecars")
    ap.add_argument("--unbundle", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    docs = ([os.path.join(a.dir, c + SUFFIX) for c in a.clip] if a.clip
            else sorted(glob.glob(os.path.join(a.dir, "*" + SUFFIX))))
    for p in docs:
        if a.unbundle:
            unbundle(p, a.sidecars_dir)
        else:
            bundle(p, a.sidecars_dir, a.all_sidecars, a.dry_run)


if __name__ == "__main__":
    main()
