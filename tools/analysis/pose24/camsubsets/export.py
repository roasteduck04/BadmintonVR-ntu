"""Export a chosen camera subset as a real pose24 build (spec § 7): a registry entry in clips.json (inserted as
text, so the hand-formatted file keeps every other line) + pb.build(name, timeline_of=clip), so cvdebug,
overlay_video and the Unity ghosts accept it like singles_05x_2cam. A mono method (1 view) writes a
`"mono": true` entry (+ mono_lengths, + mono_prior with a learned prior) and runs
pb.build(clip, mono=view, out_clip=name). A view the full build excluded (its debug cache's `excluded`, D5) is
refused before anything is written."""
from __future__ import annotations

import json
import os

import numpy as np

from analysis import pose_v2 as pv
from analysis.pose24 import build as pb
from analysis.pose24 import config, track
from analysis.pose24.camsubsets import subsets as cs


def _same(e, clip, views, mono, lengths, prior=None):
    return (e.get("base") == clip and sorted(e.get("views", ())) == views and bool(e.get("mono")) == mono
            and (not mono or (e.get("mono_lengths", "oracle") == lengths and e.get("mono_prior") == prior)))


def export_name(clip, subset, registry, mono=False, lengths="oracle", prior=None) -> str:
    """The registry entry of this export (matched on views, the mono flag, mono lengths and the mono prior path),
    else a new name <clip>_<n>cam_<abbr> (+ _<lengths> for a non-oracle mono) not taken by another entry."""
    want = sorted(subset)
    for name, e in registry.items():
        if _same(e, clip, want, mono, lengths, prior):
            return name
    name = f"{clip}_{len(want)}cam_" + "_".join(cs.abbr(v) for v in want) + (f"_{lengths}" if mono and
                                                                                lengths != "oracle" else "")
    k, out = 2, name
    while out in registry:
        out, k = f"{name}_{k}", k + 1
    return out


def _entry(base, views, timeline_of, extra=None) -> dict:
    return dict({"base": base, "views": list(views), "timeline_of": timeline_of}, **(extra or {}))


def add_registry_entry(path, name, base, views, timeline_of, extra=None) -> bool:
    """Insert `"name": {...}` as the last line of "clips"; False if already there with the same content.
    The write is atomic (temp file in the same directory + os.replace)."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    data = json.loads(text)
    entry = _entry(base, views, timeline_of, extra)
    if name in data["clips"]:
        if data["clips"][name] == entry:
            return False
        raise ValueError(f"registry {path} already has {name!r} with different content: {data['clips'][name]}")
    nl = "\r\n" if "\r\n" in text else "\n"
    body = text.rstrip("\r\n")
    lines = body.split(nl)
    if (len(lines) < 4 or list(data)[-1] != "clips" or lines[-1] != "}" or lines[-2] != "  }"
            or lines[-3].rstrip().endswith(",")):
        raise ValueError(f"registry {path}: layout not recognised (expected \"clips\" last, closed by '  }}' + '}}'); "
                         "add the entry by hand")
    prev = lines[-3].rstrip()
    new_line = f"    {json.dumps(name)}: {json.dumps(entry, separators=(', ', ': '))}"
    out = lines[:-3] + [lines[-3] if prev.endswith("{") else lines[-3] + ",", new_line] + lines[-2:]
    new_text = nl.join(out) + text[len(body):]
    try:
        got = json.loads(new_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"registry {path}: insertion would not parse ({e}); nothing written") from e
    want = dict(data, clips=dict(data["clips"], **{name: entry}))
    if got != want or list(got["clips"]) != list(want["clips"]):
        raise ValueError(f"registry {path}: insertion changed other content; nothing written")
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True


def _excluded(clip, cache_dir) -> set:
    """Views excluded by `clip`'s full build (debug cache meta only); empty when there is no cache yet."""
    p = track.cache_path(clip, cache_dir or pv.CACHE_DIR)
    if not os.path.exists(p):
        return set()
    with np.load(p) as z:
        return set(json.loads(str(z["meta"])).get("excluded") or {})


def _mono_lengths(method, mono_opts):
    from analysis.pose24 import mono
    ln = (mono_opts or {}).get("lengths") or {"mono-len-oracle": "oracle", "mono-len-prior": "prior"}.get(method)
    if ln is None and method == "mono":
        ln = "oracle"
    if ln not in mono.LENGTH_SOURCES:
        raise ValueError(f"export method {method!r}: lengths {ln!r} not one of {mono.LENGTH_SOURCES} "
                         "(mono-len-oracle, mono-len-prior, or mono_opts lengths)")
    return ln


def export(clip, subset, registry_path=config.REGISTRY_PATH, method="fit", mono_opts=None, name=None, log=print,
           **build_kw) -> str:
    """Registry entry + real pose24 build of `subset` on `clip`'s timeline; returns the doc path.
    method "fit" (>= 2 views) or "mono*" (exactly 1 view; lengths from the label mono-len-oracle / mono-len-prior
    or mono_opts["lengths"]; mono_opts also go to build_mono)."""
    mono = method.startswith("mono")
    if not mono and method != "fit":
        raise ValueError(f"unknown export method {method!r}")
    lengths = _mono_lengths(method, mono_opts) if mono else None
    reg = config.load_registry(registry_path)
    if clip not in reg or "base" in reg[clip]:
        raise ValueError(f"{clip!r} is not a full clip in {registry_path}")
    views = sorted(subset)
    unknown = sorted(set(views) - set(reg[clip]["views"]))
    n_ok = len(views) == 1 if mono else len(views) >= 2
    if unknown or not n_ok or len(set(views)) != len(views):
        raise ValueError(f"subset {', '.join(views)} of {clip}: {method} needs "
                         + ("exactly 1 view" if mono else ">= 2 distinct views") + " of the clip"
                         + (f" (unknown: {', '.join(unknown)})" if unknown else ""))
    excl = sorted(set(views) & _excluded(clip, build_kw.get("cache_dir")))
    if excl:
        raise ValueError(f"subset {', '.join(views)} of {clip}: {', '.join(excl)} excluded by the full build "
                         "(view health, D5)")
    prior = (mono_opts or {}).get("prior_path") if mono else None
    extra = dict({"mono": True, "mono_lengths": lengths}, **({"mono_prior": prior} if prior else {})) if mono else None
    name = name or export_name(clip, views, reg, mono=mono, lengths=lengths, prior=prior)
    if name in reg:
        e = reg[name]
        if not _same(e, clip, views, mono, lengths, prior):
            raise ValueError(f"registry entry {name!r} is {e}, not {clip} with views {', '.join(views)}"
                             + (f" (mono, lengths {lengths}, prior {prior})" if mono else ""))
        log(f"[export] reusing registry entry {name}")
    elif add_registry_entry(registry_path, name, clip, views, clip, extra=extra):
        log(f"[export] added registry entry {name} ({', '.join(views)} on the {clip} timeline"
            + (f", mono, lengths {lengths}" + (f", prior {prior}" if prior else "") if mono else "")
            + f") to {registry_path}")
    if mono:
        doc = pb.build(clip, mono=views[0], out_clip=name, mono_opts=dict(mono_opts or {}, lengths=lengths),
                       registry=config.load_registry(registry_path), log=log, **build_kw)
    else:
        doc = pb.build(name, timeline_of=clip, registry=config.load_registry(registry_path), log=log, **build_kw)
    log(f"[export] {doc}\n  next: PYTHONPATH=tools python -m analysis.cvdebug build --clip {name} --variant pose24\n"
        f"        PYTHONPATH=tools python -m analysis.overlay_video render --clip {name} --variant pose24 "
        f"--compare {clip}:pose24 --frames 300:600")
    return doc
