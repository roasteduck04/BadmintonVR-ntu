"""Find old scripts left in a Unity project that is updated by copying (the lab PC's BadmintonVR-unity: robocopy adds
and replaces files but never removes retired ones, so e.g. the old IMGUI viewer AnalysisViewerUI.cs kept drawing its
panel there after v0.8.0).

    python tools/unity_checks/stale_scripts.py --unity-project /mnt/c/Users/sports_1/BadmintonVR-unity            # report
    python tools/unity_checks/stale_scripts.py --unity-project ... --archive        # move the leftovers out of Assets
    python tools/unity_checks/stale_scripts.py --unity-project ... --archive --update   # + copy the repo's scripts in

Compares the .cs files under Assets/Scripts and Assets/Editor of the Unity project with this repository's:
- leftovers: in the project, not in the repo (old-style IMGUI UI flagged: OnGUI, or a retired viewer name);
- out of date: in both, different content; missing: in the repo only;
- the main scene: the AnalysisPlayer's useNewViewer switch, and components whose scripts are leftovers.
--archive moves each leftover (+ its .meta) to <project>/_archive_old_scripts/<date>/... (outside Assets: Unity stops
compiling it; nothing is deleted). --update copies the repo's out-of-date / missing scripts in (the replaced ones are
archived the same way first). Close Unity before --archive / --update."""
from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOLDERS = (os.path.join("Assets", "Scripts"), os.path.join("Assets", "Editor"))
RETIRED = {"AnalysisViewerUI.cs": "old IMGUI viewer panel (archived in v0.8.0)"}
SCENE = os.path.join("Assets", "badminton.unity")


def scripts(root):
    out = {}
    for f in FOLDERS:
        base = os.path.join(root, f)
        for d, _, files in os.walk(base):
            if "_archive" in d.replace("\\", "/").split("/"):
                continue
            for n in files:
                if n.endswith(".cs"):
                    p = os.path.join(d, n)
                    out[os.path.relpath(p, root).replace("\\", "/")] = p
    return out


def old_ui(path):
    name = os.path.basename(path)
    if name in RETIRED:
        return RETIRED[name]
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "draws IMGUI (OnGUI)" if re.search(r"\bvoid\s+OnGUI\s*\(", fh.read()) else ""
    except OSError:
        return ""


def guid(meta):
    try:
        with open(meta, encoding="utf-8", errors="replace") as fh:
            m = re.search(r"^guid:\s*([0-9a-f]+)", fh.read(), re.M)
            return m.group(1) if m else None
    except OSError:
        return None


def move_out(project, rel, stamp, log):
    dst_root = os.path.join(project, "_archive_old_scripts", stamp)
    for suffix in ("", ".meta"):
        src = os.path.join(project, rel + suffix)
        if os.path.exists(src):
            dst = os.path.join(dst_root, rel + suffix)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
    log(f"  moved {rel} -> _archive_old_scripts/{stamp}/{rel}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unity-project", required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--archive", action="store_true", help="move the leftovers out of Assets")
    ap.add_argument("--update", action="store_true", help="copy the repo's out-of-date / missing scripts in")
    a = ap.parse_args(argv)
    proj = os.path.abspath(a.unity_project)
    if not os.path.isdir(os.path.join(proj, "Assets")):
        sys.exit(f"{proj} has no Assets folder (is it the Unity project?)")
    if os.path.normcase(proj) == os.path.normcase(os.path.abspath(a.repo)):
        sys.exit("--unity-project is this repository itself: nothing to compare")
    mine, theirs = scripts(a.repo), scripts(proj)
    left = sorted(set(theirs) - set(mine))
    stale = sorted(r for r in set(theirs) & set(mine) if not filecmp.cmp(theirs[r], mine[r], shallow=False))
    missing = sorted(set(mine) - set(theirs))
    print(f"Unity project {proj}\n  {len(theirs)} scripts; repo {a.repo}: {len(mine)}")
    print(f"\nLeftovers (in the project only): {len(left)}")
    for r in left:
        tag = old_ui(theirs[r])
        print(f"  {r}{'   <- OLD UI: ' + tag if tag else ''}")
    print(f"\nOut of date (different from the repo): {len(stale)}")
    for r in stale:
        print(f"  {r}")
    print(f"\nMissing (in the repo only): {len(missing)}")
    for r in missing:
        print(f"  {r}")
    scene = os.path.join(proj, SCENE)
    if os.path.exists(scene):
        with open(scene, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        m = re.search(r"useNewViewer:\s*(\d)", txt)
        print(f"\nScene {SCENE}: useNewViewer = {m.group(1) if m else '?'}" +
              ("   <- 0 shows the old viewer: tick 'Use New Viewer' on AnalysisPlayer" if m and m.group(1) == "0" else ""))
        for r in left:
            g = guid(theirs[r] + ".meta")
            if g and g in txt:
                print(f"  the scene uses leftover {os.path.basename(r)} (remove the component after archiving it)")
    if not (a.archive or a.update):
        print("\nNothing changed. --archive moves the leftovers out of Assets; --update also copies the repo's scripts in"
              " (close Unity first).")
        return
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    if a.archive:
        print(f"\nArchiving {len(left)} leftovers:")
        for r in left:
            move_out(proj, r, stamp, print)
    if a.update:
        print(f"\nUpdating {len(stale) + len(missing)} scripts from the repo:")
        for r in stale:
            move_out(proj, r, stamp, print)
        for r in stale + missing:
            dst = os.path.join(proj, r)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(mine[r], dst)
            if os.path.exists(mine[r] + ".meta"):
                shutil.copy2(mine[r] + ".meta", dst + ".meta")
            print(f"  copied {r}")
    print("\nDone. Open Unity: it recompiles; if a component shows 'missing script', remove it from AnalysisPlayer.")


if __name__ == "__main__":
    main()
