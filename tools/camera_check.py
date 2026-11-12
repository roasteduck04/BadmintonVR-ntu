"""What the live cameras really deliver (v0.8.7): frame rate, gaps, freezes, repeated frames and delay.

Reads every camera of the saved source the same way the live engine does (OpenCV / FFmpeg, all at once) for a few
seconds and prints one block per camera in plain words, then saves a report and one picture per camera.

    python tools/camera_check.py                                   # the cameras saved on the server page, 20 s
    python tools/camera_check.py --input /mnt/c/Users/<user>/BadmintonVR-unity/runs/input.json --seconds 30
    python tools/camera_check.py --url unit1=rtsp://192.168.0.10:8554/unit1 --url unit2=rtsp://192.168.0.10:8554/unit2
    python tools/camera_check.py --latency                         # delay from the scene to this PC (see below)

--latency: maximise this terminal and aim one or more cameras at this screen. The terminal turns white and black
every few seconds; the time until each camera's picture turns bright / dark is its delay (camera + Wi-Fi + bridge +
RTSP + decoding, plus about 0.05 s of this screen). Cameras that do not see the screen say so.

Needs only OpenCV and NumPy. Stop the live engine first when the cameras allow one reader only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time

ROTATE_TAG = "#rotate=180"
STILL_DIFF = 0.25                # mean grey-level change (0-255, 32x18 picture) below which two frames look the same
FLASH_MIN_STEP = 8.0             # grey levels the picture must change by to count as seeing a screen flash
FREEZE_S = 0.5                   # a gap this long is a freeze
OPEN_MS = 15000


def split_rotate(url):
    url = str(url or "")
    return (url[:-len(ROTATE_TAG)], 180) if url.endswith(ROTATE_TAG) else (url, 0)


def hide_login(url):
    """The address without its user name / password (reports are sent around)."""
    import re
    return re.sub(r"//[^@/]*@", "//<login>@", str(url))


def cameras_from_input(path) -> tuple:
    """({name: address}, fps) of the server page's saved source (runs/input.json): the used live camera rows."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    cams = {}
    for c in d.get("cameras") or []:
        if c.get("use", True) and c.get("url"):
            cams[c["name"]] = c["url"] + (ROTATE_TAG if str(c.get("rotate")) == "180" else "")
    return cams, float(d.get("fps") or 60)


class Watch(threading.Thread):
    """Reads one camera: arrival time, brightness and change of every frame; keeps the first and the last picture."""

    def __init__(self, name, url):
        super().__init__(daemon=True, name=f"check-{name}")
        self.name_, (self.url, self.rot) = name, split_rotate(url)
        self.t, self.bright, self.diff = [], [], []
        self.first, self.last, self.size = None, None, None
        self.open_s, self.first_s, self.error, self.fails = None, None, None, 0
        self.declared_fps, self.codec = None, None
        self.running, self.lock = True, threading.Lock()

    def run(self):
        import cv2
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, OPEN_MS])
        try:
            if not cap.isOpened():
                self.error = f"did not open in {OPEN_MS / 1000:g} s"
                return
            self.open_s = time.perf_counter() - t0
            self.declared_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            cc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            self.codec = "".join(chr((cc >> (8 * i)) & 255) for i in range(4)).strip("\x00 ") or None
            prev = None
            while self.running:
                ok, img = cap.read()
                now = time.perf_counter()
                if not ok or img is None:
                    self.fails += 1
                    if self.fails > 200:
                        self.error = self.error or "stopped sending frames"
                        return
                    time.sleep(0.01)
                    continue
                if self.rot == 180:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA)
                small = small.astype("float32")
                with self.lock:
                    if self.first is None:
                        self.first, self.first_s, self.size = img, now - t0, (img.shape[1], img.shape[0])
                    self.last = img
                    self.t.append(now)
                    self.bright.append(float(small.mean()))
                    self.diff.append(float(abs(small - prev).mean()) if prev is not None else 255.0)
                prev = small
        finally:
            cap.release()

    def series(self):
        with self.lock:
            return list(self.t), list(self.bright), list(self.diff)


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stats(times, diffs, fps_expected) -> dict:
    """Numbers of one camera from its frame arrival times (s) and frame-to-frame change."""
    n = len(times)
    if n < 2:
        return {"frames": n}
    nominal = 1.0 / float(fps_expected)
    gaps = [b - a for a, b in zip(times, times[1:])]
    span = times[-1] - times[0]
    per_s = {}
    for t in times:
        per_s[int(t - times[0])] = per_s.get(int(t - times[0]), 0) + 1
    whole = [c for s, c in sorted(per_s.items()) if s < int(span)] or list(per_s.values())
    freezes = [g for g in gaps if g >= FREEZE_S]
    return {"frames": n, "seconds": round(span, 2), "fps": round((n - 1) / span, 1) if span > 0 else None,
            "fps_worst_second": min(whole), "fps_best_second": max(whole),
            "gap_ms_p50": round(1000 * pct(gaps, 50), 1), "gap_ms_p95": round(1000 * pct(gaps, 95), 1),
            "gap_ms_max": round(1000 * max(gaps), 1),
            "late_gaps": sum(1 for g in gaps if g > 2.5 * nominal),          # at least one frame time missing
            "freezes": len(freezes), "longest_freeze_s": round(max(freezes), 2) if freezes else 0.0,
            "burst_pct": round(100.0 * sum(1 for g in gaps if g < 0.25 * nominal) / len(gaps), 1),
            "still_pct": round(100.0 * sum(1 for d in diffs[1:] if d < STILL_DIFF) / max(1, len(diffs) - 1), 1)}


def verdict(st, fps_expected) -> list:
    """Plain-word findings of one camera's numbers."""
    if st.get("frames", 0) < 2:
        return ["no frames arrived"]
    out = []
    if st["fps"] < 0.9 * fps_expected:
        out.append(f"delivers {st['fps']:g} of the {fps_expected:g} frames per second asked for")
    if st["freezes"]:
        out.append(f"froze {st['freezes']} time(s), longest {st['longest_freeze_s']:g} s")
    elif st["late_gaps"] > 0.02 * st["frames"]:
        out.append(f"{st['late_gaps']} gaps where at least one frame is missing (longest {st['gap_ms_max']:g} ms)")
    if st["burst_pct"] > 30:
        out.append(f"{st['burst_pct']:g} % of frames arrive in bursts (buffered on the way): uneven and delayed")
    if st["still_pct"] > 10:
        out.append(f"{st['still_pct']:g} % of frames show no change: lost frames repeated by the bridge, or nothing "
                   "moved in front of the camera (have someone move, then compare)")
    if st["fps_worst_second"] < 0.6 * fps_expected:
        out.append(f"worst second had {st['fps_worst_second']} frames")
    return out or ["steady"]


def flash_delays(times, bright, toggles, settle_s=0.5) -> list:
    """Delay of each screen flash in one camera's brightness: toggles = [(time, +1 white | -1 black), ...]. A delay is
    the first frame after the flash whose brightness is halfway to the level it settles at; None = not seen."""
    out = []
    for k, (t0, sign) in enumerate(toggles):
        t1 = toggles[k + 1][0] if k + 1 < len(toggles) else (times[-1] if times else t0)
        pre = [b for t, b in zip(times, bright) if t0 - settle_s <= t < t0]
        post = [b for t, b in zip(times, bright) if t1 - settle_s <= t < t1]
        if len(pre) < 3 or len(post) < 3:
            out.append(None)
            continue
        a, b = pct(pre, 50), pct(post, 50)
        if (b - a) * sign < FLASH_MIN_STEP:              # the picture did not follow the screen
            out.append(None)
            continue
        half = a + 0.5 * (b - a)
        hit = next((t for t, v in zip(times, bright) if t0 <= t < t1 and (v - half) * sign >= 0), None)
        out.append(None if hit is None else hit - t0)
    return out


def run_latency(watches, n=8, gap_s=(3.0, 4.0), out=sys.stdout, clock=time.perf_counter, sleep=time.sleep) -> list:
    """Flash this terminal white / black n times; returns [(time, sign)]."""
    def paint(white):
        out.write(("\x1b[48;5;15m" if white else "\x1b[48;5;0m") + "\x1b[2J\x1b[H")
        out.flush()
    paint(False)
    sleep(3.0)
    toggles, white = [], False
    for _ in range(n):
        white = not white
        paint(white)
        toggles.append((clock(), 1 if white else -1))
        sleep(random.uniform(*gap_s))
    out.write("\x1b[0m\x1b[2J\x1b[H")
    out.flush()
    return toggles


def report_lines(name, w, st, fps_expected, delays=None) -> list:
    lines = [f"{name}  {hide_login(w.url)}" + ("  (turned 180 degrees)" if w.rot else "")]
    if w.error and st.get("frames", 0) < 2:
        return lines + [f"   FAILED: {w.error}"]
    lines.append(f"   picture {w.size[0]}x{w.size[1]}, {w.codec or '?'}, stream says {w.declared_fps:g} fps; opened in "
                 f"{w.open_s:.1f} s, first picture after {w.first_s:.1f} s")
    lines.append(f"   {st['frames']} frames in {st['seconds']:g} s = {st['fps']:g} fps (worst second "
                 f"{st['fps_worst_second']}, best {st['fps_best_second']})")
    lines.append(f"   time between frames: usual {st['gap_ms_p50']:g} ms, 1 in 20 above {st['gap_ms_p95']:g} ms, "
                 f"longest {st['gap_ms_max']:g} ms (even = {1000 / fps_expected:.1f} ms)")
    if delays is not None:
        seen = [d for d in delays if d is not None]
        if len(seen) >= min(3, len(delays)):
            lines.append(f"   DELAY scene -> this PC: usual {pct(seen, 50):.2f} s (shortest {min(seen):.2f}, longest "
                         f"{max(seen):.2f}; {len(seen)} of {len(delays)} flashes seen)")
        else:
            lines.append("   delay: this camera did not see the screen flash (or it is more than ~3 s behind)")
    for v in verdict(st, fps_expected):
        lines.append(f"   -> {v}")
    if w.error:
        lines.append(f"   -> {w.error}")
    return lines


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--input", default=os.path.join(here, "runs", "input.json"),
                    help="the server page's saved source (default: this repository's runs/input.json)")
    ap.add_argument("--url", action="append", default=[], metavar="NAME=ADDRESS", help="a camera by hand (repeatable)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=float, help="frames per second the cameras should send (default: the saved source's)")
    ap.add_argument("--latency", action="store_true", help="flash this terminal and time each camera's delay")
    ap.add_argument("--out", help="report folder (default: camera_check_<time> next to the input file)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    cams, fps = {}, a.fps or 60.0
    if a.url:
        for u in a.url:
            name, _, addr = u.partition("=")
            cams[name.strip()] = addr.strip()
    else:
        if not os.path.exists(a.input):
            sys.exit(f"no saved source at {a.input}: give --input <unity project>/runs/input.json or --url name=address")
        cams, saved_fps = cameras_from_input(a.input)
        fps = a.fps or saved_fps
    if not cams:
        sys.exit("no live cameras in the saved source (step 1 of the page: Live broadcast, tick Use, Save source)")
    import cv2
    print(f"checking {len(cams)} camera(s) for {'the flash test' if a.latency else f'{a.seconds:g} s'}, "
          f"{fps:g} fps expected ...", flush=True)
    watches = {n: Watch(n, u) for n, u in cams.items()}
    for w in watches.values():
        w.start()
    t_end = time.perf_counter() + OPEN_MS / 1000 + 2
    while time.perf_counter() < t_end and any(w.first is None and w.error is None and w.is_alive()
                                               for w in watches.values()):
        time.sleep(0.1)
    toggles = None
    if a.latency:
        toggles = run_latency(watches)
        time.sleep(3.0)
    else:
        time.sleep(a.seconds)
    for w in watches.values():
        w.running = False
    for w in watches.values():
        w.join(OPEN_MS / 1000 + 2)
    out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.input)),
                                    "camera_check_" + time.strftime("%Y-%m-%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    lines, doc = [f"camera check {time.strftime('%Y-%m-%d %H:%M:%S')}, {fps:g} fps expected"], {"fps": fps, "cameras": {}}
    for n, w in watches.items():
        t, b, d = w.series()
        st = stats(t, d, fps)
        delays = flash_delays(t, b, toggles) if toggles else None
        lines += [""] + report_lines(n, w, st, fps, delays)
        doc["cameras"][n] = {"url": hide_login(w.url), "rotate": w.rot, "size": w.size, "codec": w.codec,
                             "declared_fps": w.declared_fps, "open_s": w.open_s, "first_frame_s": w.first_s,
                             "error": w.error, "stats": st, "delays_s": delays, "verdict": verdict(st, fps)}
        if w.last is not None:
            cv2.imwrite(os.path.join(out_dir, f"{n}.jpg"), w.last, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with open(os.path.join(out_dir, f"{n}_frames.csv"), "w", encoding="utf-8") as fh:
            fh.write("t_s,gap_ms,brightness,change\n")
            for i in range(len(t)):
                fh.write(f"{t[i] - t[0]:.4f},{1000 * (t[i] - t[i - 1]) if i else 0:.1f},{b[i]:.1f},{d[i]:.2f}\n")
    firsts = {n: w.t[0] for n, w in watches.items() if w.t}
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\nreport, one picture and a frame list per camera: {out_dir}")
    return 0 if firsts else 1


if __name__ == "__main__":
    sys.exit(main())
