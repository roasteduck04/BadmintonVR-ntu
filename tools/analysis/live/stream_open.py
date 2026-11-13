"""Live camera streams read through PyAV, with damaged pictures found and kept away from the pose model (v0.8.7).

Over Wi-Fi, packets of a camera's H.264 stream get lost before they reach this PC. FFmpeg then prints lines such as
"[h264] error while decoding MB 14 14 / corrupted macroblock / cbp too large" and hands over a picture with smeared
or grey blocks; every following picture builds on it, so the smear stays until the camera's next keyframe. OpenCV
cannot tell those pictures from good ones. This reader can:

- a picture is DAMAGED when the decoder reported an error while making it (FFmpeg's error messages, caught per
  decoding thread) or flagged it (PyAV frame.is_corrupt);
- the pictures after a damaged one are TAINTED until a keyframe arrives (or `hold_max_s` has passed: streams with
  intra refresh have no keyframes, they heal over about a second);
- damaged = "hold" (default): a damaged or tainted picture is replaced by the last clean one (flag "held"), so the pose
  model never sees smears and the frame rate stays even; damaged = "use": every picture as decoded (flag says which).

Settings (input.* in the live config, "Damaged frames" / "Camera reader" / "Transport" in step 1 of the server page):
    reader     pyav (this module) | opencv (the old OpenCV reader: no damage check)
    transport  tcp (default) | udp | auto - how RTSP is pulled from the server (MediaMTX); TCP never loses packets on
               this last leg, so damage that remains was lost before MediaMTX (camera / bridge / Wi-Fi)
    damaged    hold | use
    hold_max_s longest a tainted stretch is held without a keyframe (default 1.0 s)

Each camera decodes on one thread (decoder threads would hide which camera an error message belongs to; one 720p
H.264 stream needs a few ms per picture). The FFmpeg error lines are not printed one by one any more: each camera logs
one summary line every few seconds while it has damage.
"""
from __future__ import annotations

import logging
import os
import threading
import time

READERS = ("pyav", "opencv")
TRANSPORTS = ("tcp", "udp", "auto")
DAMAGED = ("hold", "use")
DEFAULTS = {"reader": "pyav", "transport": "tcp", "damaged": "hold", "hold_max_s": 1.0}
RETRY_S = 1.0                    # a stream that failed or ended is opened again after this long
LOG_S = 5.0                      # a camera with damage logs one summary line this often

_lock = threading.Lock()
_by_thread = {}                  # decoding thread id -> Stream (whose decoder errors land there)
_installed = False


def options(cfg=None) -> dict:
    """The camera reading settings, checked, defaults filled: {reader, transport, damaged, hold_max_s}."""
    c = dict(DEFAULTS)
    c.update({k: v for k, v in (cfg or {}).items() if k in DEFAULTS and v is not None})
    for k, allowed in (("reader", READERS), ("transport", TRANSPORTS), ("damaged", DAMAGED)):
        if c[k] not in allowed:
            raise ValueError(f"input.{k} must be one of {allowed}, got {c[k]!r}")
    try:
        c["hold_max_s"] = float(c["hold_max_s"])
    except (TypeError, ValueError):
        raise ValueError(f"input.hold_max_s must be a number of seconds, got {c['hold_max_s']!r}") from None
    if not 0.05 <= c["hold_max_s"] <= 10.0:
        raise ValueError(f"input.hold_max_s must be 0.05-10 s, got {c['hold_max_s']}")
    return c


def pyav_available() -> bool:
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


def is_rtsp(url) -> bool:
    return str(url).lower().startswith(("rtsp://", "rtsps://"))


def open_options(url, transport="tcp") -> dict:
    """FFmpeg options of one stream: no input buffering (the newest picture, not a queue), RTSP over the transport."""
    o = {"fflags": "nobuffer"}
    if is_rtsp(url) and transport in ("tcp", "udp"):
        o["rtsp_transport"] = transport
    return o


def opencv_capture(url, transport="tcp", timeout_ms=None):
    """An OpenCV capture of a stream. OpenCV pulls RTSP over TCP unless OPENCV_FFMPEG_CAPTURE_OPTIONS says
    otherwise; transport udp / auto sets it for this one open."""
    import cv2
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_ms), cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_ms)] \
        if timeout_ms else []
    if not is_rtsp(url) or transport == "tcp":
        return cv2.VideoCapture(str(url), cv2.CAP_FFMPEG, params)
    with _lock:                                          # the variable is read when the stream opens
        old = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp" if transport == "udp" else "rtsp_flags;prefer_tcp"
        try:
            return cv2.VideoCapture(str(url), cv2.CAP_FFMPEG, params)
        finally:
            if old is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = old


class _DecodeErrors(logging.Handler):
    """FFmpeg's error messages (PyAV sends them to the 'libav' logger) -> the Stream decoding on that thread. A
    message from a thread no Stream decodes on (a decoder helper thread) counts for the only Stream when there is one."""

    def emit(self, record):
        if record.levelno < logging.ERROR:
            return
        with _lock:
            s = _by_thread.get(threading.get_ident())
            if s is None and len(set(map(id, _by_thread.values()))) == 1:
                s = next(iter(_by_thread.values()))
        if s is not None:
            s.decode_error(record.getMessage())


def _install():
    global _installed
    with _lock:
        if _installed:
            return
        import av.logging
        lg = logging.getLogger("libav")
        lg.addHandler(_DecodeErrors())
        lg.setLevel(logging.ERROR)
        lg.propagate = False                             # summarised per camera instead of one line per macroblock
        av.logging.set_level(av.logging.ERROR)
        _installed = True


def _register(stream):
    with _lock:
        _by_thread[threading.get_ident()] = stream


def _unregister(stream):
    with _lock:
        for k in [k for k, v in _by_thread.items() if v is stream]:
            del _by_thread[k]


class Damage:
    """The damaged / tainted / clean decision of one camera, picture by picture (no video needed: tested directly)."""

    def __init__(self, hold_max_s=1.0):
        self.hold_max_s = float(hold_max_s)
        self.errors = 0                  # decoder error messages since the last picture
        self.dirty_since = None          # time of the last damaged picture while not healed yet

    def classify(self, corrupt, key, now) -> str:
        errs, self.errors = self.errors, 0
        if corrupt or errs:
            self.dirty_since = now
            return "damaged"
        if self.dirty_since is not None:
            if key or now - self.dirty_since >= self.hold_max_s:
                self.dirty_since = None              # a keyframe (or enough time with intra refresh) heals it
            else:
                return "tainted"
        return "clean"


class Stream:
    """One camera stream: open (and open again when it fails or ends), decode on the calling thread, flag every
    picture. frames() yields (BGR image, flag, arrival time): flag clean | damaged | tainted (damaged "use") or
    clean | held (damaged "hold": the last clean picture instead; nothing is yielded before the first clean one)."""

    def __init__(self, url, name=None, transport="tcp", damaged="hold", hold_max_s=1.0, open_s=15.0, read_s=5.0,
                 threads=1, log=print):
        self.url, self.name = str(url), name or str(url)
        self.transport, self.damaged = transport, damaged
        self.open_s, self.read_s, self.threads = float(open_s), float(read_s), max(1, int(threads or 1))
        self.log = log
        self.dmg = Damage(hold_max_s)
        self.running = True
        self.container, self.vstream = None, None
        self.error, self.codec, self.size, self.declared_fps = None, None, None, None
        self.counts = {"frames": 0, "clean": 0, "damaged": 0, "tainted": 0, "held": 0, "errors": 0, "reconnects": 0}
        self._win = {"frames": 0, "damaged": 0, "tainted": 0, "errors": 0, "t": time.monotonic(), "example": None}
        self.last_clean = None

    # ------------------------------------------------------------------ decoder errors
    def decode_error(self, message):
        self.dmg.errors += 1
        self.counts["errors"] += 1
        self._win["errors"] += 1
        if self._win["example"] is None:
            self._win["example"] = str(message).strip()[:80]

    # ------------------------------------------------------------------ open / close
    def open(self):
        import av
        _install()
        _register(self)
        self.container = av.open(self.url, options=open_options(self.url, self.transport),
                                 timeout=(self.open_s, self.read_s))
        if not self.container.streams.video:
            raise ValueError(f"{self.name}: the stream has no video")
        self.vstream = self.container.streams.video[0]
        cc = self.vstream.codec_context
        cc.thread_count = self.threads                    # 1: each decoder error message belongs to this camera
        if self.threads > 1:
            cc.thread_type = "SLICE"                       # no frame threading: it holds pictures back (delay)
        self.codec = cc.name
        rate = self.vstream.average_rate or self.vstream.base_rate
        self.declared_fps = float(rate) if rate else None
        self.error = None

    def _close_container(self):
        c, self.container, self.vstream = self.container, None, None
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001 - a broken stream may fail to close; nothing to keep
                pass

    def close(self):
        """Ask frames() to stop (it returns within read_s at most); the container is closed by the decoding thread."""
        self.running = False

    # ------------------------------------------------------------------ pictures
    def frames(self):
        _register(self)
        try:
            while self.running:
                if self.container is None:
                    try:
                        self.open()
                    except Exception as e:  # noqa: BLE001 - camera off / wrong address: keep trying
                        self.error = f"{type(e).__name__}: {e}"
                        self._close_container()
                        if self.counts["frames"] == 0 and self.counts["reconnects"] == 0:
                            raise
                        time.sleep(RETRY_S)
                        continue
                try:
                    for fr in self.container.decode(self.vstream):
                        if not self.running:
                            break
                        out = self._one(fr)
                        if out is not None:
                            yield out
                except Exception as e:  # noqa: BLE001 - read timeout / connection dropped: open it again
                    self.error = f"{type(e).__name__}: {e}"
                if not self.running:
                    break
                self._close_container()
                self.counts["reconnects"] += 1
                self._say(f"camera {self.name}: stream ended or stalled ({self.error or 'end of stream'}); opening it "
                          "again")
                time.sleep(RETRY_S)
        finally:
            self._close_container()
            _unregister(self)

    def _one(self, fr):
        now = time.monotonic()
        flag = self.dmg.classify(bool(getattr(fr, "is_corrupt", False)), bool(fr.key_frame), now)
        self.counts["frames"] += 1
        self.counts[flag] += 1
        self._win["frames"] += 1
        if flag != "clean":
            self._win[flag] += 1
        self._maybe_log(now)
        if flag == "clean" or self.damaged == "use":
            img = fr.to_ndarray(format="bgr24")
            self.size = (int(img.shape[1]), int(img.shape[0]))
            if flag == "clean":
                self.last_clean = img
            return img, flag, now
        if self.last_clean is None:                       # nothing clean to hold yet: wait for a good picture
            return None
        self.counts["held"] += 1
        return self.last_clean, "held", now

    def _maybe_log(self, now):
        w = self._win
        if now - w["t"] < LOG_S:
            return
        if w["damaged"] or w["tainted"]:
            bad = w["damaged"] + w["tainted"]
            what = "held (last clean picture kept)" if self.damaged == "hold" else "passed on as they are"
            self._say(f"camera {self.name}: {bad} of {w['frames']} pictures in {now - w['t']:.0f} s damaged by lost "
                      f"packets ({100.0 * bad / max(1, w['frames']):.0f} %, {w['errors']} decoder errors"
                      + (f", e.g. '{w['example']}'" if w["example"] else "") + f"), {what}")
        self._win = {"frames": 0, "damaged": 0, "tainted": 0, "errors": 0, "t": now, "example": None}

    def _say(self, msg):
        if self.log is not None:
            try:
                self.log(msg)
            except Exception:  # noqa: BLE001 - a log must never stop the camera
                pass

    def health(self) -> dict:
        """Totals so far: frames, clean, damaged, tainted, held, decoder errors, reconnects + damaged_pct."""
        c = dict(self.counts)
        c["damaged_pct"] = round(100.0 * (c["damaged"] + c["tainted"]) / c["frames"], 1) if c["frames"] else 0.0
        return c


def grab_clean(url, drain_s=1.0, transport="tcp", open_s=10.0, read_s=5.0):
    """The newest CLEAN picture a stream sends within `drain_s` after its first clean one (a stream opens on frames
    it had buffered): (BGR image, {"held": damaged pictures skipped}) or (None, {"error": ...})."""
    s = Stream(url, transport=transport, damaged="hold", open_s=open_s, read_s=read_s, log=None)
    img, t_end = None, None
    gen = s.frames()
    try:
        for got, flag, now in gen:
            if flag == "clean":
                img = got
                if t_end is None:
                    t_end = now + float(drain_s)
            if t_end is not None and now >= t_end:
                break
            if t_end is None and s.counts["frames"] > 600:   # 10 s at 60 fps without one clean picture
                break
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"{type(e).__name__}: {e}"}
    finally:
        s.close()
        gen.close()                                      # closes the stream now, on this thread
    if img is None:
        return None, {"error": s.error or "no clean picture (every picture damaged)", **s.health()}
    return img, s.health()
