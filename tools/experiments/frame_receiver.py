"""Round-5 experiment (test branch only): count the frames a stream sends with output.exp_socket=udp://127.0.0.1:PORT.

    python tools/experiments/frame_receiver.py [--port 8899]      # Ctrl+C prints the totals

Prints every second: frames received, frames/s, and the send -> receive delay (the frame's t_written_ns is the sender's
monotonic clock, so the delay is only meaningful on the same machine)."""
from __future__ import annotations

import argparse
import json
import socket
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    a = ap.parse_args(argv)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((a.host, a.port))
    s.settimeout(1.0)
    n, last_n, t_last, delays, steps = 0, 0, time.monotonic(), [], set()
    print(f"listening on udp://{a.host}:{a.port}", flush=True)
    try:
        while True:
            try:
                data, _ = s.recvfrom(65535)
                f = json.loads(data)
                n += 1
                steps.add(f.get("step"))
                if f.get("t_written_ns"):
                    delays.append((time.monotonic_ns() - f["t_written_ns"]) / 1e6)
            except socket.timeout:
                pass
            now = time.monotonic()
            if now - t_last >= 1.0 and n != last_n:
                d = sorted(delays[-200:])
                print(f"{n} frames ({(n - last_n) / (now - t_last):.1f}/s), send -> receive "
                      f"{d[len(d) // 2]:.2f} ms median" if d else f"{n} frames", flush=True)
                last_n, t_last = n, now
    except KeyboardInterrupt:
        print(f"total {n} frames, {len(steps)} distinct steps", flush=True)


if __name__ == "__main__":
    main()
