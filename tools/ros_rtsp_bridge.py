"""ROS 2 camera topics -> RTSP streams the live server reads (v0.8.7).

Each Raspberry Pi camera publishes JPEG frames on /<unit>/camera/image_raw/compressed (camera_ros). This node pipes
every unit's frames into its own ffmpeg, which encodes H.264 and publishes rtsp://<server>/<unit> on an RTSP server
(MediaMTX) - those addresses go into the server page's camera rows (step 1, Type Network).

    python tools/ros_rtsp_bridge.py                                   # unit1..unit4 -> rtsp://127.0.0.1:8554/unitN
    python tools/ros_rtsp_bridge.py --units unit1,unit3 --flip        # two cameras, mounted upside down
    python tools/ros_rtsp_bridge.py --server rtsp://192.168.0.10:8554 # MediaMTX on another PC

Run it in a terminal where ROS 2 is active (rclpy), with ffmpeg 5.1 or newer on PATH and MediaMTX running. MediaMTX is
on this PC by default, so the node publishes to 127.0.0.1; only the readers (the live server, VLC) need this PC's
network address: rtsp://<this PC's address>:8554/unit1.

What it does differently from a plain pipe:
- frames are stamped with the time they arrive (-use_wallclock_as_timestamps) and sent at a constant rate, so a frame
  lost on Wi-Fi does not make the stream run ahead of real time;
- --flip turns the picture 180 degrees for a camera mounted upside down (the pose model needs upright people);
- an ffmpeg that exits is started again, and each unit's real frame rate is printed every few seconds.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

TOPIC = "/{unit}/camera/image_raw/compressed"
RESTART_S = 2.0                  # an ffmpeg that exited is started again after this long
REPORT_S = 5.0


def ffmpeg_cmd(unit, server="rtsp://127.0.0.1:8554", fps=60, bitrate="8M", preset="veryfast", flip=False,
               ffmpeg="ffmpeg") -> list:
    """The ffmpeg command of one unit: JPEG frames on stdin -> H.264 -> <server>/<unit>."""
    fps = int(round(float(fps)))
    cmd = [ffmpeg, "-loglevel", "warning",
           "-use_wallclock_as_timestamps", "1",          # the time each frame arrived, not a counted 1/fps
           "-f", "image2pipe", "-c:v", "mjpeg", "-i", "-"]
    if flip:
        cmd += ["-vf", "hflip,vflip"]                    # 180 degrees: a camera mounted upside down
    cmd += ["-fps_mode", "cfr", "-r", str(fps),          # constant rate: a lost frame is repeated, not skipped
            "-c:v", "libx264", "-preset", preset, "-tune", "zerolatency",
            "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", bitrate,
            "-g", str(fps), "-pix_fmt", "yuv420p",       # a keyframe every second: a reader joins within 1 s
            "-f", "rtsp", "-rtsp_transport", "tcp", f"{server.rstrip('/')}/{unit}"]
    return cmd


class Pipe:
    """One unit's ffmpeg: write(jpeg) feeds it, restarts it when it has exited, counts the frames."""

    def __init__(self, unit, cmd, log=print, popen=subprocess.Popen, clock=time.monotonic):
        self.unit, self.cmd, self.log, self.popen, self.clock = unit, list(cmd), log, popen, clock
        self.lock = threading.Lock()
        self.proc, self.down_since, self.frames, self.restarts = None, None, 0, 0
        self._start()

    def _start(self):
        self.proc = self.popen(self.cmd, stdin=subprocess.PIPE)
        self.down_since = None

    def write(self, data) -> bool:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:        # ffmpeg exited (RTSP server not up yet?)
                now = self.clock()
                if self.down_since is None:
                    self.down_since = now
                    self.log(f"{self.unit}: ffmpeg exited - is the RTSP server (MediaMTX) running? trying again")
                if now - self.down_since < RESTART_S:
                    return False
                self.restarts += 1
                self._start()
            try:
                self.proc.stdin.write(data)
                self.frames += 1
                return True
            except (BrokenPipeError, OSError, ValueError):
                return False

    def close(self):
        with self.lock:
            p, self.proc = self.proc, None
        if p is not None:
            try:
                p.stdin.close()
            except (OSError, ValueError):
                pass
            p.terminate()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", default="unit1,unit2,unit3,unit4", help="camera namespaces, comma-separated")
    ap.add_argument("--topic", default=TOPIC, help="topic of one unit; {unit} is replaced (default %(default)s)")
    ap.add_argument("--server", default="rtsp://127.0.0.1:8554", help="RTSP server to publish to (default: this PC)")
    ap.add_argument("--fps", type=float, default=60.0, help="frame rate the cameras send")
    ap.add_argument("--bitrate", default="8M", help="H.264 bitrate per camera (default %(default)s)")
    ap.add_argument("--preset", default="veryfast", help="x264 preset; ultrafast is lighter but blockier")
    ap.add_argument("--flip", action="store_true", help="turn every picture 180 degrees (cameras mounted upside down)")
    ap.add_argument("--flip-units", default="", help="turn only these units 180 degrees, comma-separated")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    units = [u.strip() for u in a.units.split(",") if u.strip()]
    flips = set(units) if a.flip else {u.strip() for u in a.flip_units.split(",") if u.strip()}
    import rclpy                                         # here, so the helpers above import without ROS
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    class Bridge(Node):
        def __init__(self):
            super().__init__("rtsp_bridge")
            self.pipes = {}
            for u in units:
                self.pipes[u] = Pipe(u, ffmpeg_cmd(u, a.server, a.fps, a.bitrate, a.preset, u in flips, a.ffmpeg),
                                     log=lambda m: self.get_logger().warning(m))
                # a callback group per unit: one slow ffmpeg does not hold the other cameras back
                self.create_subscription(CompressedImage, a.topic.format(unit=u),
                                         lambda msg, u=u: self.pipes[u].write(bytes(msg.data)),
                                         qos_profile_sensor_data, callback_group=MutuallyExclusiveCallbackGroup())
            self.last = {u: 0 for u in units}
            self.create_timer(REPORT_S, self.report)
            for u in units:
                self.get_logger().info(f"{a.topic.format(unit=u)} -> {a.server.rstrip('/')}/{u}"
                                       + (" (turned 180 degrees)" if u in flips else ""))

        def report(self):
            parts = []
            for u, p in self.pipes.items():
                n, self.last[u] = p.frames - self.last[u], p.frames
                parts.append(f"{u} {n / REPORT_S:.0f} fps" + (" (NO FRAMES)" if n == 0 else ""))
            self.get_logger().info(", ".join(parts))

        def destroy_node(self):
            for p in self.pipes.values():
                p.close()
            super().destroy_node()

    rclpy.init()
    node = Bridge()
    ex = MultiThreadedExecutor(num_threads=len(units) + 1)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
