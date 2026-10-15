"""serve_calibrator: loopback-only by default, Host/Origin checks, no static file server, JSON 400s (review 4 M1/L1/L2)."""
import http.client
import json
import threading
from http.server import HTTPServer

import pytest

import serve_calibrator as sc


@pytest.fixture
def server(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    monkeypatch.setattr(sc, "FRAMES_DIR", str(frames))
    monkeypatch.setattr(sc, "CALIB_DIR", str(tmp_path))
    srv = HTTPServer(("127.0.0.1", 0), sc.Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv.server_address[1], tmp_path
    finally:
        srv.shutdown()
        srv.server_close()


def _req(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Content-Type": "application/json"} if body is not None else {}
    h.update(headers or {})
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    conn.request(method, path, body=data, headers=h)
    r = conn.getresponse()
    payload = r.read()
    conn.close()
    return r.status, payload


def _json(payload):
    return json.loads(payload.decode())


def test_default_bind_is_loopback():
    args = sc.parse_args([])
    assert args.host == "127.0.0.1" and args.port == 8001


def test_pages_still_served(server):
    port, _ = server
    st, body = _req(port, "GET", "/")
    assert st == 200 and b"<html" in body.lower()
    st, body = _req(port, "GET", "/labeler")
    assert st == 200


def test_untrusted_host_header_rejected(server):
    port, _ = server
    st, body = _req(port, "GET", "/api/clips", headers={"Host": "evil.example"})
    assert st == 403
    st, body = _req(port, "GET", "/api/clips", headers={"Host": f"localhost:{port}"})
    assert st == 200


def test_cross_site_post_rejected_and_nothing_written(server):
    port, tmp = server
    body = {"clip": "x"}
    st, _ = _req(port, "POST", "/api/seeds", body, headers={"Origin": "http://evil.example"})
    assert st == 403
    st, _ = _req(port, "POST", "/api/seeds", body, headers={"Origin": f"http://localhost:{port + 1}"})
    assert st == 403
    assert not (tmp / "x_seeds.json").exists()
    st, _ = _req(port, "POST", "/api/seeds", body, headers={"Origin": f"http://127.0.0.1:{port}"})
    assert st == 200 and (tmp / "x_seeds.json").exists()


def test_simple_request_content_type_rejected(server):
    """A no-cors cross-site fetch can only send text/plain etc.: require application/json."""
    port, tmp = server
    st, _ = _req(port, "POST", "/api/seeds", json.dumps({"clip": "y"}).encode(),
                 headers={"Content-Type": "text/plain"})
    assert st == 415 and not (tmp / "y_seeds.json").exists()


def test_head_and_arbitrary_files_not_served(server):
    port, _ = server
    st, body = _req(port, "HEAD", "/README.md")
    assert st in (404, 405, 501)
    st, body = _req(port, "GET", "/README.md")
    assert st == 404
    st, body = _req(port, "GET", "/tools/serve_calibrator.py")
    assert st == 404


def test_bad_input_returns_json_400(server):
    port, tmp = server
    # a frame exists for an unregistered view: used to raise StopIteration and drop the connection
    import cv2
    import numpy as np
    cv2.imwrite(str(tmp / "frames" / "nope.jpg"), np.zeros((20, 30, 3), np.uint8))
    pts = [{"name": n, "px": 1.0 + i, "py": 2.0 + i} for i, n in enumerate(list(sc.COURT_POINTS)[:4])]
    st, body = _req(port, "POST", "/api/calibrate", {"view": "nope", "fov_deg": 70, "points": pts})
    assert st == 400 and "error" in _json(body)
    view = sc.CAMERAS[0]["view"]
    bad = pts[:3] + [{"name": "not_a_point", "px": 1, "py": 1}]
    for p in ("/api/calibrate", "/api/auto-fov"):
        st, body = _req(port, "POST", p, {"view": view, "fov_deg": 70, "points": bad})
        assert st == 400 and "not_a_point" in _json(body)["error"]
    st, body = _req(port, "POST", "/api/auto-fov", {"view": "nope", "points": pts})
    assert st == 400 and "error" in _json(body)
    st, body = _req(port, "POST", "/api/label/save", b"{not json")
    assert st == 400 and "error" in _json(body)
    st, body = _req(port, "POST", "/api/calibrate", {"view": view})
    assert st == 400 and "error" in _json(body)
    st, body = _req(port, "POST", "/api/seeds", [1, 2])
    assert st == 400 and "error" in _json(body)


def test_body_size_capped(server, monkeypatch):
    port, tmp = server
    monkeypatch.setattr(sc, "MAX_BODY", 100)
    st, body = _req(port, "POST", "/api/seeds", {"clip": "z", "pad": "x" * 200})
    assert st == 413 and not (tmp / "z_seeds.json").exists()
