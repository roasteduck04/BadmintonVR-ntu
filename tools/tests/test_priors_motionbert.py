"""v0.7.5 monoprior: the MotionBERT CPU prior producer (priors/motionbert.py). A fake nn.Module stands in for
DSTformer; only test_real_checkpoint_loads touches models/motionbert's checkpoints (skipped when missing)."""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from analysis.pose24 import mono
from analysis.pose24.priors import __main__ as pcli
from analysis.pose24.priors import common, motionbert as mb

needs_mb = pytest.mark.skipif(not os.path.isdir(os.path.join(mb.MB_DIR, "lib")),
                              reason="models/motionbert code not present")


class _ConstPerCall(nn.Module):
    """Returns the call index (1, 2, ...) everywhere: overlapping windows must average."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.full(x.shape[:-1] + (3,), float(self.calls), dtype=x.dtype)


@pytest.mark.parametrize("F", [100, 243, 1877])
def test_windows_cover_all_frames(F):
    W = mb.windows(F)
    assert W[0][0] == 0 and W[-1][1] == F
    covered = np.zeros(F, int)
    for s, e in W:
        assert e - s == min(F, mb.CLIP_LEN)
        covered[s:e] += 1
    assert (covered >= 1).all()
    if F > mb.CLIP_LEN:
        assert W[-1] == (F - mb.CLIP_LEN, F)                   # the last window is right-aligned
        assert all(W[i + 1][0] == W[i][1] for i in range(len(W) - 2))
    # overlaps averaged: the fake returns its call index; frames seen by windows i and j get (i + j) / 2
    x = np.zeros((F, 17, 3), np.float32)
    out = mb.run_windows(x, _ConstPerCall(), flip=False)
    assert out.shape == (F, 17, 3)
    exp = np.zeros(F)
    for i, (s, e) in enumerate(W):
        exp[s:e] += i + 1
    np.testing.assert_allclose(out[:, 0, 0], exp / covered)


def test_normalise_uses_view_image_size():
    rng = np.random.default_rng(0)
    kp = np.zeros((4, 17, 3))
    kp[..., :2] = rng.uniform(0, 1000, (4, 17, 2))
    kp[..., 2] = rng.uniform(0.2, 1, (4, 17))
    w, h = 1920, 1080
    y = mb.normalise(kp, (w, h))
    np.testing.assert_allclose(y[..., :2], (kp[..., :2] - np.array([w, h]) / 2) / (min(w, h) / 2))
    np.testing.assert_allclose(y[..., 2], kp[..., 2])


@needs_mb
def test_normalise_crop_matches_crop_scale():
    """infer_wild's default path: crop_scale(motion, [1, 1]) once over the whole track (bbox of score != 0 points)."""
    mb.flip_fn()                                                         # models/motionbert's `lib` on sys.path
    from lib.utils.utils_data import crop_scale
    rng = np.random.default_rng(3)
    F = 300
    kp = np.zeros((F, 17, 3))
    kp[..., 0] = 700 + 40 * np.sin(np.arange(F) / 30)[:, None] + rng.uniform(-60, 60, (F, 17))
    kp[..., 1] = 400 + rng.uniform(-150, 150, (F, 17))
    kp[..., 2] = rng.uniform(0.3, 1, (F, 17))
    np.testing.assert_allclose(mb.crop_normalise(kp), crop_scale(kp, [1, 1]))
    # missing points (score 0, NaN xy) stay out of the bbox; filled points are normalised + clipped like the rest
    raw = kp.copy()
    raw[50:60, :12] = [np.nan, np.nan, 0.0]
    raw[5, 3] = [5000.0, -900.0, 0.0]                                  # an outlier with score 0: not in the bbox
    ref = crop_scale(np.where(np.isfinite(raw), raw, 0.0), [1, 1])
    filled, _ = mb.fill_gaps(raw)
    seen = np.isfinite(raw[..., :2]).all(-1) & (raw[..., 2] > 0)
    y = mb.crop_normalise(filled, seen)
    kept = seen.copy()
    kept[50:60] = False                                                  # mostly-missing frames are rebuilt by fill_gaps
    np.testing.assert_allclose(y[kept][:, :2], ref[kept][:, :2])
    assert np.isfinite(y).all() and (np.abs(y[..., :2]) <= 1).all()
    assert y.shape == kp.shape


def test_mb_lib_survives_a_foreign_lib_package(monkeypatch):
    import types
    monkeypatch.setitem(__import__("sys").modules, "lib", types.ModuleType("lib"))   # e.g. pywin32's win32/lib
    if not os.path.isdir(os.path.join(mb.MB_DIR, "lib")):
        pytest.skip("models/motionbert code not present")
    assert callable(mb.flip_fn())


def test_method_labels_and_paths():
    assert mb.method_label("lite") == "motionbert-lite"
    assert mb.method_label("full", "image") == "motionbert-full-image"
    assert mb.output_path("c", "v", "lite", "d") == os.path.join("d", "c", "v_motionbert-lite.npz")
    assert mb.output_path("c", "v", "lite", "d", norm="image") == os.path.join("d", "c", "v_motionbert-lite-image.npz")
    with pytest.raises(ValueError):
        mb.method_label("lite", "nope")


def test_image_size_falls_back_to_principal_point():
    K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])
    cams = {"v1": (K, np.eye(3), np.zeros(3))}
    assert mb.image_size({"cams": cams}, "v1") == (1920, 1080)                       # no image_size key
    assert mb.image_size({"cams": cams, "image_size": {"v1": [None, None]}}, "v1") == (1920, 1080)
    assert mb.image_size({"cams": cams, "image_size": {"v1": [1280, 720]}}, "v1") == (1280, 720)


def test_fill_gaps_conf_zero_on_mostly_missing_frames():
    kp = np.zeros((6, 17, 3))
    kp[..., 0] = np.arange(6)[:, None] * 10.0
    kp[..., 1] = 5.0
    kp[..., 2] = 0.8
    kp[2, :10] = [np.nan, np.nan, 0.0]          # 10 of 17 missing: > 50 %
    kp[4, :3] = [np.nan, np.nan, 0.0]           # 3 missing: kept
    filled, conf = mb.fill_gaps(kp)
    assert np.isfinite(filled[..., :2]).all()
    np.testing.assert_allclose(filled[2, :, 0], 20.0)                  # linear between frames 1 and 3
    np.testing.assert_allclose(filled[4, :3, 0], 40.0)
    assert conf[2] == 0
    np.testing.assert_allclose(conf[4], 0.8 * 14 / 17)
    np.testing.assert_allclose(conf[[0, 1, 3, 5]], 0.8)


# ---- axes: a synthetic body seen by a known camera; the fake model returns the root-relative truth --------------

def _rot(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rx @ Ry


def _scene(F=30):
    """COCO-17 body in a Y-down world (right arm reaching towards the camera) + camera; -> obs133, cam, X (F,17,3)."""
    K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
    R = _rot(0.3, 0.25)
    C = np.array([0.0, -3.0, -9.0])                                      # camera centre (world, Y down)
    t = -R @ C
    body = np.zeros((17, 3))
    body[[11, 12]] = [[0.12, -0.95, 0], [-0.12, -0.95, 0]]               # hips (left +x)
    body[[13, 14]] = [[0.13, -0.5, 0], [-0.13, -0.5, 0]]
    body[[15, 16]] = [[0.13, -0.08, 0], [-0.13, -0.08, 0]]
    body[[5, 6]] = [[0.2, -1.45, 0], [-0.2, -1.45, 0]]
    body[[7, 9]] = [[0.25, -1.2, 0.05], [0.25, -1.0, 0.1]]               # left arm down
    body[[8, 10]] = [[-0.3, -1.45, -0.3], [-0.3, -1.45, -0.6]]          # right arm towards the camera (-z world)
    body[0] = [0, -1.62, -0.1]
    body[1:5] = [[0.03, -1.65, -0.08], [-0.03, -1.65, -0.08], [0.08, -1.62, 0], [-0.08, -1.62, 0]]
    X = np.stack([body + [0.02 * f, 0, 0.01 * f] for f in range(F)])
    Xc = X @ R.T + t
    uv = Xc @ K.T
    uv = uv[..., :2] / uv[..., 2:]
    obs = np.zeros((F, 133, 3))
    obs[:, :17, :2], obs[:, :17, 2] = uv, 0.9
    obs[:, 17:, :2], obs[:, 17:, 2] = np.nan, 0.0
    return obs, (K, R, t), Xc


_H36M_LEFT, _H36M_RIGHT = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]


def _flip_h36m(data):
    """Local copy of MotionBERT's flip_data (x -> -x, H36M left <-> right joints) for tensors or arrays, so the axes
    test runs without models/motionbert."""
    out = data.clone() if isinstance(data, torch.Tensor) else np.array(data, copy=True)
    out[..., 0] *= -1
    out[..., _H36M_LEFT + _H36M_RIGHT, :] = out[..., _H36M_RIGHT + _H36M_LEFT, :]
    return out


@needs_mb
def test_flip_stand_in_matches_motionbert_flip_data():
    x = torch.randn(1, 5, 17, 3)
    torch.testing.assert_close(_flip_h36m(x), mb.flip_fn()(x))
    a = np.random.default_rng(1).standard_normal((5, 17, 3))
    np.testing.assert_allclose(_flip_h36m(a), mb.flip_fn()(a))


class _TruthModel(nn.Module):
    """Returns `truth` for the expected input, flip_data(truth) for the flipped input (as DSTformer would)."""

    def __init__(self, x_in, truth, flip_fn):
        super().__init__()
        self.x_in, self.truth, self.flip_fn = torch.as_tensor(x_in), torch.as_tensor(truth), flip_fn

    def forward(self, x):
        T = x.shape[1]
        same = torch.allclose(x[0, :, :, :2], self.x_in[:T, :, :2].to(x.dtype), atol=1e-5)
        out = self.truth[:T] if same else self.flip_fn(self.truth[:T])
        return out[None].to(x.dtype).clone()


@pytest.mark.parametrize("norm", ["crop", "image"])
def test_motionbert_axes_match_projection(norm, monkeypatch):
    monkeypatch.setattr(mb, "flip_fn", lambda: _flip_h36m)              # runs without models/motionbert
    obs, cam, Xc = _scene()
    K = cam[0]
    wh = (1920, 1080)
    h36 = common.coco133_to_h36m(obs)
    x_in = mb.normalise(h36, wh) if norm == "image" else mb.crop_normalise(h36)
    # truth: H36M joints in the camera frame, root-relative, in normalised image units (scale f / Z_root)
    Xh = np.full((len(Xc), 17, 3), np.nan)
    for h, c in {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 9: 0, 11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10}.items():
        Xh[:, h] = Xc[:, c]
    Xh[:, 0] = Xc[:, [11, 12]].mean(1)
    Xh[:, 8] = Xc[:, [5, 6]].mean(1)
    Xh[:, 7] = (Xh[:, 0] + Xh[:, 8]) / 2
    Xh[:, 10] = Xc[:, [3, 4]].mean(1)
    rel_true = Xh - Xh[:, :1]
    truth = rel_true * (K[0, 0] / Xh[:, :1, 2:]) / (min(wh) / 2)
    model = _TruthModel(x_in.astype(np.float32), truth.astype(np.float32), _flip_h36m)
    out = mb.player_prior(obs, wh, model, norm=norm)
    rel = out["rel_cam"]
    assert rel.shape == (len(obs), 13, 3)
    assert out["kp"].tolist() == [k for _, k in common.H36M_TO_KP]
    hk = [h for h, _ in common.H36M_TO_KP]
    z_true = rel_true[:, hk, 2]
    # z increases away from the camera (OpenCV): same sign and order as the true camera-frame depths
    assert np.corrcoef(rel[..., 2].ravel(), z_true.ravel())[0, 1] > 0.999
    wr = hk.index(16)                                                     # right wrist: nearer the camera than the hip
    assert (z_true[:, wr] < 0).all() and (rel[:, wr, 2] < 0).all()
    # x right, y down: rel xy follows the image offsets from the root
    for c in (0, 1):
        assert np.corrcoef(rel[..., c].ravel(), rel_true[:, hk, c].ravel())[0, 1] > 0.999
    np.testing.assert_allclose(out["conf"], 0.9)
    # mono's depth rows accept it (F frames on the cache timeline) and recover metric depths via the leg ratio
    L = {"edges": {"thigh": 0.45, "shank": 0.42}}
    rows = mono._depth_rows(out, L, np.array([0, 0, 1.0]), len(obs))
    col = {k: i for i, k in enumerate(out["kp"])}
    d_true = z_true[rows["t"], [col[k] for k in rows["k"]]]
    assert len(rows["d"]) == len(obs) * 13
    assert np.corrcoef(rows["d"], d_true)[0, 1] > 0.999


# ---- produce / CLI ---------------------------------------------------------------------------------------------

class _Echo(nn.Module):
    """xy echo + z = x (flip-consistent enough for shape / bookkeeping tests)."""

    def forward(self, x):
        return torch.cat([x[..., :2], x[..., :1]], -1)


def _cache(F=50):
    obs, cam, _ = _scene(F)
    gap = obs.copy()
    gap[10:15, :12] = [np.nan, np.nan, 0.0]                              # mostly missing frames
    none = np.full_like(obs, np.nan)
    none[..., 2] = 0.0
    return {"views": ["v1", "v2"], "cams": {"v1": cam, "v2": cam}, "fps": 60.0, "frame_count": F,
            "build_id": "b123", "timeline_of": "c0", "excluded": {"v3": "health"},
            "obs": {"P1": {"v1": gap, "v2": obs}, "P2": {"v1": none, "v2": obs}}}


@pytest.mark.parametrize("norm", ["crop", "image"])
def test_produce_writes_prior_contract(tmp_path, norm, monkeypatch):
    monkeypatch.setattr(mb, "flip_fn", lambda: _flip_h36m)              # runs without models/motionbert
    cache = _cache()
    path = mb.produce("c1", "v1", "lite", model=_Echo(), cache=cache, out_dir=str(tmp_path), log=lambda *_: None,
                      norm=norm)
    label = "motionbert-lite" + ("-image" if norm == "image" else "")
    assert path == os.path.join(str(tmp_path), "c1", f"v1_{label}.npz")
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        assert "P2_kp" not in z.files                                   # no 2D in v1: left out
    assert meta == {"method": label, "clip": "c1", "view": "v1", "timeline_of": "c0", "fps": 60.0,
                    "frame_count": 50, "build_id": "b123", "norm": norm}
    p = mono.load_prior(path, "P1")
    assert p["rel_cam"].shape == (50, 13, 3) and p["conf"].shape == (50,)
    assert (p["conf"][10:15] == 0).all() and (p["conf"][:10] > 0).all()
    assert np.isfinite(p["rel_cam"]).all()
    assert mono.load_prior(path, "P2") is None


def test_cli_default_views_and_method(monkeypatch, tmp_path):
    cache = _cache()
    calls = []
    monkeypatch.setattr(pcli, "_load_cache", lambda clip: cache)
    monkeypatch.setattr(mb, "load_model", lambda kind: _Echo())
    monkeypatch.setattr(mb, "produce",
                        lambda clip, view, kind, **kw: calls.append((clip, view, kind, kw["norm"])) or "p")
    pcli.main(["motionbert", "--clip", "c1"])
    assert calls == [("c1", "v1", "lite", "crop"), ("c1", "v2", "lite", "crop")]   # kept views; excluded v3 not run
    calls.clear()
    pcli.main(["motionbert", "--clip", "c1", "--views", "v2", "--model", "full", "--norm", "image"])
    assert calls == [("c1", "v2", "full", "image")]
    with pytest.raises(SystemExit):
        pcli.main(["motionbert", "--clip", "c1", "--views", "nope"])


@pytest.mark.slow
@pytest.mark.needs_data
@pytest.mark.parametrize("kind", ["lite", "full"])
def test_real_checkpoint_loads(kind):
    cfg, ckpt = mb.model_paths(kind)
    if not (os.path.exists(cfg) and os.path.exists(ckpt)):
        pytest.skip(f"MotionBERT {kind} config / checkpoint missing")
    model = mb.load_model(kind)
    rng = np.random.default_rng(0)
    x = np.zeros((mb.CLIP_LEN, 17, 3), np.float32)
    x[..., :2] = rng.uniform(-0.5, 0.5, (mb.CLIP_LEN, 17, 2))
    x[..., 2] = rng.uniform(0.5, 1, (mb.CLIP_LEN, 17))
    out = mb.run_windows(x, model, flip=True)
    assert out.shape == (mb.CLIP_LEN, 17, 3) and np.isfinite(out).all()
