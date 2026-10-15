"""OSNet-x0.25 person re-identification embeddings for tracker3d (--reid osnet, SP3 spec § 2.6).

Model definition vendored from torchreid (https://github.com/KaiyangZhou/deep-person-reid,
torchreid/models/osnet.py), Copyright (c) 2018 Kaiyang Zhou, MIT License - only the parts osnet_x0_25 needs.

Copyright (c) 2018 Kaiyang Zhou

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Weights: models/reid/osnet_x0_25_msmt17.pth (torchreid model zoo, MIT; downloaded 2026-09-27, user-approved);
the MSMT17 classifier is dropped, the 512-d feature is used. Input 256 x 128 RGB, ImageNet normalisation.

The checkpoint (567 tensors, 2.31 M params without the classifier) is plain BatchNorm throughout (IN=False), one
ChannelGate shared by the four streams of each OSBlock, fc = Linear(128, 512) + BatchNorm1d + ReLU; it loads
strictly into the definition below. torch is imported lazily so analysis.people imports without it."""
from __future__ import annotations

import os

import numpy as np

from analysis import appearance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WEIGHTS = os.path.join(REPO_ROOT, "models", "reid", "osnet_x0_25_msmt17.pth")    # not CWD-relative (review 5 A3)
CACHE_DIR = os.path.join("data", "pose_cache")
_MEAN, _STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def osnet_x0_25():
    import torch.nn as nn
    import torch.nn.functional as Fn

    class ConvLayer(nn.Module):
        def __init__(self, i, o, k, stride=1, padding=0, groups=1):
            super().__init__()
            self.conv = nn.Conv2d(i, o, k, stride=stride, padding=padding, bias=False, groups=groups)
            self.bn = nn.BatchNorm2d(o)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Conv1x1(nn.Module):
        def __init__(self, i, o, stride=1, groups=1):
            super().__init__()
            self.conv = nn.Conv2d(i, o, 1, stride=stride, padding=0, bias=False, groups=groups)
            self.bn = nn.BatchNorm2d(o)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Conv1x1Linear(nn.Module):
        def __init__(self, i, o, stride=1):
            super().__init__()
            self.conv = nn.Conv2d(i, o, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(o)

        def forward(self, x):
            return self.bn(self.conv(x))

    class LightConv3x3(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv1 = nn.Conv2d(i, o, 1, stride=1, padding=0, bias=False)
            self.conv2 = nn.Conv2d(o, o, 3, stride=1, padding=1, bias=False, groups=o)
            self.bn = nn.BatchNorm2d(o)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv2(self.conv1(x))))

    class ChannelGate(nn.Module):
        def __init__(self, c, reduction=16):
            super().__init__()
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(c, c // reduction, 1, bias=True, padding=0)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(c // reduction, c, 1, bias=True, padding=0)
            self.gate_activation = nn.Sigmoid()

        def forward(self, x):
            g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
            return x * g

    class OSBlock(nn.Module):
        def __init__(self, i, o, bottleneck_reduction=4):
            super().__init__()
            mid = o // bottleneck_reduction
            self.conv1 = Conv1x1(i, mid)
            self.conv2a = LightConv3x3(mid, mid)
            self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
            self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
            self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
            self.gate = ChannelGate(mid)
            self.conv3 = Conv1x1Linear(mid, o)
            self.downsample = Conv1x1Linear(i, o) if i != o else None

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) \
                + self.gate(self.conv2d(x1))
            x3 = self.conv3(x2)
            identity = self.downsample(x) if self.downsample is not None else x
            return Fn.relu(x3 + identity)

    def layer(n, i, o, reduce):
        mods = [OSBlock(i, o)] + [OSBlock(o, o) for _ in range(n - 1)]
        if reduce:
            mods.append(nn.Sequential(Conv1x1(o, o), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*mods)

    class OSNet(nn.Module):
        def __init__(self, channels=(16, 64, 96, 128), layers=(2, 2, 2), feature_dim=512):
            super().__init__()
            self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.conv2 = layer(layers[0], channels[0], channels[1], True)
            self.conv3 = layer(layers[1], channels[1], channels[2], True)
            self.conv4 = layer(layers[2], channels[2], channels[3], False)
            self.conv5 = Conv1x1(channels[3], channels[3])
            self.global_avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim),
                                    nn.ReLU(inplace=True))

        def forward(self, x):
            x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
            return self.fc(self.global_avgpool(x).flatten(1))

    return OSNet()


def load_model(path=WEIGHTS):
    """The vendored OSNet-x0.25 with the checkpoint loaded strictly (classifier dropped), in eval mode."""
    import torch
    m = osnet_x0_25()
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()
          if not k.replace("module.", "").startswith("classifier")}
    m.load_state_dict(sd, strict=True)
    return m.eval()


def embed(model, crops):
    """BGR crops (any size) -> (N, 512) float32 L2-normalised embeddings."""
    import cv2
    import torch
    if not len(crops):
        return np.zeros((0, 512), np.float32)
    x = np.stack([cv2.resize(c, (128, 256), interpolation=cv2.INTER_LINEAR)[..., ::-1] / 255.0 for c in crops])
    x = ((x - _MEAN) / _STD).transpose(0, 3, 1, 2).astype(np.float32)
    with torch.no_grad():
        e = model(torch.from_numpy(np.ascontiguousarray(x))).numpy()
    return (e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def crop(img, kps, s_min=0.3, pad=0.1):
    """Person crop from the >= 5 body joints (COCO-17 part of the 133) with calibrated score >= s_min, padded by
    pad x the joint box height; None when too few joints or the crop is tiny."""
    ok = kps[:17, 2] >= s_min
    if ok.sum() < 5:
        return None
    p = kps[:17][ok, :2]
    lo, hi = p.min(axis=0), p.max(axis=0)
    h = hi[1] - lo[1]
    lo, hi = lo - pad * h, hi + pad * h
    x0, y0 = int(max(0, lo[0])), int(max(0, lo[1]))
    x1, y1 = int(min(img.shape[1], hi[0])), int(min(img.shape[0], hi[1]))
    if x1 - x0 < 8 or y1 - y0 < 16:
        return None
    return img[y0:y1, x0:x1]


def cache_path(clip, cache_dir=None):
    return os.path.join(cache_dir or CACHE_DIR, f"{clip}_osnet.npz")


def extract_clip(clip, views, registry=None, step=5, batch=32, log=print, data_dir="data", cache_dir=None,
                 weights=WEIGHTS):
    """Embeddings of every detection on every `step`-th raw frame of each view's video -> <clip>_osnet.npz
    ({view}_idx (raw frame, original det index) int32 (N,2), {view}_feat float16 (N,512)). Views already in the
    cache or without a video are skipped; the cache is saved after each view. Decodes the full-resolution video
    (slow on 4K)."""
    from analysis import video_io
    from analysis.cvdebug import load_raw_kp2d
    out = cache_path(clip, cache_dir)
    model = load_model(weights)
    vids = video_io.resolve_videos(clip, registry=registry)
    have = appearance.read_npz(out) if os.path.exists(out) else {}
    for v in views:
        kp = os.path.join(data_dir, "pose2d", f"{v}_kp2d.json")
        if f"{v}_idx" in have and v in appearance.fresh_views(
                {k: a for k, a in have.items() if k.startswith(f"{v}_")}, {v: kp}, log=log, what="OSNet"):
            log(f"    osnet {v}: cached")
            continue
        if not vids.get(v):
            log(f"    osnet {v}: no video, skipped")
            continue
        frames, header = load_raw_kp2d(kp)
        video_io.check_video(v, vids[v], header, log=log)       # kp2d pixels == decoded pixels, same frame count
        idx, feats, buf, keys = [], [], [], []

        def flush():
            if buf:
                feats.append(embed(model, buf))
                idx.extend(keys)
            buf.clear()
            keys.clear()

        with video_io.VideoReader(vids[v]) as rd:
            for n, r in enumerate(range(0, len(frames), step)):
                if not len(frames[r]):
                    continue
                img = rd.get(r)
                if img is None:
                    continue
                for d, k in enumerate(frames[r]):
                    c = crop(img, k)
                    if c is not None:
                        buf.append(c)
                        keys.append((r, d))
                if len(buf) >= batch:
                    flush()
                if n % 60 == 0:
                    log(f"    osnet {v}: {r}/{len(frames)}")
            flush()
        have[f"{v}_idx"] = np.array(idx, np.int32).reshape(-1, 2)
        have[f"{v}_feat"] = (np.concatenate(feats) if feats else np.zeros((0, 512))).astype(np.float16)
        have[appearance.kp2d_key(v)] = np.array(appearance.kp2d_sha1(kp))          # freshness key (L11)
        log(f"    osnet {v}: {len(idx)} embeddings")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        np.savez_compressed(out, **have)
    return out


def load_cache(clip, cache_dir=None, data_dir="data", log=print):
    """<clip>_osnet.npz -> {view: {(raw frame, det index): (512,) float64}} (the reid.Appearance osnet keying).
    A view whose kp2d file changed since extraction is dropped (appearance.fresh_views)."""
    p = cache_path(clip, cache_dir)
    if not os.path.exists(p):
        raise FileNotFoundError(f"no OSNet cache {p}: run python -m analysis.people reid-extract --clip {clip}")
    z = appearance.read_npz(p)
    paths = {k[:-4]: os.path.join(data_dir, "pose2d", f"{k[:-4]}_kp2d.json") for k in z if k.endswith("_idx")}
    out = {}
    for v in appearance.fresh_views(z, paths, log=log, what="OSNet"):       # review 1 L11
        feat = z[f"{v}_feat"].astype(np.float64)
        out[v] = {(int(a), int(b)): feat[i] for i, (a, b) in enumerate(z[f"{v}_idx"])}
    return out
