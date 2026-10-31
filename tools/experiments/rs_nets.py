"""Round-6 experiment (test branch only): the racket and shuttle networks in plain PyTorch, so they can be exported to
ONNX (and run by TensorRT) without the OpenMMLab stack or the TrackNetV3 repository.

- RacketVision racket detector: RTMDet-M (CSPNeXt-P5 deepen 0.67 / widen 0.75 + CSPNeXtPAFPN + RTMDetSepBNHead,
  3 classes, 640x640, BGR input). Exported up to the decoded boxes + sigmoid scores of every prior (8400), before NMS:
  the NMS runs in NumPy, as for the person detector (analysis/live/det_split.py).
- RacketVision racket keypoints: RTMPose-M (same backbone, last stage only + RTMCCHead, 5 keypoints, 256x256, RGB).
  Our fine-tuned checkpoint data/racket_finetune_run/best_final_epoch_60.pth (the "bootstrap" variant). Exported to
  its SimCC vectors (simcc_x, simcc_y), the form rtmlib's RTMPose decodes.
- TrackNetV3 (MIT, github.com/qaz812345/TrackNetV3, model.py): 8 frames + the background (median) image at 288x512,
  27 channels in, 8 heatmaps out.

Layer names follow the checkpoints exactly, so their state dicts load with strict=True (the layer counts are read
from the checkpoint). mmengine checkpoints are read without mmengine (load_mm_state)."""
from __future__ import annotations

import importlib.abc
import importlib.util
import re
import sys
import types

import torch
import torch.nn as nn


# ------------------------------------------------------------------------------------------------ checkpoints
class _Meta(type):
    def __getattr__(cls, n):
        return _Meta(n, (_Stub,), {})


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.__dict__["state"] = s


class _FakeModule(types.ModuleType):
    def __getattr__(self, n):
        return _Meta(n, (_Stub,), {})


class _FakeMM(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Stands in for mmengine / mmcv / mmdet / mmpose while a checkpoint's metadata is unpickled."""
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("mmengine", "mmcv", "mmdet", "mmpose"):
            return importlib.util.spec_from_loader(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return _FakeModule(spec.name)

    def exec_module(self, m):
        m.__path__ = []


def load_mm_state(path) -> dict:
    """The `state_dict` of an mmengine checkpoint (EMA weights when trained with EMAHook: mmengine swaps them in before
    saving), without mmengine installed. Only for checkpoints we trust (ours, RacketVision's published ones)."""
    finder = _FakeMM()
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in ("mmengine", "mmcv", "mmdet", "mmpose")}
    sys.meta_path.insert(0, finder)
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        sys.meta_path.remove(finder)
        for k in [k for k in sys.modules if k.split(".")[0] in ("mmengine", "mmcv", "mmdet", "mmpose")]:
            if k not in saved:
                del sys.modules[k]
    return ck["state_dict"] if "state_dict" in ck else ck


def _count(sd, prefix) -> int:
    """Number of numbered children `prefix.N.` in a state dict."""
    pat = re.compile(re.escape(prefix) + r"\.(\d+)\.")
    return 1 + max((int(m.group(1)) for k in sd for m in [pat.match(k)] if m), default=-1)


# ------------------------------------------------------------------------------------------------ CSPNeXt parts
class ConvModule(nn.Module):
    """mmcv ConvModule: conv (no bias) + BN + SiLU."""
    def __init__(self, cin, cout, k, stride=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.depthwise_conv = ConvModule(cin, cin, k, groups=cin)
        self.pointwise_conv = ConvModule(cin, cout, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class CSPNeXtBlock(nn.Module):
    def __init__(self, c, add_identity):
        super().__init__()
        self.conv1 = ConvModule(c, c, 3)
        self.conv2 = DepthwiseSeparableConvModule(c, c, 5)
        self.add_identity = add_identity

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add_identity else y


class ChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1, bias=True)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        return x * self.act(self.fc(x.mean((2, 3), keepdim=True)))


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, n_blocks, add_identity, attention):
        super().__init__()
        mid = cout // 2
        self.main_conv = ConvModule(cin, mid, 1)
        self.short_conv = ConvModule(cin, mid, 1)
        self.final_conv = ConvModule(2 * mid, cout, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid, add_identity) for _ in range(n_blocks)])
        self.attention = ChannelAttention(2 * mid) if attention else None

    def forward(self, x):
        y = torch.cat([self.blocks(self.main_conv(x)), self.short_conv(x)], 1)
        if self.attention is not None:
            y = self.attention(y)
        return self.final_conv(y)


class SPPBottleneck(nn.Module):
    def __init__(self, cin, cout, ks=(5, 9, 13)):
        super().__init__()
        mid = cin // 2
        self.conv1 = ConvModule(cin, mid, 1)
        self.poolings = nn.ModuleList([nn.MaxPool2d(k, 1, k // 2) for k in ks])
        self.conv2 = ConvModule(mid * (len(ks) + 1), cout, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [p(x) for p in self.poolings], 1))


class CSPNeXt(nn.Module):
    """CSPNeXt-P5 backbone; channels and block counts from the state dict (prefix `backbone`)."""
    def __init__(self, sd, prefix="backbone", out_indices=(4,)):
        super().__init__()
        w = lambda k: sd[f"{prefix}.{k}"]
        s0, s2 = w("stem.0.conv.weight").shape[0], w("stem.2.conv.weight").shape[0]
        self.stem = nn.Sequential(ConvModule(3, s0, 3, 2), ConvModule(s0, s0, 3), ConvModule(s0, s2, 3))
        cin = s2
        self.out_indices = out_indices
        for i in range(1, 5):
            p = f"stage{i}"
            cout = w(f"{p}.0.conv.weight").shape[0]
            layers = [ConvModule(cin, cout, 3, 2)]
            spp = f"{prefix}.{p}.1.conv1.conv.weight" in sd
            if spp:
                layers.append(SPPBottleneck(cout, cout))
            j = len(layers)
            n = _count(sd, f"{prefix}.{p}.{j}.blocks")
            layers.append(CSPLayer(cout, cout, n, add_identity=not spp,
                                   attention=f"{prefix}.{p}.{j}.attention.fc.weight" in sd))
            setattr(self, p, nn.Sequential(*layers))
            cin = cout

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i in range(1, 5):
            x = getattr(self, f"stage{i}")(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


# ------------------------------------------------------------------------------------------------ RTMPose head
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class RTMCCBlock(nn.Module):
    """Gated attention unit (mmpose RTMCCBlock, attn_type self-attention, no rel. bias / pos. enc.)."""
    def __init__(self, dim, e, s):
        super().__init__()
        self.e, self.s = e, s
        self.ln = ScaleNorm(dim)
        self.uv = nn.Linear(dim, 2 * e + s, bias=False)
        self.gamma = nn.Parameter(torch.ones(2, s))
        self.beta = nn.Parameter(torch.zeros(2, s))
        self.o = nn.Linear(e, dim, bias=False)
        self.res_scale = Scale(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.act(self.uv(self.ln(x)))
        u, v, base = torch.split(y, [self.e, self.e, self.s], dim=2)
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        q, k = base[:, :, 0], base[:, :, 1]
        kernel = torch.square(torch.relu(torch.bmm(q, k.transpose(1, 2)) / (self.s ** 0.5)))
        return self.res_scale(x) + self.o(u * torch.bmm(kernel, v))


class RTMCCHead(nn.Module):
    def __init__(self, sd, prefix="head"):
        super().__init__()
        fl = sd[f"{prefix}.final_layer.weight"]
        k, hid = fl.shape[0], sd[f"{prefix}.mlp.1.weight"].shape[0]
        self.final_layer = nn.Conv2d(fl.shape[1], k, fl.shape[2], padding=fl.shape[2] // 2)
        flat = sd[f"{prefix}.mlp.1.weight"].shape[1]
        self.mlp = nn.Sequential(ScaleNorm(flat), nn.Linear(flat, hid, bias=False))
        e = sd[f"{prefix}.gau.o.weight"].shape[1]
        s = sd[f"{prefix}.gau.gamma"].shape[1]
        self.gau = RTMCCBlock(hid, e, s)
        self.cls_x = nn.Linear(hid, sd[f"{prefix}.cls_x.weight"].shape[0], bias=False)
        self.cls_y = nn.Linear(hid, sd[f"{prefix}.cls_y.weight"].shape[0], bias=False)

    def forward(self, feat):
        x = self.final_layer(feat).flatten(2)
        x = self.gau(self.mlp(x))
        return self.cls_x(x), self.cls_y(x)


class RacketPose(nn.Module):
    """RTMPose-M racket model: NCHW float RGB already normalised (mean/std) -> (simcc_x, simcc_y)."""
    MEAN = (123.675, 116.28, 103.53)
    STD = (58.395, 57.12, 57.375)

    def __init__(self, sd):
        super().__init__()
        self.backbone = CSPNeXt(sd, "backbone", out_indices=(4,))
        self.head = RTMCCHead(sd, "head")

    def forward(self, x):
        return self.head(self.backbone(x)[-1])


# ------------------------------------------------------------------------------------------------ RTMDet
class CSPNeXtPAFPN(nn.Module):
    def __init__(self, sd, prefix="neck"):
        super().__init__()
        w = lambda k: sd[f"{prefix}.{k}"]
        n_lv = _count(sd, f"{prefix}.out_convs")
        ins = [w(f"downsamples.{i}.conv.weight").shape[0] for i in range(n_lv - 1)]
        ins.append(w("reduce_layers.0.conv.weight").shape[1])       # [192, 384, 768]
        out = w("out_convs.0.conv.weight").shape[0]
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(n_lv - 1, 0, -1):
            j = n_lv - 1 - idx
            self.reduce_layers.append(ConvModule(ins[idx], ins[idx - 1], 1))
            self.top_down_blocks.append(CSPLayer(ins[idx - 1] * 2, ins[idx - 1],
                                                 _count(sd, f"{prefix}.top_down_blocks.{j}.blocks"), False, False))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(n_lv - 1):
            self.downsamples.append(ConvModule(ins[idx], ins[idx], 3, 2))
            self.bottom_up_blocks.append(CSPLayer(ins[idx] * 2, ins[idx + 1],
                                                  _count(sd, f"{prefix}.bottom_up_blocks.{idx}.blocks"), False, False))
        self.out_convs = nn.ModuleList([ConvModule(ins[i], out, 3) for i in range(n_lv)])
        self.n_lv = n_lv

    def forward(self, inputs):
        n = self.n_lv
        inner = [inputs[-1]]
        for idx in range(n - 1, 0, -1):
            hi = self.reduce_layers[n - 1 - idx](inner[0])
            inner[0] = hi
            up = nn.functional.interpolate(hi, scale_factor=2.0, mode="nearest")
            inner.insert(0, self.top_down_blocks[n - 1 - idx](torch.cat([up, inputs[idx - 1]], 1)))
        outs = [inner[0]]
        for idx in range(n - 1):
            outs.append(self.bottom_up_blocks[idx](torch.cat([self.downsamples[idx](outs[-1]), inner[idx + 1]], 1)))
        return [conv(o) for conv, o in zip(self.out_convs, outs)]


class RTMDetSepBNHead(nn.Module):
    """Shared convs (separate BN per level), exp on the distances, x stride."""
    def __init__(self, sd, prefix="bbox_head", strides=(8, 16, 32)):
        super().__init__()
        n_st = _count(sd, f"{prefix}.cls_convs.0")
        feat = sd[f"{prefix}.cls_convs.0.0.conv.weight"].shape[0]
        self.strides = strides
        self.cls_convs = nn.ModuleList([nn.Sequential(*[ConvModule(feat, feat, 3) for _ in range(n_st)])
                                        for _ in strides])
        self.reg_convs = nn.ModuleList([nn.Sequential(*[ConvModule(feat, feat, 3) for _ in range(n_st)])
                                        for _ in strides])
        nc = sd[f"{prefix}.rtm_cls.0.weight"].shape[0]
        k = sd[f"{prefix}.rtm_cls.0.weight"].shape[2]
        self.rtm_cls = nn.ModuleList([nn.Conv2d(feat, nc, k, padding=k // 2) for _ in strides])
        self.rtm_reg = nn.ModuleList([nn.Conv2d(feat, 4, k, padding=k // 2) for _ in strides])

    def share(self):
        """share_conv=True: every level uses level 0's conv weights (their BN stays separate)."""
        for n in range(1, len(self.strides)):
            for i in range(len(self.cls_convs[0])):
                self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def forward(self, feats):
        boxes, scores = [], []
        for i, (x, st) in enumerate(zip(feats, self.strides)):
            cls = self.rtm_cls[i](self.cls_convs[i](x))
            reg = self.rtm_reg[i](self.reg_convs[i](x)).exp() * st
            b, _, h, w = x.shape
            ys, xs = torch.meshgrid(torch.arange(h, dtype=x.dtype, device=x.device),
                                    torch.arange(w, dtype=x.dtype, device=x.device), indexing="ij")
            px, py = (xs * st).reshape(1, -1), (ys * st).reshape(1, -1)    # MlvlPointGenerator offset 0
            reg = reg.flatten(2)
            boxes.append(torch.stack([px - reg[:, 0], py - reg[:, 1], px + reg[:, 2], py + reg[:, 3]], -1))
            scores.append(cls.flatten(2).transpose(1, 2).sigmoid())
        return torch.cat(boxes, 1), torch.cat(scores, 1)


class RacketDet(nn.Module):
    """RTMDet-M racket detector: NCHW float BGR normalised (mean/std), 640x640 -> boxes (N,P,4) xyxy in input pixels,
    scores (N,P,classes)."""
    MEAN = (103.53, 116.28, 123.675)
    STD = (57.375, 57.12, 58.395)

    def __init__(self, sd):
        super().__init__()
        self.backbone = CSPNeXt(sd, "backbone", out_indices=(2, 3, 4))
        self.neck = CSPNeXtPAFPN(sd, "neck")
        self.bbox_head = RTMDetSepBNHead(sd, "bbox_head")

    def load(self, sd):
        keys = {k: v for k, v in sd.items() if k.split(".")[0] in ("backbone", "neck", "bbox_head")}
        missing, unexpected = self.load_state_dict(keys, strict=False)
        # share_conv: only level 0's conv weights are real parameters; mmdet saves the shared copies too
        missing = [k for k in missing if not re.match(r"bbox_head\.(cls|reg)_convs\.[1-9]\d*\.\d+\.conv\.weight", k)]
        if missing or unexpected:
            raise RuntimeError(f"RacketDet state dict: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.bbox_head.share()
        return self

    def forward(self, x):
        return self.bbox_head(self.neck(self.backbone(x)))


# ------------------------------------------------------------------------------------------------ TrackNetV3
class _Conv2DBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _ConvN(nn.Module):
    def __init__(self, cin, cout, n):
        super().__init__()
        names = ("conv_1", "conv_2", "conv_3")[:n]
        for i, nm in enumerate(names):
            setattr(self, nm, _Conv2DBlock(cin if i == 0 else cout, cout))
        self._names = names

    def forward(self, x):
        for nm in self._names:
            x = getattr(self, nm)(x)
        return x


class TrackNet(nn.Module):
    """TrackNetV3 tracking module (model.py of the TrackNetV3 repository, same layer names): (N, 3*(L+1), 288, 512)
    in [0, 1] (background first, then L RGB frames) -> (N, L, 288, 512) heatmaps (sigmoid)."""
    def __init__(self, in_dim=27, out_dim=8):
        super().__init__()
        self.down_block_1 = _ConvN(in_dim, 64, 2)
        self.down_block_2 = _ConvN(64, 128, 2)
        self.down_block_3 = _ConvN(128, 256, 3)
        self.bottleneck = _ConvN(256, 512, 3)
        self.up_block_1 = _ConvN(768, 256, 3)
        self.up_block_2 = _ConvN(384, 128, 2)
        self.up_block_3 = _ConvN(192, 64, 2)
        self.predictor = nn.Conv2d(64, out_dim, 1)

    def forward(self, x):
        up = lambda t: nn.functional.interpolate(t, scale_factor=2.0, mode="nearest")
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(nn.functional.max_pool2d(x1, 2))
        x3 = self.down_block_3(nn.functional.max_pool2d(x2, 2))
        x = self.bottleneck(nn.functional.max_pool2d(x3, 2))
        x = self.up_block_1(torch.cat([up(x), x3], 1))
        x = self.up_block_2(torch.cat([up(x), x2], 1))
        x = self.up_block_3(torch.cat([up(x), x1], 1))
        return torch.sigmoid(self.predictor(x))


def load_tracknet(path) -> tuple:
    """(model, param_dict) from a TrackNetV3 checkpoint (plain tensors: weights_only load)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    ins = sd["down_block_1.conv_1.conv.weight"].shape[1]
    outs = sd["predictor.weight"].shape[0]
    m = TrackNet(ins, outs)
    m.load_state_dict(sd, strict=True)
    return m.eval(), ck.get("param_dict", {})
