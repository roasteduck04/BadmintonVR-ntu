"""Round-5 experiments (test branch only): loaded automatically by every Python process started with this folder on
PYTHONPATH - the campaign, each live run and each 2D / fit process they spawn - and attaches tools/experiments/hooks.py
to the live modules as they are imported. Nothing in the existing code is edited; every hook is off unless a test sets
its exp_* key (see tools/experiments/README.md).

    PYTHONPATH="$PWD/tools/experiments" python tools/live_run.py campaign run --plan lab_5090_r5 ...
"""
import importlib.abc
import importlib.machinery
import sys

_TARGETS = {"analysis.live.pose2d_fast", "analysis.live.framewriter", "analysis.live.sources"}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, name):
        self.inner, self.name = inner, name

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        import hooks                                    # tools/experiments/hooks.py (this folder is on sys.path)
        hooks.attach(self.name, module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        for f in sys.meta_path:
            if f is self:
                continue
            spec = f.find_spec(name, path, target) if hasattr(f, "find_spec") else None
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader, name)
                return spec
        return None


if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
