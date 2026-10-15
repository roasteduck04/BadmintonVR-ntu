"""A stand-in for MakeHuman's socket API: records calls, answers getters."""

from pathlib import Path

ETHNIC = ("macrodetails/African", "macrodetails/Asian", "macrodetails/Caucasian")


class FakeMakeHuman:
    def __init__(self, modifiers=None, height_fn=None):
        self.modifiers = set(modifiers or [])
        self.height_fn = height_fn          # modifier value -> cm; None = op unsupported
        self.calls: list[tuple[str, dict]] = []
        self.values: dict[str, float] = {}
        self.clothes: list[str] = []
        self.hair: str | None = None
        self.errors: dict[str, str] = {}    # func -> error message it returns instead
        self.responses: dict = {}           # func -> data, or callable(params) -> data

    def __call__(self, func, params=None, timeout=10.0, retries=3):
        params = dict(params or {})
        self.calls.append((func, params))
        if func in self.errors:
            return {"error": self.errors[func]}
        if func in self.responses:
            data = self.responses[func]
            return {"error": "", "data": data(params) if callable(data) else data}
        ok = {"error": "", "data": "OK"}
        if func == "getProxiesInfo":
            return {"error": "", "data": []}
        if func == "exportMHM":
            Path(params["path"]).write_text("# fake MHM\n" + "\n".join(
                f"modifier {k} {v}" for k, v in sorted(self.values.items())), encoding="utf-8")
            return ok
        if func == "loadMHM":
            if not Path(params.get("path", "")).exists():
                return {"error": "loadMHM failed: no such file"}
            return ok
        if func == "getAvailableModifierNames":
            return {"error": "", "data": sorted(self.modifiers)}
        if func == "applyModifier":
            self.values[params["modifier"]] = float(params["power"])
            if params["modifier"] in ETHNIC:
                self._normalise_ethnic(params["modifier"])
            return ok
        if func == "unequipAllClothes":
            self.clothes = []
            return ok
        if func == "equipClothes":
            self.clothes.append(params["path"])
            return ok
        if func == "equipHair":
            self.hair = params["path"]
            return ok
        if func == "getEquippedClothes":
            return {"error": "", "data": list(self.clothes)}
        if func == "getEquippedHair":
            return {"error": "", "data": self.hair}
        if func == "getHeightCm":
            if self.height_fn is None:
                return {"error": "Unknown function getHeightCm"}
            return {"error": "", "data": self.height_fn(self.values.get("macrodetails-height/Height", 0.5))}
        return ok

    def _normalise_ethnic(self, changed):
        """MakeHuman's Human._setEthnicVals: the three ethnic values always sum to 1; setting one
        rescales the other two in proportion (1/3 each by default)."""
        for m in ETHNIC:
            self.values.setdefault(m, 1 / 3)
        others = [m for m in ETHNIC if m != changed]
        remaining = 1.0 - self.values[changed]
        total = sum(self.values[m] for m in others)
        for m in others:
            self.values[m] = remaining * self.values[m] / total if total else remaining / 2

    def trace(self) -> list[list]:
        out = []
        for func, params in self.calls:
            if func == "getSysDir":
                continue
            norm = {}
            for k, v in sorted(params.items()):
                if k == "path":
                    v = Path(str(v)).stem
                elif isinstance(v, float):
                    v = round(v, 4)
                norm[k] = v
            out.append([func, norm])
        return out
