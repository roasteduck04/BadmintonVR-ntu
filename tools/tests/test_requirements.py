"""tools/requirements.txt covers the hard imports of the local pipeline and the test command (review 4 M4)."""
import re
from pathlib import Path

REQ = Path(__file__).resolve().parents[1] / "requirements.txt"

# pip distribution -> where the default local route (or the documented test command) imports it unconditionally
HARD = {"opencv-python": "cv2", "numpy": "numpy", "scipy": "scipy", "networkx": "analysis.people.identity",
        "matplotlib": "analysis.pose24.camsubsets.report", "threadpoolctl": "analysis.pose24.camsubsets.oracle",
        "pyyaml": "analysis.pose24.priors.motionbert", "mediapipe": "cosmetics", "fastapi": "avatar_app",
        "pytest": "tests"}


def _listed():
    names = set()
    for line in REQ.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].lower())
    return names


def test_hard_dependencies_listed():
    missing = sorted(set(HARD) - _listed())
    assert not missing, f"missing from tools/requirements.txt: {missing}"


def test_torch_is_optional_not_required():
    assert "torch" not in _listed() and "# torch" in REQ.read_text(encoding="utf-8")
