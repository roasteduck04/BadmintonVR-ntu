"""build_real writes analysis docs with allow_nan=False (review M8)."""
import json

import pytest

from analysis import build_real


def test_write_doc_rejects_nan_and_writes_nothing(tmp_path):
    build_real._write_doc(str(tmp_path / "ok.json"), {"a": [1.0]})
    assert json.loads((tmp_path / "ok.json").read_text()) == {"a": [1.0]}
    with pytest.raises(ValueError):
        build_real._write_doc(str(tmp_path / "bad.json"), {"a": [float("nan")]})
    assert not (tmp_path / "bad.json").exists()
