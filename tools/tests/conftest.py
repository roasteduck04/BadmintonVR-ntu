import pathlib
import sys

# Make tools/*.py importable as top-level modules from tools/tests/*.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

# tools/tests/ itself, so helpers like fake_makehuman.py import as top-level modules.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# The MakeHuman install is ignored by version control, so a fresh clone has neither it nor the user's community
# packs. A few files under tools/makehuman/data are tracked (the badminton_sportswear asset), so test for the
# install's own folders rather than the data dir.
MAKEHUMAN_DATA = pathlib.Path(__file__).resolve().parents[1] / "makehuman" / "data"
MAKEHUMAN_SENTINELS = ("skins", "hair", "eyes", "eyebrows", "clothes/female_sportsuit01")


def makehuman_installed():
    return all((MAKEHUMAN_DATA / d).is_dir() for d in MAKEHUMAN_SENTINELS)


def _makehuman_user_data_dir():
    from cosmetics.asset_catalog import DEFAULT_USER_DATA_DIR
    return pathlib.Path(DEFAULT_USER_DATA_DIR)


def pytest_collection_modifyitems(config, items):
    """@pytest.mark.needs_makehuman skips when tools/makehuman/data is missing; (user=True) also needs the user dir."""
    have_mh = makehuman_installed()
    have_user = None
    for item in items:
        m = item.get_closest_marker("needs_makehuman")
        if m is None:
            continue
        if not have_mh:
            item.add_marker(pytest.mark.skip(reason=f"MakeHuman install missing ({MAKEHUMAN_DATA}, gitignored)"))
        elif m.kwargs.get("user"):
            if have_user is None:
                have_user = _makehuman_user_data_dir().is_dir()
            if not have_user:
                item.add_marker(pytest.mark.skip(
                    reason=f"MakeHuman user data dir missing ({_makehuman_user_data_dir()})"))


@pytest.fixture
def fake_mh(monkeypatch, tmp_path):
    """FakeMakeHuman installed as face_scan.mh_call; generated kits go to tmp."""
    import face_scan
    from cosmetics import kit
    from cosmetics.body_photo import BODY_PHOTO_MODIFIERS
    from fake_makehuman import FakeMakeHuman

    names = {m for _, m, _, _ in face_scan.MODIFIER_MAP} | {
        "macrodetails/Gender", "macrodetails/Age", "macrodetails-universal/Muscle",
        "macrodetails-universal/Weight", "macrodetails-height/Height", "macrodetails/Asian",
        "macrodetails/African", "macrodetails/Caucasian", *BODY_PHOTO_MODIFIERS}
    fake = FakeMakeHuman(modifiers=names)
    monkeypatch.setattr(face_scan, "mh_call", fake)
    import avatar_runs                      # face_scan takes the shared MakeHuman lock: never the live one
    monkeypatch.setattr(avatar_runs, "LOCK", avatar_runs.MakeHumanLock(tmp_path / "runs"))
    monkeypatch.setattr(kit, "KIT_DIR", tmp_path / "kits")
    monkeypatch.setattr(kit, "HAIR_DIR", tmp_path / "hair_tints")
    return fake


@pytest.fixture(autouse=True)
def _no_downloads(monkeypatch):
    """Tests never download: a missing model must skip or fail, not fetch from the network.

    Tests that exercise a download path patch urlretrieve themselves (their patch wins).
    """
    import urllib.request

    def _blocked(url, *a, **k):
        raise RuntimeError(f"network download blocked in tests: {url}")

    monkeypatch.setattr(urllib.request, "urlretrieve", _blocked)
