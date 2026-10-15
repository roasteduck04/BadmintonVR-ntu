import os
import re

import numpy as np

from analysis import draw

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_player_colours_match_unity():
    cs = open(os.path.join(REPO, "Assets", "Scripts", "Analysis", "AnalysisPlayer.cs"), encoding="utf-8").read()
    block = cs[cs.index("playerColors"):]
    # AnalysisPlayer.cs writes e.g. new Color(0.2f, 0.7f, 1f)
    rgb = [tuple(float(x.rstrip("f")) for x in m)
           for m in re.findall(r"new Color\(([\d.]+f?), ([\d.]+f?), ([\d.]+f?)\)", block)[:4]]
    assert len(rgb) == 4
    assert [draw.unity_bgr(*c) for c in rgb] == draw.PLAYER_BGR
    assert draw.player_bgr(4) == draw.PLAYER_BGR[0]


def test_err_colour_is_green_to_red():
    assert draw.err_bgr(0) == (0, 255, 0) and draw.err_bgr(40) == (0, 0, 255) and draw.err_bgr(400) == (0, 0, 255)


def test_err_colour_of_unknown_error_is_grey():
    assert draw.err_bgr(float("nan")) == draw.GREY and draw.err_bgr(float("inf")) == draw.GREY


def test_banner_height_is_the_drawn_bar():
    img = np.zeros((960, 540, 3), np.uint8)
    h = draw.banner_height(img)
    draw.draw_banner(img, "EXCLUDED")
    assert h == 40 and (img[h - 1, :, 2] == 255).all() and not (img[h + 1, 200:, 2] == 255).any()


def test_shuttle_colour_is_distinct_and_in_the_legend():
    assert draw.SHUTTLE_BGR not in (draw.WHITE, draw.YELLOW, draw.MAGENTA, draw.CYAN, draw.GREY, *draw.PLAYER_BGR)
    leg = draw.legend_strip(1000)
    assert (leg == np.array(draw.SHUTTLE_BGR, np.uint8)).all(axis=2).any()


def test_skeleton_skips_nan_points():
    img = np.zeros((100, 100, 3), np.uint8)
    uv = np.array([[10.0, 10.0], [np.nan, np.nan], [90.0, 90.0]])
    draw.draw_skeleton(img, uv, [-1, 0, 1], (255, 255, 255))
    assert img[50, 50].sum() == 0                          # no line through the missing middle joint
    assert img[10, 10].sum() > 0 and img[90, 90].sum() > 0


def test_every_primitive_draws_on_a_blank_image():
    img = np.zeros((200, 300, 3), np.uint8)
    draw.draw_keypoints(img, np.array([[20.0, 20.0], [40.0, 40.0]]), np.array([0.9, 0.1]), (255, 0, 0))
    draw.draw_residual(img, (50, 50), (80, 60), 30.0, rejected=True)
    draw.draw_bbox_tag(img, np.array([[100.0, 100.0], [150.0, 180.0]]), (0, 255, 0), "P1 p 0.82")
    draw.draw_banner(img, "EXCLUDED (health 927 px)")
    draw.draw_text_panel(img, ["x" * 120, "short"])
    assert img.any()
    leg = draw.legend_strip(300)
    assert leg.shape == (28, 300, 3) and leg.any()


def test_text_panel_wraps_long_labels():
    lines = draw.wrap_lines(["SMPL-28 · triangulated · 2 cams (backleft_051, frontright_051) · dropped none"], 46)
    assert len(lines) == 2 and all(len(s) <= 46 for s in lines)
