"""Long-row scoring for the camera-subset study (spec § 5): one row per (clip, test view, subset, method, mode,
player), from stored subset results (oracle.load_result) and the array metrics. CSV writing is elsewhere.

Column notes: `test_view` "" = deployment row (3D vs the full build only; held-out columns None). Held-out px
(R21) is scored only on frames where the full build has a detection of the player in the test view (det_index >= 0);
`heldout_n` = its keypoint sample count, and every held-out column is None below metrics.HELDOUT_MIN_SAMPLES;
`bbox_h_test_px` uses the same frames. Angle columns are
MAE in degrees; knee / elbow / shoulder_abd = mean of the left and right MAE, `_swing_` = the same inside the swing
windows. `limb_cv` is a fraction (not percent); `jerk` / `skate` come from eval_pose_mv.motion_stats. `used_frac`
= JSON {view: body-inlier frame fraction}; `excluded` = JSON list of the full build's excluded views.
Coverage (R16) is recomputed here from the stored arrays (J, inlier), never taken from the stored scalars:
`coverage` = fraction of frames with a finite pelvis, `full_body` = frames with every joint finite, `measured` = a
coverage frame with >= 2 inlier views on any keypoint (mono: >= 1 view). `player_seen` (R19) = measured >=
SEEN_MIN: a pelvis that is only filled / extrapolated (finite but hardly measured) is not "seen". Rows of a player
not seen by the subset (no match, measured < SEEN_MIN, or the reference's own pelvis coverage < SEEN_MIN, logged)
carry None in every metric column, never 0; `coverage` / `full_body` are 0.0 and `measured` None when unmatched,
and the actual values when matched but unseen. `is_reference` (R17): the subset is the row's reference (full minus the test view); its 3D-vs-reference,
angle and lr_swap columns stay None (held-out px, quality and coverage are kept). Mono rows: `stature_m` None (the
fit takes its lengths as input). In mode "e2e" players are matched only through e2e_meta's "match" (a subset key
missing from e2e_meta: every row unmatched, logged)."""
from __future__ import annotations

import json

import numpy as np

from analysis import skeleton24 as sk
from analysis.pose24.camsubsets import metrics
from analysis.pose24.camsubsets import subsets as cs

SEEN_MIN = 0.05
ROW_KEYS = ["clip", "clip_type", "ref_cams", "is_reference", "test_view", "subset", "n_cams", "method", "mode",
            "player", "player_seen", "matched", "coverage", "measured", "full_body", "heldout_px_median",
            "heldout_px_p90", "heldout_n", "heldout_per_bbox_median", "w_mpjpe_cm_median", "w_mpjpe_cm_p90", "root_mpjpe_cm_median",
            "pa_mpjpe_cm_median", "pa_mpjpe_cm_p90", "pelvis_xz_cm_median", "pelvis_xz_cm_p90", "pelvis_h_cm_median",
            "knee_mae_deg", "elbow_mae_deg", "shoulder_abd_mae_deg", "trunk_fwd_mae_deg", "trunk_lat_mae_deg",
            "knee_swing_mae_deg", "elbow_swing_mae_deg", "shoulder_abd_swing_mae_deg", "trunk_fwd_swing_mae_deg",
            "trunk_lat_swing_mae_deg", "lr_swap_frames", "jerk", "limb_cv", "skate", "mean_sigma_mm", "stature_m",
            "used_frac", "ray_angle_min_deg", "ray_angle_max_deg", "bbox_h_test_px", "wrong_person_frames",
            "id_switches", "sync_shift_f", "sync_spread_f", "view_health_carried", "excluded", "fit_runtime_s"]
_ANG = (("knee", "knee_flexion", True), ("elbow", "elbow_flexion", True), ("shoulder_abd", "shoulder_abduction", True),
        ("trunk_fwd", "trunk_forward_lean", False), ("trunk_lat", "trunk_lateral_lean", False))


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def _angles(row, ang, infix):
    for col, name, lr in _ANG:
        vals = [_get(ang, f"{s}_{name}", "mae") for s in ("left", "right")] if lr else [_get(ang, name, "mae")]
        vals = [v for v in vals if v is not None]
        row[f"{col}{infix}_mae_deg"] = float(np.mean(vals)) if vals else None


def _is_mono(res, method):
    return len(res["views"]) == 1 or str(res.get("method") or method).startswith("mono")


def _fill_metrics(row, p, J, Jr, names, swing, full_cache, pid_ref, r, fps, res, mono=False):
    if not row["is_reference"]:                                    # a reference vs itself is 0 by construction
        vs = metrics.vs_ref(J, Jr, names, swing=swing)
        for m in ("w_mpjpe_cm", "root_mpjpe_cm", "pa_mpjpe_cm", "pelvis_xz_cm", "pelvis_h_cm"):
            for s in ("median", "p90"):
                if f"{m}_{s}" in row:
                    row[f"{m}_{s}"] = _get(vs, m, s)
        _angles(row, vs.get("angle_mae_deg"), "")
        _angles(row, vs.get("swing"), "_swing")
        row["lr_swap_frames"] = metrics.lr_swap_frames(J, Jr, names)
    t = r["test_view"]
    if t:
        h = metrics.heldout_px(p["X"].astype(float), full_cache, pid_ref, t)
        row.update(heldout_px_median=_get(h, "px", "median"), heldout_px_p90=_get(h, "px", "p90"),
                   heldout_n=_get(h, "px", "n"),
                   heldout_per_bbox_median=_get(h, "per_bbox", "median"),
                   bbox_h_test_px=metrics.bbox_height_px(full_cache, pid_ref, t))
    q = metrics.quality(J, p["C"].astype(float), p["sigma"].astype(float), fps, names)
    row.update(jerk=q.get("jerk"), limb_cv=q.get("limb_cv"), skate=q.get("skate"), mean_sigma_mm=q.get("mean_sigma_mm"))
    row["stature_m"] = None if mono else p.get("stature_m")
    row["used_frac"] = json.dumps(metrics.used_frac(p["inlier"], res["views"]), sort_keys=True)
    ang = [a for a in metrics.ray_angles_deg(full_cache["cams"], res["views"], J[:, 0] * [1, -1, 1]).values()
           if a is not None]
    row.update(ray_angle_min_deg=min(ang) if ang else None, ray_angle_max_deg=max(ang) if ang else None)


def score_rows(clip, rows, results, full_cache, fps, clip_type, method, mode, rallies=None, e2e_meta=None, log=print):
    """`results` = {subset key: loaded result} (every row's subset and reference); rows without one are skipped.
    `e2e_meta` = {key: {"shift", "spread", "match": {pid_sub: pid_ref}, "identity": {pid_ref: identity_proxy}}}."""
    names = list(sk.SMPL28_NAMES)
    out = []
    swing = {}
    for r in rows:
        k, kref = cs.key(r["subset"]), cs.key(r["reference"])
        res, ref = results.get(k), results.get(kref)
        if res is None or ref is None:
            log(f"    no result for {k if res is None else kref}: row skipped")
            continue
        em = (e2e_meta or {}).get(k, {})
        if "match" in em:
            match = em["match"] or {}                    # {} = pelvis matching found nobody: all unmatched
        elif mode == "e2e":                              # e2e ids are not comparable to reference ids unmatched
            log(f"    {k}: no e2e match in e2e_meta: every player unmatched")
            match = {}
        else:
            match = {p: p for p in res["players"]}
        for pid_ref in ref["players"]:
            pid = next((s for s, q in match.items() if q == pid_ref), None)
            p = res["players"].get(pid) if pid is not None else None
            pr = ref["players"][pid_ref]
            row = dict.fromkeys(ROW_KEYS)
            row.update(clip=clip, clip_type=clip_type, ref_cams=len(r["reference"]), is_reference=k == kref,
                       test_view=r["test_view"] or "",
                       subset=k, n_cams=r["n_cams"], method=method, mode=mode, player=pid_ref,
                       matched=p is not None, view_health_carried=True,
                       excluded=json.dumps(sorted(full_cache.get("excluded") or {})),
                       fit_runtime_s=res["runtime_s"], sync_shift_f=em.get("shift"), sync_spread_f=em.get("spread"))
            mono = _is_mono(res, method)
            cov, meas, full = (metrics.coverage_stats(p["J"], p["inlier"], 1 if mono else 2) if p is not None
                               else (0.0, None, 0.0))
            row.update(coverage=cov, measured=meas, full_body=full)
            ref_cov = metrics.coverage_stats(pr["J"], pr["inlier"])[0]
            if p is not None and meas >= SEEN_MIN and ref_cov < SEEN_MIN:
                log(f"    {k} {pid_ref}: the reference {kref} does not see the player (pelvis coverage "
                    f"{ref_cov:.2f}): row unseen")
            row["player_seen"] = bool(p is not None and meas >= SEEN_MIN and ref_cov >= SEEN_MIN)
            if row["player_seen"]:
                J, Jr = p["J"].astype(float), pr["J"].astype(float)
                if (kref, pid_ref) not in swing:                 # windows from this row's own (untrimmed) reference
                    swing[kref, pid_ref] = metrics.swing_windows(Jr, fps, names[:Jr.shape[1]], rallies)
                n = min(J.shape[1], Jr.shape[1])
                J, Jr = J[:, :n], Jr[:, :n]
                _fill_metrics(row, p, J, Jr, names[:n], swing[kref, pid_ref], full_cache, pid_ref, r, fps, res,
                              mono=mono)
                d = em.get("identity", {}).get(pid_ref)
                if d:
                    row.update(wrong_person_frames=d["wrong_frames"], id_switches=d["switches"])
            out.append({k2: row[k2] for k2 in ROW_KEYS})
    return out
