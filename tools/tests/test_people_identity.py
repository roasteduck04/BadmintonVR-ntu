"""Identity: per-half flow chains, pins fixed, conflicting pins split, half-court rule, unknown when no free id."""
import numpy as np

from analysis import people
from analysis.people import identity as idn
from analysis.people.candidates import Candidate
from analysis.people.tracklets import Tracklet

P = people.load_params()


class _App:
    """Appearance stub: colour per tracklet id."""
    gate, weight = 0.5, 3.0

    def __init__(self, col):
        self.col = col

    def dist(self, a, b):
        ca, cb = self.col.get(a.tid), self.col.get(b.tid)
        return None if ca is None or cb is None else (0.0 if ca == cb else 1.0)


def _t(tid, f0, f1, x0, x1, z):
    t = Tracklet(tid=tid)
    for f, x in zip(range(f0, f1), np.linspace(x0, x1, f1 - f0)):
        t.add(Candidate(frame=f, dets={"a": tid}, floor=np.array([x, z]), support=3))
    return t


def test_one_chain_links_pieces_and_skips_a_concurrent_bystander():
    T = [_t(0, 0, 50, -1, 0, -3), _t(1, 55, 100, 0, 1, -3), _t(2, 0, 100, 3.5, 3.5, -2)]
    app = _App({0: "r", 1: "r", 2: "b"})
    ch, costs = idn.chains(T, 1, 60.0, app, P, {0: "P1"}, {0: 1.0, 1: 0.9, 2: 0.9})
    assert ch == [[0, 1]] and (0, 1) in costs


def test_pins_on_two_players_split_a_chain():
    ch = [[0, 1]]
    assert idn.split_conflicts(ch, {0: "P1", 1: "P2"}, {(0, 1): 2.0}) == [[0], [1]]
    assert idn.split_conflicts(ch, {0: "P1", 1: "P1"}, {(0, 1): 2.0}) == [[0, 1]]


def test_assign_keeps_each_player_on_its_half_and_unknowns_stay_unassigned():
    near1, near2 = _t(0, 0, 90, -1, 0, -3), _t(1, 95, 180, 0, 1, -3)
    far1 = _t(2, 0, 180, 0, 0, 3)
    at_net = _t(3, 100, 180, 0, 0, 0.3)            # a far-side tracklet close to the net
    extra = _t(4, 0, 180, -2, -2, -5)              # a second player-like person on the near half (singles)
    T = [near1, near2, far1, at_net, extra]
    labels = {t.tid: "player" for t in T}
    scores = {t.tid: 0.9 for t in T}
    app = _App({0: "r", 1: "r", 2: "b", 3: "b", 4: "g"})
    sides = {"P1": "near", "P2": "far"}
    msgs = []
    got = idn.assign(T, labels, scores, {0: "P1", 2: "P2", 3: "P1"}, sides, 60.0, app, P, log=msgs.append)
    assert got[0] == "P1" and got[1] == "P1" and got[2] == "P2"
    assert 3 not in got                            # a P1 pin on a far-half tracklet is dropped: never across the net
    assert any("tracklet 3" in m and "other half" in m for m in msgs)
    assert 4 not in got                            # no free near-side identity in singles -> stays unassigned


def test_two_pins_of_one_player_beyond_the_link_gap_both_kept():
    T = [_t(0, 0, 60, 0, 0, -3), _t(1, 360, 420, 0, 0, -3)]      # 5 s apart > max_link_gap_s
    app = _App({0: "r", 1: "r"})
    got = idn.assign(T, {0: "player", 1: "player"}, {0: 0.9, 1: 0.9}, {0: "P1", 1: "P1"}, {"P1": "near"},
                     60.0, app, P, log=lambda *a: None)
    assert got == {0: "P1", 1: "P1"}


def test_a_later_unpinned_segment_of_the_player_joins_its_identity():
    T = [_t(0, 0, 60, 0, 0, -3), _t(1, 360, 1000, 0, 0, -3)]
    app = _App({0: "r", 1: "r"})
    got = idn.assign(T, {0: "player", 1: "player"}, {0: 0.9, 1: 0.9}, {0: "P1"}, {"P1": "near"},
                     60.0, app, P, log=lambda *a: None)
    assert got == {0: "P1", 1: "P1"}


def test_unknown_only_chain_makes_no_identity():
    # P2's seeds found nothing (never seen): an "unknown" person on the far half must not become P2,
    # while an unknown bridging two P1 pieces still joins P1
    T = [_t(0, 0, 50, -1, 0, -3), _t(1, 52, 60, 0, 0.1, -3), _t(2, 62, 120, 0.1, 1, -3), _t(3, 0, 120, 4.2, 4.2, 0.2)]
    labels = {0: "player", 1: "unknown", 2: "player", 3: "unknown"}
    app = _App({0: "r", 1: "r", 2: "r", 3: "g"})
    got = idn.assign(T, labels, {0: 1.0, 1: 0.6, 2: 0.9, 3: 0.69}, {0: "P1"}, {"P1": "near", "P2": "far"},
                     60.0, app, P, log=lambda *a: None)
    assert got == {0: "P1", 1: "P1", 2: "P1"}


def test_a_player_is_never_in_two_pieces_at_once():
    # T15 doubles_2sides: a (stale-seed) pin P3 on tracklet A while the flow's other chain runs B -> C with C pinned P3:
    # both pieces were labelled P3 over frames 0-60. The better-pinned piece keeps P3, A is labelled like a free piece.
    A, B, C = _t(0, 0, 60, -1, -1, 3), _t(1, 0, 150, 1, 1, 3), _t(2, 155, 300, 1, 1, 3)
    T = [A, B, C]
    labels = {t.tid: "player" for t in T}
    scores = {t.tid: 1.0 for t in T}
    app = _App({0: "r", 1: "g", 2: "g"})
    logs = []
    got = idn.assign(T, labels, scores, {0: "P3", 2: "P3"}, {"P1": "near", "P2": "near", "P3": "far", "P4": "far"},
                     60.0, app, P, log=logs.append)
    assert got[1] == "P3" and got[2] == "P3" and got.get(0) == "P4"
    by = {}
    for tid, pid in got.items():
        by.setdefault(pid, []).append(next(t for t in T if t.tid == tid))
    for pid, ts in by.items():
        fr = [f for t in ts for f in t.frames]
        assert len(fr) == len(set(fr)), pid
    assert any("P3" in m and "same time" in m for m in logs)


def test_half_of_ignores_the_net_band():
    # a near player's tracklet that spends most frames on the net line (z ~ +0.02) is still near-side
    t = Tracklet(tid=0)
    for f in range(40):
        t.add(Candidate(frame=f, dets={"a": 0}, floor=np.array([0.0, -1.0 if f < 10 else 0.02]), support=3))
    assert idn.half_of(t, P["net_band_m"]) == "near"
    assert idn.half_of(_t(1, 0, 10, 0, 0, 0.02), P["net_band_m"]) == "far"      # only on the line: median sign


def test_short_fragments_inside_a_players_gap_are_filled():
    # T15 singles_05x: at a crowd (a self-split candidate next to P1) P1 is cut into short fragments; each extra hop
    # costs appearance (fragments' colour is noisy or missing) more than a 13-frame fragment earns, so the flow jumps
    # from one long piece to the next and P1 loses the fragment's frames. A fragment that fits the gap between two
    # consecutive pieces of one player (both hops within the link gates) joins that player.
    A, X, B = _t(0, 0, 50, 0, 0, -3), _t(1, 53, 64, 0, 0, -3), _t(2, 67, 150, 0, 0, -3)
    far = _t(3, 0, 150, 0, 0, 3)
    T = [A, X, B, far]
    labels = {0: "player", 1: "unknown", 2: "player", 3: "player"}
    scores = {0: 1.0, 1: 0.6, 2: 1.0, 3: 1.0}
    app = _App({0: "r", 2: "r", 3: "b"})                     # X: no appearance features
    ch, _ = idn.chains([A, X, B], 1, 60.0, app, P, {0: "P1"}, scores)
    assert ch == [[0, 2]]                                     # the flow alone skips X
    got = idn.assign(T, labels, scores, {0: "P1", 3: "P2"}, {"P1": "near", "P2": "far"}, 60.0, app, P, log=lambda *a: None)
    assert got == {0: "P1", 1: "P1", 2: "P1", 3: "P2"}


def test_a_fragment_that_fits_two_players_gaps_stays_unassigned():
    # doubles teammates both have a gap where the fragment sits: ambiguous, no identity
    A1, B1 = _t(0, 0, 50, 0, 0, -3), _t(1, 67, 150, 0, 0, -3)
    A2, B2 = _t(2, 0, 50, 0.3, 0.3, -3), _t(3, 67, 150, 0.3, 0.3, -3)
    X = _t(4, 53, 64, 0.15, 0.15, -3)
    T = [A1, B1, A2, B2, X]
    labels = {t.tid: "player" for t in T}
    labels[4] = "unknown"
    scores = {t.tid: 1.0 for t in T}
    scores[4] = 0.6
    app = _App({0: "r", 1: "r", 2: "g", 3: "g"})
    got = idn.assign(T, labels, scores, {0: "P1", 1: "P1", 2: "P2", 3: "P2"}, {"P1": "near", "P2": "near"},
                     60.0, app, P, log=lambda *a: None)
    assert got.get(4) is None and got[0] == got[1] == "P1" and got[2] == got[3] == "P2"


def test_a_fragment_only_on_the_net_line_can_fill_either_half():
    # T15 singles_05x: P1's fragments at the net have median z +0.02 ("far" by sign) - they still fill P1's gap
    A, X, B = _t(0, 0, 50, -2, -2, -0.5), _t(1, 53, 64, -2, -2, 0.02), _t(2, 67, 150, -2, -2, -0.5)
    far = _t(3, 0, 150, 0, 0, 3)
    labels = {0: "player", 1: "unknown", 2: "player", 3: "player"}
    scores = {0: 1.0, 1: 0.6, 2: 1.0, 3: 1.0}
    got = idn.assign([A, X, B, far], labels, scores, {0: "P1", 3: "P2"}, {"P1": "near", "P2": "far"}, 60.0,
                     _App({0: "r", 2: "r", 3: "b"}), P, log=lambda *a: None)
    assert got[1] == "P1"


def test_a_one_view_end_gets_the_one_view_slack():
    # T15 doubles_1side P3 (far, 15+ m from the near-side cameras): its 1-view floor sits 1.3 m behind the 2-view
    # floor of the frame before (along-ray error), so the next tracklet was unreachable (0.5 m slack at a 1-frame
    # gap) and the flow took a wrong fragment instead
    a = _t(0, 0, 50, 1.9, 1.9, 5.55)
    b1 = Tracklet(tid=1)
    for f in range(51, 100):
        b1.add(Candidate(frame=f, dets={"a": 1}, floor=np.array([2.06, 6.9]), support=1))
    app = _App({})
    assert idn.link_cost(a, b1, 60.0, app, P) is not None
    b3 = _t(2, 51, 100, 2.06, 2.06, 6.9)                  # the same jump between multi-view candidates: no link
    assert idn.link_cost(a, b3, 60.0, app, P) is None


def test_an_overruled_pin_is_reported():
    A, B, C = _t(0, 0, 60, -1, -1, 3), _t(1, 0, 150, 1, 1, 3), _t(2, 155, 300, 1, 1, 3)
    T = [A, B, C]
    un = set()
    idn.assign(T, {t.tid: "player" for t in T}, {t.tid: 1.0 for t in T}, {0: "P3", 2: "P3"},
               {"P1": "near", "P2": "near", "P3": "far", "P4": "far"}, 60.0, _App({0: "r", 1: "g", 2: "g"}), P,
               log=lambda *a: None, unpinned=un)
    assert un == {0}


def test_a_gap_path_with_a_contested_fragment_is_dropped_whole():
    # X2 fits both teammates' gaps; P1's path A1 -> X1 -> X2 -> B1 then has no checked hop from X1 onwards: X1 must not
    # join P1 either
    A1, B1 = _t(0, 0, 50, 0, 0, -3), _t(1, 80, 150, 0, 0, -3)
    A2, B2 = _t(2, 0, 61, 0.3, 0.3, -3), _t(3, 80, 150, 0.3, 0.3, -3)
    X1, X2 = _t(4, 53, 60, 0.0, 0.0, -3), _t(5, 63, 76, 0.15, 0.15, -3)
    T = [A1, B1, A2, B2, X1, X2]
    labels = {t.tid: "player" for t in T}
    labels[4] = labels[5] = "unknown"
    scores = {t.tid: 1.0 for t in T}
    scores[4] = scores[5] = 0.6
    app = _App({0: "r", 1: "r", 2: "g", 3: "g"})
    got = idn.fill_gaps({0: "P1", 1: "P1", 2: "P2", 3: "P2"}, T, labels, {"P1": "near", "P2": "near"}, 60.0, app, P,
                        log=lambda *a: None)
    assert got == {}
