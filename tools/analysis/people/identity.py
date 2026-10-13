"""Global assignment of player tracklets to P1..Pn (SP3 spec § 2.5).

Hard rules: each player stays on its seeded half (no net crossing inside a clip), so every half is solved on its own
with k = its number of seeded players (singles 1, doubles 2); a player can't be in two tracklets at once (chains are
time-disjoint). Per half, a min-cost flow on a node-split DAG picks up to k chains maximising reward
(reward_per_frame x frames x score) minus link costs (distance + gap time + appearance; links need <= max_speed_mps
and <= max_link_gap_s, and appearance under the gate). Seed-pinned tracklets carry a reward no chain refuses. A chain
holding pins of two players is cut at its most expensive link between them. Pieces are then labelled: pinned pieces
by their pin, the rest (longest first) by the cheapest non-overlapping player (no gap limit). Pinned or
"player" tracklets left out of the chains join as single-tracklet pieces. A piece of only "unknown" tracklets gets no
identity. Finally fill_gaps puts short left-out fragments into the gap between two pieces of one player (never when
they fit two players). Unassigned player-like tracklets end as "unknown" (tracker3d)."""
from __future__ import annotations

import networkx as nx
import numpy as np

PIN_REWARD = 10 ** 7
SCALE = 100          # networkx wants integer weights: centimetres


def half_of(t, band=0.0):
    """Median side of the candidates outside the net band (|z| > band; a player at the net line jitters around z = 0,
    T15); all candidates when none is outside."""
    z = t.floor()[:, 1]
    off = z[np.abs(z) > band]
    return "near" if np.median(off if len(off) else z) < 0 else "far"


def link_cost(a, b, fps, app, p, max_gap_s="default"):
    ip = p["identity"]
    gap_max = ip["max_link_gap_s"] if max_gap_s == "default" else max_gap_s
    gap = (b.first - a.last) / fps
    if gap <= 0 or (gap_max is not None and gap > gap_max):
        return None
    d = float(np.linalg.norm(b.cands[0].floor - a.cands[-1].floor))
    # a 1-view floor point can be ~1 m off along its ray at 15+ m (T15), a triangulated one isn't
    slack = ip["link_slack_1view_m"] if min(a.cands[-1].support, b.cands[0].support) == 1 else ip["link_slack_m"]
    if d > ip["max_speed_mps"] * gap + slack:
        return None
    h = app.dist(a, b)
    if h is not None and h > app.gate:
        return None
    return d + ip["gap_weight_m_per_s"] * gap + app.weight * (h if h is not None else ip["unknown_app"])


def chains(T, k, fps, app, p, pins, weights):
    ip = p["identity"]
    G = nx.DiGraph()
    G.add_node("s", demand=-k)
    G.add_node("t", demand=k)
    G.add_edge("s", "t", capacity=k, weight=0)
    for t in T:
        r = PIN_REWARD if t.tid in pins else int(round(SCALE * ip["reward_per_frame"] * len(t) * weights.get(t.tid, 0.5)))
        G.add_edge(("i", t.tid), ("o", t.tid), capacity=1, weight=-r)
        G.add_edge("s", ("i", t.tid), capacity=1, weight=int(round(SCALE * ip["start_cost"])))
        G.add_edge(("o", t.tid), "t", capacity=1, weight=0)
    order = sorted(T, key=lambda t: (t.first, t.tid))
    costs = {}
    for a in order:
        succ = []
        for b in order:
            if b.first <= a.last:
                continue
            if (b.first - a.last) / fps > ip["max_link_gap_s"]:
                break
            c = link_cost(a, b, fps, app, p)
            if c is not None:
                succ.append((c, b.tid))
        for c, btid in sorted(succ)[:ip["top_k_succ"]]:
            G.add_edge(("o", a.tid), ("i", btid), capacity=1, weight=int(round(SCALE * c)))
            costs[(a.tid, btid)] = c
    flow = nx.min_cost_flow(G)
    out = []
    for node, fl in flow["s"].items():
        if node == "t" or fl <= 0:
            continue
        chain = []
        while node != "t":
            tid = node[1]
            chain.append(tid)
            node = next(m for m, x in flow[("o", tid)].items() if x > 0)
        out.append(chain)
    return out, costs


def split_conflicts(chain_list, pins, costs):
    pieces = []
    for ch in chain_list:
        start, last = 0, None
        for i, tid in enumerate(ch):
            pid = pins.get(tid)
            if pid is None:
                continue
            if last is not None and last[1] != pid:
                cut = max(range(last[0], i), key=lambda j: costs.get((ch[j], ch[j + 1]), 0.0))
                pieces.append(ch[start:cut + 1])
                start = cut + 1
            last = (i, pid)
        pieces.append(ch[start:])
    return [q for q in pieces if q]


def _span(q, by_id):
    return by_id[q[0]].first, by_id[q[-1]].last


def _join_cost(q, mine, by_id, fps, app, p):
    """Cost of giving piece q to a player that already has pieces `mine` (none: orphan_cost)."""
    if not mine:
        return p["identity"]["orphan_cost"]
    a0, a1 = _span(q, by_id)
    before = [m for m in mine if _span(m, by_id)[1] < a0]
    after = [m for m in mine if _span(m, by_id)[0] > a1]
    total, n = 0.0, 0
    if before:
        c = link_cost(by_id[max(before, key=lambda m: _span(m, by_id)[1])[-1]], by_id[q[0]], fps, app, p, max_gap_s=None)
        if c is None:
            return None
        total, n = total + c, n + 1
    if after:
        c = link_cost(by_id[q[-1]], by_id[min(after, key=lambda m: _span(m, by_id)[0])[0]], fps, app, p, max_gap_s=None)
        if c is None:
            return None
        total, n = total + c, n + 1
    return total if n else None


def _overlap(a, b, by_id):
    return not (_span(a, by_id)[1] < _span(b, by_id)[0] or _span(a, by_id)[0] > _span(b, by_id)[1])


def label_pieces(pieces, pids, pins, by_id, fps, app, p, log=print, unpinned=None):
    lab = {}
    for n, q in enumerate(pieces):
        ps = sorted({pins[t] for t in q if t in pins})
        if ps:
            lab[n] = ps[0]
    # a player can't be in two pieces at once: of pieces pinned to the same player that overlap in time, the one with
    # the most pinned frames keeps the pin; the others are labelled like free pieces below
    for pid in sorted(set(lab.values())):
        mine = sorted((n for n, x in lab.items() if x == pid),
                      key=lambda n: (-sum(len(by_id[t]) for t in pieces[n] if pins.get(t) == pid), n))
        kept = []
        for n in mine:
            if any(_overlap(pieces[n], pieces[k], by_id) for k in kept):
                log(f"    tracklets {pieces[n]}: {pid} is pinned on another piece at the same time - pin not used")
                del lab[n]
                if unpinned is not None:
                    unpinned.update(t for t in pieces[n] if pins.get(t) == pid)
            else:
                kept.append(n)
    free = sorted((n for n in range(len(pieces)) if n not in lab), key=lambda n: -sum(len(by_id[t]) for t in pieces[n]))
    for n in free:
        q = pieces[n]
        a0, a1 = _span(q, by_id)
        best = None
        for pid in pids:
            mine = [pieces[m] for m, x in lab.items() if x == pid]
            if any(not (_span(m, by_id)[1] < a0 or _span(m, by_id)[0] > a1) for m in mine):
                continue
            c = _join_cost(q, mine, by_id, fps, app, p)
            if c is not None and (best is None or c < best[0]):
                best = (c, pid)
        if best:
            lab[n] = best[1]
    return {t: lab[n] for n, q in enumerate(pieces) if n in lab for t in q}


def _gap_path(a, b, pool, fps, app, p):
    """Most frames of time-disjoint tracklets from `pool` between a and b (a -> x1 -> ... -> b, every hop within
    link_cost's gates); [] when none fits."""
    xs = sorted((x for x in pool if x.first > a.last and x.last < b.first), key=lambda x: (x.first, x.tid))
    best = {}                                   # tid -> (frames, path) of the best chain a -> ... -> x
    for x in xs:
        opts = [(len(x), [x.tid])] if link_cost(a, x, fps, app, p) is not None else []
        opts += [(best[y.tid][0] + len(x), best[y.tid][1] + [x.tid]) for y in xs
                 if y.tid in best and y.last < x.first and link_cost(y, x, fps, app, p) is not None]
        if opts:
            best[x.tid] = max(opts, key=lambda o: (o[0], [-t for t in o[1]]))
    ends = [best[x.tid] for x in xs if x.tid in best and link_cost(x, b, fps, app, p) is not None]
    return max(ends, key=lambda o: (o[0], [-t for t in o[1]]))[1] if ends else []


def fill_gaps(player_of, T, labels, sides, fps, app, p, log=print):
    """Unassigned player-like / unknown tracklets inside a gap (<= max_link_gap_s) between two consecutive pieces of one
    player join it when both hops pass the link gates (T15: crowd splits leave short fragments the flow skips, since
    every hop costs appearance). A fragment must be on the player's half or only ever on the net line (|z| <=
    net_band_m: either half). A fragment that fits the gaps of two players stays unassigned (ambiguous), and so does the
    rest of every gap path it is on."""
    band = p["net_band_m"]
    pool = [t for t in T if t.tid not in player_of and labels.get(t.tid) in ("player", "unknown")]
    on_line = {t.tid for t in pool if (np.abs(t.floor()[:, 1]) <= band).all()}
    paths, claims = [], {}
    for pid in sorted(set(player_of.values())):
        mine = sorted((t for t in T if player_of.get(t.tid) == pid), key=lambda t: t.first)
        ok = [t for t in pool if t.tid in on_line or half_of(t, band) == sides.get(pid)]
        for a, b in zip(mine, mine[1:]):
            if 1 < b.first - a.last <= p["identity"]["max_link_gap_s"] * fps:
                path = _gap_path(a, b, ok, fps, app, p)
                if path:
                    paths.append((pid, path))
                    for tid in path:
                        claims.setdefault(tid, set()).add(pid)
    out = {}
    # a path with a contested fragment is dropped whole: without that fragment its remaining hops are unchecked
    for pid, path in paths:
        bad = [tid for tid in path if len(claims[tid]) > 1]
        if bad:
            log(f"    tracklets {path}: {bad} fit the gaps of {sorted(set().union(*(claims[t] for t in bad)))} - "
                f"left unassigned")
        else:
            out.update({tid: pid for tid in path})
    return out


def assign(tracklets, labels, scores, pins, sides, fps, app, p, log=print, unpinned=None):
    """-> {tid: player id}. unpinned (a set, optional) receives the pinned tracklets whose pin identity did not use
    (other half, overlapping another piece of that player): they must not be reported as pinned."""
    unpinned = set() if unpinned is None else unpinned
    player_of = {}
    all_by_id = {t.tid: t for t in tracklets}
    for half in ("near", "far"):
        pids = [pid for pid, s in sides.items() if s == half]
        if not pids:
            continue
        pins_h = {}
        for tid, pid in pins.items():
            if pid in pids:
                t = all_by_id.get(tid)
                if t is None:
                    log(f"    seed pin {pid} -> tracklet {tid} does not exist - dropped")
                elif half_of(t, p["net_band_m"]) == half:
                    pins_h[tid] = pid
                else:
                    log(f"    seed pin {pid} -> tracklet {tid} is on the other half - dropped (no net crossing)")
                    unpinned.add(tid)
        T = [t for t in tracklets if half_of(t, p["net_band_m"]) == half and (t.tid in pins_h or labels.get(t.tid) in ("player", "unknown"))]
        if not T:
            continue
        ch, costs = chains(T, len(pids), fps, app, p, pins_h, {t.tid: scores.get(t.tid, 0.5) for t in T})
        by_id = {t.tid: t for t in T}
        pieces = split_conflicts(ch, pins_h, costs)
        # a piece needs player evidence (a pin or a "player" tracklet): "unknown" tracklets only bridge a player's
        # pieces, they never make an identity on their own (an unseen player's slot must stay empty)
        pieces = [q for q in pieces if any(t in pins_h or labels.get(t) == "player" for t in q)]
        placed = {tid for q in pieces for tid in q}
        # tracklets the <= k chains left out (pins beyond max_link_gap_s, later segments of a player): singleton
        # pieces; "unknown" ones only ever enter through the chains
        for t in sorted(T, key=lambda t: (t.first, t.tid)):
            if t.tid in placed or (t.tid not in pins_h and labels.get(t.tid) != "player"):
                continue
            pid = pins_h.get(t.tid)
            if pid is not None and any(
                    pins_h.get(m) == pid and not (_span(q, by_id)[1] < t.first or _span(q, by_id)[0] > t.last)
                    for q in pieces for m in q):
                log(f"    seed pin {pid} -> tracklet {t.tid} overlaps another {pid} tracklet - dropped")
                unpinned.add(t.tid)
                continue
            pieces.append([t.tid])
            placed.add(t.tid)
        player_of.update(label_pieces(pieces, pids, pins_h, by_id, fps, app, p, log=log, unpinned=unpinned))
    player_of.update(fill_gaps(player_of, tracklets, labels, sides, fps, app, p, log=log))
    return player_of
