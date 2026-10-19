using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>Background reads keyed by file (v0.8.1): one worker task per key; Cancel drops a key's result for good (also
    /// one already finished but not yet taken); the main thread takes finished results one at a time. No Unity API, so the
    /// EditMode tests cover it.</summary>
    public sealed class KeyedLoads<T> where T : class
    {
        sealed class Job { public Task<T> task; public CancellationTokenSource cancel; }
        readonly Dictionary<string, Job> _jobs = new Dictionary<string, Job>();

        public int Count => _jobs.Count;
        public bool IsPending(string key) => key != null && _jobs.ContainsKey(key);

        /// <summary>Run read(token) on a worker for key; false (nothing started) when key is already pending.</summary>
        public bool Start(string key, Func<CancellationToken, T> read)
        {
            if (key == null || _jobs.ContainsKey(key)) return false;
            var cts = new CancellationTokenSource();
            _jobs[key] = new Job { task = Task.Run(() => read(cts.Token)), cancel = cts };
            return true;
        }

        /// <summary>Signal the key's token and forget it: its result is never returned.</summary>
        public void Cancel(string key)
        {
            if (key == null || !_jobs.TryGetValue(key, out var j)) return;
            j.cancel.Cancel(); _jobs.Remove(key);
        }

        public void CancelAll() { foreach (var k in _jobs.Keys.ToList()) Cancel(k); }
        public void CancelWhere(Func<string, bool> drop) { foreach (var k in _jobs.Keys.Where(drop).ToList()) Cancel(k); }

        /// <summary>One finished job, removed from the pending set: its result, or null + the exception when read threw.</summary>
        public bool TryTakeFinished(out string key, out T result, out Exception error)
        {
            key = null; result = null; error = null;
            foreach (var kv in _jobs)
            {
                if (!kv.Value.task.IsCompleted) continue;
                key = kv.Key;
                var t = kv.Value.task;
                if (t.Status == TaskStatus.RanToCompletion) result = t.Result;
                else error = t.Exception?.GetBaseException() ?? new OperationCanceledException();
                _jobs.Remove(key);
                return true;
            }
            return false;
        }
    }

    /// <summary>Other builds of the same scene (pose variants, camera subsets, the full build of a subset) drawn as thin line
    /// skeletons, re-timed by the docs' sync offsets so the frames show the same instant (SP4 § 2, SP2 F7). A ghost build is
    /// read on a worker thread (AnalysisPlayer.ReadClip, v0.8.1) and drawn once ready; a newly loaded clip only re-times the
    /// ghosts that stay on (their sync offsets are kept), it does not re-read them.</summary>
    [DefaultExecutionOrder(150)]
    public class VariantGhosts : MonoBehaviour
    {
        public AnalysisPlayer player;
        static readonly Color[] Palette = { Color.white, Color.yellow, new Color(0.3f, 1f, 1f), new Color(0.6f, 1f, 0.2f) };

        class Ghost
        {
            public string file, label; public AnalysisDoc doc; public float shift; public bool known;
            public float lift; public Color color; public CvLineBatch lines;
            public List<string> offNames; public List<float> offs;      // the build's sync offsets (re-timed on ClipLoaded)
        }

        /// <summary>A ghost build read on a worker: the clip + its sync offsets (from its cvdebug, else meta.pose_variant).</summary>
        sealed class GhostRead
        {
            public AnalysisPlayer.ClipData clip;
            public readonly List<string> offNames = new List<string>(); public readonly List<float> offs = new List<float>();
        }

        readonly List<string> _candidates = new List<string>();
        readonly Dictionary<string, Ghost> _on = new Dictionary<string, Ghost>();
        readonly KeyedLoads<GhostRead> _loads = new KeyedLoads<GhostRead>();
        Transform _root;
        GUIStyle _style;

        public IReadOnlyList<string> Candidates => _candidates;
        /// <summary>Turned on: drawn, or still being read (IsLoading).</summary>
        public bool IsOn(string file) => _on.ContainsKey(file) || _loads.IsPending(file);
        /// <summary>Turned on but still being read on a worker thread (not drawn yet).</summary>
        public bool IsLoading(string file) => _loads.IsPending(file);
        public bool AnyLoading => _loads.Count > 0;
        public string LabelOf(string file) => _on.TryGetValue(file, out var g) ? g.label : Path.GetFileName(file).Replace(".analysis.json", "");
        /// <summary>The line colour of a ghost that is drawn (for the viewer's legend).</summary>
        public bool TryGetColor(string file, out Color c) { c = Color.white; if (!_on.TryGetValue(file, out var g)) return false; c = g.color; return true; }
        /// <summary>A note about the ghosts that are on (empty when none applies).</summary>
        public string Note => _on.Keys.Any(k => k.Contains("doubles_2sides")) ? "doubles_2sides: P3/P4 ids may differ between builds (stale seeds)" : "";

        void Awake() { if (player == null) player = GetComponent<AnalysisPlayer>(); }
        void OnEnable() { if (player != null) { player.ClipLoaded += Rebuild; if (player.Doc != null) Rebuild(); } }
        void OnDisable() { if (player != null) player.ClipLoaded -= Rebuild; _loads.CancelAll(); }
        void OnDestroy() { _loads.CancelAll(); foreach (var g in _on.Values) g.lines.Dispose(); if (_root != null) Destroy(_root.gameObject); }

        void Rebuild()
        {
            if (_root == null) _root = new GameObject("cv_ghosts").transform;
            _candidates.Clear();
            string dir = Path.Combine(Application.streamingAssetsPath, "analysis");
            if (Directory.Exists(dir))
            {
                var files = Directory.GetFiles(dir, "*.analysis.json").Select(p => "analysis/" + Path.GetFileName(p));
                _candidates.AddRange(CvMath.GhostCandidates(files, player.analysisFile, AnalysisPlayer.PoseVariantSuffixes).OrderBy(s => s));
            }
            _loads.CancelWhere(k => !_candidates.Contains(k));
            foreach (var k in _on.Keys.Where(k => !_candidates.Contains(k)).ToList()) { _on[k].lines.Dispose(); _on.Remove(k); }
            foreach (var g in _on.Values) Retime(g);          // the shift is relative to the newly loaded doc; no re-read
        }

        /// <summary>Turn a ghost on (read on a worker thread, drawn once ready) or off (also drops a read still running).</summary>
        public void Set(string file, bool on)
        {
            if (!on) { _loads.Cancel(file); if (_on.TryGetValue(file, out var g0)) { g0.lines.Dispose(); _on.Remove(file); } return; }
            if (IsOn(file) || !_candidates.Contains(file)) return;
            _ = JsonLoad.Root;                                   // main thread: the worker uses the cached path
            _loads.Start(file, cancel =>
            {
                var r = new GhostRead { clip = AnalysisPlayer.ReadClip(file, cancel) };
                if (r.clip.doc != null) CvMath.TimelineOffsets(r.clip.cv, r.clip.doc, r.offNames, r.offs);
                r.clip.cv = null; r.clip.rackets = null; r.clip.shuttles = null;   // a ghost only draws the pose
                return r;
            });
        }

        /// <summary>Main thread: draw one finished read per frame (none while the player has no doc to re-time it against).</summary>
        void ApplyFinished()
        {
            if (player.Doc == null || !_loads.TryTakeFinished(out string file, out var r, out var error)) return;
            var d = r?.clip;
            d?.log.Replay();
            if (d?.doc == null)
            {
                if (d?.error != "cancelled")
                    Debug.LogError($"[Analysis] ghost build failed to load: {file}" + (error != null ? $" ({error.Message})" : ""));
                return;
            }
            if (_on.ContainsKey(file) || !_candidates.Contains(file)) return;
            string clip = Path.GetFileName(file).Replace(".analysis.json", "");
            // lift: one vertical reference per build, like the main twin (AnalysisDoc.GroundLift, computed by ReadClip)
            var g = new Ghost { file = file, doc = d.doc, lift = d.lift, offNames = r.offNames, offs = r.offs,
                                color = Palette[_on.Count % Palette.Length] };
            g.lines = new CvLineBatch($"ghost_{clip}", _root, true);          // on top: a comparison layer over any skin
            Retime(g);
            _on[file] = g;
        }

        void Retime(Ghost g)
        {
            var an = new List<string>(); var ao = new List<float>();
            CvMath.TimelineOffsets(player.CvDebug, player.Doc, an, ao);
            g.shift = CvMath.GhostFrameShift(an, ao, g.offNames, g.offs, out g.known);
            string clip = Path.GetFileName(g.file).Replace(".analysis.json", "");
            g.label = $"{clip} · {g.doc.ShortLabel(40)}"
                      + (g.known && Mathf.Abs(g.shift) > 0.05f ? $" · shift {g.shift:+0.0;-0.0} f" : "")
                      + (g.known ? "" : " · timeline unknown");
        }

        void LateUpdate()
        {
            ApplyFinished();
            foreach (var g in _on.Values)
            {
                g.lines.Clear();
                int fb = CvMath.GhostFrame(player.CurrentFrame, g.shift, g.doc.FrameCount);
                var par = g.doc.skeleton.parents;
                if (fb >= 0)
                    for (int pi = 0; pi < g.doc.players.Length; pi++)
                    {
                        var p = g.doc.players[pi];
                        if (!p.HasPose(fb)) continue;
                        var pos = p.PositionXZ(fb);
                        if (!pos.HasValue) continue;
                        Vector3 root = new Vector3(pos.Value.x, player.LiftToFloor ? g.lift : 0f, pos.Value.y);
                        int nj = Mathf.Min(p.PoseJointCount(fb), par.Length);
                        for (int j = 0; j < nj; j++)
                        {
                            int q = par[j];
                            if (q < 0 || q >= nj || p.JointConf(fb, j) <= 1e-6f || p.JointConf(fb, q) <= 1e-6f) continue;
                            g.lines.Add(player.transform.TransformPoint(root + p.JointLocal(fb, j)),
                                        player.transform.TransformPoint(root + p.JointLocal(fb, q)), g.color);
                        }
                    }
                g.lines.Commit();
            }
        }

        void OnGUI()
        {
            if ((_on.Count == 0 && _loads.Count == 0) || UiPointerGuard.ViewerOwnsScreen) return;   // the viewer's Builds section shows the legend
            if (_style == null) _style = new GUIStyle(GUI.skin.label) { fontSize = 13 };
            float y = 272f;
            foreach (var g in _on.Values)
            {
                _style.normal.textColor = g.color;
                GUI.Label(new Rect(12, y, 700, 20), "ghost " + g.label, _style);
                y += 18f;
            }
            if (_loads.Count > 0)
            {
                _style.normal.textColor = Color.gray;
                GUI.Label(new Rect(12, y, 700, 20), $"loading {_loads.Count} ghost build(s) ...", _style);
                y += 18f;
            }
            if (_on.Keys.Any(k => k.Contains("doubles_2sides")))
            {
                _style.normal.textColor = Color.gray;
                GUI.Label(new Rect(12, y, 700, 20), Note, _style);
            }
        }
    }
}
