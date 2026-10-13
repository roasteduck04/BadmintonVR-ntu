using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>Other builds of the same scene (pose variants, camera subsets, the full build of a subset) drawn as thin line
    /// skeletons, re-timed by the docs' sync offsets so the frames show the same instant (SP4 § 2, SP2 F7).</summary>
    [DefaultExecutionOrder(150)]
    public class VariantGhosts : MonoBehaviour
    {
        public AnalysisPlayer player;
        static readonly Color[] Palette = { Color.white, Color.yellow, new Color(0.3f, 1f, 1f), new Color(0.6f, 1f, 0.2f) };

        class Ghost
        {
            public string file, label; public AnalysisDoc doc; public float shift; public bool known;
            public float lift; public Color color; public CvLineBatch lines;
        }

        readonly List<string> _candidates = new List<string>();
        readonly Dictionary<string, Ghost> _on = new Dictionary<string, Ghost>();
        Transform _root;
        GUIStyle _style;

        public IReadOnlyList<string> Candidates => _candidates;
        public bool IsOn(string file) => _on.ContainsKey(file);
        public string LabelOf(string file) => _on.TryGetValue(file, out var g) ? g.label : Path.GetFileName(file).Replace(".analysis.json", "");

        void Awake() { if (player == null) player = GetComponent<AnalysisPlayer>(); }
        void OnEnable() { if (player != null) { player.ClipLoaded += Rebuild; if (player.Doc != null) Rebuild(); } }
        void OnDisable() { if (player != null) player.ClipLoaded -= Rebuild; }
        void OnDestroy() { foreach (var g in _on.Values) g.lines.Dispose(); if (_root != null) Destroy(_root.gameObject); }

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
            var keep = _on.Keys.Where(k => _candidates.Contains(k)).ToList();
            foreach (var k in _on.Keys.ToList()) { _on[k].lines.Dispose(); _on.Remove(k); }
            foreach (var k in keep) Set(k, true);            // re-load: the shift is relative to the newly loaded doc
        }

        public void Set(string file, bool on)
        {
            if (!on) { if (_on.TryGetValue(file, out var g0)) { g0.lines.Dispose(); _on.Remove(file); } return; }
            if (_on.ContainsKey(file) || !_candidates.Contains(file)) return;
            var doc = AnalysisDoc.Load(file);
            if (doc == null) return;
            var g = new Ghost { file = file, doc = doc, color = Palette[_on.Count % Palette.Length] };
            g.lift = doc.GroundLift();          // one vertical reference per build, like the main twin (AnalysisDoc.GroundLift)
            g.shift = Shift(doc, file, out g.known);
            string clip = Path.GetFileName(file).Replace(".analysis.json", "");
            g.label = $"{clip} · {doc.ShortLabel(40)}"
                      + (g.known && Mathf.Abs(g.shift) > 0.05f ? $" · shift {g.shift:+0.0;-0.0} f" : "")
                      + (g.known ? "" : " · timeline unknown");
            g.lines = new CvLineBatch($"ghost_{clip}", _root, true);          // on top: a comparison layer over any skin
            _on[file] = g;
        }

        float Shift(AnalysisDoc ghost, string ghostFile, out bool known)
        {
            var an = new List<string>(); var ao = new List<float>(); var bn = new List<string>(); var bo = new List<float>();
            CvMath.TimelineOffsets(player.CvDebug, player.Doc, an, ao);
            CvMath.TimelineOffsets(CvDebugDoc.TryLoad(AnalysisPlayer.CvDebugSidecarPath(ghostFile), ghost), ghost, bn, bo);
            return CvMath.GhostFrameShift(an, ao, bn, bo, out known);
        }

        void LateUpdate()
        {
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
            if (_on.Count == 0) return;
            if (_style == null) _style = new GUIStyle(GUI.skin.label) { fontSize = 13 };
            float y = 272f;
            foreach (var g in _on.Values)
            {
                _style.normal.textColor = g.color;
                GUI.Label(new Rect(12, y, 700, 20), "ghost " + g.label, _style);
                y += 18f;
            }
            if (_on.Keys.Any(k => k.Contains("doubles_2sides")))
            {
                _style.normal.textColor = Color.gray;
                GUI.Label(new Rect(12, y, 700, 20), "doubles_2sides: P3/P4 ids may differ between builds (stale seeds, SP3 F21)", _style);
            }
        }
    }
}
