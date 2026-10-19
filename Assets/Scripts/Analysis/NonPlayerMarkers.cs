using System.Collections.Generic;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>SP3 persons in the twin: grey capsules for non-players, faint dashed ones for "unknown", floor discs for
    /// learnt bystander / umpire hotspots. Reads the cvdebug sidecar's persons[] / hotspots[] (attached only when they pair).</summary>
    [DefaultExecutionOrder(140)]
    public class NonPlayerMarkers : MonoBehaviour
    {
        public AnalysisPlayer player;
        public bool show = false;
        public bool showIds = false;
        public bool showHotspots = true;

        const float Height = 1.6f, Radius = 0.22f, RingR = 0.35f;
        static readonly Color NonPlayer = new Color(0.6f, 0.6f, 0.6f, 0.6f), Unknown = new Color(0.8f, 0.8f, 0.8f, 0.2f),
                              Hot = new Color(0.5f, 0.5f, 0.5f, 0.18f), Ring = new Color(0.9f, 0.9f, 0.9f, 0.8f);

        Transform _root;
        readonly List<GameObject> _caps = new List<GameObject>();
        readonly List<GameObject> _discs = new List<GameObject>();
        readonly List<(int id, Vector3 pos)> _labels = new List<(int, Vector3)>();
        CvLineBatch _rings;
        Material _npMat, _unkMat, _hotMat;

        public int VisibleCount { get; private set; }

        void Awake() { if (player == null) player = GetComponent<AnalysisPlayer>(); }
        void OnEnable() { if (player != null) { player.ClipLoaded += Rebuild; if (player.Doc != null) Rebuild(); } }
        void OnDisable() { if (player != null) player.ClipLoaded -= Rebuild; }
        void OnDestroy() { _rings?.Dispose(); if (_root != null) Destroy(_root.gameObject); }

        void Rebuild()
        {
            if (_root != null) { _rings?.Dispose(); _rings = null; Destroy(_root.gameObject); }
            _caps.Clear(); _discs.Clear();
            _root = new GameObject("cv_persons").transform;
            if (_npMat == null) _npMat = CvMaterials.Make(NonPlayer, false);
            if (_unkMat == null) _unkMat = CvMaterials.Make(Unknown, false);
            if (_hotMat == null) _hotMat = CvMaterials.Make(Hot, false);
            _rings = new CvLineBatch("cv_unknown_rings", _root, false);
            var cv = player.CvDebug;
            if (cv == null) return;
            foreach (var h in cv.hotspots)
            {
                var d = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                Destroy(d.GetComponent<Collider>());
                d.name = $"hotspot_{h.x:F1}_{h.z:F1}";
                d.transform.SetParent(_root, false);
                d.transform.position = player.transform.TransformPoint(new Vector3(h.x, 0.005f, h.z));
                d.transform.localScale = new Vector3(1.5f, 0.003f, 1.5f);
                d.GetComponent<Renderer>().sharedMaterial = _hotMat;
                _discs.Add(d);
            }
        }

        GameObject Cap(int i)
        {
            while (_caps.Count <= i)
            {
                var c = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                Destroy(c.GetComponent<Collider>());
                c.name = $"person_{_caps.Count}";
                c.transform.SetParent(_root, false);
                c.transform.localScale = new Vector3(Radius * 2f, Height * 0.5f, Radius * 2f);
                var r = c.GetComponent<Renderer>();
                r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off; r.receiveShadows = false;
                _caps.Add(c);
            }
            return _caps[i];
        }

        void LateUpdate()
        {
            var cv = player != null ? player.CvDebug : null;
            int n = 0;
            _labels.Clear();
            if (_rings != null) _rings.Clear();
            if (show && cv != null)
            {
                int f = player.CurrentFrame;
                foreach (var t in cv.persons)
                {
                    if (t.label == "player" || f < t.first_frame || f > t.last_frame || !t.FloorAt(f, out var xz)) continue;
                    Vector3 p = player.transform.TransformPoint(new Vector3(xz.x, 0f, xz.y));
                    var c = Cap(n++);
                    c.SetActive(true);
                    c.transform.position = p + Vector3.up * (Height * 0.5f);
                    bool unknown = t.label != "nonplayer";
                    c.GetComponent<Renderer>().sharedMaterial = unknown ? _unkMat : _npMat;
                    if (unknown)
                        for (int k = 0; k < 16; k += 2)          // dashed ring: every other of 16 segments
                        {
                            float a0 = k * Mathf.PI / 8f, a1 = (k + 1) * Mathf.PI / 8f;
                            _rings.Add(p + new Vector3(Mathf.Cos(a0), 0.01f, Mathf.Sin(a0)) * RingR,
                                       p + new Vector3(Mathf.Cos(a1), 0.01f, Mathf.Sin(a1)) * RingR, Ring);
                        }
                    if (showIds) _labels.Add((t.track_id, p + Vector3.up * (Height + 0.1f)));
                }
            }
            for (int i = n; i < _caps.Count; i++) if (_caps[i].activeSelf) _caps[i].SetActive(false);
            if (_rings != null) { _rings.Commit(); _rings.SetVisible(show); }
            foreach (var d in _discs) d.SetActive(show && showHotspots);
            VisibleCount = n;
        }

        void OnGUI()
        {
            if (!show || !showIds || _labels.Count == 0 || Camera.main == null) return;
            foreach (var (id, pos) in _labels)
            {
                Vector3 s = Camera.main.WorldToScreenPoint(pos);
                if (s.z > 0f && !UiPointerGuard.UnderPanel(s)) GUI.Label(new Rect(s.x - 20, Screen.height - s.y - 10, 60, 20), $"#{id}");
            }
        }
    }
}
