using System.Collections.Generic;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>
    /// Draws the shuttle: the baseline track in analysis.json (exactly as before), or a shuttle
    /// sidecar variant ("improved", "tracker") coloured by source with an optional flight trail,
    /// a faint baseline ghost and hit markers. Attaches to the same GameObject as AnalysisPlayer.
    /// </summary>
    public class ShuttleVisualAnalysis : MonoBehaviour
    {
        public const string Baseline = "baseline";

        public Color baselineColor = new Color(0.36f, 0.9f, 0.65f);
        public Color filledColor = new Color(1f, 0.85f, 0.2f);
        public Color physicsColor = new Color(0.2f, 0.85f, 1f);
        public Color ghostColor = new Color(0.75f, 0.75f, 0.75f, 0.35f);
        public Color hitColor = new Color(1f, 1f, 1f, 0.8f);
        [Tooltip("Preferred shuttle source; falls back to baseline when the clip has no such sidecar")]
        public string preferredVariant = Baseline;
        public bool ghostBaseline;
        public bool trail = true;
        [Range(2, 120)] public int trailFrames = 20;
        [Range(1, 120)] public int hitMarkerFrames = 18;
        public float trailWidth = 0.04f;

        AnalysisDoc _doc;
        Transform _ball, _ghost, _hit;
        Renderer _ballRenderer;
        LineRenderer _trail;
        // Created once and reused across Build calls (reloading a clip calls Build again on the
        // same component); nothing outside this class touches them.
        Material _opaqueMat, _fadeMat, _ghostMat, _hitMat, _trailMat;
        MaterialPropertyBlock _pb;
        readonly Dictionary<string, ShuttleVariantDoc> _variants = new Dictionary<string, ShuttleVariantDoc>();
        Vector3[] _trailBufArr = new Vector3[0];
        float _lift;     // AnalysisPlayer.VerticalLift: the same clip-wide lift as the bodies and rackets

        public string EffectiveVariant
            => preferredVariant != null && _variants.ContainsKey(preferredVariant) ? preferredVariant : Baseline;
        public ShuttleVariantDoc VariantDoc(string name)
            => name != null && _variants.TryGetValue(name, out var d) ? d : null;

        public void SetVariants(Dictionary<string, ShuttleVariantDoc> variants)
        {
            _variants.Clear();
            if (variants == null) return;
            foreach (var kv in variants) if (kv.Value != null) _variants[kv.Key] = kv.Value;
        }

        public void Build(Transform parent, AnalysisDoc doc, float radius)
        {
            _doc = doc;
            if (_opaqueMat == null) _opaqueMat = RendererUtil.MakeMat(baselineColor);
            if (_fadeMat == null) _fadeMat = RendererUtil.MakeTransparentMat(baselineColor);
            if (_ghostMat == null) _ghostMat = RendererUtil.MakeTransparentMat(ghostColor);
            if (_hitMat == null) _hitMat = RendererUtil.MakeTransparentMat(hitColor);
            _pb = new MaterialPropertyBlock();
            _ball = MakeSphere(parent, "shuttle", radius, _opaqueMat);
            _ballRenderer = _ball.GetComponent<Renderer>();
            _ghost = MakeSphere(parent, "shuttle_ghost", radius, _ghostMat);

            var hit = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            hit.name = "shuttle_hit";
            hit.transform.SetParent(parent, false);
            hit.transform.localScale = new Vector3(0.35f, 0.004f, 0.35f);
            hit.GetComponent<Renderer>().sharedMaterial = _hitMat;
            RendererUtil.DestroyCollider(hit);
            _hit = hit.transform;

            var tr = new GameObject("shuttle_trail");
            tr.transform.SetParent(parent, false);
            _trail = tr.AddComponent<LineRenderer>();
            _trail.useWorldSpace = false;
            _trail.widthMultiplier = trailWidth;
            _trail.numCapVertices = 2;
            _trail.positionCount = 0;
            if (_trailMat == null)
            {
                var sh = Shader.Find("Sprites/Default") ?? Shader.Find("Universal Render Pipeline/Unlit");
                if (sh != null) _trailMat = new Material(sh);
            }
            if (_trailMat != null) _trail.sharedMaterial = _trailMat;
            HideAll();
        }

        static Transform MakeSphere(Transform parent, string name, float radius, Material mat)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localScale = Vector3.one * (radius * 2f);
            go.GetComponent<Renderer>().sharedMaterial = mat;
            RendererUtil.DestroyCollider(go);
            return go.transform;
        }

        public Vector3? BaselineAt(int frame)
        {
            var src = _doc != null ? _doc.BaselineShuttle : null;
            return src != null ? src.PositionAt(frame) : null;
        }

        /// <param name="lift">vertical offset added to every drawn shuttle point (AnalysisPlayer.VerticalLift), so the shuttle
        /// shares the bodies' and rackets' vertical reference</param>
        public void ShowFrame(int frame, bool visible, float lift = 0f)
        {
            if (_ball == null) return;
            _lift = lift;
            if (!visible) { HideAll(); return; }
            string active = EffectiveVariant;
            if (active == Baseline)
            {
                _ballRenderer.sharedMaterial = _opaqueMat;
                _ballRenderer.SetPropertyBlock(null);
                Place(_ball, BaselineAt(frame));
                _ghost.gameObject.SetActive(false);
                _hit.gameObject.SetActive(false);
                _trail.positionCount = 0;
                return;
            }
            var doc = _variants[active];
            var fr = doc.FrameAt(frame);
            Color c = ColorFor(fr != null ? fr.source : null);
            c.a = fr != null ? Mathf.Lerp(0.35f, 1f, Mathf.Clamp01(fr.confidence)) : 1f;
            _ballRenderer.sharedMaterial = _fadeMat;
            _pb.SetColor("_BaseColor", c);
            _pb.SetColor("_Color", c);
            _ballRenderer.SetPropertyBlock(_pb);
            Place(_ball, doc.PositionAt(frame));
            Place(_ghost, ghostBaseline ? BaselineAt(frame) : null);
            DrawTrail(doc, frame, c);
            DrawHit(doc, frame);
        }

        void Place(Transform t, Vector3? p)
        {
            t.gameObject.SetActive(p.HasValue);
            if (p.HasValue) t.localPosition = p.Value + new Vector3(0f, _lift, 0f);
        }

        Color ColorFor(string source)
            => source == "filled" ? filledColor : source == "physics" ? physicsColor : baselineColor;

        void DrawTrail(ShuttleVariantDoc doc, int frame, Color c)
        {
            if (_trailBufArr.Length != trailFrames) _trailBufArr = new Vector3[trailFrames];
            int count = 0;
            if (trail)
            {
                int f = frame;
                while (count < trailFrames && f >= 0 && frame - f < trailFrames)
                {
                    var p = doc.PositionAt(f);
                    if (!p.HasValue) break;
                    count++;
                    f--;
                }
            }
            if (count < 2) { _trail.positionCount = 0; return; }
            int start = frame - count + 1;
            for (int i = 0; i < count; i++)
                _trailBufArr[i] = doc.PositionAt(start + i).Value + new Vector3(0f, _lift, 0f);
            _trail.positionCount = count;
            _trail.SetPositions(_trailBufArr);
            Color tail = c, head = c;
            tail.a = 0f;
            head.a = 0.8f;
            _trail.startColor = tail;
            _trail.endColor = head;
        }

        void DrawHit(ShuttleVariantDoc doc, int frame)
        {
            Vector3? at = null;
            foreach (var e in doc.events)
                if (e.type == "hit" && frame >= e.frame && frame < e.frame + hitMarkerFrames
                    && e.position != null && e.position.Length == 3)
                    at = new Vector3(e.position[0], e.position[1], e.position[2]);
            Place(_hit, at);
            if (at.HasValue)
            {
                var cam = Camera.main;
                if (cam != null) _hit.up = -cam.transform.forward;
            }
        }

        public string Describe(int frame)
        {
            string active = EffectiveVariant;
            if (active == Baseline)
            {
                var b = BaselineAt(frame);
                return b.HasValue ? $"shuttle [baseline] ({b.Value.x:F2},{b.Value.y:F2},{b.Value.z:F2})"
                                  : "shuttle [baseline] (none)";
            }
            var doc = _variants[active];
            var p = doc.PositionAt(frame);
            if (!p.HasValue) return $"shuttle [{active}] (none)";
            var fr = doc.FrameAt(frame);
            string err = fr.reproj_px >= 0f ? $" {fr.reproj_px:F1}px" : "";
            return $"shuttle [{active}] ({p.Value.x:F2},{p.Value.y:F2},{p.Value.z:F2}) {fr.source} {fr.n_views}v{err}";
        }

        public void HideAll()
        {
            if (_ball != null) _ball.gameObject.SetActive(false);
            if (_ghost != null) _ghost.gameObject.SetActive(false);
            if (_hit != null) _hit.gameObject.SetActive(false);
            if (_trail != null) _trail.positionCount = 0;
        }

        /// <summary>Forget the geometry (AnalysisPlayer.Clear destroys the child objects).</summary>
        public void Clear()
        {
            _ball = _ghost = _hit = null;
            _ballRenderer = null;
            _trail = null;
        }
    }
}
