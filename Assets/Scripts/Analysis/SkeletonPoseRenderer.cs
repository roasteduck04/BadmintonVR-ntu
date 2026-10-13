using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>
    /// Procedural stick-figure skin: a sphere per joint (joint_{j}) + a capsule per
    /// non-root bone (bone_{j}). Extracted verbatim from AnalysisPlayer so the default
    /// skin stays pixel-identical. Radii are skeleton-specific (constructor).
    /// </summary>
    public class SkeletonPoseRenderer : IPoseRenderer
    {
        readonly float _jointRadius, _boneRadius;
        Transform _root;
        int[] _parents;
        int _jointCount;
        Transform[] _joints;
        Transform[] _bones;   // index 0 (root joint) unused
        Material[] _mats;     // created per Build, destroyed in Clear

        public SkeletonPoseRenderer(float jointRadius, float boneRadius)
        {
            _jointRadius = jointRadius;
            _boneRadius = boneRadius;
        }

        public void Build(Transform root, int[] parents, Color color)
        {
            _root = root;
            _parents = parents;
            _jointCount = parents.Length;
            Material jointMat = RendererUtil.MakeMat(color);
            Material boneMat = RendererUtil.MakeMat(Color.Lerp(color, Color.white, 0.35f));
            _mats = new[] { jointMat, boneMat };

            _joints = new Transform[_jointCount];
            for (int j = 0; j < _jointCount; j++)
            {
                var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                s.name = $"joint_{j}";
                s.transform.SetParent(root, false);
                float radius = (j >= 22 && j <= 51) ? _jointRadius * 0.5f : _jointRadius;
                s.transform.localScale = Vector3.one * (radius * 2f);
                s.GetComponent<Renderer>().sharedMaterial = jointMat;
                RendererUtil.DestroyCollider(s);
                _joints[j] = s.transform;
            }
            _bones = new Transform[_jointCount];
            for (int j = 1; j < _jointCount; j++)
            {
                var c = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                c.name = $"bone_{j}";
                c.transform.SetParent(root, false);
                c.GetComponent<Renderer>().sharedMaterial = boneMat;
                RendererUtil.DestroyCollider(c);
                _bones[j] = c.transform;
            }
        }

        public void Apply(Vector3[] jointLocal, float[] conf, bool hasPose, float cutoff)
        {
            for (int j = 0; j < _jointCount; j++)
            {
                bool ok = hasPose && conf[j] >= cutoff;
                _joints[j].gameObject.SetActive(ok);
                if (ok) _joints[j].localPosition = jointLocal[j];
            }
            for (int j = 1; j < _jointCount; j++)
            {
                int par = _parents[j];
                bool ok = hasPose && conf[j] >= cutoff && conf[par] >= cutoff;
                _bones[j].gameObject.SetActive(ok);
                float br = (j >= 22 && j <= 51) ? _boneRadius * 0.5f : _boneRadius;
                if (ok) RendererUtil.PlaceBone(_bones[j], jointLocal[j], jointLocal[par], br);
            }
        }

        public void SetActive(bool on) { if (_root != null) _root.gameObject.SetActive(on); }

        public void SetJointNames(string[] names) { }

        public bool TryGetRenderedJoint(int joint, out Vector3 rootLocal) { rootLocal = default; return false; }

        public void Clear()
        {
            _joints = null; _bones = null;
            if (_mats != null) foreach (var m in _mats) RendererUtil.DestroySafe(m);
            _mats = null;
        }
    }
}
