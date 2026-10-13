using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>
    /// Asset-free "solid body" skin: tapered limb capsules + a torso capsule + a head
    /// sphere, all from the same SMPL-24 joints. Visibly distinct from the skeleton,
    /// proving the IPoseRenderer strategy swaps live. No mesh asset.
    /// </summary>
    public class SolidBodyPoseRenderer : IPoseRenderer
    {
        // SMPL-24 indices (see skeleton.joint_names in analysis.json).
        const int PELVIS = 0, L_HIP = 1, R_HIP = 2, L_KNEE = 4, R_KNEE = 5,
                  L_ANKLE = 7, R_ANKLE = 8, NECK = 12, HEAD = 15,
                  L_SHO = 16, R_SHO = 17, L_ELB = 18, R_ELB = 19, L_WRI = 20, R_WRI = 21;

        static readonly (int a, int b)[] Limbs =
        {
            (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI),
            (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
        };

        const float LimbRadius = 0.06f;
        const float HeadRadius = 0.12f;

        Transform _root, _torso, _head;
        Transform[] _limbs;
        Material _mat;

        public void Build(Transform root, int[] parents, Color color)
        {
            _root = root;
            _mat = RendererUtil.MakeMat(Color.Lerp(color, Color.white, 0.15f));

            _limbs = new Transform[Limbs.Length];
            for (int i = 0; i < Limbs.Length; i++)
            {
                var c = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                c.name = $"limb_{i}";
                c.transform.SetParent(root, false);
                c.GetComponent<Renderer>().sharedMaterial = _mat;
                RendererUtil.DestroyCollider(c);
                _limbs[i] = c.transform;
            }

            var torso = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            torso.name = "torso";
            torso.transform.SetParent(root, false);
            torso.GetComponent<Renderer>().sharedMaterial = _mat;
            RendererUtil.DestroyCollider(torso);
            _torso = torso.transform;

            var head = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            head.name = "head";
            head.transform.SetParent(root, false);
            head.transform.localScale = Vector3.one * (HeadRadius * 2f);
            head.GetComponent<Renderer>().sharedMaterial = _mat;
            RendererUtil.DestroyCollider(head);
            _head = head.transform;
        }

        public void Apply(Vector3[] jointLocal, float[] conf, bool hasPose, float cutoff)
        {
            for (int i = 0; i < Limbs.Length; i++)
            {
                var (a, b) = Limbs[i];
                bool ok = hasPose && conf[a] >= cutoff && conf[b] >= cutoff;
                _limbs[i].gameObject.SetActive(ok);
                if (ok) RendererUtil.PlaceBone(_limbs[i], jointLocal[a], jointLocal[b], LimbRadius);
            }

            bool torsoOk = hasPose && conf[PELVIS] >= cutoff && conf[NECK] >= cutoff;
            _torso.gameObject.SetActive(torsoOk);
            if (torsoOk)
            {
                float halfWidth = 0.5f * Vector3.Distance(jointLocal[L_SHO], jointLocal[R_SHO]);
                RendererUtil.PlaceBone(_torso, jointLocal[PELVIS], jointLocal[NECK],
                                       Mathf.Max(halfWidth, 0.08f));
            }

            bool headOk = hasPose && conf[HEAD] >= cutoff;
            _head.gameObject.SetActive(headOk);
            if (headOk) _head.localPosition = jointLocal[HEAD];
        }

        public void SetActive(bool on) { if (_root != null) _root.gameObject.SetActive(on); }

        public void SetJointNames(string[] names) { }

        public bool TryGetRenderedJoint(int joint, out Vector3 rootLocal) { rootLocal = default; return false; }

        public void Clear() { _limbs = null; _torso = null; _head = null; }
    }
}
