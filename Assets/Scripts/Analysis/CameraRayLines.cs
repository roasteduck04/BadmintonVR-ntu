using System.Collections.Generic;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>Rays from each camera centre to the joints it saw for one player (SP4): green = inlier view,
    /// red = seen but rejected. Pooled LineRenderers, rebuilt per frame.</summary>
    [DefaultExecutionOrder(120)]
    public class CameraRayLines : MonoBehaviour
    {
        public AnalysisPlayer player;
        public bool show = false;
        public bool keyJointsOnly = true;
        public int rayPlayer = 0;
        [Range(0.001f, 0.02f)] public float width = 0.004f;

        static readonly string[] KeyJoints = { "head", "left_wrist", "right_wrist", "left_ankle", "right_ankle" };
        static readonly Color Inlier = new Color(0.2f, 1f, 0.3f, 0.8f), Rejected = new Color(1f, 0.2f, 0.2f, 0.8f);

        Transform _root;
        readonly List<LineRenderer> _pool = new List<LineRenderer>();
        Material _mat;
        int[] _map;
        CameraRigGizmos _rig;

        void Awake() { if (player == null) player = GetComponent<AnalysisPlayer>(); }
        void OnEnable() { if (player != null) { player.ClipLoaded += Rebuild; if (player.Doc != null) Rebuild(); } }
        void OnDisable() { if (player != null) player.ClipLoaded -= Rebuild; }
        void OnDestroy() { if (_root != null) Destroy(_root.gameObject); }

        void Rebuild()
        {
            if (_root == null) _root = new GameObject("cv_rays").transform;
            if (_mat == null) _mat = CvMaterials.Make(Color.white, false, vertexColors: true);
            _rig = GetComponent<CameraRigGizmos>();
            var cv = player.CvDebug;
            _map = cv != null ? CvMath.JointMap(player.Doc.skeleton.joint_names, cv.joint_names) : null;
        }

        LineRenderer Line(int i)
        {
            while (_pool.Count <= i)
            {
                var go = new GameObject($"ray_{_pool.Count}");
                go.transform.SetParent(_root, false);
                var lr = go.AddComponent<LineRenderer>();
                lr.useWorldSpace = true; lr.positionCount = 2; lr.sharedMaterial = _mat;
                lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off; lr.receiveShadows = false;
                _pool.Add(lr);
            }
            return _pool[i];
        }

        void LateUpdate()
        {
            int used = 0;
            var cv = player != null ? player.CvDebug : null;
            if (show && cv != null && _map != null && _rig != null && player.Doc != null)
            {
                int pi = CvMath.RayPlayer(player.focusPlayer, rayPlayer, player.PlayerCount);
                var q = pi >= 0 ? cv.PlayerById(player.Doc.players[pi].id) : null;
                int f = player.CurrentFrame;
                var names = player.Doc.skeleton.joint_names;
                if (q != null)
                    for (int j = 0; j < _map.Length; j++)
                    {
                        if (_map[j] < 0 || (keyJointsOnly && System.Array.IndexOf(KeyJoints, names[j]) < 0)) continue;
                        if (!player.TryJointWorld(pi, j, out var w)) continue;
                        int k = CvMath.Idx(f, _map[j], cv.J);
                        for (int v = 0; v < cv.V; v++)
                        {
                            var cam = cv.cameras[v];
                            if (!cam.Used || !cam.HasPose || !CvMath.HasBit(q.seen_mask[k], v)) continue;
                            var lr = Line(used++);
                            Color c = CvMath.HasBit(q.view_mask[k], v) ? Inlier : Rejected;
                            lr.startColor = lr.endColor = c;
                            lr.widthMultiplier = width;
                            lr.SetPosition(0, _rig.CameraWorld(v));
                            lr.SetPosition(1, w);
                            lr.gameObject.SetActive(true);
                        }
                    }
            }
            for (int i = used; i < _pool.Count; i++) if (_pool[i].gameObject.activeSelf) _pool[i].gameObject.SetActive(false);
        }
    }
}
