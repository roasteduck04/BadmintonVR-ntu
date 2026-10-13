using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>Joint spheres over any skin, coloured by what the CV measured (SP4). Confidence works for every clip
    /// (doc confidences); the other modes need the cvdebug sidecar. Replaces AnalysisDebugHUD.TintJoints. Optional
    /// AUX-14 spheres (heels, toes, knuckles) for pose24 docs.</summary>
    [DefaultExecutionOrder(100)]      // after AnalysisPlayer.Update has posed the frame
    public class CvDebugOverlay : MonoBehaviour
    {
        public AnalysisPlayer player;
        public CvColorMode mode = CvColorMode.Off;
        [Tooltip("Camera index for SelectedView mode (index in the sidecar's cameras[])")]
        public int selectedView = 0;
        [Tooltip("Draw on top of the skin (else depth-tested, like the old tint)")]
        public bool onTop = false;
        [Range(0.01f, 0.06f)] public float radius = 0.026f;
        [Tooltip("Also draw the 14 AUX keypoints (pose24 docs only)")]
        public bool showAux = false;

        const int AuxCount = 14;
        static readonly string[] States = { "missing", "measured", "gap-filled", "prior-only" };
        static readonly int ColorId = Shader.PropertyToID("_Color");
        static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");

        Transform _root;
        Renderer[,] _spheres;           // [player, doc joint]
        Renderer[,] _aux;               // [player, aux keypoint]
        int[] _map;                     // doc joint -> sidecar joint
        CvPlayer[] _cv;                 // per doc player, null when the sidecar has none
        MaterialPropertyBlock _mpb;
        Material _depthMat, _topMat;

        public bool HasSidecar => player != null && player.CvDebug != null;

        void Awake() { if (player == null) player = GetComponent<AnalysisPlayer>(); _mpb = new MaterialPropertyBlock(); }
        void OnEnable() { if (player != null) { player.ClipLoaded += Rebuild; if (player.Doc != null) Rebuild(); } }
        void OnDisable() { if (player != null) player.ClipLoaded -= Rebuild; }
        void OnDestroy() { if (_root != null) Destroy(_root.gameObject); }

        Renderer MakeSphere(string name, float scale)
        {
            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Destroy(s.GetComponent<Collider>());
            s.name = name;
            s.transform.SetParent(_root, false);
            s.transform.localScale = Vector3.one * scale;
            var r = s.GetComponent<Renderer>();
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;
            s.SetActive(false);
            return r;
        }

        void Rebuild()
        {
            if (_root != null) Destroy(_root.gameObject);
            var doc = player.Doc;
            if (doc == null) return;
            _root = new GameObject("cv_joints").transform;          // scene root: AnalysisPlayer.Build destroys its own children
            if (_depthMat == null) _depthMat = CvMaterials.Make(Color.white, false);   // no ??= on Unity objects (fake null)
            if (_topMat == null) _topMat = CvMaterials.Make(Color.white, true);
            int np = doc.players.Length, nj = doc.JointCount;
            _spheres = new Renderer[np, nj];
            _aux = new Renderer[np, AuxCount];
            for (int pi = 0; pi < np; pi++)
            {
                for (int j = 0; j < nj; j++) _spheres[pi, j] = MakeSphere($"cv_{doc.players[pi].id}_{j}", radius * 2f);
                for (int k = 0; k < AuxCount; k++) _aux[pi, k] = MakeSphere($"cv_aux_{doc.players[pi].id}_{k}", radius * 1.2f);
            }
            var cv = player.CvDebug;
            _map = cv != null ? CvMath.JointMap(doc.skeleton.joint_names, cv.joint_names) : null;
            _cv = new CvPlayer[np];
            for (int pi = 0; pi < np; pi++) _cv[pi] = cv?.PlayerById(doc.players[pi].id);
            if (!HasSidecar && mode > CvColorMode.Confidence) mode = CvColorMode.Confidence;
        }

        void Paint(Renderer r, Vector3 pos, float scale, Material mat, Color c)
        {
            r.gameObject.SetActive(true);
            r.transform.position = pos;
            r.transform.localScale = Vector3.one * scale;
            r.sharedMaterial = mat;
            _mpb.Clear(); _mpb.SetColor(ColorId, c); _mpb.SetColor(BaseColorId, c);
            r.SetPropertyBlock(_mpb);
        }

        void LateUpdate()
        {
            if (_spheres == null || player == null || player.Doc == null) return;
            var doc = player.Doc;
            var cv = player.CvDebug;
            int f = player.CurrentFrame, np = _spheres.GetLength(0), nj = _spheres.GetLength(1);
            var mat = onTop ? _topMat : _depthMat;
            for (int pi = 0; pi < np; pi++)
            {
                bool shown = mode != CvColorMode.Off && player.showPose && (player.focusPlayer < 0 || player.focusPlayer == pi);
                var p = doc.players[pi];
                bool cvOk = mode == CvColorMode.Confidence || (_cv[pi] != null && _map != null);
                for (int j = 0; j < nj; j++)
                {
                    var r = _spheres[pi, j];
                    Vector3 w = default;
                    bool ok = shown && cvOk && player.TryJointWorld(pi, j, out w) && p.JointConf(f, j) > 1e-6f
                              && (mode == CvColorMode.Confidence || _map[j] >= 0);
                    if (!ok) { if (r.gameObject.activeSelf) r.gameObject.SetActive(false); continue; }
                    Color c;
                    if (mode == CvColorMode.Confidence) c = CvColors.ForJoint(mode, p.JointConf(f, j), 0, 0, -1, -1, 0, 0, -1);
                    else
                    {
                        var q = _cv[pi];
                        int k = CvMath.Idx(f, _map[j], cv.J);
                        c = CvColors.ForJoint(mode, p.JointConf(f, j), q.seen_mask[k], q.view_mask[k], q.err_dpx[k], q.sigma_mm[k],
                                              q.filled[k], q.frame_state[f], selectedView);
                    }
                    Paint(r, w, radius * 2f, mat, c);
                }
                for (int k = 0; k < AuxCount; k++)
                {
                    var r = _aux[pi, k];
                    if (shown && showAux && player.TryAuxWorld(pi, k, out var aw, out var ac))
                        Paint(r, aw, radius * 1.2f, mat, new Color(1f, 1f, 1f, Mathf.Lerp(0.3f, 1f, Mathf.Clamp01(ac))));
                    else if (r.gameObject.activeSelf) r.gameObject.SetActive(false);
                }
            }
        }

        /// <summary>HUD line for player pi at the current frame ("" without a sidecar).</summary>
        public string Describe(int pi)
        {
            var cv = player != null ? player.CvDebug : null;
            if (cv == null || _cv == null || pi < 0 || pi >= _cv.Length || _cv[pi] == null) return "";
            return Summarise(cv, _cv[pi], player.CurrentFrame);
        }

        /// <summary>"P1: measured · 4 views (back,…) · err 3.2 px · σ̄ 24 mm" for frame f ("" when out of range).</summary>
        public static string Summarise(CvDebugDoc cv, CvPlayer q, int f)
        {
            if (cv == null || q == null || q.frame_state == null || f < 0 || f >= q.frame_state.Length) return "";
            int J = cv.J, state = q.frame_state[f], views = 0, ns = 0;
            float sig = 0f;
            for (int j = 0; j < J; j++)
            {
                int k = CvMath.Idx(f, j, J);
                views |= q.view_mask[k];
                if (q.sigma_mm[k] >= 0) { sig += q.sigma_mm[k]; ns++; }
            }
            var used = new System.Text.StringBuilder();
            for (int v = 0; v < cv.V; v++)
                if (CvMath.HasBit(views, v)) used.Append(used.Length > 0 ? "," : "").Append(ShortCam(cv.cameras[v].name));
            float err = q.frame_err_px[f] >= 0 ? q.frame_err_px[f] / 10f : -1f;
            // numbers first, view list last: the HUD box cuts long lines (SP1 F13)
            return $"{q.id}: {(state >= 0 && state < States.Length ? States[state] : "?")}" +
                   $" · err {(err >= 0 ? err.ToString("F1") + " px" : "?")} · σ̄ {(ns > 0 ? (sig / ns).ToString("F0") + " mm" : "?")}" +
                   $" · {CvMath.PopCount(views)} views ({used})";
        }

        /// <summary>Camera name without the clip's numeric suffix ("back_051" -> "back").</summary>
        public static string ShortCam(string name)
        {
            int k = name != null ? name.LastIndexOf('_') : -1;
            if (k <= 0 || k == name.Length - 1) return name;
            for (int i = k + 1; i < name.Length; i++) if (!char.IsDigit(name[i])) return name;
            return name.Substring(0, k);
        }
    }
}
