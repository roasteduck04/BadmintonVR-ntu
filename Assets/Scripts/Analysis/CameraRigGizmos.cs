using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>Calibrated cameras as frusta (SP4): cyan = used, red = excluded by view health, magenta = held out of this
    /// build. Look-through snaps the main camera to a camera's pose + vertical FOV, letterboxed to its aspect, orbit off,
    /// twins not lifted (so they line up with that camera's video). Esc / ExitLookThrough returns.</summary>
    [DefaultExecutionOrder(110)]
    public class CameraRigGizmos : MonoBehaviour
    {
        public AnalysisPlayer player;
        public bool show = false;
        [Range(0.2f, 3f)] public float depth = 0.6f;
        public bool showLabels = true;

        static readonly Color Used = new Color(0.2f, 0.9f, 1f), Excluded = Color.red, HeldOut = new Color(1f, 0.2f, 0.9f);

        CvLineBatch _lines;
        Transform _root;
        readonly Vector3[] _c = new Vector3[4];
        int _look = -1;
        Vector3 _savedPos; Quaternion _savedRot; float _savedFov; Rect _savedRect; bool _savedLift, _orbitWasEnabled;
        GUIStyle _label;

        public int LookingThrough => _look;

        void Awake() { if (player == null) player = GetComponent<AnalysisPlayer>(); }
        void OnEnable() { if (player != null) { player.ClipLoaded += Rebuild; if (player.Doc != null) Rebuild(); } }
        void OnDisable() { if (player != null) player.ClipLoaded -= Rebuild; ExitLookThrough(); }
        void OnDestroy() { _lines?.Dispose(); if (_root != null) Destroy(_root.gameObject); }

        void Rebuild()
        {
            ExitLookThrough();
            if (_root == null) _root = new GameObject("cv_cameras").transform;
            if (_lines == null) _lines = new CvLineBatch("cv_frusta", _root, false);
        }

        public Vector3 CameraWorld(int v) => player.transform.TransformPoint(player.CvDebug.cameras[v].UnityPos);
        public Quaternion CameraRotWorld(int v) => player.transform.rotation * player.CvDebug.cameras[v].UnityRot;

        static Color ColourOf(CvCamera c) => c.excluded ? Excluded : c.held_out ? HeldOut : Used;

        void LateUpdate()
        {
            var cv = player != null ? player.CvDebug : null;
            if (_lines == null) return;
            _lines.SetVisible(show && cv != null && _look < 0);
            if (cv != null && show)
            {
                _lines.Clear();
                for (int v = 0; v < cv.V; v++)
                {
                    var cam = cv.cameras[v];
                    if (!cam.HasPose) continue;
                    Vector3 o = CameraWorld(v);
                    CvMath.FrustumCorners(o, CameraRotWorld(v), cam.vfov_deg, cam.Aspect, depth, _c);
                    Color col = ColourOf(cam);
                    for (int k = 0; k < 4; k++) { _lines.Add(o, _c[k], col); _lines.Add(_c[k], _c[(k + 1) % 4], col); }
                    _lines.Add(_c[0], Vector3.Lerp(_c[0], _c[1], 0.5f) + (CameraRotWorld(v) * Vector3.up) * depth * 0.15f, col);  // "up" tick
                }
                _lines.Commit();
            }
            if (_look >= 0 && cv != null)
            {
                var cam = Camera.main;
                if (cam != null) cam.rect = CvMath.LetterboxRect((float)Screen.width / Mathf.Max(Screen.height, 1), cv.cameras[_look].Aspect);
            }
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (_look >= 0 && kb != null && kb.escapeKey.wasPressedThisFrame) ExitLookThrough();
#endif
        }

        public bool LookThrough(int v)
        {
            var cv = player != null ? player.CvDebug : null;
            var cam = Camera.main;
            if (cv == null || cam == null || v < 0 || v >= cv.V || !cv.cameras[v].HasPose) return false;
            if (_look < 0)
            {
                _savedPos = cam.transform.position; _savedRot = cam.transform.rotation; _savedFov = cam.fieldOfView;
                _savedRect = cam.rect; _savedLift = player.LiftToFloor;
                var orbit = cam.GetComponent<OrbitCamera>();
                _orbitWasEnabled = orbit != null && orbit.enabled;
                if (orbit != null) orbit.enabled = false;
            }
            _look = v;
            cam.transform.SetPositionAndRotation(CameraWorld(v), CameraRotWorld(v));
            cam.fieldOfView = cv.cameras[v].vfov_deg;        // centred principal point, so no lens shift (SP2 F6)
            cam.rect = CvMath.LetterboxRect((float)Screen.width / Mathf.Max(Screen.height, 1), cv.cameras[v].Aspect);
            player.LiftToFloor = false;
            return true;
        }

        public void ExitLookThrough()
        {
            if (_look < 0) return;
            _look = -1;
            var cam = Camera.main;
            if (cam != null)
            {
                cam.transform.SetPositionAndRotation(_savedPos, _savedRot);
                cam.fieldOfView = _savedFov; cam.rect = _savedRect;
                var orbit = cam.GetComponent<OrbitCamera>();
                if (orbit != null && _orbitWasEnabled) orbit.enabled = true;
            }
            if (player != null) player.LiftToFloor = _savedLift;
        }

        void OnGUI()
        {
            var cv = player != null ? player.CvDebug : null;
            var cam = Camera.main;
            if (cv == null || cam == null) return;
            if (_label == null) { _label = new GUIStyle(GUI.skin.label) { fontSize = 13 }; }
            if (_look >= 0)
            {
                var c = cv.cameras[_look];
                GUI.Label(new Rect(12, 8, 700, 22), $"looking through {c.name}  ({c.width}×{c.height}, vfov {c.vfov_deg:F1}°)  —  Esc to return", _label);
                return;
            }
            if (!show || !showLabels) return;
            for (int v = 0; v < cv.V; v++)
            {
                var c = cv.cameras[v];
                if (!c.HasPose) continue;
                Vector3 s = cam.WorldToScreenPoint(CameraWorld(v));
                if (s.z <= 0f) continue;
                float used = UsedFrac(cv, c.name);
                string tag = c.excluded ? " EXCLUDED" : c.held_out ? " HELD OUT" : "";
                string h = c.health_px >= 0 ? $" · {c.health_px:F0} px" : "";
                _label.normal.textColor = ColourOf(c);
                GUI.Label(new Rect(s.x + 6, Screen.height - s.y - 10, 320, 20), $"{v + 1} {c.name}{h}{(used >= 0 ? $" · {used * 100:F0}%" : "")}{tag}", _label);
            }
        }

        static float UsedFrac(CvDebugDoc cv, string name)
        {
            if (cv.stats?.views == null) return -1f;
            foreach (var s in cv.stats.views) if (s.name == name) return s.used_frac;
            return -1f;
        }
    }
}
