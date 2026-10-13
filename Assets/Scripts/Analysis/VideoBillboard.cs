using System.IO;
using UnityEngine;
using UnityEngine.Video;

namespace BadmintonVR.Analysis
{
    /// <summary>Original camera video (SP2 H.264 proxy, data/video_proxy/&lt;base clip&gt;/&lt;view&gt;.mp4) on the frustum image
    /// plane, or filling the view in look-through (a camera-facing quad drawn after the opaque scene, before the cv overlays,
    /// so joints / rays / ghosts sit on the video like the SP2 overlay). Frame rule = python video_io.view_frame (SP2 F3).
    /// Playing or paused, the decoder is seek-driven: exact frames, updated as fast as seeks land.
    /// Max two decoders (Iris Xe).</summary>
    [DefaultExecutionOrder(130)]
    public class VideoBillboard : MonoBehaviour
    {
        public AnalysisPlayer player;
        public bool show = false;
        public int planeView = 0;
        [Range(0f, 1f)] public float planeAlpha = 0.85f;
        [Tooltip("Look-through video opacity: < 1 lets the Unity court show through as an alignment check")]
        [Range(0f, 1f)] public float screenAlpha = 0.85f;

        const int VideoQueue = 3500;            // after opaque + transparent scene, before the Overlay-queue cv drawing (4000)
        const float SeekTimeout = 1.5f;         // s: give up waiting for a seek and retry
        const float ErrorRetryDelay = 3f;       // s: after a decoder error, reopen the same view this much later

        class Slot
        {
            public string name;
            public bool inRange;
            public VideoPlayer vp; public int view = -1, target = -1, landed = -1; public bool ready, seeking, pausePending, failed;
            public float seekStarted, errorAt;
            public RenderTexture rt; public GameObject quad; public Material mat;
        }

        Transform _root;
        Slot _plane, _screen;
        CameraRigGizmos _rig;
        readonly Vector3[] _c = new Vector3[4];

        public string Status { get; private set; } = "";

        void Awake() { if (player == null) player = GetComponent<AnalysisPlayer>(); }
        void OnEnable() { if (player != null) { player.ClipLoaded += Rebuild; if (player.Doc != null) Rebuild(); } }
        void OnDisable() { if (player != null) player.ClipLoaded -= Rebuild; Stop(_plane); Stop(_screen); }
        void OnDestroy() { Stop(_plane); Stop(_screen); if (_root != null) Destroy(_root.gameObject); }

        public static string ProjectRoot => Directory.GetParent(Application.dataPath).FullName;

        public static string ProxyPath(CvDebugDoc cv, int v)
            => Path.Combine(ProjectRoot, "data", "video_proxy", cv.ProxyClip, cv.cameras[v].name + ".mp4");

        /// <summary>UI status: the missing-proxy hint wins, then "loading", then the shown video frame.</summary>
        public static string StatusText(bool hasProxy, bool ready, string camera, string proxyClip, int videoFrame)
        {
            if (!hasProxy) return $"no proxy for {camera}: run overlay_video proxies --clip {proxyClip}";
            if (!ready) return $"{camera} video loading…";
            return $"{camera} video frame {videoFrame}";
        }

        public bool HasProxy(int v)
        {
            var cv = player != null ? player.CvDebug : null;
            return cv != null && v >= 0 && v < cv.V && File.Exists(ProxyPath(cv, v));
        }

        /// <summary>The frame the decoder of the look-through (screen) or plane slot last showed; -1 when none.</summary>
        public long ShownFrame(bool screen)
        {
            var s = screen ? _screen : _plane;
            return s != null && s.vp != null && s.ready ? s.vp.frame : -1;
        }

        void Rebuild()
        {
            Stop(_plane); Stop(_screen);
            if (_root == null) _root = new GameObject("cv_video").transform;
            _rig = GetComponent<CameraRigGizmos>();
            if (_plane == null) _plane = NewSlot("plane");
            if (_screen == null) _screen = NewSlot("screen");
        }

        Slot NewSlot(string name)
        {
            var go = new GameObject($"cv_video_player_{name}");
            go.transform.SetParent(_root, false);
            var vp = go.AddComponent<VideoPlayer>();
            vp.playOnAwake = false; vp.isLooping = false; vp.skipOnDrop = true; vp.waitForFirstFrame = true;
            vp.audioOutputMode = VideoAudioOutputMode.None;
            vp.source = VideoSource.Url;
            vp.renderMode = VideoRenderMode.RenderTexture;
            vp.sendFrameReadyEvents = true;
            var s = new Slot { name = name, vp = vp };
            vp.prepareCompleted += p => { s.ready = true; };
            // Media Foundation fails (0xc00d4a3e) on seeks issued while it decodes: every seek pauses first, and the
            // one-frame decode starts only once the seek has landed (frameReady pauses again at the target).
            vp.seekCompleted += p => { s.seeking = false; if (s.pausePending) p.Play(); };
            vp.frameReady += (p, frame) =>
            {
                // landed: the decoder reached (or overshot, catching up from a keyframe) this target; don't re-seek to it
                if (s.pausePending && frame >= s.target) { s.pausePending = false; s.landed = s.target; p.Pause(); }
            };
            vp.errorReceived += (p, msg) =>
            {
                Debug.LogWarning($"[Analysis] video {name}: {msg}");
                s.ready = false; s.failed = true; s.errorAt = Time.unscaledTime;     // Open retries after ErrorRetryDelay
            };

            var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
            Destroy(q.GetComponent<Collider>());
            q.name = $"cv_video_quad_{name}";
            q.transform.SetParent(_root, false);
            var r = q.GetComponent<Renderer>();
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off; r.receiveShadows = false;
            s.mat = CvMaterials.Make(Color.white, name == "screen");      // look-through quad: ZTest Always, covers the scene
            s.mat.renderQueue = VideoQueue;
            r.sharedMaterial = s.mat;
            q.SetActive(false);
            s.quad = q;
            return s;
        }

        void Open(Slot s, int v)
        {
            if (!CvMath.ShouldReopenVideo(s.view == v, s.failed, Time.unscaledTime - s.errorAt, ErrorRetryDelay)) return;
            Stop(s);
            s.view = v;
            if (!HasProxy(v)) return;
            var cam = player.CvDebug.cameras[v];
            int w = cam.width >= cam.height ? 960 : 540, h = cam.width >= cam.height ? 540 : 960;   // proxy size (SP2 F5)
            s.rt = new RenderTexture(w, h, 0) { name = $"cv_rt_{cam.name}" };
            s.vp.targetTexture = s.rt;
            s.mat.mainTexture = s.rt;
            s.vp.url = ProxyPath(player.CvDebug, v);
            s.vp.Prepare();
        }

        void Stop(Slot s)
        {
            if (s == null) return;
            if (s.vp != null) s.vp.Stop();
            s.ready = false; s.seeking = false; s.pausePending = false; s.failed = false; s.inRange = false;
            s.view = -1; s.target = -1; s.landed = -1;
            if (s.quad != null) s.quad.SetActive(false);
            if (s.rt != null) { s.rt.Release(); Destroy(s.rt); s.rt = null; }
        }

        void Drive(Slot s)
        {
            if (s.view < 0 || !s.ready) return;
            var cam = player.CvDebug.cameras[s.view];
            int target = CvMath.VideoFrame(player.CurrentFrame, cam.offset, cam.det_shift);
            long n = (long)s.vp.frameCount;
            s.inRange = CvMath.VideoFrameInRange(target, n);
            if (!s.inRange) { if (s.vp.isPlaying) s.vp.Pause(); return; }      // Place hides the quad (no stale frame)
            if ((s.seeking || s.pausePending) && Time.unscaledTime - s.seekStarted > SeekTimeout) { s.seeking = false; s.pausePending = false; }
            // Seek-driven, playing or paused: the decoder never free-runs. Each step seeks (paused) to the exact target,
            // decodes that one frame and pauses; the next step starts when it has landed. Free-running playback with
            // drift-correcting seeks tripped Media Foundation 0xc00d4a3e and the editor's Error Pause (SP4 finding).
            if (!s.seeking && !s.pausePending && CvMath.NeedsVideoSeek(s.vp.frame, target, s.landed))
                Seek(s, target);
        }

        void Seek(Slot s, int target)
        {
            if (s.vp.isPlaying) s.vp.Pause();
            // every flag before the seek is issued: seekCompleted may fire as soon as vp.frame is set
            s.seeking = true; s.pausePending = true; s.seekStarted = Time.unscaledTime; s.target = target;
            s.vp.frame = target;
        }

        void Place(Slot s, Vector3 pos, Quaternion rot, float vfov, float aspect, float depth, float alpha)
        {
            CvMath.FrustumCorners(pos, rot, vfov, aspect, depth, _c);
            s.quad.transform.position = (_c[0] + _c[2]) * 0.5f;
            s.quad.transform.rotation = rot;
            s.quad.transform.localScale = new Vector3((_c[1] - _c[0]).magnitude, (_c[0] - _c[3]).magnitude, 1f);
            s.mat.color = new Color(1f, 1f, 1f, alpha);
            s.quad.SetActive(s.ready && s.inRange && s.vp.frame >= 0);
        }

        void LateUpdate()
        {
            var cv = player != null ? player.CvDebug : null;
            if (cv == null || _plane == null) { Status = ""; return; }
            int look = _rig != null ? _rig.LookingThrough : -1;
            bool planeOn = show && look < 0 && planeView >= 0 && planeView < cv.V && cv.cameras[planeView].HasPose;
            bool screenOn = show && look >= 0;

            if (planeOn) Open(_plane, planeView); else Stop(_plane);
            if (screenOn) Open(_screen, look); else Stop(_screen);
            Drive(_plane); Drive(_screen);

            if (planeOn && _rig != null)
            {
                var cam = cv.cameras[planeView];
                Place(_plane, _rig.CameraWorld(planeView), _rig.CameraRotWorld(planeView), cam.vfov_deg, cam.Aspect, _rig.depth, planeAlpha);
            }
            if (screenOn && _rig != null)
            {
                var cam = cv.cameras[look];
                var main = Camera.main;
                float d = main != null ? Mathf.Max(main.nearClipPlane * 4f, 0.2f) : 0.3f;
                Place(_screen, _rig.CameraWorld(look), _rig.CameraRotWorld(look), cam.vfov_deg, cam.Aspect, d, screenAlpha);
            }

            int sv = screenOn ? look : planeOn ? planeView : -1;
            if (sv >= 0)
            {
                var c = cv.cameras[sv];
                var slot = screenOn ? _screen : _plane;
                Status = StatusText(HasProxy(sv), slot.ready, c.name, cv.ProxyClip,
                                    CvMath.VideoFrame(player.CurrentFrame, c.offset, c.det_shift));
            }
            else Status = "";
        }
    }
}
