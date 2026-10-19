using System.Collections.Generic;
using System.Text;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace BadmintonVR.Analysis
{
    public class AnalysisDebugHUD : MonoBehaviour
    {
        public AnalysisPlayer player;
        public bool master = true;
        public bool showStats = true, showProvenance = true, showStatus = true;
        public bool showWarnings = true, colorByConfidence = true, showAxes = true, showLog = false;
        [Tooltip("Draw the stats and toggle boxes in the screen corners (IMGUI). Off while the UI Toolkit viewer is up: it shows the same text and toggles in its Debug section")]
        public bool drawPanels = true;

        [Header("Stroke Bubble")]
        public float bubbleDuration = 1.0f;
        public float bubbleYOffset = 2.0f;

        float _fps, _frameMs;
        readonly List<string> _log = new List<string>();
        List<string> _warnings;
        AnalysisDoc _builtFor;
        Transform _axes;

        struct StrokeBubble
        {
            public string label;
            public float startTime;
            public int playerIndex;
        }
        readonly List<StrokeBubble> _bubbles = new List<StrokeBubble>();
        int _lastFrame = -1;
        Texture2D _whiteTex;
        GUIStyle _statsStyle, _logStyle, _bubbleStyle;

        void Reset() => player = GetComponent<AnalysisPlayer>();
        void Awake()
        {
            if (player == null) player = GetComponent<AnalysisPlayer>();
            _whiteTex = new Texture2D(1, 1);
            _whiteTex.SetPixel(0, 0, Color.white);
            _whiteTex.Apply();
        }
        void OnEnable() => Application.logMessageReceived += OnLog;
        void OnDisable() { Application.logMessageReceived -= OnLog; UiPointerGuard.Clear(this); }
        void OnDestroy()
        {
            if (_axes != null) Destroy(_axes.gameObject);
            foreach (var m in _axisMats) RendererUtil.DestroySafe(m);
            RendererUtil.DestroySafe(_whiteTex);
        }

        /// <summary>The last log messages (newest last), for the viewer's Debug section.</summary>
        public IReadOnlyList<string> RecentLog => _log;

        void OnLog(string msg, string stack, LogType t)
        {
            _log.Add($"[{t}] {msg}");
            if (_log.Count > 8) _log.RemoveAt(0);
        }

        void Update()
        {
            if (TogglePressed()) master = !master;
            _frameMs = Mathf.Lerp(_frameMs, Time.unscaledDeltaTime * 1000f, 0.1f);
            _fps = _frameMs > 0.001f ? 1000f / _frameMs : 0f;

            if (player == null || player.Doc == null) return;
            if (_builtFor != player.Doc) RebuildForClip(player.Doc);
            if (_axes != null) _axes.gameObject.SetActive(master && showAxes);
            SyncConfidenceColour();

            int curFrame = player.CurrentFrame;
            if (curFrame != _lastFrame)
            {
                CheckForNewStrokes(curFrame);
                _lastFrame = curFrame;
            }

            _bubbles.RemoveAll(b => Time.time - b.startTime > bubbleDuration);
        }

        void CheckForNewStrokes(int frame)
        {
            var doc = player.Doc;
            if (doc == null) return;
            for (int pi = 0; pi < doc.players.Length; pi++)
            {
                var st = doc.players[pi].StrokeAt(frame);
                if (st != null && !string.IsNullOrEmpty(st.label))
                {
                    _bubbles.Add(new StrokeBubble
                    {
                        label = FormatLabel(st.label),
                        startTime = Time.time,
                        playerIndex = pi
                    });
                }
            }
        }

        static string FormatLabel(string raw)
        {
            return raw.Replace('_', ' ').ToUpperInvariant();
        }

        bool _lastWant, _lastAllSkeleton, _confSynced, _hudOwnsConfidence;

        /// <summary>The "confidence colour" toggle drives CvDebugOverlay's Confidence mode — like the old TintJoints, only while
        /// every skin is the skeleton (old clips look as before). Applied on a change only, so a mode chosen with J / the cv row
        /// isn't overridden every frame.</summary>
        void SyncConfidenceColour()
        {
            var ov = GetComponent<CvDebugOverlay>();
            if (ov == null) return;
            bool want = master && colorByConfidence, allSkeleton = AllSkinsAreSkeleton();
            if (_confSynced && want == _lastWant && allSkeleton == _lastAllSkeleton) return;
            _confSynced = true; _lastWant = want; _lastAllSkeleton = allSkeleton;
            ov.mode = ConfidenceTarget(ov.mode, want, allSkeleton, _hudOwnsConfidence, out _hudOwnsConfidence);
        }

        /// <summary>Overlay mode after the HUD's confidence toggle / skins changed. Other modes and a Confidence mode the user
        /// picked (not set by the HUD) are left alone.</summary>
        public static CvColorMode ConfidenceTarget(CvColorMode current, bool want, bool allSkeleton, bool hudOwned, out bool owned)
        {
            owned = false;
            if (current != CvColorMode.Off && current != CvColorMode.Confidence) return current;
            if (current == CvColorMode.Confidence && !hudOwned) return current;
            var desired = want && allSkeleton ? CvColorMode.Confidence : CvColorMode.Off;
            owned = desired == CvColorMode.Confidence;
            return desired;
        }

        bool AllSkinsAreSkeleton()
        {
            if (player == null || player.Doc == null) return false;
            for (int i = 0; i < player.Doc.players.Length; i++)
                if (player.GetSkin(i) != AnalysisPlayer.SkinMode.Skeleton) return false;
            return true;
        }

        void RebuildForClip(AnalysisDoc doc)
        {
            _builtFor = doc;
            _warnings = ValidateContract(doc);
            _bubbles.Clear();
            _lastFrame = -1;
            if (_axes == null) BuildAxes();      // world axes don't depend on the clip: built once (was 3 materials per load)
        }

        static readonly string[] KnownSchemas = { "3.0", "4.0", "4.1", "5.0" };

        static List<string> ValidateContract(AnalysisDoc doc)
        {
            var w = new List<string>();
            if (System.Array.IndexOf(KnownSchemas, doc.schema_version) < 0)
                w.Add($"schema_version={doc.schema_version} (expected 3.0/4.0/4.1/5.0)");
            if (doc.meta == null || doc.meta.fps <= 0f) w.Add("meta.fps missing/<=0");
            if (doc.skeleton == null || doc.skeleton.parents == null || doc.skeleton.parents.Length < 1)
                w.Add($"skeleton.parents missing or empty");
            if (doc.players == null || doc.players.Length == 0) { w.Add("no players"); return w; }
            for (int pi = 0; pi < doc.players.Length; pi++)
            {
                var p = doc.players[pi];
                if (p.frames == null || p.frames.Length == 0) { w.Add($"player {pi}: no frames"); continue; }
                if (doc.meta != null && p.frames.Length != doc.meta.frame_count)
                    w.Add($"player {pi}: {p.frames.Length} frames != meta.frame_count {doc.meta.frame_count}");
                for (int f = 0; f < p.frames.Length; f++)
                {
                    var jf = p.frames[f].joints_flat;
                    if (jf != null && jf.Length % AnalysisDoc.Stride != 0) { w.Add($"player {pi} frame {f}: joints_flat={jf.Length} not divisible by {AnalysisDoc.Stride}"); break; }
                }
            }
            return w;
        }

        readonly List<Material> _axisMats = new List<Material>();

        void BuildAxes()
        {
            if (_axes != null) Destroy(_axes.gameObject);
            foreach (var m in _axisMats) RendererUtil.DestroySafe(m);
            _axisMats.Clear();
            _axes = new GameObject("AnalysisDebugAxes").transform;
            AddAxis(Vector3.right * 3f, Color.red);
            AddAxis(Vector3.up * 3f, Color.green);
            AddAxis(Vector3.forward * 6.7f, Color.blue);
        }

        static Material MakeUnlitMat(Color col)
        {
            var shader = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Unlit/Color");
            var m = new Material(shader);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", col); else m.color = col;
            return m;
        }

        void AddAxis(Vector3 end, Color col)
        {
            var go = new GameObject("axis");
            go.transform.SetParent(_axes, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = true; lr.widthMultiplier = 0.03f; lr.positionCount = 2;
            lr.SetPosition(0, new Vector3(0, 0.02f, 0)); lr.SetPosition(1, end + Vector3.up * 0.02f);
            var mat = MakeUnlitMat(col);
            _axisMats.Add(mat);
            lr.sharedMaterial = mat;
        }

        void OnGUI()
        {
            if (!master) { UiPointerGuard.Clear(this); return; }
            var doc = player != null ? player.Doc : null;
            if (doc == null) { UiPointerGuard.Clear(this); return; }

            DrawStrokeBubbles(doc);

            if (!drawPanels || UiPointerGuard.ViewerOwnsScreen) { UiPointerGuard.Clear(this); return; }

            var tp = new Rect(Screen.width - 190, 8, 182, 176);
            UiPointerGuard.Set(this, tp);        // the toggle box (the stats box is read-only)
            GUI.Box(tp, "debug [H]");
            GUILayout.BeginArea(new Rect(tp.x + 8, tp.y + 22, tp.width - 16, tp.height - 28));
            showStats      = GUILayout.Toggle(showStats, " stats / fps");
            showProvenance = GUILayout.Toggle(showProvenance, " provenance");
            showStatus     = GUILayout.Toggle(showStatus, " shuttle/stroke");
            showWarnings   = GUILayout.Toggle(showWarnings, " contract warnings");
            colorByConfidence = GUILayout.Toggle(colorByConfidence, " confidence colour (skeleton skin)");
            showAxes       = GUILayout.Toggle(showAxes, " world axes");
            showLog        = GUILayout.Toggle(showLog, " log console");
            GUILayout.EndArea();

            string stats = StatsText();

            if (_statsStyle == null)
            {
                _statsStyle = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.UpperLeft, fontSize = 18, richText = false };
                _statsStyle.normal.textColor = Color.white;
            }
            GUI.Box(new Rect(8, 8, 540, 260), stats, _statsStyle);

            if (showLog && _log.Count > 0)
            {
                if (_logStyle == null)
                {
                    _logStyle = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.LowerLeft, fontSize = 14 };
                    _logStyle.normal.textColor = new Color(0.8f, 0.9f, 1f);
                }
                GUI.Box(new Rect(8, 276, 560, 140), string.Join("\n", _log), _logStyle);
            }
        }

        /// <summary>The stats text (one line per item) for the toggles that are on: fps, clip, skins, cv state, producers,
        /// shuttle / stroke, contract warnings.</summary>
        public string StatsText()
        {
            var doc = player != null ? player.Doc : null;
            if (doc == null) return "";
            var sb = new StringBuilder();
            int f = player.CurrentFrame;
            if (showStats)
            {
                sb.AppendLine($"fps {_fps,5:F0}   {_frameMs:F1} ms/frame")
                  .AppendLine($"clip {doc.clip_id}   frame {f + 1}/{doc.FrameCount}   t={f / Mathf.Max(doc.Fps,1f):F2}s");
                sb.AppendLine($"skeleton {doc.ShortLabel(46)}");      // fits the HUD box (checked in play mode)
                var skinSb = new StringBuilder("skin ");
                for (int pi = 0; pi < doc.players.Length; pi++)
                    skinSb.Append($" {doc.players[pi].id}={player.GetSkin(pi)}");
                sb.AppendLine(skinSb.ToString());
                var ov = GetComponent<CvDebugOverlay>();
                if (ov != null && player.CvDebug != null)
                    for (int pi = 0; pi < doc.players.Length; pi++)
                    {
                        string s = ov.Describe(pi);
                        if (s.Length > 0) sb.AppendLine(s);
                    }
                if (ov != null && ov.mode != CvColorMode.Off) sb.AppendLine(CvColors.Legend(ov.mode));
            }
            if (showProvenance && doc.meta != null && doc.meta.producers != null)
            {
                var pr = doc.meta.producers;
                sb.AppendLine("producers: " +
                    $"pose={N(pr.pose)} pos={N(pr.position)} racket={N(pr.racket)} " +
                    $"grip={N(pr.grip)} stroke={N(pr.stroke)} shuttle={N(pr.shuttle)}");
            }
            if (showStatus)
            {
                sb.AppendLine(player.DescribeShuttle(f));
                for (int pi = 0; pi < doc.players.Length; pi++)
                {
                    var st = doc.players[pi].StrokeAt(f);
                    string lbl = (st != null && !string.IsNullOrEmpty(st.label)) ? $"{st.label} ({st.confidence:F2})" : "-";
                    sb.AppendLine($"{doc.players[pi].id} ({doc.players[pi].side}): {lbl}");
                }
            }
            if (showWarnings)
                sb.AppendLine(_warnings == null || _warnings.Count == 0
                    ? "contract: OK" : "contract WARN: " + string.Join("; ", _warnings));

            return sb.ToString().TrimEnd();
        }

        void DrawStrokeBubbles(AnalysisDoc doc)
        {
            var cam = Camera.main;
            if (cam == null) return;

            if (_bubbleStyle == null)
            {
                _bubbleStyle = new GUIStyle(GUI.skin.box)
                {
                    alignment = TextAnchor.MiddleCenter,
                    fontSize = 22,
                    fontStyle = FontStyle.Bold,
                    richText = false
                };
            }

            foreach (var b in _bubbles)
            {
                float age = Time.time - b.startTime;
                float alpha = 1f - Mathf.Clamp01(age / bubbleDuration);

                Transform root = player.PlayerRoot(b.playerIndex);
                if (root == null) continue;

                Vector3 worldPos = root.position + Vector3.up * bubbleYOffset;
                Vector3 screen = cam.WorldToScreenPoint(worldPos);
                if (screen.z < 0f || UiPointerGuard.UnderPanel(screen)) continue;

                float screenY = Screen.height - screen.y;

                Color bgCol = new Color(0f, 0f, 0f, 0.7f * alpha);
                Color txtCol = new Color(1f, 0.95f, 0.4f, alpha);

                var content = new GUIContent(b.label);
                Vector2 size = _bubbleStyle.CalcSize(content);
                size.x += 24f;
                size.y += 8f;

                Rect rect = new Rect(screen.x - size.x * 0.5f, screenY - size.y - 8f, size.x, size.y);

                GUI.color = bgCol;
                GUI.DrawTexture(rect, _whiteTex);
                GUI.color = Color.white;

                _bubbleStyle.normal.textColor = txtCol;
                GUI.Label(rect, b.label, _bubbleStyle);
            }
        }

        static string N(string s) => string.IsNullOrEmpty(s) ? "null" : s;

        bool TogglePressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current; return kb != null && !UiPointerGuard.Typing && kb.hKey.wasPressedThisFrame;
#else
            return false;
#endif
        }
    }
}
