using System.IO;
using System.Linq;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    public class AnalysisViewerUI : MonoBehaviour
    {
        public AnalysisPlayer player;
        public bool visible = true;

        string[] _clips;
        string[] _clipLabels;
        bool _minimized;

        void Reset() => player = GetComponent<AnalysisPlayer>();

        void Awake()
        {
            if (player == null) player = GetComponent<AnalysisPlayer>();
            ScanClips();
        }

        static bool Exists(string rel) => File.Exists(Path.Combine(Application.streamingAssetsPath, rel));

        void ScanClips()
        {
            string dir = Path.Combine(Application.streamingAssetsPath, "analysis");
            if (!Directory.Exists(dir)) { _clips = new string[0]; _clipLabels = new string[0]; return; }
            var files = Directory.GetFiles(dir, "*.analysis.json").Select(p => "analysis/" + Path.GetFileName(p)).ToArray();
            var clips = files.Where(p => !p.Contains("_oldracket")
                                       && !p.Contains("_smplx")
                                       && !p.Contains("_posev2")
                                       && !p.Contains("_posefit")
                                       && !p.Contains("_pose24")
                                       && !p.Contains("_bhatt")
                                       && !p.Contains("_posonly")
                                       && !p.Contains("_bonelen")
                                       && !p.Contains("_sgsmooth")
                                       && !p.Contains("_bspline")
                                       && !p.Contains("_physics_sg")
                                       && !p.Contains("_physics_bsp")
                                       && !p.Contains("_newtracker")
                                       && !p.Contains("3-1"))
                             .ToList();
            // variant-only clips (e.g. a camera-subset build with no base file): list the base name
            foreach (var f in files)
                foreach (var s in PoseSuffixes)
                {
                    string tail = s + ".analysis.json";
                    if (!f.EndsWith(tail)) continue;
                    string b = f.Substring(0, f.Length - tail.Length) + ".analysis.json";
                    if (!Exists(b) && !clips.Contains(b) && !b.Contains("3-1")) clips.Add(b);
                }
            _clips = clips.OrderBy(p => p).ToArray();
            _clipLabels = _clips.Select(p => Path.GetFileName(p).Replace(".analysis.json", "")).ToArray();
        }

        /// <summary>Load a clip with the preferred pose suffix, else its base file, else the first variant that exists.</summary>
        void LoadClipPreferring(string basePath, string suffix)
        {
            string target = PoseVariantPath(basePath, suffix);
            if (Exists(target)) { player.LoadClip(target); return; }
            if (Exists(basePath)) { player.LoadClip(basePath); return; }
            foreach (var s in PoseSuffixes)
                if (Exists(PoseVariantPath(basePath, s))) { player.LoadClip(PoseVariantPath(basePath, s)); return; }
            Debug.LogWarning($"[AnalysisViewerUI] no analysis file for {basePath} (base or any pose variant) in StreamingAssets");
        }

        /// <summary>Pose variants: whole analysis files next to the base clip, "&lt;clip&gt;&lt;suffix&gt;.analysis.json".</summary>
        static string[] PoseSuffixes => AnalysisPlayer.PoseVariantSuffixes;
        // Derived from AnalysisPlayer.PoseVariantDefs (same order as PoseSuffixes); kept as a field for tools/unity_checks.
        static readonly string[] PoseLabels = AnalysisPlayer.PoseVariantLabels;

        /// <summary>The base clip path: strips the old-racket and pose-variant suffixes.</summary>
        string BasePath(string path) => AnalysisPlayer.BaseClipPath(path);

        static string PoseVariantPath(string basePath, string suffix)
            => string.IsNullOrEmpty(suffix) ? basePath : basePath.Replace(".analysis.json", suffix + ".analysis.json");

        /// <summary>Suffix of the pose variant currently loaded ("" = base).</summary>
        string CurrentPoseSuffix()
        {
            string f = player != null ? player.analysisFile ?? "" : "";
            foreach (var s in PoseSuffixes) if (f.EndsWith(s + ".analysis.json")) return s;
            return "";
        }

        string _poseCacheFor;
        readonly System.Collections.Generic.List<int> _poseVariants = new System.Collections.Generic.List<int>();

        /// <summary>Indices into PoseSuffixes whose file exists for the current clip.</summary>
        System.Collections.Generic.List<int> PoseVariants()
        {
            string basePath = player != null ? BasePath(player.analysisFile ?? "") : "";
            if (basePath == _poseCacheFor) return _poseVariants;
            _poseCacheFor = basePath;
            _poseVariants.Clear();
            for (int i = 0; i < PoseSuffixes.Length; i++)
                if (Exists(PoseVariantPath(basePath, PoseSuffixes[i])))
                    _poseVariants.Add(i);
            return _poseVariants;
        }

        /// <summary>Switch build keeping the instant (the builds' timelines differ by up to ~3 frames on real data).</summary>
        void PickPose(string suffix)
        {
            string target = PoseVariantPath(BasePath(player.analysisFile), suffix);
            player.LoadClipKeepingInstant(target);
        }

        string OldRacketPath(string basePath)
        {
            return basePath.Replace(".analysis.json", "_oldracket.analysis.json");
        }

        string _oldRacketCacheFor;
        bool _hasOldRacket;

        /// <summary>Whether an _oldracket file exists for the current clip (checked once per loaded file, not per GUI event).</summary>
        bool HasOldRacket()
        {
            if (player == null || string.IsNullOrEmpty(player.analysisFile)) return false;
            if (player.analysisFile == _oldRacketCacheFor) return _hasOldRacket;
            _oldRacketCacheFor = player.analysisFile;
            _hasOldRacket = CurrentPoseSuffix() == ""
                            && Exists(OldRacketPath(BasePath(player.analysisFile)));
            return _hasOldRacket;
        }

        bool IsOldRacket()
        {
            return player != null && player.analysisFile != null
                && player.analysisFile.Contains("_oldracket");
        }

        void SwitchToOldRacket()
        {
            string basePath = BasePath(player.analysisFile);
            int frame = player.CurrentFrame;
            player.LoadClip(OldRacketPath(basePath));
            player.SeekFrame(frame);
        }

        void SwitchToNewRacket()
        {
            string basePath = BasePath(player.analysisFile);
            int frame = player.CurrentFrame;
            player.LoadClip(basePath);
            player.SeekFrame(frame);
        }

        string _sidecarCacheFor;
        readonly System.Collections.Generic.List<string> _sidecars = new System.Collections.Generic.List<string>();

        /// <summary>Racket sidecar variants that exist for the current clip's base file.</summary>
        System.Collections.Generic.List<string> RacketSidecars()
        {
            string basePath = player != null ? (player.analysisFile ?? "").Replace("_oldracket", "") : "";
            if (basePath == _sidecarCacheFor) return _sidecars;
            _sidecarCacheFor = basePath;
            _sidecars.Clear();
            if (string.IsNullOrEmpty(basePath)) return _sidecars;
            foreach (var v in AnalysisPlayer.RacketVariantNames)
                if (File.Exists(Path.Combine(Application.streamingAssetsPath,
                                             AnalysisPlayer.RacketSidecarPath(basePath, v))))
                    _sidecars.Add(v);
            return _sidecars;
        }

        void PickRacket(string variant)
        {
            if (IsOldRacket()) SwitchToNewRacket();
            player.SetRacketVariant(variant);
        }

        readonly System.Collections.Generic.List<string> _shuttleVariants = new System.Collections.Generic.List<string>();

        /// <summary>Shuttle sidecar variants loaded (and validated) for the current clip.</summary>
        System.Collections.Generic.List<string> ShuttleVariants()
        {
            _shuttleVariants.Clear();
            foreach (var v in AnalysisPlayer.ShuttleVariantNames)
                if (player.GetShuttleVariantDoc(v) != null) _shuttleVariants.Add(v);
            return _shuttleVariants;
        }

        string ShuttleStatsLabel(string active)
        {
            var d = player.GetShuttleVariantDoc(active);
            if (d != null && d.stats != null)
                return d.stats.reproj_rms_px >= 0f
                    ? $"{d.stats.coverage * 100f:F0}% · {d.stats.reproj_rms_px:F1}px"
                    : $"{d.stats.coverage * 100f:F0}%";
            var s = player.Doc.BaselineShuttle;
            return s != null ? $"{s.Coverage() * 100f:F0}%" : "";
        }


        // ---- SP4 cv visualisation (components live on the AnalysisPlayer object) ----
        CvDebugOverlay _ov; CameraRigGizmos _rig; CameraRayLines _rays; VideoBillboard _video; NonPlayerMarkers _np; VariantGhosts _ghosts;
        int _cvView;

        void CacheCv()
        {
            if (player == null) return;
            if (_ov == null) _ov = player.GetComponent<CvDebugOverlay>();
            if (_rig == null) _rig = player.GetComponent<CameraRigGizmos>();
            if (_rays == null) _rays = player.GetComponent<CameraRayLines>();
            if (_video == null) _video = player.GetComponent<VideoBillboard>();
            if (_np == null) _np = player.GetComponent<NonPlayerMarkers>();
            if (_ghosts == null) _ghosts = player.GetComponent<VariantGhosts>();
        }

        /// <summary>The shared selected camera: SelectedView colour mode, the video plane, "look" button.</summary>
        void SetView(int v)
        {
            var cv = player.CvDebug;
            if (cv == null || cv.V == 0) return;
            _cvView = ((v % cv.V) + cv.V) % cv.V;
            if (_ov != null) _ov.selectedView = _cvView;
            if (_video != null) _video.planeView = _cvView;
        }

        // Clip / build switches clicked in OnGUI run here, at the start of the next frame: loading mid-OnGUI changed the
        // control layout between IMGUI's Layout and input events (cached player count, "pushing more GUIClips than popping").
        System.Action _pending;

        void Defer(System.Action action) => _pending = action;

        void RunPending()
        {
            if (_pending == null) return;
            var a = _pending;
            _pending = null;
            a();
        }

        void Update()
        {
            if (player == null) return;
            RunPending();
            if (player.Doc == null) return;
            CacheCv();
            if (player.CvDebug != null && _cvView >= player.CvDebug.V) SetView(0);
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (kb == null) return;
            bool has = player.CvDebug != null;
            // J / Y instead of the spec's C / R: OrbitCamera already uses C (fly mode), R (reset), F (frame)
            if (kb.jKey.wasPressedThisFrame && _ov != null) _ov.mode = CvColors.Next(_ov.mode, has);
            if (kb.gKey.wasPressedThisFrame && _ghosts != null && _ghosts.Candidates.Count > 0)
            {
                bool any = _ghosts.Candidates.Any(_ghosts.IsOn);
                foreach (var c in _ghosts.Candidates.ToList()) _ghosts.Set(c, false);
                if (!any) _ghosts.Set(_ghosts.Candidates[0], true);
            }
            if (!has) return;
            if (kb.yKey.wasPressedThisFrame && _rays != null) _rays.show = !_rays.show;
            if (kb.kKey.wasPressedThisFrame && _rig != null) _rig.show = !_rig.show;
            if (kb.vKey.wasPressedThisFrame && _video != null) _video.show = !_video.show;
            if (kb.nKey.wasPressedThisFrame && _np != null) _np.show = !_np.show;
            var digits = new[] { kb.digit1Key, kb.digit2Key, kb.digit3Key, kb.digit4Key, kb.digit5Key };
            for (int i = 0; i < digits.Length; i++)
                if (digits[i].wasPressedThisFrame && _rig != null)
                {
                    if (_rig.LookingThrough == i) _rig.ExitLookThrough();
                    else if (_rig.LookThrough(i)) SetView(i);
                }
#endif
        }

        void CvRow()
        {
            var cv = player.CvDebug;
            GUILayout.Space(4);
            GUILayout.BeginHorizontal();
            GUILayout.Label("cv", GUILayout.Width(38), GUILayout.Height(22));
            if (GUILayout.Button($"colour: {_ov.mode} [J]", GUILayout.Width(170), GUILayout.Height(22)))
                _ov.mode = CvColors.Next(_ov.mode, cv != null);
            if (cv != null)
            {
                _ov.showAux = GUILayout.Toggle(_ov.showAux, "aux", "Button", GUILayout.Width(44), GUILayout.Height(22));
                if (_rays != null) _rays.show = GUILayout.Toggle(_rays.show, "rays [Y]", "Button", GUILayout.Width(70), GUILayout.Height(22));
                if (_rig != null) _rig.show = GUILayout.Toggle(_rig.show, "cams [K]", "Button", GUILayout.Width(70), GUILayout.Height(22));
                if (GUILayout.Button("◀", GUILayout.Width(24), GUILayout.Height(22))) SetView(_cvView - 1);
                GUILayout.Label(cv.cameras[Mathf.Clamp(_cvView, 0, cv.V - 1)].name, GUILayout.Width(110), GUILayout.Height(22));
                if (GUILayout.Button("▶", GUILayout.Width(24), GUILayout.Height(22))) SetView(_cvView + 1);
                bool looking = _rig != null && _rig.LookingThrough >= 0;
                bool proxy = _video != null && _video.HasProxy(looking ? _rig.LookingThrough : _cvView);
                GUI.enabled = proxy;
                if (_video != null)
                    _video.show = GUILayout.Toggle(_video.show && proxy,
                        new GUIContent("video [V]", proxy ? "" : $"no proxy: overlay_video proxies --clip {cv.ProxyClip}"),
                        "Button", GUILayout.Width(80), GUILayout.Height(22));
                GUI.enabled = true;
                string lookLabel = looking ? $"look: {cv.cameras[_rig.LookingThrough].name} [Esc]" : "look [1-5]";
                if (_rig != null && GUILayout.Toggle(looking, lookLabel, "Button", GUILayout.Width(170), GUILayout.Height(22)) != looking)
                {
                    if (looking) _rig.ExitLookThrough();
                    else _rig.LookThrough(_cvView);
                }
                GUI.enabled = cv.persons.Length > 0;
                if (_np != null) _np.show = GUILayout.Toggle(_np.show, "non-players [N]", "Button", GUILayout.Width(110), GUILayout.Height(22));
                GUI.enabled = true;
                string st = _video != null && _video.show ? _video.Status : (_np != null && _np.show ? $"{_np.VisibleCount} shown" : "");
                GUILayout.Label(string.IsNullOrEmpty(GUI.tooltip) ? st : GUI.tooltip, GUILayout.Height(22));
            }
            else GUILayout.Label("no cvdebug sidecar for this file (analysis.cvdebug build)", GUILayout.Height(22));
            GUILayout.EndHorizontal();
        }

        void GhostRow()
        {
            GUILayout.Space(4);
            GUILayout.BeginHorizontal();
            GUILayout.Label("ghost", GUILayout.Width(38), GUILayout.Height(22));
            foreach (var c in _ghosts.Candidates.ToList())
            {
                string name = Path.GetFileName(c).Replace(".analysis.json", "");
                bool on = _ghosts.IsOn(c);
                bool want = GUILayout.Toggle(on, name, "Button", GUILayout.MinWidth(110), GUILayout.Height(22));
                if (want != on) _ghosts.Set(c, want);
            }
            GUILayout.Label("[G]", GUILayout.Width(30), GUILayout.Height(22));
            GUILayout.EndHorizontal();
        }

        void OnDisable() => UiPointerGuard.Clear(this);

        void OnGUI()
        {
            if (!visible || player == null || player.Doc == null) { UiPointerGuard.Clear(this); return; }

            const float pad = 12f;

            if (_minimized)
            {
                var btnRect = new Rect(pad, Screen.height - 40f - pad, 30f, 30f);
                UiPointerGuard.Set(this, btnRect);
                if (GUI.Button(btnRect, "▲")) _minimized = false;
                return;
            }

            int rows = 3; // transport + scrub + visibility
            if (_clips != null && _clips.Length > 0) rows++;
            bool hasOldRacket = HasOldRacket();
            var sidecars = RacketSidecars();
            bool showRacketRow = hasOldRacket || sidecars.Count > 0;
            if (showRacketRow) rows++;
            var shuttleVariants = ShuttleVariants();
            if (shuttleVariants.Count > 0) rows++;
            var poseVariants = PoseVariants();
            bool showPoseRow = poseVariants.Count > 0;
            if (showPoseRow) rows++;
            int nP = player.Doc.players != null ? player.Doc.players.Length : 0;
            if (nP > 0) rows++;
            CacheCv();
            bool cvRow = _ov != null;
            if (cvRow) rows++;
            bool ghostRow = _ghosts != null && _ghosts.Candidates.Count > 0;
            if (ghostRow) rows++;
            float boxH = 28f * rows + 24f;
            var box = new Rect(pad, Screen.height - boxH - pad, Screen.width - pad * 2f, boxH);
            UiPointerGuard.Set(this, box);       // the orbit camera ignores drags / scroll that start on the panel

            GUI.Box(box, GUIContent.none);
            GUILayout.BeginArea(new Rect(box.x + 10, box.y + 8, box.width - 20, box.height - 16));

            // ---- row 1: transport ----
            GUILayout.BeginHorizontal();

            if (GUILayout.Button("▼", GUILayout.Width(26), GUILayout.Height(26)))
                _minimized = true;

            if (GUILayout.Button(player.IsPlaying ? "|| Pause" : "▶ Play",
                                 GUILayout.Width(90), GUILayout.Height(26)))
                player.TogglePlay();
            if (GUILayout.Button("◀", GUILayout.Width(34), GUILayout.Height(26))) player.StepFrame(-1);
            if (GUILayout.Button("▶", GUILayout.Width(34), GUILayout.Height(26))) player.StepFrame(1);

            GUILayout.Space(8);
            float t = player.CurrentFrame / Mathf.Max(player.Fps, 1f);
            GUILayout.Label($"frame {player.CurrentFrame + 1}/{player.FrameCount}   t={t:F2}s",
                            GUILayout.Width(200), GUILayout.Height(26));

            GUILayout.Label("speed", GUILayout.Width(42), GUILayout.Height(26));
            player.speed = GUILayout.HorizontalSlider(player.speed, 0.1f, 3f, GUILayout.Width(120));
            GUILayout.Label($"{player.speed:F1}x", GUILayout.Width(40), GUILayout.Height(26));

            GUILayout.FlexibleSpace();

            GUILayout.Label("show", GUILayout.Width(38), GUILayout.Height(26));
            if (FocusButton("All", player.focusPlayer < 0)) player.focusPlayer = -1;
            for (int i = 0; i < nP; i++)
            {
                string id = player.Doc.players[i].id;
                if (FocusButton(string.IsNullOrEmpty(id) ? $"P{i + 1}" : id, player.focusPlayer == i))
                    player.focusPlayer = i;
            }
            GUILayout.EndHorizontal();

            GUILayout.Space(4);

            // ---- row 2: scrub ----
            float n = GUILayout.HorizontalSlider(player.Normalized, 0f, 1f);
            if (Mathf.Abs(n - player.Normalized) > 0.0005f) { player.Pause(); player.SeekNormalized(n); }

            GUILayout.Space(4);

            // ---- row 3: clip switcher ----
            if (_clips != null && _clips.Length > 0)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label("clip", GUILayout.Width(38), GUILayout.Height(22));
                for (int i = 0; i < _clips.Length; i++)
                {
                    string baseClip = _clips[i];
                    string curBase = BasePath(player.analysisFile);
                    bool cur = curBase == baseClip;
                    bool pick = GUILayout.Toggle(cur, _clipLabels[i], "Button",
                                                 GUILayout.Height(22), GUILayout.MinWidth(120));
                    if (pick && !cur)
                    {
                        if (IsOldRacket())
                        {
                            string target = OldRacketPath(baseClip);
                            string fullTarget = Path.Combine(Application.streamingAssetsPath, target);
                            if (File.Exists(fullTarget))
                                Defer(() => player.LoadClip(target));
                            else
                                Defer(() => LoadClipPreferring(baseClip, ""));
                        }
                        else
                        {
                            string suffix = CurrentPoseSuffix();
                            Defer(() => LoadClipPreferring(baseClip, suffix));
                        }
                    }
                }
                GUILayout.EndHorizontal();
            }

            // ---- row: racket version toggle (old file | v1 | sidecar variants | ghost) ----
            if (showRacketRow)
            {
                GUILayout.Space(4);
                GUILayout.BeginHorizontal();
                GUILayout.Label("racket", GUILayout.Width(38), GUILayout.Height(22));
                bool isOld = IsOldRacket();
                string active = isOld ? "old" : player.EffectiveRacketVariant;
                if (hasOldRacket && GUILayout.Toggle(isOld, "old", "Button",
                                                     GUILayout.Width(60), GUILayout.Height(22)) && !isOld)
                    Defer(() => { player.SetRacketVariant(RacketVisualAnalysis.V1); SwitchToOldRacket(); });
                if (GUILayout.Toggle(active == RacketVisualAnalysis.V1, "v1", "Button",
                                     GUILayout.Width(60), GUILayout.Height(22)) && active != RacketVisualAnalysis.V1)
                    Defer(() => PickRacket(RacketVisualAnalysis.V1));
                foreach (var v in sidecars)
                    if (GUILayout.Toggle(active == v, v, "Button", GUILayout.Width(80), GUILayout.Height(22))
                        && active != v)
                    {
                        string variant = v;
                        Defer(() => PickRacket(variant));
                    }
                if (sidecars.Count > 0)
                {
                    GUILayout.Space(8);
                    bool canGhost = active != "old" && active != RacketVisualAnalysis.V1;
                    GUI.enabled = canGhost;
                    player.GhostV1 = GUILayout.Toggle(player.GhostV1, "ghost v1", "Button",
                                                      GUILayout.Width(80), GUILayout.Height(22));
                    player.SnapRacketToHand = GUILayout.Toggle(player.SnapRacketToHand, "snap to hand", "Button",
                                                               GUILayout.Width(100), GUILayout.Height(22));
                    GUI.enabled = true;
                    var doc = canGhost ? player.GetRacketVariantDoc(active) : null;
                    if (doc != null && !string.IsNullOrEmpty(doc.time_base))
                        GUILayout.Label($"{doc.time_base} time", GUILayout.Height(22));
                }
                GUILayout.EndHorizontal();
            }

            // ---- row: shuttle version toggle (baseline | sidecar variants | ghost | trail) ----
            if (shuttleVariants.Count > 0)
            {
                GUILayout.Space(4);
                GUILayout.BeginHorizontal();
                GUILayout.Label("shuttle", GUILayout.Width(44), GUILayout.Height(22));
                string active = player.EffectiveShuttleVariant;
                if (GUILayout.Toggle(active == ShuttleVisualAnalysis.Baseline, "baseline", "Button",
                                     GUILayout.Width(80), GUILayout.Height(22)) && active != ShuttleVisualAnalysis.Baseline)
                    player.SetShuttleVariant(ShuttleVisualAnalysis.Baseline);
                foreach (var v in shuttleVariants)
                    if (GUILayout.Toggle(active == v, v, "Button", GUILayout.Width(80), GUILayout.Height(22))
                        && active != v)
                        player.SetShuttleVariant(v);
                GUILayout.Space(8);
                GUI.enabled = active != ShuttleVisualAnalysis.Baseline;
                player.GhostShuttleBaseline = GUILayout.Toggle(player.GhostShuttleBaseline, "ghost", "Button",
                                                               GUILayout.Width(60), GUILayout.Height(22));
                player.ShuttleTrail = GUILayout.Toggle(player.ShuttleTrail, "trail", "Button",
                                                       GUILayout.Width(60), GUILayout.Height(22));
                GUI.enabled = true;
                GUILayout.Label(ShuttleStatsLabel(active), GUILayout.Height(22));
                GUILayout.EndHorizontal();
            }

            // ---- row: pose version (base | v2 robust | fit | SMPL-X 58 | SMPL-24) ----
            if (showPoseRow)
            {
                GUILayout.Space(4);
                GUILayout.BeginHorizontal();
                GUILayout.Label("pose", GUILayout.Width(38), GUILayout.Height(22));
                string cur = CurrentPoseSuffix();
                if (Exists(BasePath(player.analysisFile))
                    && GUILayout.Toggle(cur == "", "base", "Button", GUILayout.Width(80), GUILayout.Height(22)) && cur != "")
                    Defer(() => PickPose(""));
                foreach (int i in poseVariants)
                    if (GUILayout.Toggle(cur == PoseSuffixes[i], PoseLabels[i], "Button",
                                         GUILayout.Width(90), GUILayout.Height(22)) && cur != PoseSuffixes[i])
                    {
                        string suffix = PoseSuffixes[i];
                        Defer(() => PickPose(suffix));
                    }
                var prod = player.Doc.meta != null && player.Doc.meta.producers != null ? player.Doc.meta.producers.pose : null;
                string lbl = player.Doc.ShortLabel();
                GUILayout.Label(string.IsNullOrEmpty(prod) ? lbl : $"{lbl} · {prod}", GUILayout.Height(22));
                GUILayout.EndHorizontal();
            }

            // ---- rows: cv visualisation (SP4) + ghost builds ----
            if (cvRow) CvRow();
            if (ghostRow) GhostRow();

            // ---- row: per-player skin ----
            GUILayout.Space(4);
            GUILayout.BeginHorizontal();
            GUILayout.Label("skin", GUILayout.Width(38), GUILayout.Height(22));
            for (int i = 0; i < nP; i++)
            {
                string sid = player.Doc.players[i].id;
                GUILayout.Label(string.IsNullOrEmpty(sid) ? $"P{i + 1}" : sid,
                                GUILayout.Width(52), GUILayout.Height(22));
                var cur = player.GetSkin(i);
                if (GUILayout.Toggle(cur == AnalysisPlayer.SkinMode.Skeleton, "skeleton", "Button",
                                     GUILayout.Width(70), GUILayout.Height(22))
                    && cur != AnalysisPlayer.SkinMode.Skeleton)
                    player.SetSkin(i, AnalysisPlayer.SkinMode.Skeleton);
                if (GUILayout.Toggle(cur == AnalysisPlayer.SkinMode.Mixamo, "mixamo", "Button",
                                     GUILayout.Width(60), GUILayout.Height(22))
                    && cur != AnalysisPlayer.SkinMode.Mixamo)
                    player.SetSkin(i, AnalysisPlayer.SkinMode.Mixamo);
                if (GUILayout.Toggle(cur == AnalysisPlayer.SkinMode.MakeHuman, "makehuman", "Button",
                                     GUILayout.Width(80), GUILayout.Height(22))
                    && cur != AnalysisPlayer.SkinMode.MakeHuman)
                    player.SetSkin(i, AnalysisPlayer.SkinMode.MakeHuman);
                GUILayout.Space(10);
            }
            GUILayout.EndHorizontal();

            // ---- row: visibility toggles ----
            GUILayout.Space(4);
            GUILayout.BeginHorizontal();
            GUILayout.Label("show", GUILayout.Width(38), GUILayout.Height(22));
            player.showPose = GUILayout.Toggle(player.showPose, "pose", "Button",
                                               GUILayout.Width(60), GUILayout.Height(22));
            player.showRacket = GUILayout.Toggle(player.showRacket, "racket", "Button",
                                                 GUILayout.Width(70), GUILayout.Height(22));
            player.showShuttle = GUILayout.Toggle(player.showShuttle, "shuttle", "Button",
                                                  GUILayout.Width(70), GUILayout.Height(22));
            GUILayout.EndHorizontal();

            GUILayout.EndArea();
        }

        static bool FocusButton(string label, bool active)
            => GUILayout.Toggle(active, label, "Button", GUILayout.Width(52), GUILayout.Height(26)) && !active;
    }
}
