using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.UIElements;

namespace BadmintonVR.Analysis
{
    /// <summary>
    /// v0.8.0 desktop viewer in UI Toolkit (the IMGUI AnalysisViewerUI panel is archived in _archive/unity-imgui-viewer/).
    /// Built from code: a runtime PanelSettings with Unity's default runtime theme (Resources/ViewerUI/ViewerTheme.tss, so
    /// scroll views and text fields are styled and clipped) and inline styles. One flex column, so no panel can overlap
    /// another at any window size:
    ///   top    clip title + build label (+ the look-through banner); beside it, while a clip loads on a worker thread,
    ///          one "loading <clip> ... MB" line (the old clip keeps playing; the clicked card gets an accent edge)
    ///   middle left: clip browser (grouped list) + live run box (run folder, frame-file folder, status)
    ///          right: Builds, Racket and shuttle, Cameras, Display, Players, Debug (the debug HUD's stats and toggles)
    ///   bottom live row (when connected), transport (play, step, +-1 s, speed presets, loop, A-B range), timeline, legend
    ///   ?/F1   shortcut help (one table drives the help text)
    /// Typing in a text field owns the keyboard (UiPointerGuard.Typing): shortcuts and fly-mode WASD wait until the field
    /// loses focus (Enter, Esc, or a click anywhere outside it).
    /// </summary>
    [DisallowMultipleComponent]
    public class AnalysisViewerUITK : MonoBehaviour
    {
        public AnalysisPlayer player;
        [Tooltip("Folder under StreamingAssets scanned for *.analysis.json")]
        public string analysisFolder = "analysis";
        public bool panelsVisible = true;
        [Tooltip("Resources path of the theme (a .tss that imports unity-theme://default)")]
        public string themeResource = "ViewerUI/ViewerTheme";

        // ------------------------------------------------------------------ palette + sizes (one place)
        static readonly Color Bg = new Color(0.08f, 0.09f, 0.11f, 0.92f);
        static readonly Color Bg2 = new Color(0.14f, 0.15f, 0.18f, 0.95f);
        static readonly Color Line = new Color(1f, 1f, 1f, 0.08f);
        static readonly Color Text = new Color(0.92f, 0.93f, 0.95f, 1f);
        static readonly Color Dim = new Color(0.62f, 0.65f, 0.70f, 1f);
        static readonly Color Accent = new Color(0.30f, 0.60f, 1.00f, 1f);
        static readonly Color Warn = new Color(1f, 0.45f, 0.38f, 1f);
        static readonly Color BtnBg = new Color(0.20f, 0.22f, 0.26f, 1f);
        static readonly Color BtnHover = new Color(0.27f, 0.30f, 0.36f, 1f);
        static readonly Color BtnOn = new Color(0.20f, 0.42f, 0.78f, 1f);
        static readonly Color QFitted = new Color(0.30f, 0.60f, 1f, 1f), QWarm = new Color(0.6f, 0.6f, 0.6f, 1f),
                              QLate = new Color(0.90f, 0.40f, 0.35f, 1f), QNone = new Color(0f, 0f, 0f, 1f),
                              QInterp = new Color(0.95f, 0.75f, 0.25f, 1f), QMissing = new Color(0.35f, 0.08f, 0.08f, 1f);
        const float LeftW = 290f, RightW = 300f, Pad = 10f;
        static readonly float[] SpeedPresets = { 0.25f, 0.5f, 1f, 2f };

        UIDocument _ui;
        PanelSettings _ps;
        VisualElement _root, _left, _right, _bottom, _help, _toast, _liveRow, _loadBox;
        Label _title, _subtitle, _banner, _readout, _speedLabel, _liveBadge, _liveStatus, _liveFrames, _loading;
        TextField _liveField;
        Button _jumpLive, _showPanels, _liveHead;
        VisualElement _liveBody;
        LiveFolderFeed _feed;
        VisualElement _transport, _legend;
        bool _broadcastShown;
        ServerFollower _follower;
        Button _serverBtn;
        bool _standalone, _serverShown;
        /// <summary>v0.8.2: the pipeline server decides what plays (runs/now_playing.json exists): no clip browser, no
        /// live box; the user moves the camera and toggles the display. "standalone" brings the panels back.</summary>
        bool ServerMode => _follower != null && _follower.Active && !_standalone;
        ScrollView _clipList, _rightScroll;
        Button _playBtn, _loopBtn, _rangeBtn;
        readonly List<(Button b, float s)> _speedBtns = new List<(Button, float)>();
        TimelineBar _timeline;
        VisualElement _secBuilds, _secRS, _secCams, _secDisplay, _secPlayers, _secDebug, _debugLines, _toastBody;
        Button _secDebugHead;
        float _toastUntil, _liveInfoAt;
        bool _built, _dirtySections = true, _lastHudMaster;
        int _builtFocus = int.MinValue, _cvView;
        string _lastFile, _clipBeforeLive, _lastLoading;

        // ------------------------------------------------------------------ clip catalogue
        public class ClipEntry
        {
            public string path, file, group, label, variant;
            public float fps; public int frames; public long bytes; public DateTime modified;
            public bool live, bundled;
        }
        readonly List<ClipEntry> _clips = new List<ClipEntry>();
        static readonly Regex ReFps = new Regex("\"fps\":\\s*([0-9.]+)"), ReFrames = new Regex("\"frame_count\":\\s*([0-9]+)"),
                              ReLabel = new Regex("\"label\":\\s*\"([^\"]*)\"");

        void Reset() => player = GetComponent<AnalysisPlayer>();

        void Awake()
        {
            if (player == null) player = GetComponent<AnalysisPlayer>();
            var hud = FindAnyObjectByType<AnalysisDebugHUD>();
            if (hud != null) { hud.master = false; hud.drawPanels = false; }   // its text + toggles live in the Debug section
        }

        void OnEnable()
        {
            _ps = ScriptableObject.CreateInstance<PanelSettings>();
            _ps.scaleMode = PanelScaleMode.ScaleWithScreenSize;
            _ps.referenceResolution = new Vector2Int(1440, 810);
            _ps.screenMatchMode = PanelScreenMatchMode.MatchWidthOrHeight;
            _ps.match = 0.5f;
            _ps.sortingOrder = 10;
            // the default runtime theme styles ScrollView / Scroller / TextField (without one a scroll view does not clip
            // its content and a text field draws nothing); every colour and size below is still set inline
            var theme = Resources.Load<ThemeStyleSheet>(themeResource);
            if (theme != null) _ps.themeStyleSheet = theme;
            else Debug.LogWarning($"[Analysis] viewer theme Resources/{themeResource}.tss not found: text fields and lists may not draw right");
            _ui = gameObject.GetComponent<UIDocument>();
            if (_ui == null) _ui = gameObject.AddComponent<UIDocument>();
            _ui.panelSettings = _ps;
            UiPointerGuard.SetPicker(this, OverUi);
            UiPointerGuard.SetTyping(this, () => Typing);
            Application.logMessageReceived += OnLog;
            if (player != null) player.ClipLoaded += OnClipLoaded;
        }

        void OnDisable()
        {
            UiPointerGuard.ClearPicker(this);
            UiPointerGuard.ClearTyping(this);
            Application.logMessageReceived -= OnLog;
            if (player != null) player.ClipLoaded -= OnClipLoaded;
            _built = false;
        }

        void OnApplicationFocus(bool focus)
        {
            if (focus && _built) RebuildClipList();
        }

        void OnClipLoaded() { _dirtySections = true; _timeline?.Rebuild(player); }

        void OnLog(string msg, string stack, LogType type)
        {
            if ((type == LogType.Error || type == LogType.Exception || type == LogType.Warning) && msg.StartsWith("[Analysis"))
                Toast(msg, type == LogType.Warning ? 4f : 7f);
        }

        bool OverUi(Vector2 screen)
        {
            var panel = _root?.panel;
            if (panel == null) return false;
            var p = RuntimePanelUtils.ScreenToPanel(panel, new Vector2(screen.x, Screen.height - screen.y));
            var hit = panel.Pick(p);
            return hit != null && hit != _root;
        }

        /// <summary>A text field has the keyboard focus (the user is typing a run folder).</summary>
        bool Typing
        {
            get
            {
                var fe = _root?.panel?.focusController?.focusedElement as VisualElement;
                return fe != null && (fe is TextField || fe.GetFirstAncestorOfType<TextField>() != null);
            }
        }

        void Unfocus() => _root?.panel?.focusController?.focusedElement?.Blur();

        // ------------------------------------------------------------------ build
        void Build()
        {
            _root = _ui.rootVisualElement;
            if (_root == null) return;
            _root.Clear();
            _root.style.flexGrow = 1;
            _root.style.flexDirection = FlexDirection.Column;
            _root.style.paddingLeft = 12; _root.style.paddingRight = 12; _root.style.paddingTop = 10; _root.style.paddingBottom = 10;
            _root.pickingMode = PickingMode.Ignore;
            // A TextCore font asset made up front: with FontDefinition.FromFont the legacy font is converted lazily, the
            // first layout measures text with other metrics and wrapped labels / wrapping rows then overlap.
            _font = _font != null ? _font : MakeFont();
            if (_font != null) _root.style.unityFontDefinition = FontDefinition.FromSDFFont(_font);
            _root.style.color = Text;
            _root.style.fontSize = 13;
            // buttons never take focus, so a press anywhere outside the field being typed in ends the typing here
            _root.RegisterCallback<PointerDownEvent>(e =>
            {
                var fe = _root.panel?.focusController?.focusedElement as VisualElement;
                var tf = fe as TextField ?? fe?.GetFirstAncestorOfType<TextField>();
                var t = e.target as VisualElement;
                if (tf != null && (t == null || (t != tf && !tf.Contains(t)))) fe.Blur();
            }, TrickleDown.TrickleDown);
            // The Input System's UI actions map WASD / arrows to "navigate", which moves the keyboard focus between fields
            // (fly-mode WASD used to land in a text field). Keyboard focus moves only by clicking.
            _root.RegisterCallback<NavigationMoveEvent>(e =>
            {
                e.StopPropagation();
                _root.focusController?.IgnoreEvent(e);
            }, TrickleDown.TrickleDown);

            // ---- top: title bar (+ "show panels" while they are hidden)
            var topRow = Layout(FlexDirection.Row); topRow.style.alignItems = Align.FlexStart; topRow.style.flexShrink = 0;
            var top = Panel(); top.style.flexShrink = 1; top.style.minWidth = 0; top.style.maxWidth = Length.Percent(62);
            top.style.paddingLeft = 12; top.style.paddingRight = 12; top.style.paddingTop = 5; top.style.paddingBottom = 5;
            _title = L("", 16, true); _subtitle = L("", 12); _subtitle.style.color = Dim;
            _banner = L("", 12, true); _banner.style.color = Accent; _banner.style.display = DisplayStyle.None;
            // the look-through banner sits on top of the build label (same line, so the panels below never move)
            _banner.style.position = Position.Absolute; _banner.style.left = 0; _banner.style.right = 0; _banner.style.top = 0;
            var sub = new VisualElement(); sub.Add(_subtitle); sub.Add(_banner);
            top.Add(_title); top.Add(sub);
            // loading line: one line beside the title panel (shorter than it, so the row height and the panels below never move)
            _loadBox = Panel(); _loadBox.style.flexShrink = 1; _loadBox.style.minWidth = 0; _loadBox.style.marginLeft = 8;
            _loadBox.style.paddingLeft = 12; _loadBox.style.paddingRight = 12; _loadBox.style.paddingTop = 5; _loadBox.style.paddingBottom = 5;
            _loadBox.style.display = DisplayStyle.None; _loadBox.pickingMode = PickingMode.Ignore;
            _loading = L("", 13, true); _loading.style.color = Accent; _loading.pickingMode = PickingMode.Ignore;
            _loadBox.Add(_loading);
            topRow.Add(top); topRow.Add(_loadBox); topRow.Add(Spacer());
            _showPanels = Btn("show panels (P)", () => SetPanels(true)); _showPanels.style.display = DisplayStyle.None;
            _serverBtn = Btn("server", () => { _standalone = !_standalone; _serverShown = !ServerMode; },
                             "the pipeline server picks what plays; click for standalone (clip browser + live box)");
            _serverBtn.style.display = DisplayStyle.None;
            topRow.Add(_serverBtn);
            topRow.Add(_showPanels);
            _root.Add(topRow);

            // ---- middle: clip browser | (3D view) | sections
            var mid = Layout(FlexDirection.Row); mid.style.flexGrow = 1; mid.style.minHeight = 0;
            mid.style.marginTop = 8; mid.style.marginBottom = 8; mid.style.alignItems = Align.Stretch;
            _left = Dock(LeftW); BuildLeft(); mid.Add(_left);
            mid.Add(Spacer());
            _right = Dock(RightW);
            _rightScroll = new ScrollView(ScrollViewMode.Vertical); _rightScroll.style.flexGrow = 1; StyleScroll(_rightScroll);
            _secBuilds = Section(_rightScroll, "Builds", true, out _);
            _secRS = Section(_rightScroll, "Racket and shuttle", true, out _);
            _secCams = Section(_rightScroll, "Cameras", true, out _);
            _secDisplay = Section(_rightScroll, "Display", true, out _);
            _secPlayers = Section(_rightScroll, "Players", true, out _);
            _secDebug = Section(_rightScroll, "Debug (H)", false, out _secDebugHead);
            _right.Add(_rightScroll);
            mid.Add(_right);
            _root.Add(mid);

            // ---- bottom: live row, transport, timeline, legend
            _bottom = Panel(); _bottom.style.flexShrink = 0;
            _bottom.style.paddingLeft = Pad; _bottom.style.paddingRight = Pad; _bottom.style.paddingTop = 8; _bottom.style.paddingBottom = 8;
            BuildBottom();
            _root.Add(_bottom);

            // ---- overlays: help, toast
            _help = Panel(); _help.style.position = Position.Absolute; _help.style.left = Length.Percent(27);
            _help.style.right = Length.Percent(27); _help.style.top = Length.Percent(10);
            _help.style.paddingLeft = 18; _help.style.paddingRight = 18; _help.style.paddingTop = 14; _help.style.paddingBottom = 14;
            _help.style.display = DisplayStyle.None;
            var hh = Row(); hh.Add(L("Shortcuts", 16, true)); hh.Add(Spacer()); hh.Add(Btn("close", ToggleHelp, "? / F1 / Esc")); _help.Add(hh);
            foreach (var k in Keys)
            {
                var r = Row(); r.style.marginTop = 1;
                var a = L(k.keys, 13, true); a.style.width = 190; a.style.flexShrink = 0; r.Add(a);
                var w = L(k.what, 13); w.style.flexShrink = 1; w.style.minWidth = 0; r.Add(w);
                _help.Add(r);
            }
            _root.Add(_help);
            _toast = Panel(); _toast.style.position = Position.Absolute; _toast.style.top = 74;
            _toast.style.left = Length.Percent(25); _toast.style.right = Length.Percent(25);
            _toast.style.paddingLeft = 12; _toast.style.paddingRight = 12; _toast.style.paddingTop = 6; _toast.style.paddingBottom = 6;
            _toast.style.backgroundColor = new Color(0.45f, 0.12f, 0.12f, 0.95f); _toast.style.display = DisplayStyle.None;
            _toast.pickingMode = PickingMode.Ignore;
            _toastBody = new VisualElement { pickingMode = PickingMode.Ignore }; _toast.Add(_toastBody);
            _root.Add(_toast);

            _built = true;
            RebuildClipList();
            SetPanels(panelsVisible);
            _dirtySections = true;
        }

        void BuildLeft()
        {
            var head = Row(); head.Add(L("Clips", 14, true)); head.Add(Spacer());
            head.Add(Btn("rescan", RebuildClipList, "rescan StreamingAssets for new files"));
            head.Add(Btn("hide", () => SetPanels(false), "hide the panels (P)"));
            _left.Add(head);
            _clipList = new ScrollView(ScrollViewMode.Vertical); _clipList.style.flexGrow = 1; _clipList.style.flexShrink = 1;
            _clipList.style.minHeight = 60; StyleScroll(_clipList);
            _left.Add(_clipList);

            // live run (collapsible, so the clip list keeps the height): a folder the live runner is writing (or a finished
            // one); its frames are <run>/unity_live_frames/frame_NNNNNN.json, one JSON file per output frame
            var box = new VisualElement(); box.style.flexShrink = 0; box.style.marginTop = 8; box.style.paddingTop = 6;
            box.style.borderTopWidth = 1; box.style.borderTopColor = Line;
            var lh = Row();
            _liveHead = Btn("+ Live run", null, "play a run folder's per-frame files"); _liveHead.style.unityFontStyleAndWeight = FontStyle.Bold;
            _liveHead.style.backgroundColor = Bg2; _liveHead.style.marginLeft = 0;
            lh.Add(_liveHead); lh.Add(Spacer());
            lh.Add(Btn("connect", ConnectLive, "play this run's frame files as they arrive"));
            lh.Add(Btn("stop", DisconnectLive, "disconnect and go back to the clip"));
            box.Add(lh);
            _liveBody = new VisualElement(); _liveBody.style.flexShrink = 0; _liveBody.style.display = DisplayStyle.None;
            _liveHead.clicked += () => OpenSection(_liveBody, _liveHead, _liveBody.style.display == DisplayStyle.None);
            var fr0 = Row();
            _liveField = Field("run folder (has unity_live_frames/)", "run folder, relative to the project or absolute; Enter connects");
            _liveField.tabIndex = -1;                                       // never reached by Tab / navigation, only by a click
            _liveField.style.flexGrow = 1; _liveField.style.flexShrink = 1; _liveField.style.minWidth = 0;
            var newest = LiveFolderFeed.NewestRun(RunsRoot);
            _liveField.value = newest != null ? Rel(newest) : "";
            _liveField.RegisterCallback<KeyDownEvent>(e =>
            {
                if (e.keyCode == KeyCode.Return || e.keyCode == KeyCode.KeypadEnter) ConnectLive();
            }, TrickleDown.TrickleDown);
            _liveField.RegisterValueChangedCallback(_ => _liveInfoAt = 0f);
            fr0.Add(_liveField); fr0.Add(Btn("latest", PickLatestRun, "newest run folder under runs/"));
            _liveBody.Add(fr0);
            var fr = Row(); _liveFrames = Small(""); _liveFrames.style.flexGrow = 1; _liveFrames.style.flexShrink = 1; _liveFrames.style.minWidth = 0;
            fr.Add(_liveFrames); fr.Add(Btn("open", OpenFramesFolder, "open the frame-file folder in Explorer"));
            _liveBody.Add(fr);
            _liveStatus = Small("not connected"); _liveBody.Add(_liveStatus);
            box.Add(_liveBody);
            _left.Add(box);
        }

        void BuildBottom()
        {
            _liveRow = Row(); _liveRow.style.display = DisplayStyle.None; _liveRow.style.marginBottom = 4;
            _liveBadge = L("", 13, true); _liveBadge.style.color = Warn; _liveBadge.style.flexGrow = 1; _liveBadge.style.flexShrink = 1;
            _liveBadge.style.minWidth = 0; _liveRow.Add(_liveBadge);
            _jumpLive = Btn("jump to live", () => player.JumpToLive(), "back to the newest frame"); _liveRow.Add(_jumpLive);
            _bottom.Add(_liveRow);

            var tr = Row(); tr.style.flexWrap = Wrap.Wrap;                 // wraps on a narrow window instead of overflowing
            _transport = tr;
            tr.Add(Btn("|<", () => player.SeekFrame(player.RangeStart), "to the start (Home)"));
            tr.Add(Btn("-1s", () => Jump(-1f), "back 1 s (Shift+Left)"));
            tr.Add(Btn("<", () => player.StepFrame(-1), "previous frame (Left)"));
            _playBtn = Btn("Play", () => player.TogglePlay(), "play / pause (Space)"); _playBtn.style.width = 62; tr.Add(_playBtn);
            tr.Add(Btn(">", () => player.StepFrame(1), "next frame (Right)"));
            tr.Add(Btn("+1s", () => Jump(1f), "forward 1 s (Shift+Right)"));
            tr.Add(Btn(">|", () => player.SeekFrame(player.RangeEnd), "to the end (End)"));
            tr.Add(Gap(10));
            _speedBtns.Clear();
            foreach (var s in SpeedPresets)
            {
                float sp = s;
                var b = Btn($"{s:0.##}x", () => player.speed = sp, $"speed {s:0.##}x  ([ / ])");
                _speedBtns.Add((b, s)); tr.Add(b);
            }
            _speedLabel = L("", 12); _speedLabel.style.color = Dim; _speedLabel.style.width = 44; _speedLabel.style.marginLeft = 4;
            tr.Add(_speedLabel);
            tr.Add(Gap(10));
            _loopBtn = Btn("loop", () => player.loop = !player.loop, "loop at the end (L)"); tr.Add(_loopBtn);
            tr.Add(Btn("A", SetA, "range start at this frame (I)"));
            tr.Add(Btn("B", SetB, "range end at this frame (O)"));
            _rangeBtn = Btn("A-B off", () => player.ClearRange(), "clear the A-B range"); tr.Add(_rangeBtn);
            tr.Add(Spacer());
            _readout = L("", 13, true); _readout.style.flexShrink = 0; _readout.style.marginLeft = 10; tr.Add(_readout);
            tr.Add(Gap(8));
            tr.Add(Btn("?", ToggleHelp, "shortcuts (? / F1)"));
            _bottom.Add(tr);
            _timeline = new TimelineBar(this); _bottom.Add(_timeline);
            var legend = Row(); legend.style.marginTop = 3; legend.style.flexWrap = Wrap.Wrap;
            _legend = legend;
            foreach (var (c, t) in new[] { (QFitted, "fitted"), (QWarm, "warm-up"), (QLate, "late / fallback"),
                                           (QInterp, "interpolated"), (QMissing, "no pose") })
            {
                var sw = new VisualElement(); sw.style.width = 10; sw.style.height = 10; sw.style.backgroundColor = c;
                sw.style.marginRight = 4; sw.style.marginLeft = 10; legend.Add(sw);
                var lab = L(t, 11); lab.style.color = Dim; legend.Add(lab);
            }
            _bottom.Add(legend);
        }

        static UnityEngine.TextCore.Text.FontAsset _font;

        static UnityEngine.TextCore.Text.FontAsset MakeFont()
        {
            var f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (f == null) return null;
            var fa = UnityEngine.TextCore.Text.FontAsset.CreateFontAsset(f);
            if (fa != null) fa.name = "ViewerFont (LegacyRuntime)";
            return fa;
        }

        // ------------------------------------------------------------------ per frame
        void Update()
        {
            if (_ui == null || player == null) return;
            if (!_built || _root != _ui.rootVisualElement || _root.childCount == 0) Build();
            if (!_built) return;
            if (player.analysisFile != _lastFile) { _lastFile = player.analysisFile; _dirtySections = true; MarkCurrentClip(); }
            if (_feed == null) _feed = player.GetComponent<LiveFolderFeed>();     // the server follower may have added it
            UpdateServerMode();
            BlurOnOutsideClick();
            HandleKeys();
            UpdateLiveInfo();
            UpdateLoading();
            if (_toast.style.display == DisplayStyle.Flex && Time.unscaledTime > _toastUntil) _toast.style.display = DisplayStyle.None;
            if (player.Doc == null) return;
            Cache();
            if (player.CvDebug != null && _cvView >= player.CvDebug.V) SetView(0);
            if (player.focusPlayer != _builtFocus) _dirtySections = true;    // focus changed elsewhere (keys, scene, code)
            if (_dirtySections) { RebuildSections(); _dirtySections = false; _builtFocus = player.focusPlayer; }
            SyncToggles();
            _title.text = ServerMode && !string.IsNullOrEmpty(_follower.Title) ? _follower.Title
                        : Path.GetFileName(player.analysisFile ?? "").Replace(".analysis.json", "");
            _subtitle.text = player.Doc.ShortLabel(90);
            UpdateBanner();
            float fps = Mathf.Max(player.Fps, 1f);
            _readout.text = $"{Clock(player.CurrentFrame / fps)} / {Clock(player.FrameCount / fps)}    " +
                            $"frame {player.CurrentFrame + 1} / {player.FrameCount}    {fps:0.#} fps";
            _playBtn.text = player.IsPlaying ? "Pause" : "Play";
            bool preset = false;
            foreach (var (b, s) in _speedBtns) { bool on = Mathf.Approximately(player.speed, s); SetOn(b, on); preset |= on; }
            _speedLabel.text = preset ? "" : $"{player.speed:0.##}x";
            SetOn(_loopBtn, player.loop);
            SetOn(_rangeBtn, player.rangeStart >= 0 || player.rangeEnd >= 0);
            _timeline.MarkDirtyRepaint();
            UpdatePlayerCards();
            UpdateLive();
            UpdateDebug();
        }

        /// <summary>A click on the 3D view (or anywhere outside the panels) ends typing, so C / WASD go to the camera again.</summary>
        void BlurOnOutsideClick()
        {
#if ENABLE_INPUT_SYSTEM
            var m = UnityEngine.InputSystem.Mouse.current;
            if (m == null || !Typing) return;
            if ((m.leftButton.wasPressedThisFrame || m.rightButton.wasPressedThisFrame || m.middleButton.wasPressedThisFrame)
                && !OverUi(m.position.ReadValue()))
                Unfocus();
#endif
        }

        static string Clock(float s) => $"{(int)(s / 60)}:{s % 60f:00.00}";

        string ProjectRoot => Path.GetDirectoryName(Application.dataPath) ?? ".";
        string RunsRoot => Path.Combine(ProjectRoot, "runs");

        string FirstClip() => _clips.Count > 0 ? _clips[0].path : player.analysisFile;

        // ------------------------------------------------------------------ live run
        void PickLatestRun()
        {
            var r = LiveFolderFeed.NewestRun(RunsRoot);
            if (r != null) _liveField.value = Rel(r); else Toast("no run with unity_live_frames/ under " + RunsRoot);
        }

        void ConnectLive()
        {
            Unfocus();
            if (_feed == null) { _feed = player.GetComponent<LiveFolderFeed>(); if (_feed == null) _feed = player.gameObject.AddComponent<LiveFolderFeed>(); }
            _feed.player = player;
            if (!player.IsLive && !(player.analysisFile ?? "").StartsWith("live:")) _clipBeforeLive = player.analysisFile;
            if (_feed.Connect(Abs(_liveField.value))) { _dirtySections = true; _timeline.Rebuild(player); OpenSection(_liveBody, _liveHead, true); }
            else Toast(_feed.Status);
        }

        void DisconnectLive()
        {
            _feed?.Disconnect();
            if (player.Doc == null || player.IsLive || (player.analysisFile ?? "").StartsWith("live:"))
                player.LoadClip(!string.IsNullOrEmpty(_clipBeforeLive) ? _clipBeforeLive
                                : (player.analysisFile ?? "").StartsWith("live:") ? FirstClip() : player.analysisFile);
        }

        /// <summary>A run folder typed relative to the project (runs/...) or absolute.</summary>
        string Abs(string p)
        {
            p = (p ?? "").Trim().Trim('"');
            return p.Length == 0 || Path.IsPathRooted(p) ? p : Path.GetFullPath(Path.Combine(ProjectRoot, p));
        }

        /// <summary>The frame folder of a run (unity_live_frames/, or live/ for v0.8.0 / v0.8.1 runs), null when none.</summary>
        string FramesDir(string run) => string.IsNullOrWhiteSpace(run) ? null : LiveFolderFeed.FramesFolder(Abs(run));

        string Rel(string p)
        {
            string root = ProjectRoot.Replace('\\', '/').TrimEnd('/') + "/";
            string q = p.Replace('\\', '/');
            return q.StartsWith(root, StringComparison.OrdinalIgnoreCase) ? q.Substring(root.Length) : q;
        }

        void UpdateLiveInfo()
        {
            if (Time.unscaledTime < _liveInfoAt) return;
            _liveInfoAt = Time.unscaledTime + 1f;
            string fd = FramesDir(_liveField.value);
            int n = 0;
            if (fd != null) n = LiveFolderFeed.CountFrames(fd);
            _liveFrames.text = fd != null ? $"{n} frame files in {Path.GetFileName(fd)}/" : "no unity_live_frames folder in this run";
            _liveFrames.tooltip = fd != null ? $"{LiveFolderFeed.FramePath(fd, 0)} ... {LiveFolderFeed.FramePath(fd, Math.Max(n - 1, 0))}" : "";
            if (_feed != null) _liveStatus.text = _feed.Status;
        }

        void OpenFramesFolder()
        {
            string fd = FramesDir(_liveField.value);
            if (fd == null) { Toast("no unity_live_frames folder in " + Abs(_liveField.value)); return; }
            Application.OpenURL("file:///" + fd.Replace('\\', '/'));
        }

        /// <summary>v0.8.2 broadcast: while frames arrive the viewer is a live broadcast, not a playback: no transport, no
        /// timeline of the whole clip, just the LIVE line; the pose follows each frame file as it lands. When the run ends
        /// the transport and timeline come back to scrub through it.</summary>
        void UpdateBroadcast()
        {
            bool b = player.IsLive && _feed != null && _feed.Broadcasting;
            if (b != _broadcastShown)
            {
                _broadcastShown = b;
                var d = b ? DisplayStyle.None : DisplayStyle.Flex;
                if (_transport != null) _transport.style.display = d;
                if (_timeline != null) _timeline.style.display = d;
                if (_legend != null) _legend.style.display = d;
                _liveBadge.style.fontSize = b ? 16 : 13;
                if (b) player.followLive = true;
            }
        }

        void UpdateLive()
        {
            bool live = player.IsLive && _feed != null && _feed.Connected;
            _liveRow.style.display = live ? DisplayStyle.Flex : DisplayStyle.None;
            _jumpLive.style.display = live && !player.followLive && !_feed.Broadcasting ? DisplayStyle.Flex : DisplayStyle.None;
            if (_feed != null) _liveStatus.text = _feed.Status;
            UpdateBroadcast();
            if (!live) return;
            if (_feed.Broadcasting)
            {
                _liveBadge.style.color = Warn;
                _liveBadge.text = _feed.Received == 0
                    ? "\u25CF LIVE    Broadcasting: waiting for the first frames from the cameras..."
                    : $"\u25CF LIVE    Broadcasting: processing the game in real time    {_feed.Received} frames    " +
                      $"{_feed.IncomingFps:0.0} frames/s in    {_feed.BehindS:0.0} s behind the cameras";
                return;
            }
            float fps = Mathf.Max(player.Fps, 1f);
            float behind = Mathf.Max(0, player.LiveHead - player.CurrentFrame) / fps;
            string follow = player.followLive ? $"buffer {player.liveDelayS * 1000f:0} ms" : $"{behind:0.0} s behind live";
            _liveBadge.style.color = Dim;
            _liveBadge.text = $"BROADCAST ENDED    {_feed.Received} frames    scrub back with the timeline    {follow}";
        }

        /// <summary>A file age in readable units (ms, s, min, h, d).</summary>
        static string Age(float ms)
        {
            float s = ms / 1000f;
            return s < 1f ? $"{ms:0} ms" : s < 60f ? $"{s:0.0} s" : s < 3600f ? $"{s / 60f:0} min" : s < 86400f ? $"{s / 3600f:0.#} h" : $"{s / 86400f:0.#} d";
        }

        /// <summary>The loading line + the loading card's accent edge follow AnalysisPlayer's worker-thread load (errors still
        /// come as a toast: the player logs them on the main thread).</summary>
        void UpdateLoading()
        {
            string f = player.LoadingFile;
            if (f != _lastLoading)
            {
                _lastLoading = f;
                _loadBox.style.display = f != null ? DisplayStyle.Flex : DisplayStyle.None;
                MarkCurrentClip();
            }
            if (f == null) return;
            string mb = player.LoadingBytes > 0 ? $"   {player.LoadingBytes / 1048576f:0.0} MB" : "";
            _loading.text = $"loading {Path.GetFileName(f).Replace(".analysis.json", "")} ...{mb}   {player.LoadingStage} {player.LoadingSeconds:0.0} s";
            _loading.tooltip = f;
        }

        void UpdateBanner()
        {
            var cv = player.CvDebug;
            int look = _rig != null ? _rig.LookingThrough : -1;
            bool on = cv != null && look >= 0 && look < cv.V;
            _banner.style.display = on ? DisplayStyle.Flex : DisplayStyle.None;
            _subtitle.style.visibility = on ? Visibility.Hidden : Visibility.Visible;
            if (on)
            {
                var c = cv.cameras[look];
                _banner.text = $"looking through {c.name}  ({c.width} x {c.height}, vfov {c.vfov_deg:0.0} deg)  -  Esc to return";
            }
        }

        void Jump(float seconds) { player.Pause(); player.SeekFrame(player.CurrentFrame + Mathf.RoundToInt(seconds * player.Fps)); }
        void SetA() { player.rangeStart = player.CurrentFrame; if (player.rangeEnd >= 0 && player.rangeEnd < player.rangeStart) player.rangeEnd = -1; }
        void SetB() { player.rangeEnd = player.CurrentFrame; if (player.rangeStart > player.rangeEnd) player.rangeStart = -1; }
        void ToggleHelp() => _help.style.display = _help.style.display == DisplayStyle.None ? DisplayStyle.Flex : DisplayStyle.None;

        /// <summary>Server mode hides the left dock (clip browser + live run box); the button shows server / standalone.</summary>
        void UpdateServerMode()
        {
            if (_follower == null) _follower = player.GetComponent<ServerFollower>();
            bool active = _follower != null && _follower.Active;
            if (_serverBtn != null)
            {
                _serverBtn.style.display = active ? DisplayStyle.Flex : DisplayStyle.None;
                _serverBtn.text = ServerMode ? "server mode" : "standalone";
                SetOn(_serverBtn, ServerMode);
            }
            bool server = ServerMode;
            if (server == _serverShown || _left == null) return;
            _serverShown = server;
            _left.style.display = server ? DisplayStyle.None : DisplayStyle.Flex;
            if (server && !string.IsNullOrEmpty(_follower.LastError)) Toast(_follower.LastError);
        }

        void SetPanels(bool on)
        {
            panelsVisible = on;
            var d = on ? DisplayStyle.Flex : DisplayStyle.None;
            // the middle row keeps its place (visibility, not display) so the bottom bar stays at the bottom
            if (_left != null) _left.style.visibility = on ? Visibility.Visible : Visibility.Hidden;
            if (_right != null) _right.style.visibility = on ? Visibility.Visible : Visibility.Hidden;
            if (_bottom != null) _bottom.style.display = d;
            if (_showPanels != null) _showPanels.style.display = on ? DisplayStyle.None : DisplayStyle.Flex;
        }

        public void Toast(string msg, float seconds = 5f)
        {
            if (_toast == null) return;
            _toastBody.Clear();
            if (msg.Length > 400) msg = msg.Substring(0, 400) + "...";
            foreach (var line in WrapText(msg, 96)) { var l = L(line, 13); l.pickingMode = PickingMode.Ignore; _toastBody.Add(l); }
            _toast.style.display = DisplayStyle.Flex;
            _toastUntil = Time.unscaledTime + seconds;
        }

        // ------------------------------------------------------------------ shortcuts (help table + handling)
        struct Key { public string keys, what; public Key(string k, string w) { keys = k; what = w; } }
        static readonly Key[] Keys =
        {
            new Key("Space", "play / pause"), new Key("Left / Right", "previous / next frame"),
            new Key("Shift + Left / Right", "back / forward 1 s"), new Key("Home / End", "range start / end"),
            new Key("[ / ]", "slower / faster"), new Key("L", "loop on / off"), new Key("I / O", "A-B range start / end"),
            new Key("P", "hide / show the panels"), new Key("? or F1", "this help"),
            new Key("C", "fly mode on / off"), new Key("W A S D / Q E", "fly: move (hold the right mouse button to look)"),
            new Key("F / R", "frame the players / reset the camera"),
            new Key("J", "joint colour mode"), new Key("G", "ghost builds on / off"),
            new Key("Y / K / V / N", "rays / cameras / video / non-players"),
            new Key("1 - 5", "look through a camera (Esc: back)"), new Key("H", "debug HUD on / off"),
            new Key("Enter / Esc", "stop typing in a text field (or click the 3D view)"),
        };

        void HandleKeys()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (kb == null || player == null) return;
            if (Typing) return;                                              // the letters belong to the text field
            bool shift = kb.leftShiftKey.isPressed || kb.rightShiftKey.isPressed;
            if (kb.spaceKey.wasPressedThisFrame) player.TogglePlay();
            if (kb.leftArrowKey.wasPressedThisFrame) { if (shift) Jump(-1f); else player.StepFrame(-1); }
            if (kb.rightArrowKey.wasPressedThisFrame) { if (shift) Jump(1f); else player.StepFrame(1); }
            if (kb.homeKey.wasPressedThisFrame) player.SeekFrame(player.RangeStart);
            if (kb.endKey.wasPressedThisFrame) player.SeekFrame(player.RangeEnd);
            if (kb.leftBracketKey.wasPressedThisFrame) player.speed = Mathf.Max(0.1f, player.speed * 0.5f);
            if (kb.rightBracketKey.wasPressedThisFrame) player.speed = Mathf.Min(3f, player.speed * 2f);
            if (kb.lKey.wasPressedThisFrame) player.loop = !player.loop;
            if (kb.iKey.wasPressedThisFrame) SetA();
            if (kb.oKey.wasPressedThisFrame) SetB();
            if (kb.pKey.wasPressedThisFrame) SetPanels(!panelsVisible);
            if (kb.f1Key.wasPressedThisFrame || (shift && kb.slashKey.wasPressedThisFrame)) ToggleHelp();
            if (kb.escapeKey.wasPressedThisFrame && _help.style.display == DisplayStyle.Flex) ToggleHelp();
            if (player.Doc == null) return;
            Cache();
            bool has = player.CvDebug != null;
            // J / Y, not C / R: OrbitCamera already uses C (fly mode), R (reset), F (frame)
            if (kb.jKey.wasPressedThisFrame && _ov != null) _ov.mode = CvColors.Next(_ov.mode, has);
            if (kb.gKey.wasPressedThisFrame && _ghosts != null && _ghosts.Candidates.Count > 0)
            {
                bool any = _ghosts.Candidates.Any(_ghosts.IsOn);
                foreach (var c in _ghosts.Candidates.ToList()) _ghosts.Set(c, false);
                if (!any) _ghosts.Set(_ghosts.Candidates[0], true);
                _dirtySections = true;
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

        // ------------------------------------------------------------------ clip browser
        void RebuildClipList()
        {
            if (_clipList == null) return;
            ScanClips();
            _clipList.Clear();
            foreach (var g in _clips.GroupBy(c => c.group).OrderBy(g => g.Key))
            {
                var gl = L(g.Key, 13, true); gl.style.marginTop = 8; gl.style.color = Accent; _clipList.Add(gl);
                foreach (var c in g.OrderBy(c => c.variant))
                {
                    // a plain clickable panel, not a Button: a Button is a text element and ignores its children's height
                    string path = c.path;
                    var card = new VisualElement { userData = path };
                    card.AddManipulator(new Clickable(() => { Unfocus(); player.LoadClip(path); }));
                    card.style.backgroundColor = BtnBg; card.style.flexDirection = FlexDirection.Column; card.style.flexShrink = 0;
                    card.style.paddingLeft = 8; card.style.paddingRight = 8; card.style.paddingTop = 5; card.style.paddingBottom = 5;
                    card.style.marginTop = 2; card.style.marginBottom = 3; card.style.marginRight = 4;
                    card.style.borderLeftWidth = 3; card.style.borderLeftColor = Color.clear;     // accent while this clip loads
                    Round(card, 4);
                    card.RegisterCallback<PointerEnterEvent>(_ => { if (!IsOn(card)) card.style.backgroundColor = BtnHover; });
                    card.RegisterCallback<PointerLeaveEvent>(_ => { if (!IsOn(card)) card.style.backgroundColor = BtnBg; });
                    card.Add(L(string.IsNullOrEmpty(c.variant) ? c.file.Replace(".analysis.json", "") : c.variant, 13, true));
                    var info = Small($"{c.frames} frames   {c.fps:0.#} fps   {c.bytes / 1048576f:0.0} MB   {c.modified:dd MMM HH:mm}"
                                     + (c.live ? "   live run" : "") + (c.bundled ? "   1 file" : ""));
                    card.Add(info);
                    if (!string.IsNullOrEmpty(c.label)) card.Add(Small(c.label));
                    card.tooltip = c.path;
                    _clipList.Add(card);
                }
            }
            if (_clips.Count == 0) _clipList.Add(Small($"no *.analysis.json in StreamingAssets/{analysisFolder}"));
            MarkCurrentClip();
        }

        void MarkCurrentClip()
        {
            if (_clipList == null || player == null) return;
            foreach (var e in _clipList.Children())
                if (e.userData is string p)
                {
                    SetOn(e, p == player.analysisFile);
                    e.style.borderLeftColor = p == player.LoadingFile ? Accent : Color.clear;
                }
        }

        void ScanClips()
        {
            _clips.Clear();
            string dir = Path.Combine(Application.streamingAssetsPath, analysisFolder);
            if (!Directory.Exists(dir)) return;
            foreach (var full in Directory.GetFiles(dir, "*.analysis.json"))
            {
                var fi = new FileInfo(full);
                string file = fi.Name, rel = analysisFolder + "/" + file;
                string grp = Path.GetFileName(AnalysisPlayer.BaseClipPath(rel)).Replace(".analysis.json", "");
                string variant = "";
                for (int i = 0; i < AnalysisPlayer.PoseVariantSuffixes.Length; i++)
                    if (file.EndsWith(AnalysisPlayer.PoseVariantSuffixes[i] + ".analysis.json")) variant = AnalysisPlayer.PoseVariantLabels[i];
                var e = new ClipEntry { path = rel, file = file, group = grp, variant = variant, bytes = fi.Length, modified = fi.LastWriteTime };
                try
                {
                    string head;
                    using (var r = new StreamReader(full)) { var buf = new char[65536]; int n = r.Read(buf, 0, buf.Length); head = new string(buf, 0, n); }
                    var m = ReFps.Match(head); if (m.Success) float.TryParse(m.Groups[1].Value, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out e.fps);
                    m = ReFrames.Match(head); if (m.Success) int.TryParse(m.Groups[1].Value, out e.frames);
                    m = ReLabel.Match(head); if (m.Success) e.label = Regex.Unescape(m.Groups[1].Value);
                    e.live = head.Contains("\"name\": \"live\"") || head.Contains("simulated live");
                    e.bundled = head.Contains("\"bundle\"");
                }
                catch (IOException) { }
                _clips.Add(e);
            }
        }

        // ------------------------------------------------------------------ right-hand sections
        CvDebugOverlay _ov; CameraRigGizmos _rig; CameraRayLines _rays; VideoBillboard _video; NonPlayerMarkers _np; VariantGhosts _ghosts;
        AnalysisDebugHUD _hud;
        Label _camStatus;
        readonly List<(Label head, Label pos)> _cards = new List<(Label, Label)>();
        readonly List<(VisualElement b, Func<bool> on)> _syncs = new List<(VisualElement, Func<bool>)>();

        void Cache()
        {
            if (_ov == null) _ov = player.GetComponent<CvDebugOverlay>();
            if (_rig == null) _rig = player.GetComponent<CameraRigGizmos>();
            if (_rays == null) _rays = player.GetComponent<CameraRayLines>();
            if (_video == null) _video = player.GetComponent<VideoBillboard>();
            if (_np == null) _np = player.GetComponent<NonPlayerMarkers>();
            if (_ghosts == null) _ghosts = player.GetComponent<VariantGhosts>();
            if (_hud == null) _hud = FindAnyObjectByType<AnalysisDebugHUD>();
        }

        /// <summary>The shared selected camera: the SelectedView colour mode and the video plane.</summary>
        void SetView(int v)
        {
            var cv = player.CvDebug;
            if (cv == null || cv.V == 0) return;
            _cvView = ((v % cv.V) + cv.V) % cv.V;
            if (_ov != null) _ov.selectedView = _cvView;
            if (_video != null) _video.planeView = _cvView;
        }

        /// <summary>Every toggle button follows its state each frame, so keys (J, Y, K, V, N, H, G...) and code show on the buttons.</summary>
        void SyncToggles()
        {
            foreach (var (b, on) in _syncs) SetOn(b, on());
        }

        void RebuildSections()
        {
            Cache();
            _syncs.Clear();
            BuildBuildsSection();
            BuildRacketShuttleSection();
            BuildCamerasSection();
            BuildDisplaySection();
            _secPlayers.Clear(); _cards.Clear();
            for (int i = 0; i < player.PlayerCount; i++)
            {
                var a = L("", 12); var b = Small(""); b.style.marginBottom = 3;
                _cards.Add((a, b)); _secPlayers.Add(a); _secPlayers.Add(b);
            }
            BuildDebugSection();
            _timeline.Rebuild(player);
        }

        void BuildBuildsSection()
        {
            _secBuilds.Clear();
            string grp = Path.GetFileName(AnalysisPlayer.BaseClipPath(player.analysisFile ?? "")).Replace(".analysis.json", "");
            var same = _clips.Where(c => c.group == grp).OrderBy(c => c.variant).ToList();
            var row = Flow();
            foreach (var c in same)
            {
                string path = c.path;
                var b = Btn(string.IsNullOrEmpty(c.variant) ? "base" : c.variant, () => player.LoadClipKeepingInstant(path), c.file);
                _syncs.Add((b, () => path == player.analysisFile)); row.Add(b);
            }
            if (same.Count > 0) _secBuilds.Add(row);
            _secBuilds.Add(Small(player.Doc.DisplayLabel));
            if (_ghosts != null && _ghosts.Candidates.Count > 0)
            {
                _secBuilds.Add(Heading("ghost builds (G)"));
                foreach (var g in _ghosts.Candidates.ToList())
                {
                    string gp = g;
                    var r = Row();
                    var sw = new VisualElement(); sw.style.width = 10; sw.style.height = 10; sw.style.marginRight = 6; sw.style.flexShrink = 0;
                    var b = Btn(Path.GetFileName(gp).Replace(".analysis.json", ""), () => { _ghosts.Set(gp, !_ghosts.IsOn(gp)); _dirtySections = true; });
                    b.style.flexShrink = 1; b.style.minWidth = 0; b.style.overflow = Overflow.Hidden;
                    // a ghost build is read on a worker thread (VariantGhosts, v0.8.1): "loading ..." beside its button until
                    // it is drawn, then its swatch takes the line colour
                    var ld = L("loading ...", 11); ld.style.color = Accent; ld.style.marginLeft = 4; ld.pickingMode = PickingMode.Ignore;
                    bool? shownLoading = null;
                    _syncs.Add((b, () =>
                    {
                        bool loading = _ghosts.IsLoading(gp);
                        if (shownLoading != loading)
                        {
                            shownLoading = loading;
                            ld.style.display = loading ? DisplayStyle.Flex : DisplayStyle.None;
                            sw.style.backgroundColor = _ghosts.TryGetColor(gp, out var col) ? col : new Color(0, 0, 0, 0);
                        }
                        return _ghosts.IsOn(gp);
                    }));
                    r.Add(sw); r.Add(b); r.Add(ld); _secBuilds.Add(r);
                }
                if (!string.IsNullOrEmpty(_ghosts.Note)) _secBuilds.Add(Small(_ghosts.Note));
            }
        }

        void BuildRacketShuttleSection()
        {
            _secRS.Clear();
            var doc = player.Doc;
            bool v1 = doc.players.Any(p => Enumerable.Range(0, p.FrameCount).Any(p.HasRacket));
            var sidecars = AnalysisPlayer.RacketVariantNames.Where(v => File.Exists(Path.Combine(Application.streamingAssetsPath,
                               AnalysisPlayer.RacketSidecarPath(player.analysisFile ?? "", v)))).ToList();
            if (v1 || sidecars.Count > 0)
            {
                _secRS.Add(Heading("racket"));
                var r = Flow();
                foreach (var v in new[] { RacketVisualAnalysis.V1 }.Concat(sidecars))
                {
                    string variant = v;
                    var b = Btn(v, () => player.SetRacketVariant(variant), v == RacketVisualAnalysis.V1 ? "racket stored in the analysis file" : $"sidecar .racket_{v}.json");
                    _syncs.Add((b, () => player.EffectiveRacketVariant == variant)); r.Add(b);
                }
                if (sidecars.Count > 0)
                {
                    r.Add(Toggle("ghost v1", () => player.GhostV1, x => player.GhostV1 = x, "the stored racket as a ghost"));
                    r.Add(Toggle("snap to hand", () => player.SnapRacketToHand, x => player.SnapRacketToHand = x));
                }
                _secRS.Add(r);
                var rd = sidecars.Count > 0 ? player.GetRacketVariantDoc(player.EffectiveRacketVariant) : null;
                if (rd != null && !string.IsNullOrEmpty(rd.time_base)) _secRS.Add(Small($"{rd.time_base} time"));
            }
            var shuttles = AnalysisPlayer.ShuttleVariantNames.Where(v => player.GetShuttleVariantDoc(v) != null).ToList();
            bool baseline = doc.BaselineShuttle?.frames != null && doc.BaselineShuttle.frames.Length > 0;
            if (baseline || shuttles.Count > 0)
            {
                _secRS.Add(Heading("shuttle"));
                var r = Flow();
                foreach (var v in new[] { ShuttleVisualAnalysis.Baseline }.Concat(shuttles))
                {
                    string variant = v;
                    var b = Btn(v, () => { player.SetShuttleVariant(variant); _dirtySections = true; });
                    _syncs.Add((b, () => player.EffectiveShuttleVariant == variant)); r.Add(b);
                }
                if (shuttles.Count > 0)
                {
                    r.Add(Toggle("ghost", () => player.GhostShuttleBaseline, x => player.GhostShuttleBaseline = x, "the baseline track as a ghost"));
                    r.Add(Toggle("trail", () => player.ShuttleTrail, x => player.ShuttleTrail = x));
                }
                _secRS.Add(r);
                _secRS.Add(Small(ShuttleStats(player.EffectiveShuttleVariant)));
            }
            if (_secRS.childCount == 0) _secRS.Add(Small("No racket or shuttle data in this clip."));
        }

        string ShuttleStats(string active)
        {
            var d = player.GetShuttleVariantDoc(active);
            if (d != null && d.stats != null)
                return d.stats.reproj_rms_px >= 0f ? $"coverage {d.stats.coverage * 100f:0}%   reprojection {d.stats.reproj_rms_px:0.0} px"
                                                   : $"coverage {d.stats.coverage * 100f:0}%";
            var s = player.Doc.BaselineShuttle;
            return s != null ? $"coverage {s.Coverage() * 100f:0}%" : "";
        }

        void BuildCamerasSection()
        {
            _secCams.Clear();
            var cv = player.CvDebug;
            if (_ov != null)
            {
                Button cm = null;
                cm = Btn($"joint colour: {_ov.mode}", () => { _ov.mode = CvColors.Next(_ov.mode, player.CvDebug != null); }, "joint colour mode (J)");
                cm.style.unityTextAlign = TextAnchor.MiddleLeft;
                _syncs.Add((cm, () => { cm.text = $"joint colour: {_ov.mode}"; return _ov.mode != CvColorMode.Off; }));
                _secCams.Add(cm);
            }
            if (cv == null)
            {
                _secCams.Add(Small("No camera debug data in this file."));
                _secCams.Add(Small("Run analysis.cvdebug build, then analysis.bundle."));
                return;
            }
            var t = Flow();
            if (_ov != null) t.Add(Toggle("aux", () => _ov.showAux, v => _ov.showAux = v, "heels, toes, finger knuckles"));
            if (_rays != null) t.Add(Toggle("rays", () => _rays.show, v => _rays.show = v, "camera rays (Y)"));
            if (_rig != null) t.Add(Toggle("cameras", () => _rig.show, v => _rig.show = v, "camera frusta (K)"));
            if (_video != null) t.Add(Toggle("video", () => _video.show, v => _video.show = v, "synced video plane of the selected camera (V)"));
            if (_np != null && cv.persons != null && cv.persons.Length > 0)
                t.Add(Toggle("non-players", () => _np.show, v => _np.show = v, "people who are not players (N)"));
            _secCams.Add(t);
            _secCams.Add(Heading("selected camera (keys 1-5 look through)"));
            var lr = Flow();
            for (int v = 0; v < cv.V; v++)
            {
                int vi = v;
                var b = Btn($"{v + 1} {cv.cameras[v].name}", () => SetView(vi), "select this camera");
                _syncs.Add((b, () => _cvView == vi)); lr.Add(b);
            }
            _secCams.Add(lr);
            if (_rig != null)
            {
                var look = Btn("look through the selected camera", () =>
                {
                    if (_rig.LookingThrough >= 0) _rig.ExitLookThrough(); else _rig.LookThrough(_cvView);
                }, "1-5 for a camera, Esc to return");
                look.style.unityTextAlign = TextAnchor.MiddleLeft;
                _syncs.Add((look, () => _rig.LookingThrough >= 0));
                _secCams.Add(look);
            }
            _camStatus = Small(""); _secCams.Add(_camStatus);
            _syncs.Add((_camStatus, () =>
            {
                bool proxy = _video != null && _video.HasProxy(_cvView);
                _camStatus.text = _video != null && _video.show ? (proxy ? _video.Status : $"no video proxy (overlay_video proxies --clip {cv.ProxyClip})")
                                : _np != null && _np.show ? $"{_np.VisibleCount} non-players shown" : "";
                return false;
            }));
        }

        void BuildDisplaySection()
        {
            _secDisplay.Clear();
            int nP = player.PlayerCount;
            var fr = Flow();
            var all = Btn("all", () => { player.focusPlayer = -1; _dirtySections = true; }, "show every player");
            _syncs.Add((all, () => player.focusPlayer < 0)); fr.Add(all);
            for (int i = 0; i < nP; i++)
            {
                int pi = i;
                var b = Btn(PlayerName(i), () => { player.focusPlayer = pi; _dirtySections = true; }, "show only this player");
                _syncs.Add((b, () => player.focusPlayer == pi)); fr.Add(b);
            }
            _secDisplay.Add(Heading("players")); _secDisplay.Add(fr);
            _secDisplay.Add(Heading("skin"));
            for (int i = 0; i < nP; i++)
            {
                int pi = i;
                var sr = Row(); var nl = L(PlayerName(i), 12); nl.style.width = 34; nl.style.flexShrink = 0; sr.Add(nl);
                foreach (AnalysisPlayer.SkinMode m in Enum.GetValues(typeof(AnalysisPlayer.SkinMode)))
                {
                    var mode = m;
                    var b = Btn(m.ToString().ToLowerInvariant(), () => player.SetSkin(pi, mode));
                    _syncs.Add((b, () => player.GetSkin(pi) == mode)); sr.Add(b);
                }
                _secDisplay.Add(sr);
            }
            var lay = Flow();
            lay.Add(Toggle("pose", () => player.showPose, v => player.showPose = v));
            lay.Add(Toggle("racket", () => player.showRacket, v => player.showRacket = v));
            lay.Add(Toggle("shuttle", () => player.showShuttle, v => player.showShuttle = v));
            lay.Add(Toggle("on floor", () => player.LiftToFloor, v => player.LiftToFloor = v, "lift the bodies so the lowest joint touches the floor"));
            _secDisplay.Add(Heading("layers")); _secDisplay.Add(lay);
        }

        void BuildDebugSection()
        {
            _secDebug.Clear();
            if (_hud == null) { _secDebug.Add(Small("No AnalysisDebugHUD in the scene.")); _debugLines = null; return; }
            var t = Flow();
            t.Add(Toggle("HUD on", () => _hud.master, v => _hud.master = v, "stroke bubbles, world axes, confidence colour and these stats (H)"));
            t.Add(Toggle("stats", () => _hud.showStats, v => _hud.showStats = v, "fps, clip, skins, cv state"));
            t.Add(Toggle("provenance", () => _hud.showProvenance, v => _hud.showProvenance = v, "producer of each block"));
            t.Add(Toggle("shuttle / stroke", () => _hud.showStatus, v => _hud.showStatus = v));
            t.Add(Toggle("contract", () => _hud.showWarnings, v => _hud.showWarnings = v, "schema / frame-count checks"));
            t.Add(Toggle("confidence colour", () => _hud.colorByConfidence, v => _hud.colorByConfidence = v, "skeleton skin only"));
            t.Add(Toggle("world axes", () => _hud.showAxes, v => _hud.showAxes = v));
            t.Add(Toggle("log", () => _hud.showLog, v => _hud.showLog = v, "the last log messages"));
            _secDebug.Add(t);
            _debugLines = new VisualElement(); _debugLines.style.marginTop = 4; _debugLines.style.flexShrink = 0;
            _secDebug.Add(_debugLines);
        }

        float _debugAt;

        /// <summary>The debug HUD's text, wrapped into one-line labels (refreshed 5 times a second while the section is open).
        /// H turns the HUD on and opens + scrolls to this section.</summary>
        void UpdateDebug()
        {
            if (_hud == null || _debugLines == null) return;
            if (_hud.master != _lastHudMaster)
            {
                _lastHudMaster = _hud.master;
                if (_hud.master)
                {
                    OpenSection(_secDebug, _secDebugHead, true);
                    _rightScroll.schedule.Execute(() => _rightScroll.scrollOffset = new Vector2(0f, _secDebugHead.layout.y)).ExecuteLater(60);
                }
                _debugAt = 0f;
            }
            if (_secDebug.style.display == DisplayStyle.None || Time.unscaledTime < _debugAt) return;
            _debugAt = Time.unscaledTime + 0.2f;
            var lines = new List<(string text, bool log)>();
            if (!_hud.master) lines.Add(("HUD is off: press H or \"HUD on\".", false));
            else
            {
                foreach (var l in _hud.StatsText().Split('\n')) foreach (var w in WrapText(l.TrimEnd('\r'), 46)) lines.Add((w, false));
                if (_hud.showLog)
                    foreach (var l in _hud.RecentLog) foreach (var w in WrapText(l, 46)) lines.Add((w, true));
            }
            while (_debugLines.childCount < lines.Count) _debugLines.Add(L("", 11));
            for (int i = 0; i < _debugLines.childCount; i++)
            {
                var lab = (Label)_debugLines[i];
                bool on = i < lines.Count;
                lab.style.display = on ? DisplayStyle.Flex : DisplayStyle.None;
                if (!on) continue;
                lab.text = lines[i].text;
                lab.style.color = lines[i].log ? new Color(0.75f, 0.85f, 1f, 1f) : Dim;
            }
        }

        string PlayerName(int i)
        {
            string id = player.Doc?.players != null && i < player.Doc.players.Length ? player.Doc.players[i].id : null;
            return string.IsNullOrEmpty(id) ? $"P{i + 1}" : id;
        }

        void UpdatePlayerCards()
        {
            var doc = player.Doc;
            for (int i = 0; i < _cards.Count && i < doc.players.Length; i++)
            {
                var p = doc.players[i]; int f = Mathf.Clamp(player.CurrentFrame, 0, p.FrameCount - 1);
                var fr = p.frames[f];
                string state = !p.HasPose(f) ? "no pose" : fr.interpolated ? "interpolated" : "seen";
                float conf = 0f; int nj = p.PoseJointCount(f);
                for (int j = 0; j < nj; j++) conf += p.JointConf(f, j);
                conf = nj > 0 ? conf / nj : 0f;
                string pos = fr.position != null && fr.position.Length >= 2 ? $"court x {fr.position[0]:+0.00;-0.00} m   z {fr.position[1]:+0.00;-0.00} m" : "no position";
                _cards[i].head.text = $"<b>{PlayerName(i)}</b>   {p.side} half   {state}";
                _cards[i].pos.text = $"confidence {conf:0.00}   {pos}";
            }
        }

        // ------------------------------------------------------------------ small UI helpers (inline styles)
        /// <summary>One-line label at a fixed line height: UI Toolkit measures this runtime font's lines ~30 % short, so
        /// auto-height or wrapped labels overlap whatever follows them. Long text ends in an ellipsis (full text in the tooltip
        /// where it matters); multi-line text is split into several labels (WrapText).</summary>
        static Label L(string t, int size = 13, bool bold = false)
        {
            var l = new Label(t) { enableRichText = true };
            l.style.fontSize = size; l.style.color = Text;
            l.style.minHeight = Mathf.Ceil(size * 1.45f);
            l.style.whiteSpace = WhiteSpace.NoWrap; l.style.overflow = Overflow.Hidden;
            l.style.textOverflow = TextOverflow.Ellipsis;
            l.style.marginLeft = 0; l.style.marginRight = 0; l.style.paddingLeft = 0; l.style.paddingRight = 0;
            l.style.flexShrink = 0;
            if (bold) l.style.unityFontStyleAndWeight = FontStyle.Bold;
            l.style.unityTextAlign = TextAnchor.MiddleLeft;
            return l;
        }

        static Label Small(string t) { var l = L(t, 11); l.style.color = Dim; l.style.marginTop = 1; l.tooltip = t; return l; }

        static Label Heading(string t) { var l = L(t, 12, true); l.style.marginTop = 6; return l; }

        /// <summary>Split text into lines of at most maxChars, at spaces where possible.</summary>
        public static List<string> WrapText(string text, int maxChars)
        {
            var outLines = new List<string>();
            foreach (var raw in (text ?? "").Split('\n'))
            {
                string s = raw.TrimEnd();
                while (s.Length > maxChars)
                {
                    int cut = s.LastIndexOf(' ', maxChars);
                    if (cut < maxChars / 2) cut = maxChars;
                    outLines.Add(s.Substring(0, cut).TrimEnd());
                    s = "  " + s.Substring(cut).TrimStart();
                }
                outLines.Add(s);
            }
            return outLines;
        }

        static void Round(VisualElement v, float r)
        {
            v.style.borderTopLeftRadius = r; v.style.borderTopRightRadius = r; v.style.borderBottomLeftRadius = r; v.style.borderBottomRightRadius = r;
        }

        static VisualElement Panel()
        {
            var v = new VisualElement();
            v.style.backgroundColor = Bg;
            Round(v, 6);
            v.style.borderTopWidth = 1; v.style.borderBottomWidth = 1; v.style.borderLeftWidth = 1; v.style.borderRightWidth = 1;
            v.style.borderTopColor = Line; v.style.borderBottomColor = Line; v.style.borderLeftColor = Line; v.style.borderRightColor = Line;
            return v;
        }

        /// <summary>A side panel of the middle row: fixed width, full height of the row, its own column layout.</summary>
        static VisualElement Dock(float width)
        {
            var v = Panel();
            v.style.width = width; v.style.flexShrink = 0; v.style.flexDirection = FlexDirection.Column; v.style.minHeight = 0;
            v.style.paddingLeft = Pad; v.style.paddingRight = 6; v.style.paddingTop = 8; v.style.paddingBottom = 8;
            v.style.overflow = Overflow.Hidden;
            return v;
        }

        /// <summary>A layout-only container: never picks the pointer, so the 3D view behind it gets the clicks.</summary>
        static VisualElement Layout(FlexDirection dir)
        {
            var v = new VisualElement { pickingMode = PickingMode.Ignore };
            v.style.flexDirection = dir;
            return v;
        }

        static void StyleScroll(ScrollView sv)
        {
            sv.mode = ScrollViewMode.Vertical;
            sv.horizontalScrollerVisibility = ScrollerVisibility.Hidden;
            sv.verticalScrollerVisibility = ScrollerVisibility.Auto;
            sv.mouseWheelScrollSize = 40;
            sv.style.minHeight = 0;
            sv.contentContainer.style.flexShrink = 0;
            sv.contentContainer.style.paddingRight = 4;
            var bar = sv.verticalScroller;                     // a thin dark track: slider only
            bar.style.width = 8;
            bar.style.borderLeftWidth = 0;
            bar.style.backgroundColor = new Color(1f, 1f, 1f, 0.04f);
            foreach (var b in bar.Query<RepeatButton>().ToList()) b.style.display = DisplayStyle.None;
            var slider = bar.slider;
            if (slider != null)
            {
                slider.style.marginTop = 0; slider.style.marginBottom = 0;
                var tracker = slider.Q(className: "unity-base-slider__tracker");
                if (tracker != null) { tracker.style.backgroundColor = new Color(0, 0, 0, 0); tracker.style.borderTopWidth = 0; tracker.style.borderBottomWidth = 0; tracker.style.borderLeftWidth = 0; tracker.style.borderRightWidth = 0; }
                var drag = slider.Q(className: "unity-base-slider__dragger");
                if (drag != null) { drag.style.backgroundColor = new Color(1f, 1f, 1f, 0.28f); drag.style.borderTopWidth = 0; drag.style.borderBottomWidth = 0; drag.style.borderLeftWidth = 0; drag.style.borderRightWidth = 0; Round(drag, 3); }
            }
        }

        static VisualElement Row() { var r = new VisualElement(); r.style.flexDirection = FlexDirection.Row; r.style.alignItems = Align.Center; r.style.flexShrink = 0; return r; }
        static VisualElement Flow() { var r = Row(); r.style.flexWrap = Wrap.Wrap; r.style.marginBottom = 2; return r; }
        static VisualElement Spacer() { var s = new VisualElement { pickingMode = PickingMode.Ignore }; s.style.flexGrow = 1; s.style.flexShrink = 1; return s; }
        static VisualElement Gap(float w) { var s = new VisualElement { pickingMode = PickingMode.Ignore }; s.style.width = w; s.style.flexShrink = 0; return s; }

        static void StyleButton(Button b)
        {
            b.style.backgroundColor = BtnBg; b.style.color = Text; b.style.fontSize = 12;
            b.style.height = 24; b.style.minWidth = 28; b.style.paddingLeft = 8; b.style.paddingRight = 8;
            b.style.paddingTop = 0; b.style.paddingBottom = 0;
            b.style.marginLeft = 2; b.style.marginRight = 2; b.style.marginTop = 2; b.style.marginBottom = 2;
            b.style.borderTopWidth = 0; b.style.borderBottomWidth = 0; b.style.borderLeftWidth = 0; b.style.borderRightWidth = 0;
            Round(b, 4);
            b.style.unityTextAlign = TextAnchor.MiddleCenter;
            b.style.flexShrink = 0; b.style.whiteSpace = WhiteSpace.NoWrap;     // never squeezed into two lines
            b.style.textOverflow = TextOverflow.Ellipsis;
            b.focusable = false;                                                  // a click never steals the keyboard
            b.RegisterCallback<PointerEnterEvent>(_ => { if (!IsOn(b)) b.style.backgroundColor = BtnHover; });
            b.RegisterCallback<PointerLeaveEvent>(_ => { if (!IsOn(b)) b.style.backgroundColor = BtnBg; });
        }

        static Button Btn(string text, Action onClick, string tip = null)
        {
            var b = new Button(onClick) { text = text };
            StyleButton(b);
            if (tip != null) b.tooltip = tip;
            return b;
        }

        static readonly HashSet<VisualElement> _on = new HashSet<VisualElement>();
        static bool IsOn(VisualElement b) => _on.Contains(b);
        static void SetOn(VisualElement b, bool on)
        {
            if (b == null || b is Label) return;
            if (on) _on.Add(b); else _on.Remove(b);
            b.style.backgroundColor = on ? BtnOn : BtnBg;
        }

        /// <summary>A button that flips a bool; its look follows the state every frame (SyncToggles).</summary>
        Button Toggle(string text, Func<bool> get, Action<bool> set, string tip = null)
        {
            var b = Btn(text, () => set(!get()), tip);
            _syncs.Add((b, get));
            SetOn(b, get());
            return b;
        }

        /// <summary>A single-line text field with a placeholder. Enter or Esc ends typing; the field shows it has the keyboard
        /// with a blue border.</summary>
        static TextField Field(string placeholder, string tip)
        {
            var f = new TextField { tooltip = tip };
            f.textEdition.placeholder = placeholder;
            f.textEdition.hidePlaceholderOnFocus = true;
            f.style.marginLeft = 0; f.style.marginRight = 4; f.style.marginTop = 6; f.style.marginBottom = 4;
            f.style.height = 26; f.style.flexShrink = 0;
            var input = f.Q(className: TextField.inputUssClassName) ?? f;
            input.style.backgroundColor = Bg2; input.style.color = Text;
            input.style.borderTopWidth = 1; input.style.borderBottomWidth = 1; input.style.borderLeftWidth = 1; input.style.borderRightWidth = 1;
            SetBorder(input, Line);
            Round(input, 4);
            // the theme pads the input ~9 px top and bottom, which leaves the text line ~2 px tall (text cut in half)
            input.style.paddingLeft = 6; input.style.paddingRight = 6; input.style.paddingTop = 0; input.style.paddingBottom = 0;
            input.style.overflow = Overflow.Hidden; input.style.whiteSpace = WhiteSpace.NoWrap;
            input.style.justifyContent = Justify.Center;
            foreach (var te in f.Query<TextElement>().ToList())
            {
                te.style.whiteSpace = WhiteSpace.NoWrap; te.style.fontSize = 12; te.style.color = Text;
                te.style.flexGrow = 1; te.style.minHeight = 18; te.style.unityTextAlign = TextAnchor.MiddleLeft;
                te.style.marginTop = 0; te.style.marginBottom = 0; te.style.paddingTop = 0; te.style.paddingBottom = 0;
            }
            f.RegisterCallback<FocusInEvent>(_ => SetBorder(input, Accent));
            f.RegisterCallback<FocusOutEvent>(_ => SetBorder(input, Line));
            f.RegisterCallback<KeyDownEvent>(e =>
            {
                if (e.keyCode == KeyCode.Escape || e.keyCode == KeyCode.Return || e.keyCode == KeyCode.KeypadEnter)
                    f.schedule.Execute(() => f.Blur());
            }, TrickleDown.TrickleDown);
            return f;
        }

        static void SetBorder(VisualElement v, Color c)
        {
            v.style.borderTopColor = c; v.style.borderBottomColor = c; v.style.borderLeftColor = c; v.style.borderRightColor = c;
        }

        static void OpenSection(VisualElement body, Button head, bool open)
        {
            if (body == null || head == null) return;
            body.style.display = open ? DisplayStyle.Flex : DisplayStyle.None;
            string title = head.text.Length > 2 ? head.text.Substring(2) : head.text;
            head.text = (open ? "- " : "+ ") + title;
        }

        VisualElement Section(VisualElement parent, string title, bool open, out Button head)
        {
            var h = Btn((open ? "- " : "+ ") + title, null);
            h.style.unityTextAlign = TextAnchor.MiddleLeft; h.style.unityFontStyleAndWeight = FontStyle.Bold;
            h.style.backgroundColor = Bg2; h.style.marginTop = 6; h.style.marginLeft = 0; h.style.marginRight = 0;
            var body = new VisualElement(); body.style.paddingLeft = 4; body.style.paddingTop = 4; body.style.paddingBottom = 2; body.style.flexShrink = 0;
            body.style.display = open ? DisplayStyle.Flex : DisplayStyle.None;
            h.clicked += () => OpenSection(body, h, body.style.display == DisplayStyle.None);
            parent.Add(h); parent.Add(body);
            head = h;
            return body;
        }

        // ------------------------------------------------------------------ timeline
        /// <summary>Scrub bar: per-pixel-column quality markers (live docs: meta.live.frame_quality; else interpolated / no
        /// pose from the frames), the A-B range and the playhead. Click or drag to seek.</summary>
        public class TimelineBar : VisualElement
        {
            readonly AnalysisViewerUITK _ui;
            Color[] _cols = new Color[0];
            int _frames;
            bool _drag;

            public TimelineBar(AnalysisViewerUITK ui)
            {
                _ui = ui;
                style.height = 22; style.marginTop = 6; style.backgroundColor = Bg2; style.flexShrink = 0;
                Round(this, 3);
                generateVisualContent += Draw;
                RegisterCallback<PointerDownEvent>(e => { _drag = true; this.CapturePointer(e.pointerId); Seek(e.localPosition.x); });
                RegisterCallback<PointerMoveEvent>(e => { if (_drag) Seek(e.localPosition.x); });
                RegisterCallback<PointerUpEvent>(e => { _drag = false; this.ReleasePointer(e.pointerId); });
                tooltip = "click or drag to seek; markers: frame quality";
            }

            void Seek(float x)
            {
                var p = _ui.player;
                if (p == null || p.FrameCount == 0 || contentRect.width <= 0) return;
                p.Pause();
                p.SeekNormalized(Mathf.Clamp01(x / contentRect.width));
            }

            /// <summary>Per-frame marker colour (transparent = nothing to mark).</summary>
            public void Rebuild(AnalysisPlayer p)
            {
                var doc = p != null ? p.Doc : null;
                if (doc == null) { _cols = new Color[0]; _frames = 0; return; }
                _frames = doc.FrameCount;
                _cols = new Color[_frames];
                if (p.IsLive) { MarkDirtyRepaint(); return; }      // live: drawn straight from player.LiveQuality
                var live = doc.meta?.live;
                if (live?.ref_frames != null && live.frame_quality != null && live.ref_frames.Length == live.frame_quality.Length)
                {
                    int stride = live.ref_frames.Length > 1 ? Mathf.Max(1, live.ref_frames[1] - live.ref_frames[0]) : 1;
                    for (int k = 0; k < live.ref_frames.Length; k++)
                    {
                        var c = Q(live.frame_quality[k]);
                        for (int s = 0; s < stride; s++) { int f = live.ref_frames[k] + s; if (f >= 0 && f < _frames) _cols[f] = c; }
                    }
                }
                else
                {
                    for (int f = 0; f < _frames; f++)
                    {
                        bool any = false, interp = false;
                        foreach (var pl in doc.players) { if (pl.HasPose(f)) { any = true; interp |= pl.frames[f].interpolated; } }
                        _cols[f] = !any ? QMissing : interp ? QInterp : Color.clear;
                    }
                }
                MarkDirtyRepaint();
            }

            static Color Q(int q) => q == 0 ? QFitted : q == 1 ? QWarm : q == 2 ? QLate : QNone;

            void Draw(MeshGenerationContext mgc)
            {
                var r = contentRect; var p = _ui.player;
                if (r.width <= 0 || p == null || _frames == 0) return;
                var g = mgc.painter2D;
                if (p.IsLive && p.LiveQuality != null && p.LiveQuality.Length == _frames)
                    for (int f = 0; f < _frames; f++) _cols[f] = p.LiveQuality[f] < 0 ? Color.clear : Q(p.LiveQuality[f]);
                int W = Mathf.Max(1, Mathf.FloorToInt(r.width));
                for (int x = 0; x < W; x++)
                {
                    int f0 = x * _frames / W, f1 = Mathf.Max(f0 + 1, (x + 1) * _frames / W);
                    Color c = Color.clear;
                    for (int f = f0; f < f1 && f < _frames; f++)
                        if (_cols[f].a > 0 && (c.a == 0 || Rank(_cols[f]) > Rank(c))) c = _cols[f];
                    if (c.a == 0) continue;
                    Rect(g, x, r.height * 0.55f, 1, r.height * 0.45f, c);
                }
                if (p.rangeStart >= 0 || p.rangeEnd >= 0)
                {
                    float a = r.width * p.RangeStart / Mathf.Max(_frames - 1, 1), b = r.width * p.RangeEnd / Mathf.Max(_frames - 1, 1);
                    Rect(g, a, 0, Mathf.Max(b - a, 1), r.height * 0.5f, new Color(1f, 1f, 1f, 0.15f));
                }
                float h = r.width * p.CurrentFrame / Mathf.Max(_frames - 1, 1);
                Rect(g, h - 1, 0, 2, r.height, Accent);
            }

            static int Rank(Color c) => c == QMissing ? 5 : c == QLate ? 4 : c == QNone ? 3 : c == QInterp ? 2 : c == QWarm ? 1 : 0;

            static void Rect(Painter2D g, float x, float y, float w, float h, Color c)
            {
                g.fillColor = c;
                g.BeginPath();
                g.MoveTo(new Vector2(x, y)); g.LineTo(new Vector2(x + w, y)); g.LineTo(new Vector2(x + w, y + h)); g.LineTo(new Vector2(x, y + h));
                g.ClosePath();
                g.Fill();
            }
        }
    }
}
