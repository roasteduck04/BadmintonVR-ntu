using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>
    /// Drives the whole synthetic base from one analysis.json v3: two procedural
    /// skeletons standing at their court positions, a shuttle marker, and a HUD of
    /// the current frame + each player's stroke. Helper objects are children of this
    /// object; keep it at scene root (Build destroys its children on reload).
    /// </summary>
    public class AnalysisPlayer : MonoBehaviour
    {
        public enum SkinMode { Skeleton, Mixamo, MakeHuman }

        [Tooltip("Path under StreamingAssets, e.g. analysis/singles_05x.analysis.json")]
        public string analysisFile = "analysis/singles_05x.analysis.json";
        public bool play = true;
        [Range(0.005f, 0.12f)] public float jointRadius = 0.02f;
        [Range(0.005f, 0.08f)] public float boneRadius = 0.012f;
        [Range(0f, 1f)] public float confidenceCutoff = 0.3f;
        // Schema 4.1 (pose24) confidences are exp(-sigma / 40 mm): honest for analysis, but a far or 2-camera
        // player's hips sit near 0.03 and wrists near 1e-3, so the usual cutoff would hide whole limbs.
        // For 4.1 docs (and later 4.x, compared numerically) only exact-0 placeholders are hidden: docs round confidence
        // to 5 decimals, so any written non-zero value (>= 1e-5) passes 1e-6 (colour-by-confidence still shows the
        // uncertainty); other schemas (3.x, 4.0, the 5.x base / SMPL-X lineage) keep confidenceCutoff unchanged.
        const float Schema41Cutoff = 1e-6f;

        float EffectiveCutoff =>
            _doc != null && CvMath.SchemaUsesSigmaConfidence(_doc.schema_version)
                ? Mathf.Min(confidenceCutoff, Schema41Cutoff) : confidenceCutoff;
        public float shuttleRadius = 0.08f;
        public Color[] playerColors = {
            new Color(0.2f, 0.7f, 1f),   // P1 blue
            new Color(1f, 0.5f, 0.2f),    // P2 orange
            new Color(0.3f, 0.9f, 0.4f),  // P3 green
            new Color(0.9f, 0.3f, 0.7f),  // P4 pink
        };
        [Range(0.1f, 3f)] public float speed = 1f;
        [Tooltip("-1 = all players; otherwise the only player index shown")]
        public int focusPlayer = -1;
        [Tooltip("Humanoid FBX for the Mixamo character skin")]
        public GameObject mixamoPrefab;
        [Tooltip("Humanoid FBX for the MakeHuman character skin")]
        public GameObject makeHumanPrefab;
        public bool showPose = true;
        public bool showRacket = true;
        public bool showShuttle = true;

        AnalysisDoc _doc;
        Transform[] _roots;          // per player, positioned at court XZ
        IPoseRenderer[] _renderers;  // per player
        float _clipLift;             // one vertical lift for the whole clip: bodies, rackets and shuttle (AnalysisDoc.GroundLift)
        SkinMode[] _skins;           // per player; persists across clip reload when count matches
        Vector3[] _jointBuf;         // scratch: joints for one player, one frame
        float[] _confBuf;            // scratch: confidences for one player, one frame
        ShuttleVisualAnalysis _shuttleVis;
        RacketVisualAnalysis _racketVis;
        Quaternion[] _smoothFacing;  // per player, EMA-smoothed facing direction (advanced per data frame)
        int[] _facingFrame;          // per player, the data frame _smoothFacing belongs to; -1 = snap on the next valid frame
        const float FacingSmoothing = 0.25f;   // EMA weight per data frame
        const int FacingMaxStep = 10;          // a forward jump of more data frames than this is a seek: snap
        float _t;
        int _frame;
        bool _liftToFloor = true;

        /// <summary>Raised at the end of every clip (re)load, after frame 0 is posed: SP4 cv components rebuild on it.</summary>
        public event System.Action ClipLoaded;

        /// <summary>The loaded clip's cvdebug/1 sidecar (&lt;file&gt;.cvdebug.json next to it), null when absent or mismatched.</summary>
        public CvDebugDoc CvDebug { get; private set; }

        public static string CvDebugSidecarPath(string analysisPath)
            => analysisPath.Replace(".analysis.json", ".cvdebug.json");

        /// <summary>Stand the twins on the floor (lift everything by one clip-wide offset, the default). Look-through turns it
        /// off so the drawn joints / shuttle are the measured ones and line up with the camera's video.</summary>
        public bool LiftToFloor
        {
            get => _liftToFloor;
            set { if (_liftToFloor != value) { _liftToFloor = value; Refresh(); } }
        }

        /// <summary>The vertical lift applied this frame to everything in court coordinates — every player's joints, the
        /// rackets and the shuttle share it, so they stay on one vertical reference (0 when LiftToFloor is off).</summary>
        public float VerticalLift => _liftToFloor && _doc != null ? _clipLift : 0f;

        /// <summary>The vertical lift applied to player pi's joints this frame (= VerticalLift for every player in the clip).</summary>
        public float GroundOffset(int pi) => pi >= 0 && pi < PlayerCount ? VerticalLift : 0f;

        public int PlayerCount => _doc != null && _doc.players != null ? _doc.players.Length : 0;

        /// <summary>Fractional clock frame while playing (for video sync), else the shown frame.</summary>
        public float FrameFloat => play && _doc != null
            ? Mathf.Clamp(_t * _doc.Fps, 0f, Mathf.Max(FrameCount - 1, 0)) : _frame;

        /// <summary>Where joint `joint` of player `pi` is drawn this frame (world space, same lift as the skins; the root's
        /// facing rotation is cancelled by the renderers, so it is root position + body-local joint + lift).</summary>
        public bool TryJointWorld(int pi, int joint, out Vector3 world)
        {
            world = default;
            if (_doc == null || _roots == null || pi < 0 || pi >= _roots.Length) return false;
            var p = _doc.players[pi];
            if (_frame >= p.FrameCount || !p.HasPose(_frame) || joint < 0 || joint >= p.PoseJointCount(_frame)) return false;
            world = transform.TransformPoint(_roots[pi].localPosition + p.JointLocal(_frame, joint)
                                             + new Vector3(0f, GroundOffset(pi), 0f));
            return true;
        }

        /// <summary>AUX-14 keypoint k of player pi (pose24 aux_flat, 14 x 4 body-local), same lift; false when absent.</summary>
        public bool TryAuxWorld(int pi, int k, out Vector3 world, out float conf)
        {
            world = default; conf = 0f;
            if (_doc == null || _roots == null || pi < 0 || pi >= _roots.Length) return false;
            var fr = _doc.players[pi].frames;
            if (fr == null || _frame >= fr.Length) return false;
            var a = fr[_frame].aux_flat;
            int b = k * AnalysisDoc.Stride;
            if (a == null || k < 0 || b + 3 >= a.Length || a[b + 3] <= 0f) return false;
            conf = a[b + 3];
            world = transform.TransformPoint(_roots[pi].localPosition + new Vector3(a[b], a[b + 1], a[b + 2])
                                             + new Vector3(0f, GroundOffset(pi), 0f));
            return true;
        }

        /// <summary>SP4 cv components live on this object; added here when the scene doesn't have them yet.</summary>
        void EnsureCvComponents()
        {
            if (GetComponent<CvDebugOverlay>() == null) gameObject.AddComponent<CvDebugOverlay>();
            if (GetComponent<CameraRigGizmos>() == null) gameObject.AddComponent<CameraRigGizmos>();
            if (GetComponent<CameraRayLines>() == null) gameObject.AddComponent<CameraRayLines>();
            if (GetComponent<VideoBillboard>() == null) gameObject.AddComponent<VideoBillboard>();
            if (GetComponent<NonPlayerMarkers>() == null) gameObject.AddComponent<NonPlayerMarkers>();
            if (GetComponent<VariantGhosts>() == null) gameObject.AddComponent<VariantGhosts>();
        }

        public AnalysisDoc Doc => _doc;
        public Transform PlayerRoot(int index) => (_roots != null && index >= 0 && index < _roots.Length) ? _roots[index] : null;
        public bool IsPlaying => play;
        public int CurrentFrame => _frame;
        public int FrameCount => _doc != null ? _doc.FrameCount : 0;
        public float Fps => _doc != null ? _doc.Fps : 30f;
        public float Normalized => FrameCount > 1 ? (float)_frame / (FrameCount - 1) : 0f;

        public static readonly string[] RacketVariantNames = { "rigid", "bootstrap", "imu" };

        /// <summary>One pose variant: file suffix + the pose-row button label.</summary>
        public struct PoseVariantDef
        {
            public readonly string Suffix, Label;
            public PoseVariantDef(string suffix, string label) { Suffix = suffix; Label = label; }
        }

        /// <summary>Pose variants: whole analysis files "&lt;clip&gt;&lt;suffix&gt;.analysis.json" next to the base clip. The single
        /// source of truth: PoseVariantSuffixes / PoseVariantLabels are derived from it, so a new suffix always has a label.</summary>
        public static readonly PoseVariantDef[] PoseVariantDefs =
        {
            new PoseVariantDef("_posev2", "v2 robust"),
            new PoseVariantDef("_posefit", "fit"),
            new PoseVariantDef("_smplx", "SMPL-X 58"),
            new PoseVariantDef("_pose24", "SMPL-24"),
        };

        public static readonly string[] PoseVariantSuffixes = System.Array.ConvertAll(PoseVariantDefs, d => d.Suffix);
        public static readonly string[] PoseVariantLabels = System.Array.ConvertAll(PoseVariantDefs, d => d.Label);

        /// <summary>
        /// The base clip a loaded file belongs to (strips the old-racket and pose-variant suffixes).
        /// Resolve sidecars that do not depend on the pose (e.g. shuttle, which is in court coordinates)
        /// against this, so they keep working whichever pose variant is loaded. Racket sidecars are
        /// body-local, so each pose variant ships its own and they resolve against the loaded file.
        /// </summary>
        public static string BaseClipPath(string analysisPath)
        {
            if (string.IsNullOrEmpty(analysisPath)) return analysisPath;
            string p = analysisPath.Replace("_oldracket", "");
            foreach (var s in PoseVariantSuffixes) p = p.Replace(s + ".analysis.json", ".analysis.json");
            return p;
        }

        public static string RacketSidecarPath(string analysisPath, string variant)
            => analysisPath.Replace(".analysis.json", $".racket_{variant}.json");

        public string EffectiveRacketVariant
            => _racketVis != null ? _racketVis.EffectiveVariant : RacketVisualAnalysis.V1;

        public bool GhostV1
        {
            get => _racketVis != null && _racketVis.ghostV1;
            set { if (_racketVis != null && _racketVis.ghostV1 != value) { _racketVis.ghostV1 = value; Refresh(); } }
        }

        public bool SnapRacketToHand
        {
            get => _racketVis != null && _racketVis.snapToHand;
            set { if (_racketVis != null && _racketVis.snapToHand != value) { _racketVis.snapToHand = value; Refresh(); } }
        }

        public void SetRacketVariant(string variant)
        {
            if (_racketVis == null) return;
            _racketVis.preferredVariant = variant;
            Refresh();
        }

        public RacketVariantDoc GetRacketVariantDoc(string variant)
            => _racketVis != null ? _racketVis.VariantDoc(variant) : null;

        public static readonly string[] ShuttleVariantNames = { "improved", "tracker" };

        /// <summary>Shuttle sidecars are in court coordinates, so they resolve against the base clip.</summary>
        public static string ShuttleSidecarPath(string analysisPath, string variant)
            => BaseClipPath(analysisPath).Replace(".analysis.json", $".shuttle_{variant}.json");

        public string EffectiveShuttleVariant
            => _shuttleVis != null ? _shuttleVis.EffectiveVariant : ShuttleVisualAnalysis.Baseline;

        public void SetShuttleVariant(string variant)
        {
            if (_shuttleVis == null) return;
            _shuttleVis.preferredVariant = variant;
            Refresh();
        }

        public bool GhostShuttleBaseline
        {
            get => _shuttleVis != null && _shuttleVis.ghostBaseline;
            set { if (_shuttleVis != null && _shuttleVis.ghostBaseline != value) { _shuttleVis.ghostBaseline = value; Refresh(); } }
        }

        public bool ShuttleTrail
        {
            get => _shuttleVis != null && _shuttleVis.trail;
            set { if (_shuttleVis != null && _shuttleVis.trail != value) { _shuttleVis.trail = value; Refresh(); } }
        }

        public ShuttleVariantDoc GetShuttleVariantDoc(string variant)
            => _shuttleVis != null ? _shuttleVis.VariantDoc(variant) : null;

        public string DescribeShuttle(int frame)
            => _shuttleVis != null ? _shuttleVis.Describe(frame) : "shuttle (none)";

        /// <summary>Shuttle sidecars of a clip that is about to load (parsed + validated, nothing attached yet).</summary>
        static System.Collections.Generic.Dictionary<string, ShuttleVariantDoc> LoadShuttleVariants(string file, AnalysisDoc doc)
        {
            var found = new System.Collections.Generic.Dictionary<string, ShuttleVariantDoc>();
            foreach (var v in ShuttleVariantNames)
            {
                var sc = ShuttleVariantDoc.TryLoad(ShuttleSidecarPath(file, v), doc);
                if (sc != null) found[v] = sc;
            }
            if (found.Count > 0)
                Debug.Log($"[Analysis] shuttle variants: {string.Join(", ", found.Keys)}");
            return found;
        }

        void Refresh() { if (_doc != null && _roots != null) ShowFrame(_frame); }

        /// <summary>Racket sidecars of a clip that is about to load (parsed + validated, nothing attached yet).</summary>
        static System.Collections.Generic.Dictionary<string, RacketVariantDoc> LoadRacketVariants(string file, AnalysisDoc doc)
        {
            var found = new System.Collections.Generic.Dictionary<string, RacketVariantDoc>();
            foreach (var v in RacketVariantNames)
            {
                var sc = RacketVariantDoc.TryLoad(RacketSidecarPath(file, v), doc);
                if (sc != null) found[v] = sc;
            }
            if (found.Count > 0)
                Debug.Log($"[Analysis] racket variants: {string.Join(", ", found.Keys)}");
            return found;
        }

        public void TogglePlay() => play = !play;
        public void Play() => play = true;
        public void Pause() => play = false;

        /// <summary>Jump to frame f; the facing snaps to that frame's hips (no smoothing carried over from the old instant).</summary>
        public void SeekFrame(int f) => Seek(f, snapFacing: true);

        void Seek(int f, bool snapFacing)
        {
            if (_doc == null || FrameCount == 0) return;
            _frame = Mathf.Clamp(f, 0, FrameCount - 1);
            _t = _frame / Mathf.Max(_doc.Fps, 1f);   // keep clock in sync so play resumes here
            if (snapFacing && _facingFrame != null)
                for (int i = 0; i < _facingFrame.Length; i++) _facingFrame[i] = -1;
            if (_roots != null) ShowFrame(_frame);
        }

        public void SeekNormalized(float t01)
            => SeekFrame(Mathf.RoundToInt(Mathf.Clamp01(t01) * Mathf.Max(FrameCount - 1, 0)));

        /// <summary>Step like playback: one frame forward keeps the per-data-frame facing smoothing (backward snaps).</summary>
        public void StepFrame(int delta) { Pause(); Seek(_frame + delta, snapFacing: false); }

        void Start() => LoadClip(analysisFile);

        /// <summary>Load a different clip at runtime (StreamingAssets-relative path). Transactional: the doc and every sidecar
        /// are read, parsed and validated into locals first (each loader contains its own exceptions); only then are
        /// analysisFile and the state swapped. On failure the previous clip keeps playing, analysisFile unchanged.</summary>
        public bool LoadClip(string streamingAssetsRelativePath)
        {
            string file = streamingAssetsRelativePath;
            var doc = AnalysisDoc.Load(file);
            if (doc == null)
            {
                Debug.LogError($"[Analysis] load failed, keeping {(_doc != null ? analysisFile : "nothing")}: {file}");
                return false;
            }
            var rackets = LoadRacketVariants(file, doc);
            var shuttles = LoadShuttleVariants(file, doc);
            var cv = CvDebugDoc.TryLoad(CvDebugSidecarPath(file), doc);

            analysisFile = file;
            _doc = doc;
            _t = 0f; _frame = 0;
            Build(rackets, shuttles, cv);
            Debug.Log($"[Analysis] loaded {_doc.clip_id}: {_doc.FrameCount} frames, " +
                      $"{_doc.players.Length} players");
            return true;
        }

        /// <summary>Load another build of the same scene keeping the instant, not the frame index: builds sit on different
        /// timelines, so the frame is re-timed by their sync offsets (CvMath.GhostFrameShift, as VariantGhosts does);
        /// kept as is when the offsets share no view. False (nothing changed) when the load fails.</summary>
        public bool LoadClipKeepingInstant(string streamingAssetsRelativePath)
        {
            int frame = _frame;
            var an = new System.Collections.Generic.List<string>(); var ao = new System.Collections.Generic.List<float>();
            CvMath.TimelineOffsets(CvDebug, _doc, an, ao);
            if (!LoadClip(streamingAssetsRelativePath)) return false;
            var bn = new System.Collections.Generic.List<string>(); var bo = new System.Collections.Generic.List<float>();
            CvMath.TimelineOffsets(CvDebug, _doc, bn, bo);
            float shift = CvMath.GhostFrameShift(an, ao, bn, bo, out bool known);
            int target = CvMath.SameInstantFrame(frame, shift, known, FrameCount);
            if (target != frame) Debug.Log($"[Analysis] kept the instant: frame {frame} -> {target} (shift {shift:+0.0;-0.0} f)");
            SeekFrame(target);
            return true;
        }

        IPoseRenderer NewRenderer(SkinMode mode)
        {
            switch (mode)
            {
                case SkinMode.Mixamo: return new CharacterPoseRenderer(mixamoPrefab);
                case SkinMode.MakeHuman: return new CharacterPoseRenderer(makeHumanPrefab);
                default:             return new SkeletonPoseRenderer(jointRadius, boneRadius);
            }
        }

        public SkinMode GetSkin(int player)
            => (_skins != null && player >= 0 && player < _skins.Length)
               ? _skins[player] : SkinMode.Skeleton;

        /// <summary>Swap one player's skin live; frame + playback state are preserved.</summary>
        public void SetSkin(int player, SkinMode mode)
        {
            if (_doc == null || _renderers == null || player < 0 || player >= _renderers.Length) return;
            if (_skins[player] == mode) return;
            _skins[player] = mode;

            var root = _roots[player];
            for (int i = root.childCount - 1; i >= 0; i--)
            {
                var go = root.GetChild(i).gameObject;
                if (Application.isPlaying) Destroy(go); else DestroyImmediate(go);
            }
            _renderers[player].Clear();

            Color col = playerColors[player % playerColors.Length];
            _renderers[player] = NewRenderer(mode);
            _renderers[player].SetJointNames(_doc.skeleton.joint_names);
            // Build with the root active: focus may have deactivated it, and a character skin finds its Animator and measures
            // its renderer bounds only on active objects (else it never animates). ShowFrame re-applies the focus.
            bool wasActive = root.gameObject.activeSelf;
            root.gameObject.SetActive(true);
            _renderers[player].Build(root, _doc.skeleton.parents, col);
            root.gameObject.SetActive(wasActive);
            ShowFrame(_frame);
        }

        Vector3[] _racketJointBuf;

        /// <summary>Joints for snap-to-hand: the data joints, with the hands replaced by where
        /// the skin actually drew them (a retargeted character keeps its own arm lengths, so
        /// its hands are not at the data's hand joints).</summary>
        Vector3[] RacketHandJoints(int pi)
        {
            if (_racketJointBuf == null || _racketJointBuf.Length != _jointBuf.Length)
                _racketJointBuf = new Vector3[_jointBuf.Length];
            System.Array.Copy(_jointBuf, _racketJointBuf, _jointBuf.Length);
            foreach (int j in new[] { _racketVis.HandJoint(true), _racketVis.HandJoint(false) })
                if (j >= 0 && j < _racketJointBuf.Length && _renderers[pi].TryGetRenderedJoint(j, out var drawn))
                    _racketJointBuf[j] = drawn;
            return _racketJointBuf;
        }

        static readonly Color[] _defaultColors = {
            new Color(0.2f, 0.7f, 1f),
            new Color(1f, 0.5f, 0.2f),
            new Color(0.3f, 0.9f, 0.4f),
            new Color(0.9f, 0.3f, 0.7f),
        };

        void Build(System.Collections.Generic.Dictionary<string, RacketVariantDoc> rackets,
                   System.Collections.Generic.Dictionary<string, ShuttleVariantDoc> shuttles, CvDebugDoc cv)
        {
            Clear();
            int np = _doc.players.Length;
            focusPlayer = CvMath.ClampFocus(focusPlayer, np);
            if (playerColors == null || playerColors.Length < np)
                playerColors = _defaultColors;
            _roots = new Transform[np];
            _renderers = new IPoseRenderer[np];
            _clipLift = _doc.GroundLift();
            _jointBuf = new Vector3[_doc.JointCount];
            _confBuf = new float[_doc.JointCount];
            if (_skins == null || _skins.Length != np)
                _skins = new SkinMode[np];   // defaults to Skeleton (enum value 0)
            _smoothFacing = new Quaternion[np];
            _facingFrame = new int[np];
            for (int i = 0; i < np; i++) { _smoothFacing[i] = Quaternion.identity; _facingFrame[i] = -1; }

            for (int pi = 0; pi < np; pi++)
            {
                var p = _doc.players[pi];
                var root = new GameObject($"player_{p.id}").transform;
                root.SetParent(transform, false);
                _roots[pi] = root;

                Color col = playerColors[pi % playerColors.Length];
                _renderers[pi] = NewRenderer(_skins[pi]);
                _renderers[pi].SetJointNames(_doc.skeleton.joint_names);
                _renderers[pi].Build(root, _doc.skeleton.parents, col);
            }

            _racketVis = GetComponent<RacketVisualAnalysis>();
            if (_racketVis == null) _racketVis = gameObject.AddComponent<RacketVisualAnalysis>();
            _racketVis.Build(np);
            _racketVis.SetJointNames(_doc.skeleton.joint_names);
            _racketVis.SetVariants(rackets);

            _shuttleVis = GetComponent<ShuttleVisualAnalysis>();
            if (_shuttleVis == null) _shuttleVis = gameObject.AddComponent<ShuttleVisualAnalysis>();
            _shuttleVis.Build(transform, _doc, shuttleRadius);
            _shuttleVis.SetVariants(shuttles);

            CvDebug = cv;
            EnsureCvComponents();
            ShowFrame(0);
            ClipLoaded?.Invoke();
        }

        void Update()
        {
            if (_doc == null || _roots == null) return;
            if (play && _doc.FrameCount > 0)
            {
                _t += Time.deltaTime * Mathf.Max(speed, 0f);
                float duration = _doc.FrameCount / Mathf.Max(_doc.Fps, 1f);
                if (_t >= duration) _t -= duration;
                _frame = Mathf.Clamp(Mathf.FloorToInt(_t * _doc.Fps), 0, _doc.FrameCount - 1);
            }
            ShowFrame(_frame);
        }

        void ShowFrame(int frame)
        {
            float cutoff = EffectiveCutoff;
            // Racket roots are children of this object, not of the player roots focus deactivates: hide them explicitly
            // (all of them when the racket layer is off, an unfocused player's below), else they freeze in mid-air.
            if (_racketVis != null && !showRacket) _racketVis.HideAll();
            for (int pi = 0; pi < _doc.players.Length; pi++)
            {
                bool shown = focusPlayer < 0 || focusPlayer == pi;
                _renderers[pi].SetActive(shown);
                if (!shown)
                {
                    if (_racketVis != null) _racketVis.HidePlayer(pi);
                    continue;
                }

                var p = _doc.players[pi];
                Vector2? pos = p.PositionXZ(frame);
                if (pos.HasValue)
                    _roots[pi].localPosition = new Vector3(pos.Value.x, 0f, pos.Value.y);

                Vector3 lift = new Vector3(0, GroundOffset(pi), 0);
                bool hasPose = p.HasPose(frame);
                int nj = _doc.JointCount;
                for (int j = 0; j < nj; j++)
                {
                    _jointBuf[j] = hasPose ? p.JointLocal(frame, j) + lift : Vector3.zero;
                    _confBuf[j] = hasPose ? p.JointConf(frame, j) : 0f;
                }

                // Facing from the hip line (1=L_HIP, 2=R_HIP), EMA-smoothed per data frame (not per rendered frame:
                // the same frame re-rendered or paused keeps its facing; a seek / loop / big jump snaps).
                float blend = CvMath.FacingBlend(_facingFrame[pi], frame, FacingSmoothing, FacingMaxStep);
                bool computed = false;
                if (blend > 0f && hasPose && nj > 2 && _confBuf[1] > cutoff && _confBuf[2] > cutoff)
                {
                    Vector3 hipVec = _jointBuf[2] - _jointBuf[1]; // left→right
                    Vector3 forward = Vector3.Cross(Vector3.up, hipVec);
                    if (forward.sqrMagnitude > 1e-6f)
                    {
                        Quaternion raw = Quaternion.LookRotation(forward.normalized, Vector3.up);
                        _smoothFacing[pi] = blend >= 1f ? raw : Quaternion.Slerp(_smoothFacing[pi], raw, blend);
                        computed = true;
                    }
                }
                if (computed || blend < 1f) _facingFrame[pi] = frame;   // a pending snap waits for a frame with hips
                // Always counter-rotate by the current smoothed facing, also when the hips are missing this frame: the root
                // keeps that rotation, so joints, character skin, racket and TryJointWorld (overlay) stay consistent.
                Quaternion facing = _smoothFacing[pi];
                _roots[pi].localRotation = facing;
                Quaternion facingInv = Quaternion.Inverse(facing);
                for (int j = 0; j < nj; j++)
                    _jointBuf[j] = facingInv * _jointBuf[j];

                _renderers[pi].Apply(_jointBuf, _confBuf, hasPose && showPose, cutoff);

                if (_racketVis != null && showRacket)
                    _racketVis.ShowFrame(pi, p, frame, _roots[pi], GroundOffset(pi), facingInv,
                                         hasPose ? RacketHandJoints(pi) : null);
            }

            if (_shuttleVis != null) _shuttleVis.ShowFrame(frame, showShuttle, VerticalLift);
        }

        void Clear()
        {
            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                var go = transform.GetChild(i).gameObject;
                if (Application.isPlaying) Destroy(go); else DestroyImmediate(go);
            }
            if (_renderers != null)
                foreach (var r in _renderers) r?.Clear();      // releases the materials each skin made
            if (_racketVis != null) _racketVis.Clear();
            if (_shuttleVis != null) _shuttleVis.Clear();
            _roots = null; _renderers = null;
        }
    }
}
