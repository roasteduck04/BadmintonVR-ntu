using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>Pure maths shared by the SP4 cv components (EditMode-tested against Python-generated cases).</summary>
    public static class CvMath
    {
        /// <summary>Video frame of analysis frame f (python analysis.video_io.view_frame, SP2 F3): floor(f + offset + 0.5); on an
        /// exact half the tracker's frame f + det_shift wins when it is one of the two equidistant frames. Never Math.Round.</summary>
        public static int VideoFrame(int f, float offset, int detShift)
        {
            double x = f + (double)offset;
            int idx = (int)Math.Floor(x + 0.5);
            if (Math.Abs(Math.Abs(x - idx) - 0.5) < 1e-6)
            {
                int lo = (int)Math.Floor(x), d = f + detShift;
                if (d == lo || d == lo + 1) idx = d;
            }
            return idx;
        }

        public static bool HasBit(int mask, int v) => ((mask >> v) & 1) != 0;

        public static int PopCount(int mask)
        {
            int n = 0;
            for (uint m = (uint)mask; m != 0; m &= m - 1) n++;
            return n;
        }

        public static int Idx(int f, int j, int J) => f * J + j;
        public static int ViewIdx(int v, int f, int F) => v * F + f;

        /// <summary>Frames to add to a frame of doc A to get the same instant in doc B: analysis frame f_A shows raw frame
        /// f_A + off_A(v), so f_B = f_A + off_A(v) - off_B(v). Median over views both docs have offsets for; known = false
        /// (shift 0) when they share none.</summary>
        public static float GhostFrameShift(IList<string> aNames, IList<float> aOffsets, IList<string> bNames, IList<float> bOffsets,
                                            out bool known)
        {
            var d = new List<float>();
            for (int i = 0; i < aNames.Count; i++)
            {
                int k = bNames.IndexOf(aNames[i]);
                if (k >= 0) d.Add(aOffsets[i] - bOffsets[k]);
            }
            known = d.Count > 0;
            if (!known) return 0f;
            d.Sort();
            int n = d.Count;
            return n % 2 == 1 ? d[n / 2] : 0.5f * (d[n / 2 - 1] + d[n / 2]);
        }

        public static void OffsetsOf(CvCamera[] cams, List<string> names, List<float> offsets)
        {
            names.Clear(); offsets.Clear();
            if (cams == null) return;
            foreach (var c in cams) if (c.HasOffset) { names.Add(c.name); offsets.Add(c.offset); }
        }

        public static void OffsetsOf(PoseVariantMeta pv, List<string> names, List<float> offsets)
        {
            names.Clear(); offsets.Clear();
            if (pv == null) return;
            if (!string.IsNullOrEmpty(pv.time_ref_view)) { names.Add(pv.time_ref_view); offsets.Add(0f); }
            if (pv.offsets_list != null)
                foreach (var o in pv.offsets_list)
                    if (!names.Contains(o.view)) { names.Add(o.view); offsets.Add(o.offset); }
        }

        /// <summary>A build's timeline offsets: its cvdebug sidecar's cameras when it has one, else meta.pose_variant (the
        /// source VariantGhosts and the pose row both use, so ghosts and build switches agree on "the same instant").</summary>
        public static void TimelineOffsets(CvDebugDoc cv, AnalysisDoc doc, List<string> names, List<float> offsets)
        {
            if (cv != null) OffsetsOf(cv.cameras, names, offsets);
            else OffsetsOf(doc?.meta?.pose_variant, names, offsets);
        }

        /// <summary>Frame of build B that shows the instant frame fA of build A shows, when switching builds: GhostFrame's rule
        /// when the shift is known (else the index is kept), clamped into B (a ghost may vanish, the loaded clip can't).</summary>
        public static int SameInstantFrame(int fA, float shift, bool known, int frameCountB)
        {
            if (frameCountB <= 0) return 0;
            int f = known ? Mathf.FloorToInt(fA + shift + 0.5f) : fA;
            return Mathf.Clamp(f, 0, frameCountB - 1);
        }

        public static int GhostFrame(int fA, float shift, int frameCountB)
        {
            int f = Mathf.FloorToInt(fA + shift + 0.5f);
            return f >= 0 && f < frameCountB ? f : -1;
        }

        /// <summary>Image-plane corners at `depth` metres: top-left, top-right, bottom-right, bottom-left (camera looks +Z,
        /// +Y up in the image = pixel v decreasing).</summary>
        public static void FrustumCorners(Vector3 pos, Quaternion rot, float vfovDeg, float aspect, float depth, Vector3[] out4)
        {
            float h = depth * Mathf.Tan(0.5f * vfovDeg * Mathf.Deg2Rad), w = h * aspect;
            out4[0] = pos + rot * new Vector3(-w, h, depth);
            out4[1] = pos + rot * new Vector3(w, h, depth);
            out4[2] = pos + rot * new Vector3(w, -h, depth);
            out4[3] = pos + rot * new Vector3(-w, -h, depth);
        }

        /// <summary>SP2 F6: l = M^T (p - c), u = cx + fx lx / lz, v = cy - fy ly / lz (K row-major 3x3).</summary>
        public static bool ProjectToPixel(Vector3 pos, Quaternion rot, float[] K, Vector3 p, out Vector2 uv)
        {
            Vector3 l = Quaternion.Inverse(rot) * (p - pos);
            uv = default;
            if (l.z <= 0.1f || K == null || K.Length != 9) return false;
            uv = new Vector2(K[2] + K[0] * l.x / l.z, K[5] - K[4] * l.y / l.z);
            return true;
        }

        public static Rect LetterboxRect(float screenAspect, float camAspect)
        {
            if (camAspect <= 0f || screenAspect <= 0f) return new Rect(0, 0, 1, 1);
            if (camAspect < screenAspect) { float w = camAspect / screenAspect; return new Rect((1f - w) / 2f, 0f, w, 1f); }
            float h = screenAspect / camAspect;
            return new Rect(0f, (1f - h) / 2f, 1f, h);
        }

        /// <summary>Paused: exact. Playing: only when more than PlayingSeekTolerance frames off (or nothing decoded yet) —
        /// Windows Media Foundation errors out (0xc00d4a3e) and Error Pause stops the editor under per-frame seeks.</summary>
        public static bool NeedsSeek(long current, int target, bool playing)
            => playing ? current < 0 || Math.Abs(current - target) > PlayingSeekTolerance : current != target;

        public const int PlayingSeekTolerance = 2;

        /// <summary>Paused seek-driven video: seek unless the decoder shows the target, or its last seek to this same target
        /// already landed (it may overshoot catching up from a keyframe; re-seeking to the same frame would loop forever).</summary>
        public static bool NeedsVideoSeek(long current, int target, int landedTarget)
            => NeedsSeek(current, target, false) && landedTarget != target;

        /// <summary>Open (or reopen) a video slot: another view, or the same view after a decoder error once retryDelay passed.</summary>
        public static bool ShouldReopenVideo(bool sameView, bool failed, float secondsSinceError, float retryDelay)
            => !sameView || (failed && secondsSinceError >= retryDelay);

        /// <summary>Is video frame `target` inside a video of `frameCount` frames (0 = not known yet, don't hide)?</summary>
        public static bool VideoFrameInRange(int target, long frameCount) => target >= 0 && (frameCount <= 0 || target < frameCount);

        /// <summary>Player whose rays are drawn: the focused one when it exists in this clip, else rayPlayer clamped; -1 = none.</summary>
        public static int RayPlayer(int focusPlayer, int rayPlayer, int playerCount)
        {
            if (playerCount <= 0) return -1;
            if (focusPlayer >= 0 && focusPlayer < playerCount) return focusPlayer;
            return Mathf.Clamp(rayPlayer, 0, playerCount - 1);
        }

        /// <summary>focusPlayer after a clip with playerCount players loads: a focus past the new clip (P3/P4 of a doubles clip,
        /// then a singles clip) falls back to -1 (all), else the scene would show no players and no active "show" button.</summary>
        public static int ClampFocus(int focusPlayer, int playerCount)
            => focusPlayer >= 0 && focusPlayer < playerCount ? focusPlayer : -1;

        /// <summary>Weight of this data frame's raw facing in the per-data-frame EMA: 0 = same frame as last time (re-render /
        /// paused: keep), 1 = snap (nothing yet, backwards, a loop, or a forward jump over maxStep), else 1 - (1 - perFrame)^k
        /// for k frames advanced (the same result whether frames were rendered one by one or skipped at high speed).</summary>
        public static float FacingBlend(int lastFrame, int frame, float perFrame, int maxStep)
        {
            if (lastFrame < 0 || frame < lastFrame || frame - lastFrame > maxStep) return 1f;
            int k = frame - lastFrame;
            return k == 0 ? 0f : 1f - Mathf.Pow(1f - Mathf.Clamp01(perFrame), k);
        }

        /// <summary>"4.1" -> (4, 1); false for a missing or non-numeric version. Compared as numbers, never as strings.</summary>
        public static bool ParseSchema(string version, out int major, out int minor)
        {
            major = minor = 0;
            if (string.IsNullOrEmpty(version)) return false;
            var parts = version.Trim().Split('.');
            if (!int.TryParse(parts[0], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out major)) return false;
            if (parts.Length > 1 && !int.TryParse(parts[1], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out minor)) return false;
            return true;
        }

        /// <summary>version >= major.minor, numerically ("4.10" > "4.2"); false when unparsable.</summary>
        public static bool SchemaAtLeast(string version, int major, int minor)
            => ParseSchema(version, out int ma, out int mi) && (ma > major || (ma == major && mi >= minor));

        /// <summary>pose24 lineage (schema 4.x, x >= 1): confidences are exp(-sigma / 40 mm), so only exact-0 placeholders are
        /// hidden. Schema 5.x is the separate base / SMPL-X lineage (tools/analysis/schema.py) and keeps the normal cutoff.</summary>
        public static bool SchemaUsesSigmaConfidence(string version)
            => SchemaAtLeast(version, 4, 1) && !SchemaAtLeast(version, 5, 0);

        public static int[] JointMap(string[] docJointNames, string[] sidecarJointNames)
        {
            var m = new int[docJointNames != null ? docJointNames.Length : 0];
            for (int j = 0; j < m.Length; j++)
                m[j] = sidecarJointNames != null ? Array.IndexOf(sidecarJointNames, docJointNames[j]) : -1;
            return m;
        }

        // <clip>_<N>cam (SP1) or <clip>_<N>cam_<view abbreviations> (SP5 exports, e.g. singles_05x_2cam_bl_br, singles_05x_1cam_b);
        // lazy so the family is everything before the first _<N>cam (doubles_2sides_4angle_2cam -> doubles_2sides_4angle)
        static readonly Regex SubsetStem = new Regex(@"^(.*?)_\d+cam(?:_[a-z]+)*$");

        /// <summary>Other builds of the same scene to ghost: the base + pose variants of the loaded clip, of its camera
        /// subsets (&lt;clip&gt;_Ncam*, incl. &lt;clip&gt;_Ncam_&lt;views&gt;), and — when the loaded clip is a subset — of its full clip. Excludes the loaded file
        /// and the _oldracket copies.</summary>
        public static List<string> GhostCandidates(IEnumerable<string> files, string loadedFile, string[] poseSuffixes)
        {
            string stem = Stem(AnalysisPlayer.BaseClipPath(loadedFile), poseSuffixes);
            var m = SubsetStem.Match(stem);
            string family = m.Success ? m.Groups[1].Value : stem;
            var outp = new List<string>();
            foreach (var f in files)
            {
                if (f == loadedFile || f.Contains("_oldracket")) continue;
                string s = Stem(f, poseSuffixes);
                var ms = SubsetStem.Match(s);
                if (s == family || (ms.Success && ms.Groups[1].Value == family)) outp.Add(f);
            }
            return outp;
        }

        static string Stem(string file, string[] poseSuffixes)
        {
            string n = Path.GetFileName(file ?? "");
            if (n.EndsWith(".analysis.json")) n = n.Substring(0, n.Length - ".analysis.json".Length);
            foreach (var s in poseSuffixes) if (n.EndsWith(s)) return n.Substring(0, n.Length - s.Length);
            return n;
        }
    }
}
