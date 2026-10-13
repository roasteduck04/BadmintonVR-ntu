using System;
using System.IO;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    // Mirrors analysis.json v3 (superset of skeleton.json v2). JsonUtility ignores
    // fields we don't declare (racket, grip, camera, rallies, betas). Pose joints are
    // a flat float array (112 = 28 x 4), body-local; court placement is `position`.

    /// <summary>The one read + parse path for every JSON the viewer loads from StreamingAssets (analysis docs, racket /
    /// shuttle / cvdebug sidecars). JsonUtility throws on a NaN token or a truncated file, and racket / shuttle sidecars
    /// are written in place, so a pipeline run can leave one half-written: every exception is contained here and turned
    /// into a reason string. Each loader parses + validates into a local and only returns a doc that fits.</summary>
    public static class JsonLoad
    {
        public static string FullPath(string streamingAssetsRelativePath)
            => Path.Combine(Application.streamingAssetsPath, streamingAssetsRelativePath);

        /// <summary>Parse json into T; false with the reason on empty input, malformed JSON or a null result.</summary>
        public static bool TryParse<T>(string json, out T doc, out string why) where T : class
        {
            doc = null; why = "";
            if (string.IsNullOrEmpty(json)) { why = "empty file"; return false; }
            try { doc = JsonUtility.FromJson<T>(json); }
            catch (Exception e) { why = $"not valid JSON ({e.Message})"; doc = null; return false; }
            if (doc == null) { why = "empty"; return false; }
            return true;
        }

        /// <summary>Read a StreamingAssets file; false when it is missing (why = "not found") or unreadable.</summary>
        public static bool TryRead(string streamingAssetsRelativePath, out string text, out string path, out string why)
        {
            text = null; why = "";
            path = FullPath(streamingAssetsRelativePath);
            if (!File.Exists(path)) { why = "not found"; return false; }
            try { text = File.ReadAllText(path); return true; }
            catch (Exception e) { why = $"unreadable ({e.Message})"; return false; }
        }
    }

    [Serializable] public class Producers { public string pose, position, racket, grip, stroke, shuttle; }
    [Serializable] public class ViewOffset { public string view; public float offset; }
    /// <summary>meta.pose_variant: JsonUtility-safe offsets_list (the offsets dict is Python-only) + the timeline's reference view.</summary>
    [Serializable] public class PoseVariantMeta { public string time_ref_view; public ViewOffset[] offsets_list; }
    /// <summary>meta.pose24: only the build id (pairs the doc with its cvdebug sidecar / debug cache).</summary>
    [Serializable] public class Pose24Meta { public string build_id; }
    [Serializable] public class AnalysisMeta { public float fps; public int frame_count; public string source_clip; public Producers producers; public string label; public PoseVariantMeta pose_variant; public Pose24Meta pose24; }
    [Serializable] public class AnalysisCourt { public float length_z, width_x, singles_half_width_x, baseline_z, short_service_z, net_z; }
    [Serializable] public class AnalysisSkeleton { public string type; public string family; public string label; public string[] joint_names; public int[] parents; public string[] aux_names; }
    [Serializable] public class Stroke { public string label; public float confidence; }

    [Serializable]
    public class RacketFrame
    {
        public float[] grip;         // [x, y, z] body-local
        public float[] head;         // [x, y, z] body-local
        public float[] side;         // [x, y, z] body-local
        public float[] axis;         // [x, y, z] unit shaft direction
        public float[] face_normal;  // [x, y, z] unit
        public string which_hand;    // "left" | "right"
        public float confidence;
        public string status;        // "detected" | "missing"
    }

    [Serializable]
    public class RacketTemplateInfo
    {
        public float length = 0.665f, head_length = 0.25f, head_width = 0.21f, width_z = 0.545f, grip_z = 0.075f;
    }

    [Serializable]
    public class RacketVariantPlayer
    {
        public string id;
        public string which_hand;
        public RacketFrame[] frames;
    }

    /// <summary>Racket-only sidecar (&lt;clip&gt;.racket_&lt;variant&gt;.json) overlaid on an analysis doc.</summary>
    [Serializable]
    public class RacketVariantDoc
    {
        public string format;
        public string variant;
        public string clip_id;
        public string time_base;
        public int frame_count;
        public RacketTemplateInfo template;
        public RacketVariantPlayer[] players;

        public static bool Valid(RacketFrame r)
            => r != null && r.grip != null && r.grip.Length == 3 && r.head != null && r.head.Length == 3
               && r.side != null && r.side.Length == 3;

        public RacketFrame At(int playerIdx, int frame)
        {
            if (players == null || playerIdx < 0 || playerIdx >= players.Length) return null;
            var fr = players[playerIdx].frames;
            return (fr != null && frame >= 0 && frame < fr.Length) ? fr[frame] : null;
        }

        /// <summary>Parses the sidecar and checks it against the analysis doc's players / frames; null with the reason when
        /// it is malformed or doesn't fit.</summary>
        public static RacketVariantDoc Parse(string json, AnalysisDoc analysis, out string why)
        {
            if (!JsonLoad.TryParse(json, out RacketVariantDoc doc, out why)) return null;
            if (doc.players == null || analysis == null || analysis.players == null
                || doc.players.Length != analysis.players.Length || doc.frame_count != analysis.FrameCount)
            { why = "does not match clip"; return null; }
            for (int i = 0; i < doc.players.Length; i++)
                if (doc.players[i] == null || doc.players[i].id != analysis.players[i].id)
                { why = "player order differs"; return null; }
            if (doc.template == null || doc.template.length <= 0f) doc.template = new RacketTemplateInfo();
            return doc;
        }

        /// <summary>Loads the sidecar if it exists, parses and matches the analysis doc's players/frames; else null.</summary>
        public static RacketVariantDoc TryLoad(string streamingAssetsRelativePath, AnalysisDoc analysis)
        {
            if (!JsonLoad.TryRead(streamingAssetsRelativePath, out string json, out string path, out string why))
            {
                if (why != "not found") Debug.LogWarning($"[Analysis] racket sidecar {why}, ignored: {path}");
                return null;
            }
            var doc = Parse(json, analysis, out why);
            if (doc == null) Debug.LogWarning($"[Analysis] racket sidecar {why}, ignored: {path}");
            return doc;
        }
    }

    [Serializable]
    public class PlayerFrame
    {
        public int frame_id;
        public float time;
        public float[] joints_flat;  // 112 or null
        public bool interpolated;    // pose24/posev2: pelvis gap-filled (no camera saw it)
        public float[] aux_flat;     // pose24: 14 x 4 auxiliary keypoints (heels, toes, finger MCPs), body-local; JSON null (-> empty array) when absent
        public float[] position;     // [x, z] court meters or null
        public RacketFrame racket;   // null when no detection
        public Stroke stroke;        // JSON null on non-contact frames, but JsonUtility
                                     // materializes a default Stroke (empty label) — callers
                                     // must gate on label presence, not on stroke == null.
    }

    [Serializable]
    public class Player
    {
        public string id;
        public string side;          // "near" | "far"
        public PlayerFrame[] frames;

        public int FrameCount => frames != null ? frames.Length : 0;

        public Vector3 JointLocal(int frame, int joint)
        {
            int b = joint * AnalysisDoc.Stride;
            var f = frames[frame].joints_flat;
            return new Vector3(f[b], f[b + 1], f[b + 2]);
        }

        public float JointConf(int frame, int joint)
            => frames[frame].joints_flat[joint * AnalysisDoc.Stride + 3];

        public bool HasPose(int frame)
        {
            var f = frames[frame].joints_flat;
            return f != null && f.Length >= AnalysisDoc.Stride && f.Length % AnalysisDoc.Stride == 0;
        }

        public int PoseJointCount(int frame)
        {
            var f = frames[frame].joints_flat;
            return (f != null) ? f.Length / AnalysisDoc.Stride : 0;
        }

        public Vector2? PositionXZ(int frame)
        {
            var p = frames[frame].position;
            return (p != null && p.Length == 2) ? new Vector2(p[0], p[1]) : (Vector2?)null;
        }

        public Stroke StrokeAt(int frame) => frames[frame].stroke;

        public RacketFrame RacketAt(int frame) => frames[frame].racket;

        public bool HasRacket(int frame)
        {
            var r = frames[frame].racket;
            return r != null && r.grip != null && r.grip.Length == 3
                   && r.head != null && r.head.Length == 3
                   && r.side != null && r.side.Length == 3;
        }

        /// <summary>Robust lowest joint Y (5th percentile of per-frame minima); 0 when the player has no pose.</summary>
        public float MinY() => TryMinY(out float y) ? y : 0f;

        /// <summary>Robust lowest joint Y (5th percentile of per-frame minima); false when the player has no pose.</summary>
        public bool TryMinY(out float minY)
        {
            minY = 0f;
            var mins = new System.Collections.Generic.List<float>();
            for (int i = 0; i < FrameCount; i++)
            {
                if (!HasPose(i)) continue;
                int nj = PoseJointCount(i);
                float fmin = float.MaxValue;
                for (int j = 0; j < nj; j++)
                    fmin = Mathf.Min(fmin, frames[i].joints_flat[j * AnalysisDoc.Stride + 1]);
                if (fmin < float.MaxValue) mins.Add(fmin);
            }
            if (mins.Count == 0) return false;
            mins.Sort();
            int idx = Mathf.Clamp(Mathf.FloorToInt(mins.Count * 0.05f), 0, mins.Count - 1);
            minY = mins[idx];
            return true;
        }
    }

    [Serializable]
    public class ShuttleFrame
    {
        public int frame_id;
        public float time;
        public float[] position;   // [x, y, z] or null
        public float confidence;
    }

    [Serializable]
    public class Shuttle
    {
        public ShuttleFrame[] frames;

        public Vector3? PositionAt(int frame)
        {
            if (frames == null || frame < 0 || frame >= frames.Length) return null;
            var p = frames[frame].position;
            return (p != null && p.Length == 3) ? new Vector3(p[0], p[1], p[2]) : (Vector3?)null;
        }

        /// <summary>Fraction of frames with a position.</summary>
        public float Coverage()
        {
            if (frames == null || frames.Length == 0) return 0f;
            int n = 0;
            foreach (var f in frames) if (f.position != null && f.position.Length == 3) n++;
            return (float)n / frames.Length;
        }
    }

    [Serializable]
    public class ShuttleVariantFrame
    {
        public float[] position;   // [x, y, z] court metres, Y-up; empty/null when source == "none"
        public float confidence;
        public string source;      // observed | filled | physics | none
        public int n_views;
        public float reproj_px;    // -1 when unknown
    }

    [Serializable]
    public class ShuttleVariantEvent
    {
        public int frame;
        public string type;        // hit | floor
        public float[] position;
    }

    [Serializable]
    public class ShuttleVariantStats
    {
        public float coverage;
        public int observed, filled, physics;
        public float reproj_rms_px;   // -1 when unknown
        public int jumps_gt_1m;
    }

    /// <summary>Alternative shuttle track (tools/analysis/shuttle_variants.py, format shuttle_variant/1).</summary>
    [Serializable]
    public class ShuttleVariantDoc
    {
        public const string Format = "shuttle_variant/1";

        public string format, variant, clip_id, time_base;
        public float fps;
        public int frame_count;
        public ShuttleVariantStats stats;
        public ShuttleVariantFrame[] frames;
        public ShuttleVariantEvent[] events;

        public ShuttleVariantFrame FrameAt(int frame)
            => (frames != null && frame >= 0 && frame < frames.Length) ? frames[frame] : null;

        public Vector3? PositionAt(int frame)
        {
            var fr = FrameAt(frame);
            if (fr == null || fr.source == "none" || fr.position == null || fr.position.Length != 3) return null;
            return new Vector3(fr.position[0], fr.position[1], fr.position[2]);
        }

        /// <summary>Parses the sidecar and checks format + frame count against the analysis doc; null with the reason.</summary>
        public static ShuttleVariantDoc Parse(string json, AnalysisDoc analysis, out string why)
        {
            if (!JsonLoad.TryParse(json, out ShuttleVariantDoc doc, out why)) return null;
            if (doc.format != Format || doc.frames == null || analysis == null
                || doc.frame_count != analysis.FrameCount || doc.frames.Length != doc.frame_count)
            { why = "does not match clip"; return null; }
            if (doc.events == null) doc.events = new ShuttleVariantEvent[0];
            return doc;
        }

        /// <summary>Loads the sidecar if it exists, parses and matches the analysis doc's frame count; else null.</summary>
        public static ShuttleVariantDoc TryLoad(string streamingAssetsRelativePath, AnalysisDoc analysis)
        {
            if (!JsonLoad.TryRead(streamingAssetsRelativePath, out string json, out string path, out string why))
            {
                if (why != "not found") Debug.LogWarning($"[Analysis] shuttle sidecar {why}, ignored: {path}");
                return null;
            }
            var doc = Parse(json, analysis, out why);
            if (doc == null) Debug.LogWarning($"[Analysis] shuttle sidecar {why}, ignored: {path}");
            return doc;
        }
    }

    [Serializable]
    public class AnalysisDoc
    {
        public string schema_version;
        public string clip_id;
        public AnalysisMeta meta;
        public AnalysisCourt court;
        public AnalysisSkeleton skeleton;
        public Player[] players;
        public Shuttle shuttle;
        public Shuttle shuttle_legacy;

        public bool HasShuttleLegacy => shuttle_legacy?.frames != null && shuttle_legacy.frames.Length > 0;

        /// <summary>The shuttle track the viewer has always drawn: legacy when present, else shuttle.</summary>
        public Shuttle BaselineShuttle => HasShuttleLegacy ? shuttle_legacy : shuttle;

        public int JointCount => skeleton != null && skeleton.parents != null ? skeleton.parents.Length : 0;

        /// <summary>One vertical lift for the whole clip, applied to every player's joints, the rackets and the shuttle so they
        /// share one vertical reference (racket-shuttle contact stays as measured): the mean over posed players of their
        /// robust lowest joint (-MinY). Per-player lifts put each body on the floor but moved it ~4-5 cm relative to the
        /// shuttle, which stays in raw court coordinates.</summary>
        public float GroundLift()
        {
            if (players == null) return 0f;
            float sum = 0f; int n = 0;
            foreach (var p in players)
                if (p != null && p.TryMinY(out float y)) { sum += -y; n++; }
            return n > 0 ? sum / n : 0f;
        }

        /// <summary>Which skeleton this file is ("SMPL-28 (SMPL-24 + 4 face)", "SMPL-X 58", ...). Files without a
        /// label are named by joint count — older files carry wrong `type` strings.</summary>
        public string SkeletonLabel
        {
            get
            {
                if (skeleton != null && !string.IsNullOrEmpty(skeleton.label)) return skeleton.label;
                switch (JointCount)
                {
                    case 24: return "SMPL-24";
                    case 28: return "SMPL-28 (SMPL-24 + 4 face)";
                    case 58: return "SMPL-X 58";
                    default: return $"{(skeleton != null ? skeleton.type : "?")} ({JointCount} joints)";
                }
            }
        }

        /// <summary>meta.label when the producer wrote one (skeleton · method · cameras), else the skeleton label.</summary>
        public string DisplayLabel => meta != null && !string.IsNullOrEmpty(meta.label) ? meta.label : SkeletonLabel;

        /// <summary>DisplayLabel cut to maxChars with an ellipsis (5-camera pose24 labels list every camera name).</summary>
        public string ShortLabel(int maxChars = 64)
        {
            string s = DisplayLabel ?? "";
            return s.Length <= maxChars ? s : s.Substring(0, Mathf.Max(maxChars - 1, 1)) + "…";
        }

        public const int Stride = 4;

        public float Fps => (meta != null && meta.fps > 1f) ? meta.fps : 30f;
        public int FrameCount => meta != null ? meta.frame_count : 0;

        /// <summary>Parses an analysis doc and checks what the viewer indexes without bounds checks; null with the reason.</summary>
        public static AnalysisDoc Parse(string json, out string why)
        {
            if (!JsonLoad.TryParse(json, out AnalysisDoc doc, out why)) return null;
            if (doc.players == null || doc.players.Length == 0
                || doc.skeleton == null || doc.skeleton.parents == null
                || doc.skeleton.parents.Length < 1)
            { why = "wrong topology (no players or no skeleton.parents)"; return null; }
            // ShowFrame / JointLocal index frames and joints_flat without bounds checks: a short player or a 24-joint
            // frame in a 28-joint doc would throw every Update, so such a doc is refused here.
            if (doc.meta == null || doc.meta.frame_count <= 0) { why = "meta.frame_count missing"; return null; }
            int F = doc.meta.frame_count, J = doc.JointCount;
            for (int pi = 0; pi < doc.players.Length; pi++)
            {
                var p = doc.players[pi];
                if (p == null || p.frames == null || p.frames.Length != F)
                { why = $"player {pi}: {(p?.frames != null ? p.frames.Length : 0)} frames != meta.frame_count {F}"; return null; }
                for (int f = 0; f < F; f++)
                {
                    var fr = p.frames[f];
                    if (fr == null) { why = $"player {pi} frame {f}: null"; return null; }
                    int n = fr.joints_flat != null ? fr.joints_flat.Length : 0;
                    if (n != 0 && n != Stride * J)
                    { why = $"player {pi} frame {f}: joints_flat has {n} values, expected 0 or {Stride * J} ({J} joints x {Stride})"; return null; }
                }
            }
            return doc;
        }

        /// <summary>Loads + validates an analysis doc; null (error logged) when it is missing, malformed or doesn't fit.</summary>
        public static AnalysisDoc Load(string streamingAssetsRelativePath)
        {
            if (!JsonLoad.TryRead(streamingAssetsRelativePath, out string json, out string path, out string why))
            {
                Debug.LogError($"[Analysis] file {why}: {path}");
                return null;
            }
            var doc = Parse(json, out why);
            if (doc == null) Debug.LogError($"[Analysis] failed to load ({why}): {path}");
            return doc;
        }
    }
}
