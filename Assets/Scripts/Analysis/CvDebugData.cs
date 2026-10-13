using System;
using System.IO;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    // Mirrors the cvdebug/1 sidecar (tools/analysis/cvdebug.py; field table in tools/README.md § CV debug + video overlay).
    // Flat arrays: joint facts at f*J + j, per-view facts at v*F + f, contact at f*6 + k; bit / row v = index in cameras[]
    // (every view of the base clip, sorted — held-out and excluded cameras included). -1 = unknown.

    [Serializable]
    public class CvCamera
    {
        public string name, video, offset_source;
        public int width, height, det_shift;
        public float[] K, R, t;
        public float offset, health_px, vfov_deg;
        public bool excluded, held_out;
        public float[] unity_pos, unity_rot;

        public bool HasPose => unity_pos != null && unity_pos.Length == 3 && unity_rot != null && unity_rot.Length == 4 && vfov_deg > 0f;
        public bool HasOffset => offset_source != "none";
        public bool Used => !excluded && !held_out;
        public Vector3 UnityPos => new Vector3(unity_pos[0], unity_pos[1], unity_pos[2]);
        // Python already converted to the Y-up analysis world (SP2 F6): assign directly, no flips.
        public Quaternion UnityRot => new Quaternion(unity_rot[0], unity_rot[1], unity_rot[2], unity_rot[3]);
        public float Aspect => height > 0 ? (float)width / height : 1f;
    }

    [Serializable]
    public class CvPlayer
    {
        public string id, label;
        public int[] frame_state, seen_mask, view_mask, err_dpx, sigma_mm, filled, det_index, lr_swapped, frame_err_px, contact;
    }

    [Serializable]
    public class PersonTrack
    {
        public int track_id;
        public string label, player_id;       // label: player | nonplayer | unknown
        public float score;                   // not a confidence for short tracks (SP3 F9): draw by label
        public int first_frame, last_frame, ambiguous, pinned;
        public float[] floor_xz;              // 2 per frame first..last; NoFloor = no candidate that frame
        public int[] support, det_index;
        public const float NoFloor = -9999f;

        public bool FloorAt(int frame, out Vector2 xz)
        {
            xz = default;
            int k = frame - first_frame;
            if (frame < first_frame || frame > last_frame || floor_xz == null || 2 * k + 1 >= floor_xz.Length) return false;
            float x = floor_xz[2 * k], z = floor_xz[2 * k + 1];
            if (x <= NoFloor + 1f || z <= NoFloor + 1f) return false;
            xz = new Vector2(x, z);
            return true;
        }
    }

    [Serializable] public class CvHotspot { public float x, z; public int frames; }
    [Serializable] public class CvViewStat { public string name; public float used_frac; }
    [Serializable] public class CvStats { public CvViewStat[] views; public float[] coverage_per_player; }

    [Serializable]
    public class CvDebugDoc
    {
        public const string Format = "cvdebug/1";
        const int MaxViews = 31;

        public string format, clip_id, source_analysis, source_build_id, variant, skeleton_type, time_ref_view, banner, proxy_clip;
        public float fps;
        public int frame_count, persons_dropped;
        public string[] joint_names, aux_names;
        public CvCamera[] cameras;
        public CvPlayer[] players;
        public PersonTrack[] persons;
        public CvHotspot[] hotspots;
        public CvStats stats;

        public int J => joint_names != null ? joint_names.Length : 0;
        public int V => cameras != null ? cameras.Length : 0;
        public string ProxyClip => string.IsNullOrEmpty(proxy_clip) ? clip_id : proxy_clip;

        public CvPlayer PlayerById(string id)
        {
            if (players == null) return null;
            foreach (var p in players) if (p.id == id) return p;
            return null;
        }

        public int CameraIndex(string name)
        {
            for (int v = 0; v < V; v++) if (cameras[v].name == name) return v;
            return -1;
        }

        /// <summary>Parses and checks the sidecar against the loaded analysis doc (analysis may be null in tests).
        /// Returns null with the reason when it doesn't fit — a mismatched sidecar would colour the wrong joints.</summary>
        public static CvDebugDoc Parse(string json, AnalysisDoc analysis, out string why)
        {
            if (!JsonLoad.TryParse(json, out CvDebugDoc d, out why)) return null;
            if (d.format != Format) { why = $"format {d.format} != {Format}"; return null; }
            if (d.V == 0 || d.V > MaxViews) { why = $"{d.V} cameras"; return null; }
            if (d.J == 0) { why = "no joint_names"; return null; }
            if (analysis != null && d.frame_count != analysis.FrameCount)
            { why = $"frame_count {d.frame_count} != analysis {analysis.FrameCount}"; return null; }
            if (analysis != null && !string.IsNullOrEmpty(analysis.clip_id) && !string.IsNullOrEmpty(d.clip_id) && d.clip_id != analysis.clip_id)
            { why = $"clip {d.clip_id} != analysis {analysis.clip_id}"; return null; }
            // pose24 docs carry meta.pose24.build_id: a sidecar left over from an earlier build would colour the wrong joints
            // (other docs' sidecars hold a sha1 of the doc bytes, not checked here)
            string bid = analysis?.meta?.pose24?.build_id;
            if (!string.IsNullOrEmpty(bid) && !string.IsNullOrEmpty(d.source_build_id) && d.source_build_id != bid)
            { why = $"from another build ({d.source_build_id} != {bid}): rerun analysis.cvdebug build"; return null; }
            int F = d.frame_count, J = d.J, V = d.V;
            foreach (var p in d.players ?? new CvPlayer[0])
            {
                if (Len(p.frame_state) != F || Len(p.seen_mask) != F * J || Len(p.view_mask) != F * J || Len(p.err_dpx) != F * J
                    || Len(p.sigma_mm) != F * J || Len(p.filled) != F * J || Len(p.det_index) != V * F
                    || Len(p.frame_err_px) != F                                // read every frame by the HUD summary
                    || !EmptyOr(p.lr_swapped, V * F) || !EmptyOr(p.contact, F * 6))
                { why = $"player {p.id}: array lengths don't match F={F} J={J} V={V}"; return null; }
            }
            if (d.players == null) d.players = new CvPlayer[0];
            if (d.persons == null) d.persons = new PersonTrack[0];
            if (d.hotspots == null) d.hotspots = new CvHotspot[0];
            return d;
        }

        static int Len(int[] a) => a != null ? a.Length : -1;
        static bool EmptyOr(int[] a, int n) => a == null || a.Length == 0 || a.Length == n;

        public static CvDebugDoc TryLoad(string streamingAssetsRelativePath, AnalysisDoc analysis)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            if (!JsonLoad.TryRead(streamingAssetsRelativePath, out string json, out string path, out string why))
            {
                if (why != "not found") Debug.LogWarning($"[Analysis] cvdebug sidecar {why}, ignored: {path}");
                return null;
            }
            var d = Parse(json, analysis, out why);
            if (d == null) { Debug.LogWarning($"[Analysis] cvdebug sidecar ignored ({why}): {path}"); return null; }
            Debug.Log($"[Analysis] cvdebug: {d.V} cameras, {d.players.Length} players, {d.persons.Length} persons " +
                      $"({sw.ElapsedMilliseconds} ms)");
            return d;
        }
    }
}
