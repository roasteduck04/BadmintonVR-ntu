using System;
using System.IO;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>
    /// v0.8.2: Unity follows what the pipeline server plays (tools/pipeline_server). The server writes
    /// &lt;project&gt;/runs/now_playing.json; on every new "seq" this loads the build or connects the live run:
    ///   {"seq": 7, "kind": "run",  "run": "live_singles_05x_60fps_…", "frames": "unity_live_frames", "title": "…"}
    ///   {"seq": 8, "kind": "clip", "file": "singles_05x_pose24.analysis.json", "title": "…"}   (StreamingAssets/analysis)
    ///   {"seq": 9, "kind": "stop"}                                                            (keep the current view)
    /// Paths are names relative to the runs folder / StreamingAssets, so a server in WSL and Unity on Windows agree.
    /// While a pointer file exists the viewer runs in server mode (no clip browser, no live box): the user only moves
    /// the camera and toggles the display. Added to the AnalysisPlayer at runtime.
    /// </summary>
    public class ServerFollower : MonoBehaviour
    {
        public AnalysisPlayer player;
        [Tooltip("now_playing.json written by the pipeline server; empty = <project>/runs/now_playing.json")]
        public string pointerPath = "";
        public float pollSeconds = 0.5f;

        /// <summary>A pointer file exists (the server has played something since it was started).</summary>
        public bool Active { get; private set; }
        public string Title { get; private set; } = "";
        public string Kind { get; private set; } = "";
        public int Seq { get; private set; } = -1;
        public string LastError { get; private set; } = "";

        [Serializable]
        public class NowPlaying { public int v, seq; public string kind, run, frames, file, title, written; }

        float _next, _nextBeat;
        DateTime _stamp;

        /// <summary>Written next to the pointer every second while Unity is in Play mode; the server reads it to tell
        /// whether Unity is ready before "Start streaming to Unity" (stale after a few seconds = not in Play mode).</summary>
        [Serializable]
        public class UnityStatus { public int v = 1; public bool playing; public double t; public string mode, title; public int received, seq; public float behind_s, fps_in; }

        public static string StatusPath(string pointer) => Path.Combine(Path.GetDirectoryName(pointer) ?? ".", "unity_status.json");

        public static string DefaultPointer() =>
            Path.Combine(Path.GetDirectoryName(Application.dataPath) ?? ".", "runs", "now_playing.json");

        string Pointer => string.IsNullOrEmpty(pointerPath) ? DefaultPointer() : pointerPath;

        /// <summary>The pointer's contents, or null when it is missing or unreadable (the server may be mid-write).</summary>
        public static NowPlaying Read(string path)
        {
            try
            {
                if (!File.Exists(path)) return null;
                using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
                using (var sr = new StreamReader(fs))
                {
                    var np = JsonUtility.FromJson<NowPlaying>(sr.ReadToEnd());
                    return np != null && !string.IsNullOrEmpty(np.kind) ? np : null;
                }
            }
            catch (Exception) { return null; }
        }

        /// <summary>The run folder a "run" pointer names (relative to the pointer's folder, or absolute).</summary>
        public static string RunFolder(string pointer, NowPlaying np)
        {
            if (np == null || string.IsNullOrEmpty(np.run)) return null;
            return Path.IsPathRooted(np.run) ? np.run : Path.Combine(Path.GetDirectoryName(pointer) ?? ".", np.run);
        }

        /// <summary>The StreamingAssets-relative path a "clip" pointer names (analysis/&lt;file&gt;).</summary>
        public static string ClipPath(NowPlaying np) =>
            np == null || string.IsNullOrEmpty(np.file) ? null : "analysis/" + Path.GetFileName(np.file);

        void Reset() => player = GetComponent<AnalysisPlayer>();

        void Update()
        {
            if (Time.unscaledTime >= _nextBeat) { _nextBeat = Time.unscaledTime + 1f; WriteStatus(true); }
            if (Time.unscaledTime < _next) return;
            _next = Time.unscaledTime + Mathf.Max(0.1f, pollSeconds);
            if (player == null) player = GetComponent<AnalysisPlayer>();
            string p = Pointer;
            if (!File.Exists(p)) { Active = false; return; }
            Active = true;
            DateTime t;
            try { t = File.GetLastWriteTimeUtc(p); } catch (Exception) { return; }
            if (t == _stamp && Seq >= 0) return;
            var np = Read(p);
            if (np == null) return;                       // half written: try again next poll
            _stamp = t;
            if (np.seq == Seq) return;
            Seq = np.seq;
            Apply(p, np);
        }

        void OnDisable() => WriteStatus(false);
        void OnApplicationQuit() => WriteStatus(false);

        void WriteStatus(bool playing)
        {
            try
            {
                string pointer = Pointer;
                string dir = Path.GetDirectoryName(pointer);
                if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return;
                var feed = player != null ? player.GetComponent<LiveFolderFeed>() : null;
                var st = new UnityStatus
                {
                    playing = playing,
                    t = (DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds,
                    mode = feed != null && feed.Broadcasting ? "broadcast" : feed != null && feed.Connected ? "replay" : player != null && player.Doc != null ? "clip" : "idle",
                    title = Title, seq = Seq,
                    received = feed != null ? feed.Received : 0,
                    behind_s = feed != null && feed.Broadcasting ? feed.BehindS : 0f,
                    fps_in = feed != null ? feed.IncomingFps : 0f,
                };
                string path = StatusPath(pointer), tmp = path + ".tmp";
                File.WriteAllText(tmp, JsonUtility.ToJson(st));
                if (File.Exists(path)) File.Replace(tmp, path, null); else File.Move(tmp, path);
            }
            catch (Exception) { }                         // never break playback over a status file
        }

        void Apply(string pointer, NowPlaying np)
        {
            Kind = np.kind;
            if (np.kind == "stop") return;
            Title = string.IsNullOrEmpty(np.title) ? (np.run ?? np.file ?? "") : np.title;
            var feed = player.GetComponent<LiveFolderFeed>();
            if (np.kind == "run")
            {
                if (feed == null) feed = player.gameObject.AddComponent<LiveFolderFeed>();
                feed.player = player;
                string run = RunFolder(pointer, np);
                if (!feed.Connect(run)) { LastError = feed.Status; Debug.LogWarning($"[Analysis] server: {feed.Status}"); }
                else LastError = "";
            }
            else if (np.kind == "clip")
            {
                if (feed != null && feed.Connected) feed.Disconnect();
                string clip = ClipPath(np);
                if (clip != null) player.LoadClip(clip);
                LastError = "";
            }
        }
    }
}
