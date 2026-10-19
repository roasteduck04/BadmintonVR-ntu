using System;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Threading;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>
    /// Plays a live run folder while the runner writes it (simulated live; works on a finished run too).
    ///   &lt;run&gt;/unity_live_frames/manifest.json          header (skeleton, cameras, players, timeline length)
    ///   &lt;run&gt;/unity_live_frames/frame_000123.json     one output frame; written atomically (tmp + rename), in step order
    /// Runs from v0.8.0 / v0.8.1 use &lt;run&gt;/live/manifest.json + &lt;run&gt;/live/frames/000123.json; both are read.
    /// A background thread polls for the next frame file (no FileSystemWatcher: unreliable on Windows / Mono) and parses
    /// it off the main thread; Update hands the frames to AnalysisPlayer. Stats for the LIVE badge: received, incoming
    /// fps, file age (write time -> shown, same machine) and whether the run has finished (final doc written + idle).
    /// </summary>
    public class LiveFolderFeed : MonoBehaviour
    {
        public AnalysisPlayer player;
        public string liveDir;
        public bool Connected { get; private set; }
        public bool Finished { get; private set; }
        public int Received { get; private set; }
        public float IncomingFps { get; private set; }
        public float FileAgeMs { get; private set; }
        public string Status { get; private set; } = "not connected";

        [Serializable] class LivePlayerEntry { public string id; public bool valid; public int quality; public bool warmup, late, interpolated; public PlayerFrame frame; }
        [Serializable] class LiveFrameFile { public int step, ref_frame, frame_id, quality; public float time; public LivePlayerEntry[] players; }
        struct Item { public LiveFrameFile f; public DateTime written; }

        readonly ConcurrentQueue<Item> _q = new ConcurrentQueue<Item>();
        Thread _thread;
        volatile bool _stop;
        volatile bool _idleEnd;
        int[] _lastFrame;
        float _rateT0; int _rateN;

        void Reset() => player = GetComponent<AnalysisPlayer>();

        public const string FramesDir = "unity_live_frames", OldDir = "live";

        /// <summary>The frame folder of a run (or the folder itself when it holds manifest.json); null when none.</summary>
        public static string FramesFolder(string runOrFramesDir)
        {
            if (string.IsNullOrEmpty(runOrFramesDir)) return null;
            foreach (var d in new[] { runOrFramesDir, Path.Combine(runOrFramesDir, FramesDir), Path.Combine(runOrFramesDir, OldDir) })
                if (File.Exists(Path.Combine(d, "manifest.json"))) return d;
            return null;
        }

        /// <summary>Path of frame `step` in a frame folder (new layout: frame_000123.json; old: frames/000123.json).</summary>
        public static string FramePath(string framesDir, int step) =>
            Path.GetFileName(framesDir.TrimEnd('/', '\\')) == OldDir
                ? Path.Combine(framesDir, "frames", $"{step:000000}.json")
                : Path.Combine(framesDir, $"frame_{step:000000}.json");

        /// <summary>Number of frame files written so far in a frame folder (either layout).</summary>
        public static int CountFrames(string framesDir)
        {
            try
            {
                return Path.GetFileName(framesDir.TrimEnd('/', '\\')) == OldDir
                    ? (Directory.Exists(Path.Combine(framesDir, "frames")) ? Directory.GetFiles(Path.Combine(framesDir, "frames"), "*.json").Length : 0)
                    : Directory.GetFiles(framesDir, "frame_*.json").Length;
            }
            catch (Exception) { return 0; }
        }

        /// <summary>The newest run folder under &lt;project&gt;/runs that has unity_live_frames/manifest.json (or the old
        /// live/manifest.json); null when none.</summary>
        public static string NewestRun(string runsRoot)
        {
            if (string.IsNullOrEmpty(runsRoot) || !Directory.Exists(runsRoot)) return null;
            try
            {
                return Directory.GetFiles(runsRoot, "manifest.json", SearchOption.AllDirectories)
                    .Where(p => { var n = Path.GetFileName(Path.GetDirectoryName(p)); return n == FramesDir || n == OldDir; })
                    .OrderByDescending(File.GetLastWriteTimeUtc)
                    .Select(p => Path.GetDirectoryName(Path.GetDirectoryName(p))).FirstOrDefault();
            }
            catch (Exception) { return null; }
        }

        public bool Connect(string runOrLiveDir)
        {
            Disconnect();
            if (player == null) player = GetComponent<AnalysisPlayer>();
            string dir = FramesFolder(runOrLiveDir);
            if (dir == null) { Status = $"no {FramesDir}/manifest.json in {runOrLiveDir}"; Debug.LogWarning($"[Analysis] live: {Status}"); return false; }
            string man = Path.Combine(dir, "manifest.json");
            AnalysisDoc header;
            try { header = JsonUtility.FromJson<AnalysisDoc>(File.ReadAllText(man)); }
            catch (Exception e) { Status = $"manifest unreadable: {e.Message}"; Debug.LogError($"[Analysis] live: {Status}"); return false; }
            string runName = Path.GetFileName(Path.GetDirectoryName(dir.TrimEnd('/', '\\')) ?? dir);
            if (!player.AttachLive(header, $"live:{runName}")) { Status = "manifest not usable"; return false; }
            liveDir = dir;
            _lastFrame = Enumerable.Repeat(-1, header.players.Length).ToArray();
            Received = 0; Finished = false; _idleEnd = false; _stop = false;
            _rateT0 = Time.unscaledTime; _rateN = 0;
            _thread = new Thread(Poll) { IsBackground = true, Name = "live-folder-feed" };
            _thread.Start();
            Connected = true;
            Status = "waiting for frames";
            return true;
        }

        public void Disconnect()
        {
            _stop = true;
            if (_thread != null && _thread.IsAlive) _thread.Join(500);
            _thread = null;
            if (Connected && player != null) player.DetachLive();
            Connected = false;
            Status = "not connected";
            while (_q.TryDequeue(out _)) { }
        }

        void OnDisable() => Disconnect();

        void Poll()
        {
            string runDir = Path.GetDirectoryName(liveDir.TrimEnd('/', '\\'));
            int next = 0;
            DateTime lastNew = DateTime.UtcNow;
            while (!_stop)
            {
                string p = FramePath(liveDir, next);
                if (File.Exists(p))
                {
                    LiveFrameFile f = null;
                    for (int k = 0; k < 20 && f == null && !_stop; k++)
                    {
                        try { f = JsonUtility.FromJson<LiveFrameFile>(File.ReadAllText(p)); }
                        catch (IOException) { Thread.Sleep(5); }               // the writer is still renaming it
                        catch (Exception) { break; }                           // malformed: skip this frame
                    }
                    if (f != null) _q.Enqueue(new Item { f = f, written = File.GetLastWriteTimeUtc(p) });
                    next++;
                    lastNew = DateTime.UtcNow;
                    continue;
                }
                // the run is over when its final analysis doc exists and nothing new arrived for 2 s
                if ((DateTime.UtcNow - lastNew).TotalSeconds > 2 && runDir != null
                    && Directory.GetFiles(runDir, "*.analysis.json").Length > 0)
                { _idleEnd = true; break; }
                Thread.Sleep(5);
            }
        }

        void Update()
        {
            if (!Connected || player == null) return;
            int n = 0;
            while (_q.TryDequeue(out var it) && n < 240)
            {
                var f = it.f;
                if (f.players != null && player.Doc != null)
                    foreach (var e in f.players)
                    {
                        int pi = Array.FindIndex(player.Doc.players, p => p.id == e.id);
                        if (pi < 0 || e.frame == null) continue;
                        if (!e.valid) e.frame.joints_flat = null;
                        player.PushLiveFrame(f.frame_id, pi, e.frame, e.quality, _lastFrame[pi]);
                        _lastFrame[pi] = f.frame_id;
                    }
                FileAgeMs = (float)(DateTime.UtcNow - it.written).TotalMilliseconds;
                Received++; n++; _rateN++;
            }
            float dt = Time.unscaledTime - _rateT0;
            if (dt >= 1f) { IncomingFps = _rateN / dt; _rateN = 0; _rateT0 = Time.unscaledTime; }
            if (_idleEnd) Finished = true;
            Status = Finished ? $"run finished · {Received} frames" : Received == 0 ? "waiting for frames" : $"{Received} frames";
        }
    }
}
