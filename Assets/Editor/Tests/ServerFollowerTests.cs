using System.IO;
using NUnit.Framework;

namespace BadmintonVR.Analysis.Tests
{
    /// <summary>v0.8.2: the pipeline server's now_playing.json pointer and the frame-folder layouts Unity reads.</summary>
    public class ServerFollowerTests
    {
        string _dir;

        [SetUp] public void SetUp()
        {
            _dir = Path.Combine(Path.GetTempPath(), "bvr_follow_" + System.Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
        }

        [TearDown] public void TearDown() { try { Directory.Delete(_dir, true); } catch (IOException) { } }

        [Test] public void Run_pointer_resolves_relative_to_the_runs_folder()
        {
            string p = Path.Combine(_dir, "now_playing.json");
            File.WriteAllText(p, "{\"v\":1,\"seq\":7,\"kind\":\"run\",\"run\":\"live_singles_05x_60fps_x\",\"frames\":\"unity_live_frames\",\"title\":\"t\"}");
            var np = ServerFollower.Read(p);
            Assert.AreEqual(7, np.seq);
            Assert.AreEqual("run", np.kind);
            Assert.AreEqual(Path.Combine(_dir, "live_singles_05x_60fps_x"), ServerFollower.RunFolder(p, np));
        }

        [Test] public void Clip_pointer_maps_into_streaming_assets_analysis_and_drops_folders()
        {
            var np = new ServerFollower.NowPlaying { kind = "clip", file = "../../evil/singles_05x_pose24.analysis.json" };
            Assert.AreEqual("analysis/singles_05x_pose24.analysis.json", ServerFollower.ClipPath(np));
        }

        [Test] public void Missing_or_half_written_pointer_reads_as_null()
        {
            string p = Path.Combine(_dir, "now_playing.json");
            Assert.IsNull(ServerFollower.Read(p));
            File.WriteAllText(p, "{\"v\":1,\"se");
            Assert.IsNull(ServerFollower.Read(p));
        }

        [Test] public void Frame_folders_of_both_layouts_are_found()
        {
            string run = Path.Combine(_dir, "run_new"), old = Path.Combine(_dir, "run_old");
            Directory.CreateDirectory(Path.Combine(run, "unity_live_frames"));
            Directory.CreateDirectory(Path.Combine(old, "live", "frames"));
            File.WriteAllText(Path.Combine(run, "unity_live_frames", "manifest.json"), "{}");
            File.WriteAllText(Path.Combine(old, "live", "manifest.json"), "{}");
            File.WriteAllText(Path.Combine(run, "unity_live_frames", "frame_000000.json"), "{}");
            File.WriteAllText(Path.Combine(old, "live", "frames", "000000.json"), "{}");
            string fn = LiveFolderFeed.FramesFolder(run), fo = LiveFolderFeed.FramesFolder(old);
            Assert.AreEqual(Path.Combine(run, "unity_live_frames"), fn);
            Assert.AreEqual(Path.Combine(old, "live"), fo);
            Assert.AreEqual(Path.Combine(fn, "frame_000012.json"), LiveFolderFeed.FramePath(fn, 12));
            Assert.AreEqual(Path.Combine(fo, "frames", "000012.json"), LiveFolderFeed.FramePath(fo, 12));
            Assert.AreEqual(1, LiveFolderFeed.CountFrames(fn));
            Assert.AreEqual(1, LiveFolderFeed.CountFrames(fo));
            Assert.IsNull(LiveFolderFeed.FramesFolder(_dir));
        }
    }
}
