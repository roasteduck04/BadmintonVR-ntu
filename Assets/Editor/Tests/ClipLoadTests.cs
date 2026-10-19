using System.IO;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace BadmintonVR.Analysis.Tests
{
    /// <summary>v0.8.1: AnalysisPlayer.ReadClip, the clip read + parse + validate that runs on a worker thread.</summary>
    public class ClipLoadTests
    {
        string _dir;

        [SetUp] public void MakeDir()
        {
            _dir = Path.Combine(Path.GetTempPath(), "bvr_clipload_" + System.Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_dir);
            _ = JsonLoad.Root;             // as AnalysisPlayer.StartLoad does: the worker only reads the cached path
        }

        [TearDown] public void RemoveDir() { if (Directory.Exists(_dir)) Directory.Delete(_dir, true); }

        static string Fixture(string name) =>
            File.ReadAllText(Path.Combine(Directory.GetParent(Application.dataPath).FullName, "tools", "tests", "fixtures", name));

        /// <summary>An analysis doc with nFrames frames of 2 players (2 joints, lowest joint at y), optionally with a cvdebug block.</summary>
        static string DocJson(string clipId, int nFrames, float y, string cvdebug = null)
        {
            var sb = new System.Text.StringBuilder();
            sb.Append("{\"schema_version\":\"4.1\",\"clip_id\":\"").Append(clipId).Append("\",\"meta\":{\"fps\":60,\"frame_count\":")
              .Append(nFrames).Append("},\"skeleton\":{\"parents\":[-1,0]},\"players\":[");
            for (int p = 0; p < 2; p++)
            {
                sb.Append(p > 0 ? "," : "").Append("{\"id\":\"P").Append(p + 1).Append("\",\"frames\":[");
                for (int f = 0; f < nFrames; f++)
                    sb.Append(f > 0 ? "," : "").Append("{\"frame_id\":").Append(f).Append(",\"joints_flat\":[0,")
                      .Append(y.ToString(System.Globalization.CultureInfo.InvariantCulture)).Append(",0,1,0,1,0,1],\"position\":[0,0]}");
                sb.Append("]}");
            }
            sb.Append("]");
            if (cvdebug != null) sb.Append(",\"cvdebug\":").Append(cvdebug);
            return sb.Append("}").ToString();
        }

        static AnalysisPlayer.ClipData ReadOnWorker(string file, CancellationToken cancel, out int workerThread)
        {
            int tid = -1;
            var task = Task.Run(() => AnalysisPlayer.ReadClip(file, cancel, _ => tid = Thread.CurrentThread.ManagedThreadId));
            Assert.IsTrue(task.Wait(30000), "ReadClip did not finish");
            workerThread = tid;
            return task.Result;
        }

        [Test] public void Clip_and_sidecars_parse_on_a_worker_thread()
        {
            string cvJson = Fixture("cvdebug_small.json");
            var cvHead = JsonUtility.FromJson<CvDebugDoc>(cvJson);
            string file = Path.Combine(_dir, "syn.analysis.json");
            File.WriteAllText(file, DocJson(cvHead.clip_id, cvHead.frame_count, 0.05f, cvJson));
            File.WriteAllText(Path.Combine(_dir, "syn.shuttle_improved.json"),
                "{\"format\":\"shuttle_variant/1\",\"frame_count\":" + cvHead.frame_count + ",\"frames\":["
                + string.Join(",", System.Linq.Enumerable.Repeat("{}", cvHead.frame_count)) + "]}");
            File.WriteAllText(Path.Combine(_dir, "syn.racket_rigid.json"), "{\"frame_count\":40,\"players\":[");  // half-written

            var d = ReadOnWorker(file, CancellationToken.None, out int worker);

            Assert.AreNotEqual(Thread.CurrentThread.ManagedThreadId, worker, "ReadClip ran on the main thread");
            Assert.IsNotNull(d.doc, d.error);
            Assert.IsNull(d.error);
            Assert.AreEqual(cvHead.frame_count, d.doc.FrameCount);
            Assert.AreEqual(2, d.doc.players.Length);
            Assert.AreEqual(new FileInfo(file).Length, d.bytes);
            Assert.AreEqual(-0.05f, d.lift, 1e-6f);                                      // GroundLift, computed on the worker
            Assert.IsNotNull(d.cv, "embedded cvdebug");
            Assert.AreEqual(cvHead.cameras.Length, d.cv.V);
            CollectionAssert.AreEquivalent(new[] { "improved" }, d.shuttles.Keys);
            Assert.AreEqual(0, d.rackets.Count);
            // the worker logs nothing itself: the half-written racket sidecar is a warning in the load's log, replayed on
            // the main thread when the result is applied (that is when the viewer's toast can see it)
            Assert.IsTrue(System.Linq.Enumerable.Any(d.log.Items, m => m.type == LogType.Warning && m.msg.Contains("racket sidecar")));
            LogAssert.NoUnexpectedReceived();
            LogAssert.Expect(LogType.Warning, new System.Text.RegularExpressions.Regex("racket sidecar"));
            d.log.Replay();
        }

        [Test] public void A_missing_or_broken_clip_is_an_error_not_an_exception()
        {
            var d = ReadOnWorker(Path.Combine(_dir, "none.analysis.json"), CancellationToken.None, out _);
            Assert.IsNull(d.doc);
            StringAssert.Contains("not found", d.error);
            Assert.AreEqual(LogType.Error, d.log.Items[0].type);

            string file = Path.Combine(_dir, "half.analysis.json");
            string json = DocJson("t", 3, 0f);
            File.WriteAllText(file, json.Substring(0, json.Length / 2));                 // a pipeline run still writing it
            d = ReadOnWorker(file, CancellationToken.None, out _);
            Assert.IsNull(d.doc);
            StringAssert.Contains("failed to load", d.error);
            LogAssert.NoUnexpectedReceived();
        }

        [Test] public void The_reader_skips_a_bom_and_reuses_its_buffer_safely()
        {
            string big = Path.Combine(_dir, "big.json"), small = Path.Combine(_dir, "small.json");
            File.WriteAllText(big, "{\"clip_id\":\"" + new string('x', 5000) + "\"}", new System.Text.UTF8Encoding(true));   // with BOM
            File.WriteAllText(small, "{\"clip_id\":\"é\"}", new System.Text.UTF8Encoding(false));
            Assert.IsTrue(JsonLoad.TryRead(big, out string a, out _, out string why), why);
            Assert.AreEqual('{', a[0]);
            Assert.IsTrue(JsonLoad.TryRead(small, out string b, out _, out why), why);     // a shorter file in the reused buffer
            Assert.AreEqual("{\"clip_id\":\"é\"}", b);
            var reads = new Task<string>[8];
            for (int i = 0; i < reads.Length; i++)
            {
                string f = i % 2 == 0 ? big : small;
                reads[i] = Task.Run(() => JsonLoad.TryRead(f, out string t, out _, out _) ? t : null);
            }
            Assert.IsTrue(Task.WaitAll(reads, 30000));
            for (int i = 0; i < reads.Length; i++) Assert.AreEqual(i % 2 == 0 ? a : b, reads[i].Result, $"read {i}");
            Assert.IsFalse(JsonLoad.TryRead(Path.Combine(_dir, "none.json"), out _, out _, out why));
            Assert.AreEqual("not found", why);
        }

        [Test] public void A_cancelled_read_returns_no_doc()
        {
            string file = Path.Combine(_dir, "t.analysis.json");
            File.WriteAllText(file, DocJson("t", 5, 0f));
            var cts = new CancellationTokenSource();
            cts.Cancel();                                                                // superseded by a newer click
            var d = ReadOnWorker(file, cts.Token, out _);
            Assert.IsNull(d.doc);
            Assert.AreEqual("cancelled", d.error);
            Assert.AreEqual(0, d.log.Items.Count);
        }
    }
}
