using System;
using System.Diagnostics;
using System.Threading;
using NUnit.Framework;

namespace BadmintonVR.Analysis.Tests
{
    /// <summary>v0.8.1: KeyedLoads, the per-file worker reads VariantGhosts uses to turn ghost builds on without a freeze.</summary>
    public class GhostLoadTests
    {
        sealed class Box { public string v; public Box(string v) { this.v = v; } }

        /// <summary>Poll like the main thread does (one take per call) until a finished job comes out, or time out.</summary>
        static bool Take(KeyedLoads<Box> loads, out string key, out Box result, out Exception error, int timeoutMs = 10000)
        {
            var sw = Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                if (loads.TryTakeFinished(out key, out result, out error)) return true;
                Thread.Sleep(2);
            }
            key = null; result = null; error = null;
            return false;
        }

        [Test] public void A_read_runs_on_a_worker_and_is_taken_once()
        {
            var loads = new KeyedLoads<Box>();
            int main = Thread.CurrentThread.ManagedThreadId, worker = main;
            Assert.IsTrue(loads.Start("a", _ => { worker = Thread.CurrentThread.ManagedThreadId; return new Box("A"); }));
            Assert.IsTrue(loads.IsPending("a"));
            Assert.IsTrue(Take(loads, out var key, out var r, out var err));
            Assert.AreEqual("a", key); Assert.AreEqual("A", r.v); Assert.IsNull(err);
            Assert.AreNotEqual(main, worker, "the read ran on the main thread");
            Assert.IsFalse(loads.IsPending("a")); Assert.AreEqual(0, loads.Count);
            Assert.IsFalse(loads.TryTakeFinished(out _, out _, out _), "a result came out twice");
        }

        [Test] public void A_key_already_pending_is_not_started_again()
        {
            var loads = new KeyedLoads<Box>();
            var gate = new ManualResetEventSlim(false);
            int reads = 0;
            Func<CancellationToken, Box> read = _ => { Interlocked.Increment(ref reads); gate.Wait(5000); return new Box("x"); };
            Assert.IsTrue(loads.Start("a", read));
            Assert.IsFalse(loads.Start("a", read));
            gate.Set();
            Assert.IsTrue(Take(loads, out _, out _, out _));
            Assert.AreEqual(1, reads);
        }

        [Test] public void A_cancelled_read_is_signalled_and_its_result_never_taken()
        {
            var loads = new KeyedLoads<Box>();
            var started = new ManualResetEventSlim(false); var gate = new ManualResetEventSlim(false);
            bool sawCancel = false;
            loads.Start("a", tok => { started.Set(); gate.Wait(5000); sawCancel = tok.IsCancellationRequested; return new Box("stale"); });
            Assert.IsTrue(started.Wait(5000));
            loads.Cancel("a");                                   // ghost toggled off while its file is being read
            Assert.IsFalse(loads.IsPending("a"));
            gate.Set();
            Assert.IsFalse(Take(loads, out _, out _, out _, 300), "a cancelled read was taken");
            Assert.IsTrue(sawCancel, "the worker's token was not cancelled");
        }

        [Test] public void A_finished_but_untaken_result_is_dropped_by_cancel()
        {
            var loads = new KeyedLoads<Box>();
            var done = new ManualResetEventSlim(false);
            loads.Start("a", _ => { done.Set(); return new Box("stale"); });
            Assert.IsTrue(done.Wait(5000));
            Thread.Sleep(20);                                    // the task completes right after the read returns
            loads.Cancel("a");
            Assert.IsFalse(loads.TryTakeFinished(out _, out _, out _));
            // turned on again: a new read, its own result
            loads.Start("a", _ => new Box("fresh"));
            Assert.IsTrue(Take(loads, out _, out var r, out _));
            Assert.AreEqual("fresh", r.v);
        }

        [Test] public void A_read_that_throws_comes_out_as_an_error()
        {
            var loads = new KeyedLoads<Box>();
            loads.Start("bad", _ => throw new InvalidOperationException("boom"));
            Assert.IsTrue(Take(loads, out var key, out var r, out var err));
            Assert.AreEqual("bad", key); Assert.IsNull(r);
            Assert.IsInstanceOf<InvalidOperationException>(err); Assert.AreEqual("boom", err.Message);
        }

        [Test] public void CancelWhere_drops_only_the_rejected_keys()
        {
            var loads = new KeyedLoads<Box>();
            var gate = new ManualResetEventSlim(false);
            foreach (var k in new[] { "keep1", "drop", "keep2" }) loads.Start(k, _ => { gate.Wait(5000); return new Box(k); });
            loads.CancelWhere(k => k.StartsWith("drop"));        // a newer clip loaded: no longer a ghost candidate
            Assert.AreEqual(2, loads.Count); Assert.IsFalse(loads.IsPending("drop"));
            gate.Set();
            Assert.IsTrue(Take(loads, out var k1, out _, out _)); Assert.IsTrue(Take(loads, out var k2, out _, out _));
            CollectionAssert.AreEquivalent(new[] { "keep1", "keep2" }, new[] { k1, k2 });
            loads.CancelAll(); Assert.AreEqual(0, loads.Count);
        }
    }
}
