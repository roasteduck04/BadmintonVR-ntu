using NUnit.Framework;

namespace BadmintonVR.Analysis.Tests
{
    /// <summary>SP4 harness check: EditMode tests run and see the runtime types.</summary>
    public class CvSmokeTests
    {
        [Test] public void Harness_runs() => Assert.AreEqual(2, 1 + 1);

        [Test] public void Runtime_types_are_visible() => Assert.IsNotNull(typeof(BadmintonVR.Analysis.AnalysisDoc));
    }
}
