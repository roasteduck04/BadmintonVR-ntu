using UnityEngine;

namespace BadmintonVR.Analysis
{
    public enum CvColorMode { Off, Confidence, ViewsUsed, SelectedView, ReprojError, Sigma, Filled, FrameState }

    /// <summary>Per-joint colours for CvDebugOverlay. Grey = unknown / not measured. σ ranks errors only weakly
    /// (SP1 F17: Spearman ≤ 0.27): Confidence and Sigma are coarse cues.</summary>
    public static class CvColors
    {
        public static readonly Color Unknown = new Color(0.55f, 0.55f, 0.55f, 1f);
        static readonly Color Orange = new Color(1f, 0.55f, 0.1f, 1f);
        static readonly Color Magenta = new Color(1f, 0.2f, 0.9f, 1f);

        /// <summary>green at `good`, yellow at `mid`, red at `bad` and above.</summary>
        public static Color Ramp(float x, float good, float mid, float bad)
        {
            if (x <= good) return Color.green;
            if (x <= mid) return Color.Lerp(Color.green, Color.yellow, (x - good) / Mathf.Max(mid - good, 1e-6f));
            return Color.Lerp(Color.yellow, Color.red, Mathf.Clamp01((x - mid) / Mathf.Max(bad - mid, 1e-6f)));
        }

        public static Color ForJoint(CvColorMode mode, float conf, int seen, int view, int errDpx, int sigmaMm, int filled,
                                     int frameState, int selectedView)
        {
            switch (mode)
            {
                case CvColorMode.Confidence: return Color.Lerp(Color.red, Color.green, Mathf.Clamp01(conf));
                case CvColorMode.ViewsUsed:
                    switch (CvMath.PopCount(view))
                    {
                        case 0: return Unknown;
                        case 1: return Color.red;
                        case 2: return Orange;
                        case 3: return Color.yellow;
                        default: return Color.green;
                    }
                case CvColorMode.SelectedView:
                    if (selectedView < 0 || !CvMath.HasBit(seen, selectedView)) return Unknown;
                    return CvMath.HasBit(view, selectedView) ? Color.green : Color.red;
                case CvColorMode.ReprojError: return errDpx < 0 ? Unknown : Ramp(errDpx / 10f, 2f, 10f, 20f);
                case CvColorMode.Sigma: return sigmaMm < 0 ? Unknown : Ramp(sigmaMm, 20f, 60f, 150f);
                case CvColorMode.Filled: return filled != 0 ? Magenta : Color.white;
                case CvColorMode.FrameState:
                    switch (frameState)
                    {
                        case 1: return Color.green;      // measured
                        case 2: return Color.yellow;     // gap-filled
                        case 3: return Orange;           // prior-only
                        default: return Unknown;         // missing
                    }
                default: return Color.white;
            }
        }

        public static string Legend(CvColorMode mode)
        {
            switch (mode)
            {
                case CvColorMode.Confidence: return "confidence: red 0 → green 1";
                case CvColorMode.ViewsUsed: return "inlier views: red 1 · orange 2 · yellow 3 · green 4+ · grey 0";
                case CvColorMode.SelectedView: return "selected view: green inlier · red rejected · grey not seen";
                case CvColorMode.ReprojError: return "reprojection: green ≤2 px · yellow 10 · red ≥20 · grey unknown";
                case CvColorMode.Sigma: return "σ: green ≤20 mm · yellow 60 · red ≥150 · grey unknown";
                case CvColorMode.Filled: return "magenta = gap-filled";
                case CvColorMode.FrameState: return "frame: green measured · yellow gap-filled · orange prior-only · grey missing";
                default: return "";
            }
        }

        public static CvColorMode Next(CvColorMode m, bool hasSidecar)
        {
            if (!hasSidecar) return m == CvColorMode.Off ? CvColorMode.Confidence : CvColorMode.Off;
            return m == CvColorMode.FrameState ? CvColorMode.Off : (CvColorMode)((int)m + 1);
        }
    }
}
