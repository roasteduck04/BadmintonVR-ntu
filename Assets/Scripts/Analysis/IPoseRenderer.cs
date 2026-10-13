using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>
    /// A per-player pose renderer. Owns the GameObjects it creates under `root`.
    /// AnalysisPlayer computes ground-lifted joint positions + confidences each frame and
    /// hands them here; the concrete renderer decides how to draw them (skeleton, solid
    /// body, and later asset-backed skins). Radii/proportions are the renderer's own
    /// concern and stay off this interface.
    /// </summary>
    public interface IPoseRenderer
    {
        /// <summary>Create GameObjects under `root`. `parents` is skeleton.parents;
        /// `color` the player's base colour.</summary>
        void Build(Transform root, int[] parents, Color color);

        /// <summary>Update to the current frame. `jointLocal` is already ground-lifted;
        /// `conf` is per-joint confidence; `hasPose` false hides the body.</summary>
        void Apply(Vector3[] jointLocal, float[] conf, bool hasPose, float confidenceCutoff);

        /// <summary>Show/hide the whole renderer (used by player-focus).</summary>
        void SetActive(bool on);

        /// <summary>skeleton.joint_names of the loaded doc (SMPL-24, 28 or 58 joints), set
        /// before Build; renderers that map joints to rig bones resolve them by name.</summary>
        void SetJointNames(string[] names);

        /// <summary>Where this renderer actually drew joint `joint` last Apply, in `root`
        /// space. False when it draws the data joints as given (skeleton, solid body) —
        /// a retargeted character keeps its own bone lengths, so its hands differ.</summary>
        bool TryGetRenderedJoint(int joint, out Vector3 rootLocal);

        /// <summary>Release references to owned GameObjects (the root's children are
        /// destroyed by the caller).</summary>
        void Clear();
    }
}
