using UnityEditor;
using UnityEngine;
using BadmintonVR.Analysis;

namespace BadmintonVR.EditorTools
{
    /// <summary>
    /// Assembles the Phase-1 analysis scene without top-bar clicks: full court +
    /// one AnalysisPlayer at the scene root + a camera framing the court. Idempotent.
    /// </summary>
    public static class AnalysisSceneSetup
    {
        [MenuItem("Tools/Badminton/Setup Analysis Scene")]
        public static GameObject Build()
        {
            CourtBuilder.BuildCourt(CourtBuilder.Scope.Full);

            var existing = Object.FindFirstObjectByType<AnalysisPlayer>();
            GameObject go = existing != null ? existing.gameObject : new GameObject("AnalysisPlayer");
            var ap = existing != null ? existing : go.AddComponent<AnalysisPlayer>();
            go.transform.position = Vector3.zero;

            // viewer UI + debug HUD live on the same object, drive the same player
            // explicit null checks: GetComponent returns a fake-null object in the editor, so `??` would never add
            var ui = go.GetComponent<AnalysisViewerUI>(); if (ui == null) ui = go.AddComponent<AnalysisViewerUI>();
            ui.player = ap;
            var hud = go.GetComponent<AnalysisDebugHUD>(); if (hud == null) hud = go.AddComponent<AnalysisDebugHUD>();
            hud.player = ap;
            // SP4 cv visualisation (AnalysisPlayer also adds them at runtime when missing)
            if (go.GetComponent<CvDebugOverlay>() == null) go.AddComponent<CvDebugOverlay>();
            if (go.GetComponent<CameraRigGizmos>() == null) go.AddComponent<CameraRigGizmos>();
            if (go.GetComponent<CameraRayLines>() == null) go.AddComponent<CameraRayLines>();
            if (go.GetComponent<VideoBillboard>() == null) go.AddComponent<VideoBillboard>();
            if (go.GetComponent<NonPlayerMarkers>() == null) go.AddComponent<NonPlayerMarkers>();
            if (go.GetComponent<VariantGhosts>() == null) go.AddComponent<VariantGhosts>();

            var cam = Camera.main;
            if (cam == null)
            {
                var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
                cam = camGo.AddComponent<Camera>();
            }
            var orbit = cam.GetComponent<OrbitCamera>(); if (orbit == null) orbit = cam.gameObject.AddComponent<OrbitCamera>();
            orbit.Frame();

            Debug.Log("[Analysis] scene assembled: court + AnalysisPlayer + UI + HUD + cv components + orbit camera");
            return go;
        }
    }
}
