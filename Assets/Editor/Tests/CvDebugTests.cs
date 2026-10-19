using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace BadmintonVR.Analysis.Tests
{
    public class CvDebugTests
    {
        static string Fix(string name) =>
            Path.Combine(Directory.GetParent(Application.dataPath).FullName, "tools", "tests", "fixtures", name);

        [System.Serializable] class FrameCase { public int f; public float offset; public int det_shift; public int expected; }
        [System.Serializable] class ProjCase { public string camera; public float[] point, uv; }
        [System.Serializable] class ShiftCase { public string[] a_names, b_names; public float[] a_offsets, b_offsets; public float shift; public bool known; }
        [System.Serializable] class Cases { public FrameCase[] frames; public ProjCase[] projections; public ShiftCase[] shifts; }

        static Cases LoadCases() => JsonUtility.FromJson<Cases>(File.ReadAllText(Fix("cv_cases.json")));

        static CvDebugDoc LoadFixture()
        {
            var doc = CvDebugDoc.Parse(File.ReadAllText(Fix("cvdebug_small.json")), null, out string why);
            Assert.IsNotNull(doc, why);
            return doc;
        }

        [Test] public void Parse_reads_the_python_fixture()
        {
            var d = LoadFixture();
            Assert.AreEqual(CvDebugDoc.Format, d.format);
            Assert.AreEqual(28, d.J);
            Assert.AreEqual(4, d.V);
            Assert.AreEqual("syn", d.ProxyClip);
            foreach (var p in d.players)
            {
                Assert.AreEqual(d.frame_count * d.J, p.seen_mask.Length);
                Assert.AreEqual(d.V * d.frame_count, p.det_index.Length);
                Assert.AreEqual(d.frame_count * 6, p.contact.Length);
            }
            Assert.AreEqual(3, d.persons.Length);
            Assert.AreEqual(1, d.hotspots.Length);
        }

        [Test] public void Parse_rejects_mismatched_sidecar()
        {
            string json = File.ReadAllText(Fix("cvdebug_small.json"));
            var doc = new AnalysisDoc { meta = new AnalysisMeta { frame_count = 999 }, players = new Player[0] };
            Assert.IsNull(CvDebugDoc.Parse(json, doc, out string why));
            StringAssert.Contains("frame", why);
            Assert.IsNull(CvDebugDoc.Parse("{\"format\":\"other/1\"}", null, out why));
            StringAssert.Contains("format", why);
            Assert.IsNull(CvDebugDoc.Parse("not json", null, out why));
        }

        [Test] public void Unknown_sentinels_stay_minus_one()
        {
            var d = LoadFixture();
            int spine2 = System.Array.IndexOf(d.joint_names, "spine2");
            var p = d.players[0];
            Assert.AreEqual(-1, p.err_dpx[CvMath.Idx(0, spine2, d.J)]);   // spine joints have no 2D error source (SP2)
        }

        [Test] public void Rays_and_view_counts_ignore_held_out_cameras()
        {
            var d = LoadFixture();
            int c = d.CameraIndex("c"), dd = d.CameraIndex("d");
            Assert.IsTrue(d.cameras[c].excluded && !d.cameras[c].Used);
            Assert.IsTrue(d.cameras[dd].held_out && !d.cameras[dd].Used);
            foreach (var p in d.players)
                for (int i = 0; i < p.seen_mask.Length; i++)
                {
                    Assert.IsFalse(CvMath.HasBit(p.seen_mask[i], c));
                    Assert.IsFalse(CvMath.HasBit(p.seen_mask[i], dd));
                    Assert.LessOrEqual(CvMath.PopCount(p.view_mask[i]), 2);
                    Assert.AreEqual(0, p.view_mask[i] & ~p.seen_mask[i]);   // an inlier view was always seen
                }
        }

        [Test] public void VideoFrame_matches_python_cases()
        {
            foreach (var c in LoadCases().frames)
                Assert.AreEqual(c.expected, CvMath.VideoFrame(c.f, c.offset, c.det_shift), $"f={c.f} off={c.offset} ds={c.det_shift}");
        }

        [Test] public void Projection_matches_python_unity_project()
        {
            var d = LoadFixture();
            foreach (var pc in LoadCases().projections)
            {
                var cam = d.cameras[d.CameraIndex(pc.camera)];
                Assert.IsTrue(CvMath.ProjectToPixel(cam.UnityPos, cam.UnityRot, cam.K,
                    new Vector3(pc.point[0], pc.point[1], pc.point[2]), out var uv));
                Assert.AreEqual(pc.uv[0], uv.x, 0.05f);
                Assert.AreEqual(pc.uv[1], uv.y, 0.05f);
            }
        }

        [Test] public void GhostShift_uses_shared_views_median()
        {
            foreach (var s in LoadCases().shifts)
            {
                float sh = CvMath.GhostFrameShift(s.a_names, s.a_offsets, s.b_names, s.b_offsets, out bool known);
                Assert.AreEqual(s.known, known);
                Assert.AreEqual(s.shift, sh, 1e-4f);
            }
        }

        [Test] public void GhostShift_unknown_without_shared_offsets()
        {
            float sh = CvMath.GhostFrameShift(new[] { "a" }, new[] { 1f }, new string[0], new float[0], out bool known);
            Assert.IsFalse(known);
            Assert.AreEqual(0f, sh);
            Assert.AreEqual(-1, CvMath.GhostFrame(5, -10f, 100));
            Assert.AreEqual(2, CvMath.GhostFrame(5, -3.2f, 100));    // floor(5 - 3.2 + 0.5) = 2
            Assert.AreEqual(-1, CvMath.GhostFrame(99, 1f, 100));
        }

        [Test] public void OffsetsOf_pose_variant_puts_the_reference_view_at_zero()
        {
            var pv = new PoseVariantMeta { time_ref_view = "back_41",
                offsets_list = new[] { new ViewOffset { view = "backright_41", offset = -3.1f } } };
            var n = new List<string>(); var o = new List<float>();
            CvMath.OffsetsOf(pv, n, o);
            CollectionAssert.AreEquivalent(new[] { "back_41", "backright_41" }, n);
            Assert.AreEqual(0f, o[n.IndexOf("back_41")]);
        }

        [Test] public void Frustum_corners_project_to_the_image_corners()
        {
            var d = LoadFixture();
            var cam = d.cameras[0];
            var corners = new Vector3[4];
            CvMath.FrustumCorners(cam.UnityPos, cam.UnityRot, cam.vfov_deg, cam.Aspect, 2f, corners);
            Assert.IsTrue(CvMath.ProjectToPixel(cam.UnityPos, cam.UnityRot, cam.K, corners[0], out var tl));
            Assert.IsTrue(CvMath.ProjectToPixel(cam.UnityPos, cam.UnityRot, cam.K, corners[2], out var br));
            Assert.AreEqual(0f, tl.x, 1f); Assert.AreEqual(0f, tl.y, 1f);                    // top-left pixel
            Assert.AreEqual(cam.width, br.x, 1f); Assert.AreEqual(cam.height, br.y, 1f);     // bottom-right pixel
        }

        [Test] public void Letterbox_keeps_the_camera_aspect()
        {
            var r = CvMath.LetterboxRect(16f / 9f, 9f / 16f);             // portrait camera on a landscape screen
            Assert.AreEqual(1f, r.height, 1e-5f);
            Assert.AreEqual((9f / 16f) / (16f / 9f), r.width, 1e-5f);
            Assert.AreEqual((1f - r.width) / 2f, r.x, 1e-5f);
            var w = CvMath.LetterboxRect(4f / 3f, 16f / 9f);              // wider camera: bars top and bottom
            Assert.AreEqual(1f, w.width, 1e-5f);
            Assert.Less(w.height, 1f);
        }

        [Test] public void Seek_rule()
        {
            Assert.IsFalse(CvMath.NeedsSeek(10, 11, true));
            Assert.IsFalse(CvMath.NeedsSeek(10, 12, true));      // playing: tolerate 2 f (Media Foundation seek storms)
            Assert.IsTrue(CvMath.NeedsSeek(10, 13, true));
            Assert.IsTrue(CvMath.NeedsSeek(-1, 5, true));        // nothing decoded yet
            Assert.IsTrue(CvMath.NeedsSeek(10, 11, false));
            Assert.IsFalse(CvMath.NeedsSeek(10, 10, false));
        }

        [Test] public void JointMap_is_by_name()
        {
            var m = CvMath.JointMap(new[] { "pelvis", "left_hip", "jaw" }, new[] { "left_hip", "pelvis" });
            CollectionAssert.AreEqual(new[] { 1, 0, -1 }, m);
        }

        [Test] public void Person_floor_sentinel_means_no_candidate()
        {
            var d = LoadFixture();
            var np = System.Array.Find(d.persons, p => p.label == "nonplayer");
            Assert.IsTrue(np.FloorAt(np.first_frame, out var xz));
            Assert.AreEqual(3.4f, xz.x, 1e-4f);
            Assert.IsFalse(np.FloorAt(np.first_frame + 3, out _));      // -9999 in the fixture
            Assert.IsFalse(np.FloorAt(np.last_frame + 1, out _));
        }

        [Test] public void GhostCandidates_include_variants_and_camera_subsets()
        {
            var files = new[] { "analysis/singles_05x.analysis.json", "analysis/singles_05x_pose24.analysis.json",
                "analysis/singles_05x_posefit.analysis.json", "analysis/singles_05x_2cam_pose24.analysis.json",
                "analysis/singles_1x_pose24.analysis.json", "analysis/singles_05x_oldracket.analysis.json" };
            var c = CvMath.GhostCandidates(files, "analysis/singles_05x_pose24.analysis.json", AnalysisPlayer.PoseVariantSuffixes);
            CollectionAssert.AreEquivalent(new[] { "analysis/singles_05x.analysis.json", "analysis/singles_05x_posefit.analysis.json",
                "analysis/singles_05x_2cam_pose24.analysis.json" }, c);
            var back = CvMath.GhostCandidates(files, "analysis/singles_05x_2cam_pose24.analysis.json", AnalysisPlayer.PoseVariantSuffixes);
            CollectionAssert.Contains(back, "analysis/singles_05x_pose24.analysis.json");     // subset -> its full build
            CollectionAssert.DoesNotContain(back, "analysis/singles_05x_2cam_pose24.analysis.json");
        }

        [Test] public void GhostCandidates_pair_sp5_view_named_subsets_with_their_base()
        {
            var files = new[] { "analysis/singles_05x_pose24.analysis.json", "analysis/singles_05x_2cam_pose24.analysis.json",
                "analysis/singles_05x_2cam_bl_br_pose24.analysis.json", "analysis/singles_05x_1cam_b_pose24.analysis.json",
                "analysis/doubles_2sides_pose24.analysis.json", "analysis/doubles_2sides_2cam_br_fr_pose24.analysis.json",
                "analysis/doubles_2sides_4angle_pose24.analysis.json", "analysis/doubles_2sides_4angle_2cam_pose24.analysis.json" };
            var sfx = AnalysisPlayer.PoseVariantSuffixes;
            // base -> its subsets (both SP1 and SP5 names), nothing from other clips
            CollectionAssert.AreEquivalent(new[] { "analysis/singles_05x_2cam_pose24.analysis.json",
                "analysis/singles_05x_2cam_bl_br_pose24.analysis.json", "analysis/singles_05x_1cam_b_pose24.analysis.json" },
                CvMath.GhostCandidates(files, "analysis/singles_05x_pose24.analysis.json", sfx));
            CollectionAssert.AreEquivalent(new[] { "analysis/doubles_2sides_2cam_br_fr_pose24.analysis.json" },
                CvMath.GhostCandidates(files, "analysis/doubles_2sides_pose24.analysis.json", sfx));
            // subset -> its full build and sibling subsets
            foreach (var sub in new[] { "analysis/singles_05x_2cam_bl_br_pose24.analysis.json", "analysis/singles_05x_1cam_b_pose24.analysis.json" })
            {
                var back = CvMath.GhostCandidates(files, sub, sfx);
                CollectionAssert.Contains(back, "analysis/singles_05x_pose24.analysis.json");
                CollectionAssert.DoesNotContain(back, sub);
                CollectionAssert.DoesNotContain(back, "analysis/doubles_2sides_pose24.analysis.json");
            }
            CollectionAssert.AreEquivalent(new[] { "analysis/doubles_2sides_pose24.analysis.json" },
                CvMath.GhostCandidates(files, "analysis/doubles_2sides_2cam_br_fr_pose24.analysis.json", sfx));
            // lazy family: the 4angle 2-cam subset still belongs to doubles_2sides_4angle, not doubles_2sides
            CollectionAssert.AreEquivalent(new[] { "analysis/doubles_2sides_4angle_pose24.analysis.json" },
                CvMath.GhostCandidates(files, "analysis/doubles_2sides_4angle_2cam_pose24.analysis.json", sfx));
        }

        [Test] public void Summary_names_only_the_views_used()
        {
            var d = LoadFixture();
            var q = d.PlayerById("P1");
            string s = CvDebugOverlay.Summarise(d, q, 0);
            StringAssert.StartsWith("P1: measured · err ", s);          // numbers first: a narrow HUD cuts the view list only
            StringAssert.Contains(" px · σ̄ ", s);
            StringAssert.EndsWith(" mm · 2 views (a,b)", s);
            Assert.AreEqual("back", CvDebugOverlay.ShortCam("back_051"));
            Assert.AreEqual("frontleft", CvDebugOverlay.ShortCam("frontleft_41"));
            Assert.AreEqual("a", CvDebugOverlay.ShortCam("a"));
            Assert.AreEqual("", CvDebugOverlay.Summarise(d, q, d.frame_count));     // out of range
        }

        [Test] public void Video_status_keeps_the_missing_proxy_hint()
        {
            StringAssert.Contains("no proxy for back_051", VideoBillboard.StatusText(false, false, "back_051", "singles_05x", 100));
            StringAssert.Contains("proxies --clip singles_05x", VideoBillboard.StatusText(false, false, "back_051", "singles_05x", 100));
            StringAssert.Contains("loading", VideoBillboard.StatusText(true, false, "back_051", "singles_05x", 100));
            Assert.AreEqual("back_051 video frame 100", VideoBillboard.StatusText(true, true, "back_051", "singles_05x", 100));
        }

        [Test] public void Parse_rejects_a_sidecar_from_another_build_or_clip()
        {
            string json = File.ReadAllText(Fix("cvdebug_small.json"));
            var d0 = LoadFixture();
            AnalysisDoc Doc(string clip, string build) => new AnalysisDoc { clip_id = clip, players = new Player[0],
                meta = new AnalysisMeta { frame_count = d0.frame_count, pose24 = new Pose24Meta { build_id = build } } };
            Assert.IsNotNull(CvDebugDoc.Parse(json, Doc("syn", d0.source_build_id), out string why), why);
            Assert.IsNotNull(CvDebugDoc.Parse(json, Doc("syn", ""), out why), why);           // non-pose24 doc: no build id to compare
            Assert.IsNull(CvDebugDoc.Parse(json, Doc("syn", "someotherbuild"), out why));
            StringAssert.Contains("build", why);
            Assert.IsNull(CvDebugDoc.Parse(json, Doc("other_clip", d0.source_build_id), out why));
            StringAssert.Contains("clip", why);
        }

        [Test] public void Video_target_outside_the_video_hides_it()
        {
            Assert.IsTrue(CvMath.VideoFrameInRange(0, 1877));
            Assert.IsTrue(CvMath.VideoFrameInRange(1876, 1877));
            Assert.IsFalse(CvMath.VideoFrameInRange(-2, 1877));     // frontright_051 offset -1.7 at analysis frame 0
            Assert.IsFalse(CvMath.VideoFrameInRange(1877, 1877));
            Assert.IsTrue(CvMath.VideoFrameInRange(5, 0));          // frame count not known yet: don't hide
        }

        [Test] public void Ray_player_is_clamped_to_the_clip()
        {
            Assert.AreEqual(2, CvMath.RayPlayer(2, 0, 4));
            Assert.AreEqual(0, CvMath.RayPlayer(2, 0, 2));          // P3 focused, then a singles clip was loaded
            Assert.AreEqual(1, CvMath.RayPlayer(-1, 5, 2));
            Assert.AreEqual(-1, CvMath.RayPlayer(-1, 0, 0));        // no players: nothing to draw
        }

        [Test] public void Hud_confidence_tint_keeps_the_old_skeleton_only_default()
        {
            bool owned;
            Assert.AreEqual(CvColorMode.Confidence, AnalysisDebugHUD.ConfidenceTarget(CvColorMode.Off, true, true, false, out owned));
            Assert.IsTrue(owned);
            Assert.AreEqual(CvColorMode.Off, AnalysisDebugHUD.ConfidenceTarget(CvColorMode.Confidence, true, false, true, out owned));
            Assert.IsFalse(owned);                                   // a character skin: the HUD's tint goes away, as before SP4
            Assert.AreEqual(CvColorMode.Confidence, AnalysisDebugHUD.ConfidenceTarget(CvColorMode.Confidence, true, false, false, out owned));
            Assert.IsFalse(owned);                                   // chosen by the user (J): kept on any skin
            Assert.AreEqual(CvColorMode.Sigma, AnalysisDebugHUD.ConfidenceTarget(CvColorMode.Sigma, false, true, false, out owned));
            Assert.AreEqual(CvColorMode.Off, AnalysisDebugHUD.ConfidenceTarget(CvColorMode.Confidence, false, true, true, out owned));
        }

        [Test] public void Colours()
        {
            Assert.AreEqual(CvColors.Unknown, CvColors.ForJoint(CvColorMode.ReprojError, 1f, 1, 1, -1, 10, 0, 1, 0));
            Assert.AreEqual(Color.green, CvColors.ForJoint(CvColorMode.SelectedView, 1f, 0b11, 0b01, 5, 10, 0, 1, 0));
            Assert.AreEqual(Color.red, CvColors.ForJoint(CvColorMode.SelectedView, 1f, 0b11, 0b01, 5, 10, 0, 1, 1));
            Assert.AreEqual(CvColors.Unknown, CvColors.ForJoint(CvColorMode.SelectedView, 1f, 0b01, 0b01, 5, 10, 0, 1, 2));
            Assert.AreEqual(CvColorMode.Confidence, CvColors.Next(CvColorMode.Off, false));
            Assert.AreEqual(CvColorMode.Off, CvColors.Next(CvColorMode.Confidence, false));
            Assert.AreEqual(CvColorMode.ViewsUsed, CvColors.Next(CvColorMode.Confidence, true));
            Assert.AreEqual(CvColorMode.Off, CvColors.Next(CvColorMode.FrameState, true));
        }
    }
}
