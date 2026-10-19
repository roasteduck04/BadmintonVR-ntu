using NUnit.Framework;

namespace BadmintonVR.Analysis.Tests
{
    /// <summary>v0.7.5 review-3 fixes: pure viewer logic (pose-variant table, focus, facing, loaders, timelines).</summary>
    public class ViewerFixTests
    {
        [Test] public void Every_pose_variant_suffix_has_a_label()
        {
            Assert.AreEqual(AnalysisPlayer.PoseVariantSuffixes.Length, AnalysisPlayer.PoseVariantLabels.Length);
            for (int i = 0; i < AnalysisPlayer.PoseVariantDefs.Length; i++)
            {
                Assert.AreEqual(AnalysisPlayer.PoseVariantDefs[i].Suffix, AnalysisPlayer.PoseVariantSuffixes[i]);
                Assert.IsFalse(string.IsNullOrEmpty(AnalysisPlayer.PoseVariantLabels[i]), AnalysisPlayer.PoseVariantSuffixes[i]);
            }
        }

        [Test] public void Focus_is_clamped_when_a_clip_with_fewer_players_loads()
        {
            Assert.AreEqual(-1, CvMath.ClampFocus(2, 2));      // P3 focused, then a singles clip
            Assert.AreEqual(-1, CvMath.ClampFocus(3, 2));
            Assert.AreEqual(1, CvMath.ClampFocus(1, 2));
            Assert.AreEqual(-1, CvMath.ClampFocus(-1, 4));
            Assert.AreEqual(-1, CvMath.ClampFocus(-5, 4));
            Assert.AreEqual(-1, CvMath.ClampFocus(0, 0));
        }

        [Test] public void Facing_smoothing_advances_per_data_frame_and_snaps_on_seek()
        {
            Assert.AreEqual(1f, CvMath.FacingBlend(-1, 10, 0.25f, 10));            // nothing yet / seek requested
            Assert.AreEqual(0f, CvMath.FacingBlend(10, 10, 0.25f, 10));            // same frame re-rendered / paused: keep
            Assert.AreEqual(0.25f, CvMath.FacingBlend(10, 11, 0.25f, 10), 1e-6f);
            Assert.AreEqual(1f - 0.75f * 0.75f, CvMath.FacingBlend(10, 12, 0.25f, 10), 1e-6f);   // == two single steps
            Assert.AreEqual(1f, CvMath.FacingBlend(10, 9, 0.25f, 10));             // backwards / loop wrap
            Assert.AreEqual(1f, CvMath.FacingBlend(10, 21, 0.25f, 10));            // big jump = seek
            Assert.Less(CvMath.FacingBlend(10, 20, 0.25f, 10), 1f);
        }

        /// <summary>A small analysis doc as JSON: nPlayers players with ids P1.., nFrames frames of nJoints joints (+ 4 floats each).</summary>
        static string DocJson(int nPlayers, int nFrames, int nJoints, string schema = "4.1", int metaFrames = -1, int jointsInFrame = -1)
        {
            var sb = new System.Text.StringBuilder();
            sb.Append("{\"schema_version\":\"").Append(schema).Append("\",\"clip_id\":\"t\",\"meta\":{\"fps\":60,\"frame_count\":")
              .Append(metaFrames >= 0 ? metaFrames : nFrames).Append("},\"skeleton\":{\"parents\":[");
            for (int j = 0; j < nJoints; j++) sb.Append(j == 0 ? "-1" : ",0");
            sb.Append("]},\"players\":[");
            int jf = jointsInFrame >= 0 ? jointsInFrame : nJoints;
            for (int p = 0; p < nPlayers; p++)
            {
                sb.Append(p > 0 ? "," : "").Append("{\"id\":\"P").Append(p + 1).Append("\",\"frames\":[");
                for (int f = 0; f < nFrames; f++)
                {
                    sb.Append(f > 0 ? "," : "").Append("{\"frame_id\":").Append(f).Append(",\"joints_flat\":[");
                    for (int k = 0; k < jf * 4; k++) sb.Append(k > 0 ? ",0.5" : "0.5");
                    sb.Append("],\"position\":[0,0]}");
                }
                sb.Append("]}");
            }
            return sb.Append("]}").ToString();
        }

        [Test] public void Loaders_contain_malformed_json_instead_of_throwing()
        {
            string why;
            Assert.IsNull(AnalysisDoc.Parse("not json", out why));
            StringAssert.Contains("JSON", why);
            string good = DocJson(2, 3, 4);
            Assert.IsNull(AnalysisDoc.Parse(good.Substring(0, good.Length / 2), out why));     // truncated mid-write
            Assert.IsNull(AnalysisDoc.Parse("", out why));
            var doc = AnalysisDoc.Parse(good, out why);
            Assert.IsNotNull(doc, why);

            // racket / shuttle sidecars: a NaN token or a half-written file is rejected with a reason, never thrown
            Assert.IsNull(RacketVariantDoc.Parse("{\"frame_count\":3,\"players\":[{\"id\":\"P1\",\"frames\":[{\"confidence\":NaN}]}]}", doc, out why));
            Assert.IsNull(RacketVariantDoc.Parse("{\"frame_count\":3,\"players\":[", doc, out why));
            Assert.IsNull(ShuttleVariantDoc.Parse("{\"format\":\"shuttle_variant/1\",\"frame_count\":3,\"frames\":[{\"confidence\":NaN}]}", doc, out why));
            Assert.IsNull(CvDebugDoc.Parse("{\"format\":\"cvdebug/1\",", doc, out why));
        }

        [Test] public void Sidecar_parsers_check_the_doc_they_attach_to()
        {
            var doc = AnalysisDoc.Parse(DocJson(2, 3, 4), out string why);
            Assert.IsNotNull(RacketVariantDoc.Parse("{\"frame_count\":3,\"players\":[{\"id\":\"P1\"},{\"id\":\"P2\"}]}", doc, out why), why);
            Assert.IsNull(RacketVariantDoc.Parse("{\"frame_count\":3,\"players\":[{\"id\":\"P2\"},{\"id\":\"P1\"}]}", doc, out why));
            StringAssert.Contains("order", why);
            Assert.IsNull(RacketVariantDoc.Parse("{\"frame_count\":4,\"players\":[{\"id\":\"P1\"},{\"id\":\"P2\"}]}", doc, out why));
            Assert.IsNotNull(ShuttleVariantDoc.Parse("{\"format\":\"shuttle_variant/1\",\"frame_count\":3,\"frames\":[{},{},{}]}", doc, out why), why);
            Assert.IsNull(ShuttleVariantDoc.Parse("{\"format\":\"shuttle_variant/1\",\"frame_count\":3,\"frames\":[{},{}]}", doc, out why));
        }

        [Test] public void Analysis_doc_validation_refuses_what_ShowFrame_would_index_out_of_range()
        {
            string why;
            Assert.IsNotNull(AnalysisDoc.Parse(DocJson(2, 3, 28), out why), why);
            Assert.IsNotNull(AnalysisDoc.Parse(DocJson(1, 3, 28, jointsInFrame: 0), out why), why);   // no pose: allowed
            Assert.IsNull(AnalysisDoc.Parse(DocJson(2, 3, 28, metaFrames: 5), out why));               // short player
            StringAssert.Contains("frame_count", why);
            Assert.IsNull(AnalysisDoc.Parse(DocJson(2, 3, 28, jointsInFrame: 24), out why));           // 24-joint frame, 28-joint doc
            StringAssert.Contains("joints_flat", why);
        }

        [Test] public void Schema_versions_compare_numerically()
        {
            Assert.IsTrue(CvMath.SchemaAtLeast("4.1", 4, 1));
            Assert.IsTrue(CvMath.SchemaAtLeast("4.2", 4, 1));
            Assert.IsTrue(CvMath.SchemaAtLeast("4.10", 4, 2));       // numeric, not string order
            Assert.IsFalse(CvMath.SchemaAtLeast("4.0", 4, 1));
            Assert.IsFalse(CvMath.SchemaAtLeast("3.9", 4, 1));
            Assert.IsTrue(CvMath.SchemaAtLeast("5", 4, 1));
            Assert.IsFalse(CvMath.SchemaAtLeast(null, 4, 1));
            Assert.IsFalse(CvMath.SchemaAtLeast("v4.1", 4, 1));
            Assert.IsTrue(CvMath.SchemaUsesSigmaConfidence("4.1"));
            Assert.IsTrue(CvMath.SchemaUsesSigmaConfidence("4.2"));  // a future pose24 minor keeps the 1e-6 cutoff
            Assert.IsFalse(CvMath.SchemaUsesSigmaConfidence("4.0"));
            Assert.IsFalse(CvMath.SchemaUsesSigmaConfidence("5.0")); // base / SMPL-X lineage
            Assert.IsFalse(CvMath.SchemaUsesSigmaConfidence("3.0"));
        }

        [Test] public void Switching_builds_keeps_the_instant()
        {
            // doubles_2sides_4angle_2cam SMPL-24 -> fit (review 3 A5): offsets (0, -3.1, -0.4, -3.5) vs (2.0, 0.0, 2.9, 1.0)
            var views = new[] { "a", "b", "c", "d" };
            float shift = CvMath.GhostFrameShift(views, new[] { 0f, -3.1f, -0.4f, -3.5f }, views, new[] { 2f, 0f, 2.9f, 1f }, out bool known);
            Assert.IsTrue(known);
            Assert.AreEqual(-3.2f, shift, 1e-4f);
            Assert.AreEqual(97, CvMath.SameInstantFrame(100, shift, known, 1370));       // same rounding as GhostFrame
            Assert.AreEqual(CvMath.GhostFrame(100, shift, 1370), CvMath.SameInstantFrame(100, shift, known, 1370));
            Assert.AreEqual(100, CvMath.SameInstantFrame(100, 5f, false, 1370));        // unknown timeline: keep the index
            Assert.AreEqual(0, CvMath.SameInstantFrame(1, shift, known, 1370));         // clamped, never -1
            Assert.AreEqual(1369, CvMath.SameInstantFrame(1369, 2f, true, 1370));
            Assert.AreEqual(0, CvMath.SameInstantFrame(5, 0f, true, 0));
        }

        [Test] public void Timeline_offsets_prefer_the_cvdebug_sidecar()
        {
            var doc = new AnalysisDoc { meta = new AnalysisMeta { pose_variant = new PoseVariantMeta { time_ref_view = "a",
                offsets_list = new[] { new ViewOffset { view = "b", offset = 1.5f } } } } };
            var n = new System.Collections.Generic.List<string>(); var o = new System.Collections.Generic.List<float>();
            CvMath.TimelineOffsets(null, doc, n, o);
            CollectionAssert.AreEqual(new[] { "a", "b" }, n);
            var cv = new CvDebugDoc { cameras = new[] { new CvCamera { name = "a", offset = 2f, offset_source = "sync" },
                                                         new CvCamera { name = "z", offset = 9f, offset_source = "none" } } };
            CvMath.TimelineOffsets(cv, doc, n, o);
            CollectionAssert.AreEqual(new[] { "a" }, n);
            Assert.AreEqual(2f, o[0]);
        }

        [Test] public void Ui_guard_maps_the_pointer_into_gui_space()
        {
            // a 1080-high screen: the viewer panel sits at the bottom (GUI y near 1080), the pointer near screen y = 0
            var panel = new UnityEngine.Rect(12, 1080 - 12 - 300, 1896, 300);
            var gui = UiPointerGuard.ScreenToGui(new UnityEngine.Vector2(500, 50), 1080);
            Assert.AreEqual(1030f, gui.y);
            Assert.IsTrue(UiPointerGuard.Contains(new[] { panel }, gui));
            Assert.IsFalse(UiPointerGuard.Contains(new[] { panel }, UiPointerGuard.ScreenToGui(new UnityEngine.Vector2(500, 900), 1080)));
            Assert.IsFalse(UiPointerGuard.Contains(new UnityEngine.Rect[0], gui));
        }

        [Test] public void Video_seek_does_not_loop_on_an_overshoot_and_retries_after_an_error()
        {
            Assert.IsTrue(CvMath.NeedsVideoSeek(-1, 40, -1));      // nothing decoded yet
            Assert.IsFalse(CvMath.NeedsVideoSeek(40, 40, -1));     // on target
            Assert.IsFalse(CvMath.NeedsVideoSeek(42, 40, 40));     // overshot catching up from a keyframe: accept, no re-seek loop
            Assert.IsTrue(CvMath.NeedsVideoSeek(42, 41, 40));      // the next step seeks again
            Assert.IsTrue(CvMath.ShouldReopenVideo(false, false, 0f, 3f));   // another view
            Assert.IsFalse(CvMath.ShouldReopenVideo(true, false, 99f, 3f));  // same view, healthy
            Assert.IsFalse(CvMath.ShouldReopenVideo(true, true, 1f, 3f));    // same view, error just now
            Assert.IsTrue(CvMath.ShouldReopenVideo(true, true, 3f, 3f));     // same view, retry after the delay
        }

        static Player PlayerWithLowestJoint(float y)
            => new Player { id = "P", frames = new[] { new PlayerFrame { joints_flat = new[] { 0f, y, 0f, 1f, 0f, y + 1f, 0f, 1f } } } };

        [Test] public void One_vertical_lift_for_the_whole_clip()
        {
            // singles_05x_pose24: P1 / P2 lowest joints about 4.4 / 5.2 cm above the floor -> one lift of -4.8 cm for
            // bodies, rackets and shuttle (per-player lifts were -4.4 / -5.2 with the shuttle unlifted)
            var doc = new AnalysisDoc { players = new[] { PlayerWithLowestJoint(0.044f), PlayerWithLowestJoint(0.052f),
                                                          new Player { id = "P3", frames = new[] { new PlayerFrame() } } } };
            Assert.AreEqual(-0.048f, doc.GroundLift(), 1e-6f);                // the unposed player doesn't count
            Assert.IsFalse(doc.players[2].TryMinY(out _));
            Assert.AreEqual(0f, doc.players[2].MinY());
            Assert.AreEqual(0f, new AnalysisDoc { players = new Player[0] }.GroundLift());
        }

        [Test] public void Cvdebug_parse_checks_the_per_frame_arrays_the_hud_reads()
        {
            string json = System.IO.File.ReadAllText(System.IO.Path.Combine(
                System.IO.Directory.GetParent(UnityEngine.Application.dataPath).FullName, "tools", "tests", "fixtures", "cvdebug_small.json"));
            Assert.IsNotNull(CvDebugDoc.Parse(json, null, out string why), why);
            var d = UnityEngine.JsonUtility.FromJson<CvDebugDoc>(json);
            d.players[0].frame_err_px = new int[d.frame_count - 1];
            Assert.IsNull(CvDebugDoc.Parse(UnityEngine.JsonUtility.ToJson(d), null, out why));
            StringAssert.Contains("array lengths", why);
            d = UnityEngine.JsonUtility.FromJson<CvDebugDoc>(json);
            d.players[0].contact = new int[3];
            Assert.IsNull(CvDebugDoc.Parse(UnityEngine.JsonUtility.ToJson(d), null, out why));
        }

        [Test] public void Viewer_wraps_long_text_at_spaces_and_keeps_every_word()
        {
            var lines = AnalysisViewerUITK.WrapText("producers: pose=pose24_multiview_fit pos=pose24_multiview_fit racket=x", 30);
            foreach (var l in lines) Assert.LessOrEqual(l.Length, 30, l);
            Assert.AreEqual("producers: pose=pose24_multiview_fit pos=pose24_multiview_fit racket=x".Replace(" ", ""),
                            string.Join("", lines).Replace(" ", ""));
            Assert.AreEqual(new[] { "a", "b" }, AnalysisViewerUITK.WrapText("a\nb", 10).ToArray());
            Assert.AreEqual(3, AnalysisViewerUITK.WrapText(new string('x', 25), 10).Count);    // no space: hard cut
        }

        [Test] public void Typing_in_a_viewer_field_blocks_the_keyboard_shortcuts()
        {
            var owner = new object();
            bool typing = true;
            UiPointerGuard.SetTyping(owner, () => typing);
            try
            {
                Assert.IsTrue(UiPointerGuard.Typing);
                typing = false;
                Assert.IsFalse(UiPointerGuard.Typing);
            }
            finally { UiPointerGuard.ClearTyping(owner); }
            Assert.IsFalse(UiPointerGuard.Typing);
        }
    }
}
