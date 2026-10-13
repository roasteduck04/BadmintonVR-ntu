using System.Collections.Generic;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>
    /// Draws a racket per player: either the triangulated grip/head/side in analysis.json
    /// ("v1") or a racket sidecar variant ("rigid", "bootstrap"), optionally with v1 as a
    /// faint ghost for comparison. Attaches to the same GameObject as AnalysisPlayer.
    /// </summary>
    public class RacketVisualAnalysis : MonoBehaviour
    {
        public const string V1 = "v1";

        public Color racketColor = new Color(1f, 0.45f, 0.05f);
        public Color rigidColor = new Color(0.1f, 0.85f, 1f);
        public Color bootstrapColor = new Color(0.95f, 0.25f, 0.85f);
        public Color imuColor = new Color(1f, 0.9f, 0.15f);
        [Range(0f, 1f)] public float confidenceCutoff = 0.15f;
        [Range(0f, 1f)] public float ghostAlpha = 0.3f;
        [Tooltip("Preferred racket source; falls back to v1 when the clip has no such sidecar")]
        public string preferredVariant = V1;
        public bool ghostV1;
        [Tooltip("Display only: move sidecar rackets so the grip sits on the rendered (smoothed) hand; orientation unchanged")]
        public bool snapToHand;

        const float ShaftRadius = 0.012f;
        const float HeadWidth = 0.22f;
        const float HeadLength = 0.28f;
        const float HeadThickness = 0.025f;

        struct RacketGeo
        {
            public Transform root;
            public Transform shaft;
            public Transform head;
        }

        RacketGeo[] _rackets;
        RacketGeo[] _ghosts;
        Vector3[] _renderedJoints;
        Material _mat;
        MaterialPropertyBlock _propBlock;
        readonly Dictionary<string, RacketVariantDoc> _variants = new Dictionary<string, RacketVariantDoc>();

        int _leftHand = -1, _rightHand = -1;

        public void SetJointNames(string[] names)
        {
            _leftHand = _rightHand = -1;
            if (names == null) return;
            foreach (var pair in new[] { ("left_hand", "right_hand"), ("left_middle1", "right_middle1"),
                                         ("left_wrist", "right_wrist") })
            {
                int l = System.Array.IndexOf(names, pair.Item1), r = System.Array.IndexOf(names, pair.Item2);
                if (l >= 0 && r >= 0) { _leftHand = l; _rightHand = r; return; }
            }
        }

        /// <summary>Joint index snap-to-hand uses for that hand (-1 if the doc has none).</summary>
        public int HandJoint(bool left) => left ? _leftHand : _rightHand;

        public string EffectiveVariant
            => preferredVariant != null && _variants.ContainsKey(preferredVariant) ? preferredVariant : V1;
        public RacketVariantDoc VariantDoc(string name)
            => name != null && _variants.TryGetValue(name, out var d) ? d : null;

        public void SetVariants(Dictionary<string, RacketVariantDoc> variants)
        {
            _variants.Clear();
            if (variants == null) return;
            foreach (var kv in variants) if (kv.Value != null) _variants[kv.Key] = kv.Value;
        }

        public void Build(int playerCount)
        {
            Clear();
            // NOTE: was RendererUtil.MakeMat(racketColor) (opaque); switched to transparent
            // material so per-frame alpha fading via MaterialPropertyBlock takes effect.
            // Created once and reused across clip loads (the colour comes from the property block).
            if (_mat == null) _mat = RendererUtil.MakeTransparentMat(racketColor);
            _propBlock = new MaterialPropertyBlock();
            _rackets = new RacketGeo[playerCount];
            _ghosts = new RacketGeo[playerCount];
            for (int i = 0; i < playerCount; i++)
            {
                _rackets[i] = BuildOne($"racket_P{i}");
                _ghosts[i] = BuildOne($"racket_ghost_P{i}");
            }
        }

        RacketGeo BuildOne(string name)
        {
            var root = new GameObject(name).transform;
            root.SetParent(transform, false);
            root.gameObject.SetActive(false);

            var shaftGo = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            shaftGo.name = "shaft";
            shaftGo.transform.SetParent(root, false);
            shaftGo.GetComponent<Renderer>().sharedMaterial = _mat;
            RendererUtil.DestroyCollider(shaftGo);
            var headGo = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            headGo.name = "head";
            headGo.transform.SetParent(root, false);
            headGo.GetComponent<Renderer>().sharedMaterial = _mat;
            RendererUtil.DestroyCollider(headGo);

            return new RacketGeo { root = root, shaft = shaftGo.transform, head = headGo.transform };
        }

        // _archive: pre-alpha-fade ShowFrame (no MaterialPropertyBlock, hard confidence
        // cutoff only, confidenceCutoff default 0.1). Superseded 2026-08-29 by the version
        // below which adds confidence-based alpha fading (Racket Sub-Project C, Task 3).
        //
        // public void ShowFrame(int playerIdx, Player player, int frame,
        //                       Transform playerRoot, float groundOffset,
        //                       Quaternion facingInv = default)
        // {
        //     if (_rackets == null || playerIdx >= _rackets.Length) return;
        //     var geo = _rackets[playerIdx];
        //
        //     if (!player.HasRacket(frame) || !player.HasPose(frame))
        //     {
        //         geo.root.gameObject.SetActive(false);
        //         return;
        //     }
        //
        //     var r = player.RacketAt(frame);
        //     if (r.confidence < confidenceCutoff)
        //     {
        //         geo.root.gameObject.SetActive(false);
        //         return;
        //     }
        //
        //     Vector3 grip = new Vector3(r.grip[0], r.grip[1], r.grip[2]);
        //     Vector3 head = new Vector3(r.head[0], r.head[1], r.head[2]);
        //     Vector3 side = new Vector3(r.side[0], r.side[1], r.side[2]);
        //     Vector3 lift = new Vector3(0, groundOffset, 0);
        //
        //     grip += lift;
        //     head += lift;
        //     side += lift;
        //
        //     // Rotate from world orientation into root-local space
        //     if (facingInv.w != 0f || facingInv.x != 0f || facingInv.y != 0f || facingInv.z != 0f)
        //     {
        //         grip = facingInv * grip;
        //         head = facingInv * head;
        //         side = facingInv * side;
        //     }
        //
        //     Vector3 shaftVec = head - grip;
        //     float shaftLen = shaftVec.magnitude;
        //     if (shaftLen < 0.01f) { geo.root.gameObject.SetActive(false); return; }
        //
        //     geo.root.gameObject.SetActive(true);
        //     geo.root.position = playerRoot.TransformPoint(Vector3.zero);
        //     geo.root.rotation = playerRoot.rotation;
        //
        //     Vector3 mid = (grip + head) * 0.5f;
        //     Vector3 shaftDir = shaftVec / shaftLen;
        //     Quaternion shaftRot = Quaternion.FromToRotation(Vector3.up, shaftDir);
        //
        //     geo.shaft.localPosition = mid;
        //     geo.shaft.localRotation = shaftRot;
        //     geo.shaft.localScale = new Vector3(ShaftRadius * 2f, shaftLen * 0.5f, ShaftRadius * 2f);
        //
        //     Vector3 sideDir = (side - head);
        //     float sideLen = sideDir.magnitude;
        //     Vector3 faceNormal = Vector3.Cross(shaftDir, sideLen > 0.001f ? sideDir.normalized : Vector3.right);
        //     if (faceNormal.sqrMagnitude < 1e-6f) faceNormal = Vector3.forward;
        //     faceNormal.Normalize();
        //
        //     geo.head.localPosition = head;
        //     geo.head.localRotation = Quaternion.LookRotation(faceNormal, shaftDir);
        //     geo.head.localScale = new Vector3(HeadWidth, HeadLength, HeadThickness);
        // }

        /// <param name="jointsLocal">the player's rendered joints (lifted, facing-rotated), or null</param>
        public void ShowFrame(int playerIdx, Player player, int frame,
                              Transform playerRoot, float groundOffset,
                              Quaternion facingInv = default, Vector3[] jointsLocal = null)
        {
            if (_rackets == null || playerIdx >= _rackets.Length) return;
            bool hasPose = player.HasPose(frame);
            string active = EffectiveVariant;
            _renderedJoints = jointsLocal;

            if (active == V1)
            {
                DrawV1(_rackets[playerIdx], player, frame, hasPose, playerRoot, groundOffset, facingInv,
                       racketColor, 1f);
                _ghosts[playerIdx].root.gameObject.SetActive(false);
                return;
            }

            var doc = _variants[active];
            DrawModel(_rackets[playerIdx], doc.At(playerIdx, frame), doc.template, hasPose, playerRoot,
                      groundOffset, facingInv, ColorFor(active));
            if (ghostV1)
                DrawV1(_ghosts[playerIdx], player, frame, hasPose, playerRoot, groundOffset, facingInv,
                       racketColor, ghostAlpha);
            else
                _ghosts[playerIdx].root.gameObject.SetActive(false);
        }

        Color ColorFor(string variant)
            => variant == "rigid" ? rigidColor : variant == "bootstrap" ? bootstrapColor
             : variant == "imu" ? imuColor : racketColor;

        float ConfidenceAlpha(float confidence) => confidence >= 0.25f ? 1f : 0.5f;

        static bool HasFacing(Quaternion q) => q.w != 0f || q.x != 0f || q.y != 0f || q.z != 0f;

        void ReadPoints(RacketFrame r, float groundOffset, Quaternion facingInv,
                        out Vector3 grip, out Vector3 head, out Vector3 side)
        {
            Vector3 lift = new Vector3(0, groundOffset, 0);
            grip = new Vector3(r.grip[0], r.grip[1], r.grip[2]) + lift;
            head = new Vector3(r.head[0], r.head[1], r.head[2]) + lift;
            side = new Vector3(r.side[0], r.side[1], r.side[2]) + lift;
            if (HasFacing(facingInv))
            {
                grip = facingInv * grip;
                head = facingInv * head;
                side = facingInv * side;
            }
        }

        void Place(RacketGeo geo, Transform playerRoot, Color c)
        {
            geo.root.gameObject.SetActive(true);
            geo.root.position = playerRoot.TransformPoint(Vector3.zero);
            geo.root.rotation = playerRoot.rotation;
            _propBlock.SetColor("_BaseColor", c);
        }

        static Vector3 FaceNormal(Vector3 shaftDir, Vector3 head, Vector3 side)
        {
            Vector3 sideDir = side - head;
            Vector3 n = Vector3.Cross(shaftDir, sideDir.magnitude > 0.001f ? sideDir.normalized : Vector3.right);
            return n.sqrMagnitude < 1e-6f ? Vector3.forward : n.normalized;
        }

        void DrawV1(RacketGeo geo, Player player, int frame, bool hasPose, Transform playerRoot,
                    float groundOffset, Quaternion facingInv, Color color, float alphaMul)
        {
            if (!player.HasRacket(frame) || !hasPose) { geo.root.gameObject.SetActive(false); return; }
            var r = player.RacketAt(frame);
            if (r.confidence < confidenceCutoff) { geo.root.gameObject.SetActive(false); return; }

            ReadPoints(r, groundOffset, facingInv, out var grip, out var head, out var side);
            Vector3 shaftVec = head - grip;
            float shaftLen = shaftVec.magnitude;
            if (shaftLen < 0.01f) { geo.root.gameObject.SetActive(false); return; }

            Color c = color;
            c.a = ConfidenceAlpha(r.confidence) * alphaMul;
            Place(geo, playerRoot, c);

            Vector3 shaftDir = shaftVec / shaftLen;
            geo.shaft.localPosition = (grip + head) * 0.5f;
            geo.shaft.localRotation = Quaternion.FromToRotation(Vector3.up, shaftDir);
            geo.shaft.localScale = new Vector3(ShaftRadius * 2f, shaftLen * 0.5f, ShaftRadius * 2f);
            geo.shaft.GetComponent<Renderer>().SetPropertyBlock(_propBlock);

            geo.head.localPosition = head;
            geo.head.localRotation = Quaternion.LookRotation(FaceNormal(shaftDir, head, side), shaftDir);
            geo.head.localScale = new Vector3(HeadWidth, HeadLength, HeadThickness);
            geo.head.GetComponent<Renderer>().SetPropertyBlock(_propBlock);
        }

        /// <summary>Sidecar racket: grip is the hand point, head is the head centre, drawn to template size.</summary>
        void DrawModel(RacketGeo geo, RacketFrame r, RacketTemplateInfo tpl, bool hasPose, Transform playerRoot,
                       float groundOffset, Quaternion facingInv, Color color)
        {
            if (!hasPose || !RacketVariantDoc.Valid(r) || r.confidence < confidenceCutoff)
            {
                geo.root.gameObject.SetActive(false);
                return;
            }
            ReadPoints(r, groundOffset, facingInv, out var grip, out var head, out var side);
            Vector3 axis = head - grip;
            if (axis.magnitude < 0.01f) { geo.root.gameObject.SetActive(false); return; }
            axis.Normalize();

            int hi = r.which_hand == "left" ? _leftHand : _rightHand;
            if (snapToHand && _renderedJoints != null && hi >= 0 && hi < _renderedJoints.Length)
            {
                Vector3 shift = _renderedJoints[hi] - grip;
                grip += shift; head += shift; side += shift;
            }

            Color c = color;
            c.a = ConfidenceAlpha(r.confidence);
            Place(geo, playerRoot, c);

            Vector3 butt = grip - axis * tpl.grip_z;
            Vector3 throat = head - axis * (tpl.head_length * 0.5f);
            float shaftLen = Vector3.Distance(butt, throat);
            geo.shaft.localPosition = (butt + throat) * 0.5f;
            geo.shaft.localRotation = Quaternion.FromToRotation(Vector3.up, axis);
            geo.shaft.localScale = new Vector3(ShaftRadius * 2f, shaftLen * 0.5f, ShaftRadius * 2f);
            geo.shaft.GetComponent<Renderer>().SetPropertyBlock(_propBlock);

            geo.head.localPosition = head;
            geo.head.localRotation = Quaternion.LookRotation(FaceNormal(axis, head, side), axis);
            geo.head.localScale = new Vector3(tpl.head_width, tpl.head_length, HeadThickness);
            geo.head.GetComponent<Renderer>().SetPropertyBlock(_propBlock);
        }

        public void HideAll()
        {
            Hide(_rackets);
            Hide(_ghosts);
        }

        /// <summary>Hide one player's racket and its v1 ghost (an unfocused player).</summary>
        public void HidePlayer(int playerIdx)
        {
            if (_rackets != null && playerIdx >= 0 && playerIdx < _rackets.Length && _rackets[playerIdx].root != null)
                _rackets[playerIdx].root.gameObject.SetActive(false);
            if (_ghosts != null && playerIdx >= 0 && playerIdx < _ghosts.Length && _ghosts[playerIdx].root != null)
                _ghosts[playerIdx].root.gameObject.SetActive(false);
        }

        static void Hide(RacketGeo[] set)
        {
            if (set == null) return;
            for (int i = 0; i < set.Length; i++)
                if (set[i].root != null) set[i].root.gameObject.SetActive(false);
        }

        public void Clear()
        {
            DestroySet(_rackets);
            DestroySet(_ghosts);
            _rackets = null;
            _ghosts = null;
        }

        static void DestroySet(RacketGeo[] set)
        {
            if (set == null) return;
            for (int i = 0; i < set.Length; i++)
                if (set[i].root != null)
                {
                    if (Application.isPlaying) Object.Destroy(set[i].root.gameObject);
                    else Object.DestroyImmediate(set[i].root.gameObject);
                }
        }

        void OnDestroy() { Clear(); RendererUtil.DestroySafe(_mat); }
    }
}
