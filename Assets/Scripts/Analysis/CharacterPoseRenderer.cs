using UnityEngine;

namespace BadmintonVR.Analysis
{
    public class CharacterPoseRenderer : IPoseRenderer
    {
        const int PELVIS = 0, L_HIP = 1, R_HIP = 2, SPINE1 = 3,
                  L_KNEE = 4, R_KNEE = 5, SPINE2 = 6,
                  L_ANKLE = 7, R_ANKLE = 8, SPINE3 = 9,
                  L_FOOT = 10, R_FOOT = 11, NECK = 12,
                  L_COLLAR = 13, R_COLLAR = 14, HEAD = 15,
                  L_SHO = 16, R_SHO = 17, L_ELB = 18, R_ELB = 19,
                  L_WRI = 20, R_WRI = 21, L_HAND = 22, R_HAND = 23;

        // Each entry: which Unity bone to rotate, using the direction from SMPL joint `from` to `to`.
        // The `child` is the next HumanBodyBones in the chain — used to read the rig's
        // rest-pose bone direction (bone.position → child.position).
        static readonly (HumanBodyBones bone, HumanBodyBones child, int from, int to)[] BoneMap =
        {
            (HumanBodyBones.Hips,           HumanBodyBones.Spine,          PELVIS,  SPINE1),
            (HumanBodyBones.Spine,          HumanBodyBones.Chest,          SPINE1,  SPINE2),
            (HumanBodyBones.Chest,          HumanBodyBones.UpperChest,     SPINE2,  SPINE3),
            (HumanBodyBones.UpperChest,     HumanBodyBones.Neck,           SPINE3,  NECK),
            (HumanBodyBones.Neck,           HumanBodyBones.Head,           NECK,    HEAD),

            (HumanBodyBones.LeftUpperLeg,   HumanBodyBones.LeftLowerLeg,   L_HIP,   L_KNEE),
            (HumanBodyBones.LeftLowerLeg,   HumanBodyBones.LeftFoot,       L_KNEE,  L_ANKLE),
            (HumanBodyBones.LeftFoot,       HumanBodyBones.LeftToes,       L_ANKLE, L_FOOT),

            (HumanBodyBones.RightUpperLeg,  HumanBodyBones.RightLowerLeg,  R_HIP,   R_KNEE),
            (HumanBodyBones.RightLowerLeg,  HumanBodyBones.RightFoot,      R_KNEE,  R_ANKLE),
            (HumanBodyBones.RightFoot,      HumanBodyBones.RightToes,      R_ANKLE, R_FOOT),

            (HumanBodyBones.LeftShoulder,   HumanBodyBones.LeftUpperArm,   L_COLLAR, L_SHO),
            (HumanBodyBones.LeftUpperArm,   HumanBodyBones.LeftLowerArm,   L_SHO,   L_ELB),
            (HumanBodyBones.LeftLowerArm,   HumanBodyBones.LeftHand,       L_ELB,   L_WRI),
            (HumanBodyBones.LeftHand,       HumanBodyBones.LeftMiddleProximal, L_WRI, L_HAND),

            (HumanBodyBones.RightShoulder,  HumanBodyBones.RightUpperArm,  R_COLLAR, R_SHO),
            (HumanBodyBones.RightUpperArm,  HumanBodyBones.RightLowerArm,  R_SHO,   R_ELB),
            (HumanBodyBones.RightLowerArm,  HumanBodyBones.RightHand,      R_ELB,   R_WRI),
            (HumanBodyBones.RightHand,      HumanBodyBones.RightMiddleProximal, R_WRI, R_HAND),
        };

        // Names per SMPL-24 slot above (first match wins). 58-joint SMPL-X docs have no
        // "hand" joint and put thumbs at 22/23, so slots are resolved by name, not index.
        static readonly string[][] SlotNames =
        {
            new[] { "pelvis" }, new[] { "left_hip" }, new[] { "right_hip" }, new[] { "spine1" },
            new[] { "left_knee" }, new[] { "right_knee" }, new[] { "spine2" },
            new[] { "left_ankle" }, new[] { "right_ankle" }, new[] { "spine3" },
            new[] { "left_foot", "left_big_toe" }, new[] { "right_foot", "right_big_toe" }, new[] { "neck" },
            new[] { "left_collar" }, new[] { "right_collar" }, new[] { "head" },
            new[] { "left_shoulder" }, new[] { "right_shoulder" }, new[] { "left_elbow" }, new[] { "right_elbow" },
            new[] { "left_wrist" }, new[] { "right_wrist" },
            new[] { "left_hand", "left_middle1" }, new[] { "right_hand", "right_middle1" },
        };

        int[] _slot = DefaultSlots();   // SMPL-24 slot -> joint index in the loaded doc (-1 = absent)
        bool _posed;

        static int[] DefaultSlots()
        {
            var s = new int[SlotNames.Length];
            for (int i = 0; i < s.Length; i++) s[i] = i;
            return s;
        }

        public void SetJointNames(string[] names)
        {
            _slot = DefaultSlots();
            if (names == null) return;
            for (int i = 0; i < SlotNames.Length; i++)
            {
                _slot[i] = -1;
                foreach (var n in SlotNames[i])
                {
                    int j = System.Array.IndexOf(names, n);
                    if (j >= 0) { _slot[i] = j; break; }
                }
            }
        }

        public bool TryGetRenderedJoint(int joint, out Vector3 rootLocal)
        {
            rootLocal = default;
            if (!_posed || _animator == null || _root == null || joint < 0) return false;
            int slot = System.Array.IndexOf(_slot, joint);
            if (slot < 0) return false;
            foreach (var (bone, child, from, to) in BoneMap)
            {
                var t = from == slot ? _animator.GetBoneTransform(bone)
                      : to == slot ? _animator.GetBoneTransform(child) : null;
                if (t == null) continue;
                rootLocal = _root.InverseTransformPoint(t.position);
                return true;
            }
            return false;
        }

        static GameObject _sharedPrefab;
        GameObject _prefab;
        Transform _root;
        GameObject _instance;
        Animator _animator;
        Transform[] _boneTransforms;
        Quaternion[] _restLocalRot;
        Vector3[] _restBoneDir;
        float _feetOffset;
        readonly System.Collections.Generic.List<Material> _ownedMaterials = new System.Collections.Generic.List<Material>();

        public static void SetPrefab(GameObject prefab) => _sharedPrefab = prefab;

        public CharacterPoseRenderer(GameObject prefab = null)
        {
            _prefab = prefab ?? _sharedPrefab;
        }

        public void Build(Transform root, int[] parents, Color color)
        {
            _root = root;

            if (_prefab == null) _prefab = _sharedPrefab;
            if (_prefab == null) _prefab = Resources.Load<GameObject>("character");
            if (_prefab == null)
            {
                Debug.LogWarning("[CharacterPose] No character prefab found.");
                return;
            }

            _instance = Object.Instantiate(_prefab, root);
            _instance.name = "CharacterSkin";
            _instance.transform.localPosition = Vector3.zero;
            _instance.transform.localRotation = Quaternion.Euler(0f, 180f, 0f);
            _instance.transform.localScale = Vector3.one;

            _animator = _instance.GetComponentInChildren<Animator>(true);     // includeInactive: the root may be hidden by focus
            if (_animator == null || !_animator.isHuman)
            {
                Debug.LogWarning("[CharacterPose] FBX is not Humanoid.");
                return;
            }

            _boneTransforms = new Transform[BoneMap.Length];
            _restLocalRot = new Quaternion[BoneMap.Length];
            _restBoneDir = new Vector3[BoneMap.Length];

            for (int i = 0; i < BoneMap.Length; i++)
            {
                var t = _animator.GetBoneTransform(BoneMap[i].bone);
                _boneTransforms[i] = t;
                if (t != null)
                    _restLocalRot[i] = t.localRotation;
            }

            // Compute how far the feet sit below the instance origin so we can
            // shift hips up by this amount and keep feet on the ground plane.
            _feetOffset = 0f;
            var lFoot = _animator.GetBoneTransform(HumanBodyBones.LeftFoot);
            var rFoot = _animator.GetBoneTransform(HumanBodyBones.RightFoot);
            if (lFoot != null || rFoot != null)
            {
                float lowestY;
                if (lFoot != null && rFoot != null)
                    lowestY = Mathf.Min(lFoot.position.y, rFoot.position.y);
                else
                    lowestY = (lFoot ?? rFoot).position.y;

                // Find the actual lowest renderer point for a more accurate ground plane
                float meshBottom = lowestY;
                foreach (var rend in _instance.GetComponentsInChildren<Renderer>())
                {
                    float bottom = rend.bounds.min.y;
                    if (bottom < meshBottom) meshBottom = bottom;
                }

                _feetOffset = _instance.transform.position.y - meshBottom + 0.05f;
            }

            UpgradeMaterialsToURP();
            CaptureRestDirections();
        }

        void UpgradeMaterialsToURP()
        {
            if (_instance == null) return;
            var urpLit = Shader.Find("Universal Render Pipeline/Lit");
            if (urpLit == null) return;

            foreach (var rend in _instance.GetComponentsInChildren<Renderer>(true))
            {
                string objName = rend.gameObject.name.ToLower();
                // sharedMaterials + an explicit copy only for the materials changed below: `rend.materials` instantiated a
                // copy of every material on every Build / SetSkin and nothing destroyed them. The copies are owned here and
                // destroyed in Clear (never edit the shared ones: in the editor they are the FBX's assets).
                var mats = rend.sharedMaterials;
                bool changed = false;
                for (int i = 0; i < mats.Length; i++)
                {
                    if (mats[i] == null) continue;
                    bool isURP = mats[i].shader.name.StartsWith("Universal Render Pipeline");
                    string matName = mats[i].name.ToLower();
                    bool preEye = objName.Contains("high-poly") || objName.Contains("eye")
                                  || (matName.Contains("eye") && !matName.Contains("brow") && !matName.Contains("lash"));
                    bool preSkin = matName.Contains("skin") || objName.Contains("badmintonplayer");
                    if (isURP && !preEye && !preSkin) continue;          // nothing to change: keep the shared material
                    var copy = new Material(mats[i]) { name = mats[i].name };
                    _ownedMaterials.Add(copy);
                    mats[i] = copy;

                    if (!isURP)
                    {
                        Texture oldTex = null;
                        foreach (var pn in mats[i].GetTexturePropertyNames())
                        {
                            var t = mats[i].GetTexture(pn);
                            if (t != null) { oldTex = t; break; }
                        }
                        var oldColor = mats[i].HasProperty("_Color") ? mats[i].GetColor("_Color") : Color.white;

                        mats[i].shader = urpLit;
                        if (oldTex != null) mats[i].SetTexture("_BaseMap", oldTex);
                        mats[i].SetColor("_BaseColor", oldColor);
                        changed = true;
                    }

                    if (preEye)
                    {
                        mats[i].SetFloat("_Smoothness", 0.95f);
                        mats[i].SetFloat("_Metallic", 0.05f);
                        mats[i].SetColor("_BaseColor", Color.white);
                        changed = true;
                    }
                    else if (preSkin)
                    {
                        mats[i].SetFloat("_Smoothness", 0.3f);
                        changed = true;
                    }
                }
                if (changed) rend.sharedMaterials = mats;
            }
        }

        void CaptureRestDirections()
        {
            for (int i = 0; i < BoneMap.Length; i++)
            {
                var boneT = _boneTransforms[i];
                if (boneT == null) continue;

                Transform childT = _animator.GetBoneTransform(BoneMap[i].child);
                if (childT == null && boneT.childCount > 0)
                    childT = boneT.GetChild(0);

                if (childT != null)
                {
                    _restBoneDir[i] = (childT.position - boneT.position).normalized;
                }
                else
                {
                    _restBoneDir[i] = boneT.rotation * Vector3.forward;
                }
            }
        }

        public void Apply(Vector3[] jointLocal, float[] conf, bool hasPose, float cutoff)
        {
            _posed = false;
            if (_instance == null || _animator == null || !_animator.isHuman) return;
            _instance.SetActive(hasPose);
            if (!hasPose) return;

            // Position hips at pelvis, shifted up so feet sit on the ground plane
            var hipsT = _boneTransforms[0];
            int pelvis = _slot[PELVIS];
            if (hipsT != null && pelvis >= 0 && pelvis < jointLocal.Length)
                hipsT.position = _root.TransformPoint(jointLocal[pelvis]) + Vector3.up * _feetOffset;

            // Rotate each bone so its rest direction aligns with the SMPL target direction
            for (int i = 0; i < BoneMap.Length; i++)
            {
                var t = _boneTransforms[i];
                if (t == null) continue;

                int from = _slot[BoneMap[i].from], to = _slot[BoneMap[i].to];
                if (from < 0 || to < 0 || from >= jointLocal.Length || to >= jointLocal.Length) continue;
                if (conf[from] < cutoff || conf[to] < cutoff)
                    continue;

                Vector3 targetDir = (jointLocal[to] - jointLocal[from]).normalized;
                if (targetDir.sqrMagnitude < 1e-6f) continue;

                Vector3 targetDirWorld = _root.TransformDirection(targetDir);
                Vector3 restDir = _restBoneDir[i];
                if (restDir.sqrMagnitude < 1e-6f) continue;

                // Reset to rest pose first, then apply delta
                t.localRotation = _restLocalRot[i];

                // After resetting, the bone now points along restDir (world).
                // Recompute restDir from current state since parent rotations may have changed.
                Vector3 currentRestDir = _restBoneDir[i];

                Transform childT = _animator.GetBoneTransform(BoneMap[i].child);
                if (childT == null && t.childCount > 0)
                    childT = t.GetChild(0);
                if (childT != null)
                    currentRestDir = (childT.position - t.position).normalized;
                else
                    currentRestDir = t.rotation * Vector3.forward;

                if (currentRestDir.sqrMagnitude < 1e-6f) continue;

                Quaternion delta = Quaternion.FromToRotation(currentRestDir, targetDirWorld);
                t.rotation = delta * t.rotation;
            }
            _posed = true;
        }

        public void SetActive(bool on)
        {
            if (_root != null) _root.gameObject.SetActive(on);
        }

        public void Clear()
        {
            _boneTransforms = null;
            _instance = null;
            _animator = null;
            foreach (var m in _ownedMaterials) RendererUtil.DestroySafe(m);
            _ownedMaterials.Clear();
        }
    }
}
