using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace BadmintonVR.Editor
{
    public static class UpgradeCharacterMaterials
    {
        [MenuItem("Tools/Badminton/Upgrade Character Materials to URP")]
        static void Upgrade()
        {
            // Extract materials from FBX so they can be edited
            string fbxPath = "Assets/Models/character.fbx";
            var importer = AssetImporter.GetAtPath(fbxPath) as ModelImporter;
            if (importer == null)
            {
                Debug.LogError("character.fbx not found at " + fbxPath);
                return;
            }

            // Set material import to extract
            importer.materialImportMode = ModelImporterMaterialImportMode.ImportStandard;
            importer.materialLocation = ModelImporterMaterialLocation.External;

            // Create materials folder
            string matDir = "Assets/Models/Materials";
            if (!AssetDatabase.IsValidFolder(matDir))
                AssetDatabase.CreateFolder("Assets/Models", "Materials");

            importer.ExtractTextures(matDir);
            AssetDatabase.Refresh();

            // Now find and upgrade all materials
            var urpLit = Shader.Find("Universal Render Pipeline/Lit");
            if (urpLit == null)
            {
                Debug.LogError("URP/Lit shader not found");
                return;
            }

            string[] matGuids = AssetDatabase.FindAssets("t:Material", new[] { matDir });
            int count = 0;
            foreach (string guid in matGuids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
                if (mat == null) continue;

                // Save the texture before switching shader
                var mainTex = mat.mainTexture;
                var mainColor = mat.color;

                mat.shader = urpLit;
                if (mainTex != null)
                    mat.SetTexture("_BaseMap", mainTex);
                mat.SetColor("_BaseColor", mainColor);

                EditorUtility.SetDirty(mat);
                count++;
                Debug.Log($"[UpgradeCharMat] Upgraded {mat.name} -> URP/Lit" +
                          (mainTex != null ? $" tex={mainTex.name}" : ""));
            }

            // Also upgrade materials embedded in the FBX (remapped ones)
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(fbxPath);
            if (prefab != null)
            {
                var renderers = prefab.GetComponentsInChildren<Renderer>(true);
                foreach (var r in renderers)
                {
                    foreach (var mat in r.sharedMaterials)
                    {
                        if (mat == null || mat.shader == urpLit) continue;
                        var mainTex = mat.mainTexture;
                        var mainColor = mat.color;
                        mat.shader = urpLit;
                        if (mainTex != null)
                            mat.SetTexture("_BaseMap", mainTex);
                        mat.SetColor("_BaseColor", mainColor);
                        EditorUtility.SetDirty(mat);
                        count++;
                    }
                }
            }

            importer.SaveAndReimport();
            AssetDatabase.SaveAssets();
            Debug.Log($"[UpgradeCharMat] Done. Upgraded {count} materials.");
        }
    }
}
