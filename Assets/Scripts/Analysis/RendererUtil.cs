using UnityEngine;

namespace BadmintonVR.Analysis
{
    /// <summary>Shared primitives for pose renderers + the shuttle marker.</summary>
    public static class RendererUtil
    {
        public static Material MakeMat(Color col)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            var m = new Material(shader);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", col); else m.color = col;
            return m;
        }

        /// <summary>URP-transparent material with alpha blending enabled (for confidence-based fading).</summary>
        public static Material MakeTransparentMat(Color col)
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            var m = new Material(shader);
            // Enable transparency
            m.SetFloat("_Surface", 1f); // 0=Opaque, 1=Transparent
            m.SetFloat("_Blend", 0f);   // Alpha blend
            m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetInt("_ZWrite", 0);
            m.renderQueue = 3000; // Transparent queue
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", col);
            else m.color = col;
            return m;
        }

        /// <summary>Destroy a runtime-created object (material, mesh, ...) in play or edit mode; null is ignored.</summary>
        public static void DestroySafe(Object o)
        {
            if (o == null) return;
            if (Application.isPlaying) Object.Destroy(o);
            else Object.DestroyImmediate(o);
        }

        public static void DestroyCollider(GameObject go)
        {
            var col = go.GetComponent<Collider>();
            if (col != null)
            {
                if (Application.isPlaying) Object.Destroy(col);
                else Object.DestroyImmediate(col);
            }
        }

        /// <summary>Stretch/orient a unit capsule (Y-axis) to span p0->p1 with the given radius.</summary>
        public static void PlaceBone(Transform bone, Vector3 p0, Vector3 p1, float radius)
        {
            Vector3 dir = p1 - p0;
            float len = dir.magnitude;
            bone.localPosition = (p0 + p1) * 0.5f;
            bone.localScale = new Vector3(radius * 2f, Mathf.Max(len * 0.5f, 0.001f), radius * 2f);
            bone.localRotation = len > 1e-5f ? Quaternion.FromToRotation(Vector3.up, dir) : Quaternion.identity;
        }
    }
}
