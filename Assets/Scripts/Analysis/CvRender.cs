using System.Collections.Generic;
using UnityEngine;

namespace BadmintonVR.Analysis
{
    public static class CvMaterials
    {
        static Shader _shader;
        static readonly int ColorId = Shader.PropertyToID("_Color");

        public static Material Make(Color c, bool onTop, bool vertexColors = false, Texture tex = null)
        {
            if (_shader == null) _shader = Shader.Find("BadmintonVR/CvOverlay");
            if (_shader == null) return RendererUtil.MakeTransparentMat(c);     // shader not compiled: plain URP transparent
            var m = new Material(_shader);
            m.SetColor(ColorId, c);
            m.SetFloat("_ZTest", onTop ? (float)UnityEngine.Rendering.CompareFunction.Always
                                       : (float)UnityEngine.Rendering.CompareFunction.LessEqual);
            m.SetFloat("_UseVertexColor", vertexColors ? 1f : 0f);
            if (tex != null) m.mainTexture = tex;
            return m;
        }
    }

    /// <summary>Many coloured 1-px lines in one mesh (frusta, ghosts, rings): one draw call, rebuilt per frame.</summary>
    public sealed class CvLineBatch
    {
        readonly GameObject _go;
        readonly Mesh _mesh;
        readonly Material _mat;
        readonly List<Vector3> _v = new List<Vector3>();
        readonly List<Color> _c = new List<Color>();
        readonly List<int> _i = new List<int>();

        public CvLineBatch(string name, Transform parent, bool onTop)
        {
            _go = new GameObject(name);
            _go.transform.SetParent(parent, false);
            _mesh = new Mesh { name = name };
            _mesh.MarkDynamic();
            _go.AddComponent<MeshFilter>().sharedMesh = _mesh;
            _mat = CvMaterials.Make(Color.white, onTop, vertexColors: true);
            _go.AddComponent<MeshRenderer>().sharedMaterial = _mat;
        }

        public void Clear() { _v.Clear(); _c.Clear(); _i.Clear(); }

        public void Add(Vector3 a, Vector3 b, Color c)
        {
            _i.Add(_v.Count); _v.Add(a); _c.Add(c);
            _i.Add(_v.Count); _v.Add(b); _c.Add(c);
        }

        public void Commit()
        {
            _mesh.Clear();
            if (_v.Count > 65000) _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            _mesh.SetVertices(_v);
            _mesh.SetColors(_c);
            _mesh.SetIndices(_i, MeshTopology.Lines, 0);
            _mesh.RecalculateBounds();
        }

        public void SetVisible(bool on) { if (_go != null) _go.SetActive(on); }

        public void Dispose()
        {
            if (_go == null) return;
            RendererUtil.DestroySafe(_go);
            RendererUtil.DestroySafe(_mesh);
            RendererUtil.DestroySafe(_mat);      // each batch owns its material (a ghost / rings rebuild made a new one)
        }
    }
}
