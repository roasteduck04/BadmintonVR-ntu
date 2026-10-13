using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

/// <summary>
/// Builds a simple, regulation-proportioned badminton court out of primitives.
/// Menu: Tools > Badminton > Build Court  (full court)
///       Tools > Badminton > Build Court (Tracked Half)  (+Z half only)
///
/// Low-fidelity on purpose: flat colored surface, painted lines, a net, posts.
///
/// Every build ALSO writes data/calib/court_geometry.json (the shared source of
/// truth for corner coordinates) so tools/calibrate_court.py calibrates against
/// the exact same corners this floor is drawn from. Origin is court center in
/// both, so the two can never drift apart.
/// </summary>
public static class CourtBuilder
{
    // Doubles court dimensions (metres). Court runs along Z, width along X.
    const float LENGTH = 13.40f;         // full length (Z)
    const float WIDTH = 6.10f;           // full width (X)
    const float HALF_L = LENGTH / 2f;    // 6.70  (baseline)
    const float HALF_W = WIDTH / 2f;     // 3.05  (doubles sideline)
    const float LINE = 0.04f;            // 40 mm line width
    const float SHORT_SERVICE = 1.98f;   // from net
    const float DOUBLES_BACK = 0.76f;    // doubles long-service inset from back line
    const float SINGLES_HALF_W = 2.59f;  // singles sideline
    const float NET_TOP = 1.524f;        // net height at centre
    const float NET_BOTTOM = 0.76f;      // bottom edge of the net mesh
    const float POST_HEIGHT = 1.55f;     // post height

    const float LINE_Y = 0.011f;         // lines sit just above the floor surface (top at y=0)

    // The tracked half is the +Z half (net z=0 -> baseline z=6.70). See
    // court_geometry.json / docs/for-claude/PROGRESS.md 2026-07-14.
    public enum Scope { Full, TrackedHalf }

    [MenuItem("Tools/Badminton/Build Court")]
    public static void BuildCourt() => BuildCourt(Scope.Full);

    [MenuItem("Tools/Badminton/Build Court (Tracked Half)")]
    public static void BuildCourtHalf() => BuildCourt(Scope.TrackedHalf);

    public static void BuildCourt(Scope scope)
    {
        // Remove a previous build so re-running is clean.
        var existing = GameObject.Find("BadmintonCourt");
        if (existing != null) Undo.DestroyObjectImmediate(existing);

        var root = new GameObject("BadmintonCourt");
        Undo.RegisterCreatedObjectUndo(root, "Build Badminton Court");

        Material surface = MakeMat(new Color(0.16f, 0.42f, 0.22f)); // green
        Material white = MakeMat(Color.white);
        Material netMat = MakeNetMat();
        Material postMat = MakeMat(new Color(0.20f, 0.20f, 0.20f));

        // Drawn Z-extent. Full court spans the whole length; the tracked half
        // only covers +Z (net -> baseline) so the floor matches the recorded box.
        bool half = scope == Scope.TrackedHalf;
        float zNear = half ? 0f : -HALF_L;   // near edge of the drawn floor
        float zFar = HALF_L;                 // far edge (+Z baseline) either way
        float zMid = (zNear + zFar) * 0.5f;
        float zLen = zFar - zNear;

        // Floor slab (slightly larger than the drawn court so lines have a margin).
        var floor = Box(root, surface, "Floor",
            new Vector3(0f, -0.05f, zMid),
            new Vector3(WIDTH + 1.0f, 0.10f, zLen + 1.0f));

        // --- Painted lines ---
        // Sidelines (doubles) + singles sidelines run the full drawn Z-extent.
        Box(root, white, "Sideline_R", new Vector3(HALF_W, LINE_Y, zMid), new Vector3(LINE, 0.02f, zLen));
        Box(root, white, "Sideline_L", new Vector3(-HALF_W, LINE_Y, zMid), new Vector3(LINE, 0.02f, zLen));
        Box(root, white, "SinglesSide_R", new Vector3(SINGLES_HALF_W, LINE_Y, zMid), new Vector3(LINE, 0.02f, zLen));
        Box(root, white, "SinglesSide_L", new Vector3(-SINGLES_HALF_W, LINE_Y, zMid), new Vector3(LINE, 0.02f, zLen));

        // Net centre line (z = 0) — the front edge of the tracked half.
        Box(root, white, "NetLine", new Vector3(0f, LINE_Y, 0f), new Vector3(WIDTH, 0.02f, LINE));

        // Transverse lines on the +Z (far) half — always drawn.
        Box(root, white, "Baseline_F", new Vector3(0f, LINE_Y, HALF_L), new Vector3(WIDTH, 0.02f, LINE));
        Box(root, white, "ShortService_F", new Vector3(0f, LINE_Y, SHORT_SERVICE), new Vector3(WIDTH, 0.02f, LINE));
        float dbl = HALF_L - DOUBLES_BACK;
        Box(root, white, "DoublesLong_F", new Vector3(0f, LINE_Y, dbl), new Vector3(WIDTH, 0.02f, LINE));

        // Centre line on the +Z half (short service -> baseline).
        float cLenF = HALF_L - SHORT_SERVICE;
        Box(root, white, "CentreLine_F", new Vector3(0f, LINE_Y, (SHORT_SERVICE + HALF_L) / 2f), new Vector3(LINE, 0.02f, cLenF));

        // -Z (near) half — only on the full court.
        if (!half)
        {
            Box(root, white, "Baseline_B", new Vector3(0f, LINE_Y, -HALF_L), new Vector3(WIDTH, 0.02f, LINE));
            Box(root, white, "ShortService_B", new Vector3(0f, LINE_Y, -SHORT_SERVICE), new Vector3(WIDTH, 0.02f, LINE));
            Box(root, white, "DoublesLong_B", new Vector3(0f, LINE_Y, -dbl), new Vector3(WIDTH, 0.02f, LINE));
            Box(root, white, "CentreLine_B", new Vector3(0f, LINE_Y, -(SHORT_SERVICE + HALF_L) / 2f), new Vector3(LINE, 0.02f, cLenF));
        }

        // --- Net + posts (at z = 0) ---
        Box(root, netMat, "Net",
            new Vector3(0f, (NET_TOP + NET_BOTTOM) / 2f, 0f),
            new Vector3(WIDTH, NET_TOP - NET_BOTTOM, 0.02f));
        Post(root, postMat, "Post_R", new Vector3(HALF_W, 0f, 0f));
        Post(root, postMat, "Post_L", new Vector3(-HALF_W, 0f, 0f));

        WriteGeometryJson(scope);

        Selection.activeGameObject = root;
        EditorSceneManager.MarkSceneDirty(floor.scene);
        Debug.Log($"[CourtBuilder] Built {(half ? "TRACKED-HALF (+Z)" : "full")} court. " +
            "Tracked-half corners (X,Z m): " +
            $"ssl_fl(-{HALF_W:0.00},{SHORT_SERVICE:0.00}) ssl_fr({HALF_W:0.00},{SHORT_SERVICE:0.00}) " +
            $"corner_fr({HALF_W:0.00},{HALF_L:0.00}) corner_fl(-{HALF_W:0.00},{HALF_L:0.00}). " +
            "Wrote data/calib/court_geometry.json for calibrate_court.py.");
    }

    /// <summary>
    /// Export the court geometry (dimensions + every named corner) to
    /// data/calib/court_geometry.json so tools/calibrate_court.py uses the exact
    /// same corner coordinates this floor is built from.
    /// </summary>
    static void WriteGeometryJson(Scope scope)
    {
        var ci = CultureInfo.InvariantCulture;
        float XD = HALF_W, XS = SINGLES_HALF_W, ZB = HALF_L,
              ZL = HALF_L - DOUBLES_BACK, ZS = SHORT_SERVICE;

        // name, x, z — mirrors COURT_POINTS in calibrate_court.py.
        (string, float, float)[] pts =
        {
            ("corner_nl", -XD, -ZB), ("corner_nr", XD, -ZB),
            ("corner_fl", -XD, ZB),  ("corner_fr", XD, ZB),
            ("sing_bl_nl", -XS, -ZB), ("sing_bl_nr", XS, -ZB),
            ("sing_bl_fl", -XS, ZB),  ("sing_bl_fr", XS, ZB),
            ("ctr_bl_n", 0f, -ZB), ("ctr_bl_f", 0f, ZB),
            ("lsl_nl", -XD, -ZL), ("lsl_nr", XD, -ZL),
            ("lsl_fl", -XD, ZL),  ("lsl_fr", XD, ZL),
            ("lsl_sing_nl", -XS, -ZL), ("lsl_sing_nr", XS, -ZL),
            ("lsl_sing_fl", -XS, ZL),  ("lsl_sing_fr", XS, ZL),
            ("lsl_ctr_n", 0f, -ZL), ("lsl_ctr_f", 0f, ZL),
            ("ssl_nl", -XD, -ZS), ("ssl_nr", XD, -ZS),
            ("ssl_fl", -XD, ZS),  ("ssl_fr", XD, ZS),
            ("ssl_sing_nl", -XS, -ZS), ("ssl_sing_nr", XS, -ZS),
            ("ssl_sing_fl", -XS, ZS),  ("ssl_sing_fr", XS, ZS),
            ("ssl_ctr_n", 0f, -ZS), ("ssl_ctr_f", 0f, ZS),
            ("net_l", -XD, 0f), ("net_r", XD, 0f),
        };

        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append("  \"schema_version\": \"1.0\",\n");
        sb.Append("  \"generated_by\": \"CourtBuilder.cs (Tools > Badminton > Build Court). Regenerated on every court build; read by tools/calibrate_court.py.\",\n");
        sb.Append("  \"convention\": \"court XZ meters, origin at court center, +Z away from camera, +X right of camera; camera side is -Z. Same axes as Unity CourtBuilder and skeleton.json.\",\n");
        sb.AppendFormat(ci, "  \"tracked_half\": \"far\",\n");
        sb.AppendFormat(ci, "  \"last_build_scope\": \"{0}\",\n", scope == Scope.TrackedHalf ? "tracked_half" : "full");
        sb.Append("  \"tracked_half_note\": \"We track ONE half-court: the +Z half box between the short service line (z=1.98) and the baseline (z=6.70). Its four corners are ssl_fl, ssl_fr, corner_fr, corner_fl.\",\n");
        sb.Append("  \"dimensions_m\": {\n");
        sb.AppendFormat(ci, "    \"length_z\": {0:0.00},\n", LENGTH);
        sb.AppendFormat(ci, "    \"width_x\": {0:0.00},\n", WIDTH);
        sb.AppendFormat(ci, "    \"doubles_half_width_x\": {0:0.00},\n", XD);
        sb.AppendFormat(ci, "    \"singles_half_width_x\": {0:0.00},\n", XS);
        sb.AppendFormat(ci, "    \"baseline_z\": {0:0.00},\n", ZB);
        sb.AppendFormat(ci, "    \"doubles_long_service_z\": {0:0.00},\n", ZL);
        sb.AppendFormat(ci, "    \"short_service_z\": {0:0.00},\n", ZS);
        sb.Append("    \"net_z\": 0.0\n");
        sb.Append("  },\n");
        sb.Append("  \"tracked_half_corners\": {\n");
        sb.AppendFormat(ci, "    \"ssl_fl\": [{0:0.00}, {1:0.00}],\n", -XD, ZS);
        sb.AppendFormat(ci, "    \"ssl_fr\": [{0:0.00}, {1:0.00}],\n", XD, ZS);
        sb.AppendFormat(ci, "    \"corner_fr\": [{0:0.00}, {1:0.00}],\n", XD, ZB);
        sb.AppendFormat(ci, "    \"corner_fl\": [{0:0.00}, {1:0.00}]\n", -XD, ZB);
        sb.Append("  },\n");
        sb.Append("  \"points\": {\n");
        for (int i = 0; i < pts.Length; i++)
        {
            var (name, x, z) = pts[i];
            sb.AppendFormat(ci, "    \"{0}\": [{1:0.00}, {2:0.00}]{3}\n",
                name, x, z, i == pts.Length - 1 ? "" : ",");
        }
        sb.Append("  }\n");
        sb.Append("}\n");

        string dir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "data", "calib"));
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "court_geometry.json"), sb.ToString());
    }

    static Material MakeMat(Color c)
    {
        var shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        var m = new Material(shader);
        if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
        else m.color = c;
        return m;
    }

    static Material MakeNetMat()
    {
        var shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        var m = new Material(shader);
        var c = new Color(0.05f, 0.05f, 0.05f, 0.35f);
        if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
        else m.color = c;
        // URP transparent surface
        if (m.HasProperty("_Surface"))
        {
            m.SetFloat("_Surface", 1f); // 0=Opaque, 1=Transparent
            m.SetFloat("_Blend", 0f);   // 0=Alpha
            m.SetFloat("_AlphaClip", 0f);
            m.SetOverrideTag("RenderType", "Transparent");
            m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetInt("_ZWrite", 0);
            m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        }
        else
        {
            // Standard shader fallback
            m.SetFloat("_Mode", 3f); // Transparent
            m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetInt("_ZWrite", 0);
            m.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            m.EnableKeyword("_ALPHABLEND_ON");
        }
        return m;
    }

    static GameObject Box(GameObject parent, Material mat, string name, Vector3 pos, Vector3 scale)
    {
        var g = GameObject.CreatePrimitive(PrimitiveType.Cube);
        g.name = name;
        g.transform.SetParent(parent.transform);
        g.transform.localScale = scale;
        g.transform.position = pos;
        g.GetComponent<Renderer>().sharedMaterial = mat;
        Object.DestroyImmediate(g.GetComponent<Collider>());
        return g;
    }

    static void Post(GameObject parent, Material mat, string name, Vector3 basePos)
    {
        var g = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        g.name = name;
        g.transform.SetParent(parent.transform);
        g.transform.localScale = new Vector3(0.06f, POST_HEIGHT / 2f, 0.06f);
        g.transform.position = new Vector3(basePos.x, POST_HEIGHT / 2f, basePos.z);
        g.GetComponent<Renderer>().sharedMaterial = mat;
        Object.DestroyImmediate(g.GetComponent<Collider>());
    }
}
