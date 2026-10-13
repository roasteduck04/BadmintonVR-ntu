using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace BadmintonVR.Analysis
{
    /// <summary>Screen regions owned by IMGUI panels (viewer rows, debug toggles). The orbit camera ignores a mouse drag that
    /// starts over one of them or while an IMGUI control is hot (a slider being dragged), and scroll over them.</summary>
    public static class UiPointerGuard
    {
        static readonly System.Collections.Generic.Dictionary<object, Rect> _rects = new System.Collections.Generic.Dictionary<object, Rect>();

        /// <summary>Set (each OnGUI) the GUI-space rect (origin top-left) an owner draws its controls in.</summary>
        public static void Set(object owner, Rect guiRect) => _rects[owner] = guiRect;
        public static void Clear(object owner) => _rects.Remove(owner);

        /// <summary>GUI space (origin top-left, y down) of a screen / Input System pointer position (origin bottom-left).</summary>
        public static Vector2 ScreenToGui(Vector2 screen, float screenHeight) => new Vector2(screen.x, screenHeight - screen.y);

        public static bool Contains(System.Collections.Generic.IEnumerable<Rect> rects, Vector2 guiPoint)
        {
            foreach (var r in rects) if (r.Contains(guiPoint)) return true;
            return false;
        }

        /// <summary>Is the pointer (screen coordinates) over a registered panel, or is an IMGUI control being dragged?</summary>
        public static bool PointerOverUi(Vector2 screenPos)
            => GUIUtility.hotControl != 0 || Contains(_rects.Values, ScreenToGui(screenPos, Screen.height));
    }

    /// <summary>
    /// Desktop camera for the analysis viewer. Two modes toggled with C:
    ///   Orbit (default) — left-drag orbits, scroll zooms, right/middle-drag pans, F reframes, R resets.
    ///   Fly — right-click + WASD/QE moves, mouse look while held, scroll changes speed.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class OrbitCamera : MonoBehaviour
    {
        [Header("Orbit")]
        public Vector3 target = new Vector3(0f, 1f, 0f);
        public float distance = 14f;
        public float yaw = 0f;
        public float pitch = 24f;
        public float minPitch = 5f, maxPitch = 85f;
        public float minDistance = 3f, maxDistance = 40f;
        public float orbitSpeed = 0.25f, panSpeed = 0.01f, zoomSpeed = 1.2f;

        [Header("Fly")]
        public float flySpeed = 5f;
        public float flyLookSpeed = 0.15f;
        public float flySpeedMin = 1f, flySpeedMax = 50f;

        public bool acceptInput = true;
        public bool flyMode { get; private set; }

        float _flyYaw, _flyPitch;
        bool _dragFromUi;      // the held mouse drag started over an IMGUI panel: it belongs to the UI, not the camera

        // Fly mode owns the transform: re-enabling (e.g. leaving look-through, which restores the saved fly pose first) must
        // not snap it to the orbit pose.
        void OnEnable() { if (!flyMode) Apply(); }

        void LateUpdate()
        {
            if (acceptInput) HandleInput();
            if (!flyMode) Apply();
        }

        public void Apply()
        {
            pitch = Mathf.Clamp(pitch, minPitch, maxPitch);
            distance = Mathf.Clamp(distance, minDistance, maxDistance);
            Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);
            transform.position = target - (rot * Vector3.forward) * distance;
            transform.rotation = rot;
        }

        public void Frame()
        {
            target = new Vector3(0f, 1f, 0f);
            yaw = 0f; pitch = 24f; distance = 15f;
            Apply();
        }

#if ENABLE_INPUT_SYSTEM
        void HandleInput()
        {
            var kb = Keyboard.current;
            if (kb != null && kb.cKey.wasPressedThisFrame)
            {
                flyMode = !flyMode;
                if (flyMode)
                {
                    _flyYaw = transform.eulerAngles.y;
                    _flyPitch = transform.eulerAngles.x;
                    if (_flyPitch > 180f) _flyPitch -= 360f;
                    Cursor.lockState = CursorLockMode.None;
                }
            }

            if (flyMode)
                HandleFlyInput();
            else
                HandleOrbitInput();
        }

        /// <summary>True while the current mouse drag belongs to the IMGUI panels (started over one, or a control is hot).
        /// Latched on press, so a camera drag that passes over the panel keeps going and a slider drag never orbits.</summary>
        bool DragBelongsToUi(Mouse m)
        {
            Vector2 pos = m.position.ReadValue();
            if (m.leftButton.wasPressedThisFrame || m.rightButton.wasPressedThisFrame || m.middleButton.wasPressedThisFrame)
                _dragFromUi = UiPointerGuard.PointerOverUi(pos);
            if (!m.leftButton.isPressed && !m.rightButton.isPressed && !m.middleButton.isPressed) _dragFromUi = false;
            return _dragFromUi || GUIUtility.hotControl != 0;
        }

        void HandleOrbitInput()
        {
            var m = Mouse.current; var kb = Keyboard.current;
            if (m != null)
            {
                Vector2 d = m.delta.ReadValue();
                if (!DragBelongsToUi(m))
                {
                    if (m.leftButton.isPressed) { yaw += d.x * orbitSpeed; pitch -= d.y * orbitSpeed; }
                    if (m.rightButton.isPressed || m.middleButton.isPressed)
                    {
                        Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);
                        Vector3 right = rot * Vector3.right, up = rot * Vector3.up;
                        target -= (right * d.x + up * d.y) * panSpeed * distance;
                    }
                }
                float scroll = m.scroll.ReadValue().y;
                if (Mathf.Abs(scroll) > 0.01f && !UiPointerGuard.PointerOverUi(m.position.ReadValue()))
                    distance -= scroll * 0.01f * zoomSpeed * distance;
            }
            if (kb != null)
            {
                if (kb.fKey.wasPressedThisFrame) Frame();
                if (kb.rKey.wasPressedThisFrame) { yaw = 0f; pitch = 24f; distance = 14f; target = new Vector3(0, 1, 0); }
            }
        }

        void HandleFlyInput()
        {
            var m = Mouse.current; var kb = Keyboard.current;
            if (m == null || kb == null) return;

            if (m.rightButton.isPressed && !DragBelongsToUi(m))
            {
                Vector2 d = m.delta.ReadValue();
                _flyYaw += d.x * flyLookSpeed;
                _flyPitch -= d.y * flyLookSpeed;
                _flyPitch = Mathf.Clamp(_flyPitch, -89f, 89f);
            }

            float scroll = m.scroll.ReadValue().y;
            if (Mathf.Abs(scroll) > 0.01f && !UiPointerGuard.PointerOverUi(m.position.ReadValue()))
                flySpeed = Mathf.Clamp(flySpeed * (1f + scroll * 0.002f), flySpeedMin, flySpeedMax);

            Quaternion rot = Quaternion.Euler(_flyPitch, _flyYaw, 0f);
            transform.rotation = rot;

            Vector3 move = Vector3.zero;
            if (kb.wKey.isPressed) move += Vector3.forward;
            if (kb.sKey.isPressed) move += Vector3.back;
            if (kb.aKey.isPressed) move += Vector3.left;
            if (kb.dKey.isPressed) move += Vector3.right;
            if (kb.eKey.isPressed) move += Vector3.up;
            if (kb.qKey.isPressed) move += Vector3.down;

            float speed = flySpeed;
            if (kb.leftShiftKey.isPressed) speed *= 2.5f;

            if (move.sqrMagnitude > 0.001f)
                transform.position += rot * move.normalized * speed * Time.deltaTime;

            if (kb.fKey.wasPressedThisFrame) Frame();
            if (kb.rKey.wasPressedThisFrame)
            {
                transform.position = new Vector3(0f, 2f, -10f);
                _flyYaw = 0f; _flyPitch = 10f;
            }
        }
#else
        void HandleInput() { }
#endif
    }
}
