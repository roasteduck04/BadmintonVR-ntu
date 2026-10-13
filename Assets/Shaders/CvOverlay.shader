Shader "BadmintonVR/CvOverlay"
{
    // Unlit, alpha-blended overlay for the SP4 cv visualisation: joint spheres, frusta / rays / ghosts (vertex colours),
    // video billboards (_MainTex). _ZTest 8 = Always (drawn on top of any skin), 4 = LessEqual.
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
        _MainTex ("Texture", 2D) = "white" {}
        [Enum(UnityEngine.Rendering.CompareFunction)] _ZTest ("ZTest", Float) = 4
        _UseVertexColor ("Use vertex colour", Float) = 0
    }
    SubShader
    {
        Tags { "Queue"="Overlay" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest [_ZTest]
            Cull Off
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            sampler2D _MainTex; float4 _MainTex_ST; fixed4 _Color; float _UseVertexColor;
            struct appdata { float4 vertex : POSITION; float4 color : COLOR; float2 uv : TEXCOORD0; };
            struct v2f { float4 pos : SV_POSITION; fixed4 color : COLOR; float2 uv : TEXCOORD0; };
            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.color = lerp(fixed4(1,1,1,1), v.color, _UseVertexColor) * _Color;
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }
            fixed4 frag (v2f i) : SV_Target { return tex2D(_MainTex, i.uv) * i.color; }
            ENDCG
        }
    }
}
