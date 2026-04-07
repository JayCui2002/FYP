#version 330 core
in vec3 vPosition;
in vec3 vNormal;
in vec2 vUv;
in vec4 vLightSpacePosition;

out vec4 FragColor;

uniform vec3 lightPosition;
uniform vec3 lightIntensity;
uniform vec3 cameraPosition;
uniform sampler2D uBaseColorTex;
uniform sampler2D uNormalTex;
uniform sampler2D uMetallicRoughnessTex;
uniform sampler2D shadowMap;
uniform int uUseBaseColorTex;
uniform int uUseNormalTex;
uniform int uUseMetallicRoughnessTex;
uniform vec4 uBaseColorFactor;
uniform float uMetallicFactor;
uniform float uRoughnessFactor;
uniform float uNormalScale;

float shadowVisibility(vec3 normal, vec3 lightDirection) {
    vec3 projected = vLightSpacePosition.xyz / vLightSpacePosition.w;
    projected = projected * 0.5 + 0.5;
    if (projected.z > 1.0 || projected.x < 0.0 || projected.x > 1.0 || projected.y < 0.0 || projected.y > 1.0) {
        return 1.0;
    }
    float bias = max(0.002 * (1.0 - max(dot(normal, lightDirection), 0.0)), 0.0005);
    vec2 texelSize = 1.0 / vec2(textureSize(shadowMap, 0));
    float visibility = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float closestDepth = texture(shadowMap, projected.xy + vec2(x, y) * texelSize).r;
            visibility += (projected.z - bias) <= closestDepth ? 1.0 : 0.0;
        }
    }
    return visibility / 9.0;
}

vec3 sampleNormal(vec2 uv, vec3 geometricNormal) {
    vec3 N = normalize(geometricNormal);
    if (uUseNormalTex == 0) {
        return N;
    }
    vec3 mapped = texture(uNormalTex, uv).xyz * 2.0 - 1.0;
    mapped.xy *= uNormalScale;

    vec3 dp1 = dFdx(vPosition);
    vec3 dp2 = dFdy(vPosition);
    vec2 duv1 = dFdx(vUv);
    vec2 duv2 = dFdy(vUv);
    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) < 1e-6) {
        return N;
    }
    vec3 T = normalize((dp1 * duv2.y - dp2 * duv1.y) / det);
    vec3 B = normalize((-dp1 * duv2.x + dp2 * duv1.x) / det);
    return normalize(mat3(T, B, N) * mapped);
}

void main() {
    vec2 uv = vec2(vUv.x, 1.0 - vUv.y);
    vec4 base = uBaseColorFactor;
    if (uUseBaseColorTex != 0) {
        vec4 texel = texture(uBaseColorTex, uv);
        if (texel.g == 0.0 && texel.b == 0.0) {
            texel.rgb = vec3(texel.r);
        }
        base *= texel;
    }

    float metallic = clamp(uMetallicFactor, 0.0, 1.0);
    float roughness = clamp(uRoughnessFactor, 0.04, 1.0);
    if (uUseMetallicRoughnessTex != 0) {
        vec4 mr = texture(uMetallicRoughnessTex, uv);
        roughness = clamp(uRoughnessFactor * mr.g, 0.04, 1.0);
        metallic = clamp(uMetallicFactor * mr.b, 0.0, 1.0);
    }

    vec3 normal = sampleNormal(uv, vNormal);
    vec3 lightVector = lightPosition - vPosition;
    float distanceSquared = max(dot(lightVector, lightVector), 0.0001);
    vec3 lightDirection = normalize(lightVector);
    vec3 viewDirection = normalize(cameraPosition - vPosition);
    vec3 halfVector = normalize(lightDirection + viewDirection);
    float ndl = max(dot(normal, lightDirection), 0.0);
    float ndh = max(dot(normal, halfVector), 0.0);
    float visibility = shadowVisibility(normal, lightDirection);
    float shininess = mix(128.0, 8.0, roughness);
    vec3 F0 = mix(vec3(0.04), base.rgb, metallic);
    vec3 diffuse = base.rgb * (1.0 - metallic) * ndl;
    vec3 specular = F0 * pow(ndh, shininess) * ndl;
    vec3 ambient = base.rgb * 0.03 * (1.0 - metallic * 0.5);
    vec3 radiance = (ambient + (diffuse + specular) * visibility) * lightIntensity / distanceSquared;
    vec3 color = radiance / (1.0 + radiance);
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, base.a);
}
