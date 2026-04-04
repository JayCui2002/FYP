#version 330 core
in vec3 vPosition;
in vec3 vNormal;
in vec2 vUv;
in vec4 vLightSpacePosition;

out vec4 FragColor;

uniform vec3 lightPosition;
uniform vec3 lightIntensity;
uniform sampler2D uBaseColorTex;
uniform sampler2D shadowMap;
uniform int uUseBaseColorTex;
uniform vec4 uBaseColorFactor;

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

void main() {
    vec4 base = uBaseColorFactor;
    if (uUseBaseColorTex != 0) {
        base *= texture(uBaseColorTex, vec2(vUv.x, 1.0 - vUv.y));
    }

    vec3 normal = normalize(vNormal);
    vec3 lightVector = lightPosition - vPosition;
    float distanceSquared = max(dot(lightVector, lightVector), 0.0001);
    vec3 lightDirection = normalize(lightVector);
    float ndl = max(dot(normal, lightDirection), 0.0);
    float visibility = shadowVisibility(normal, lightDirection);
    vec3 radiance = base.rgb * lightIntensity * ndl * visibility / distanceSquared;
    vec3 color = radiance / (1.0 + radiance);
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, base.a);
}
