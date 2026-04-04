#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUv;

uniform mat4 MVP;
uniform mat4 modelMatrix;
uniform mat4 lightSpaceMatrix;

out vec3 vPosition;
out vec3 vNormal;
out vec2 vUv;
out vec4 vLightSpacePosition;

void main() {
    vec4 worldPosition = modelMatrix * vec4(aPos, 1.0);
    vPosition = worldPosition.xyz;
    vNormal = mat3(transpose(inverse(modelMatrix))) * aNormal;
    vUv = aUv;
    vLightSpacePosition = lightSpaceMatrix * worldPosition;
    gl_Position = MVP * vec4(aPos, 1.0);
}
