#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aUV;

out vec2 vUV;

uniform mat4 uViewNoTranslation;
uniform mat4 uProj;
uniform mat4 uSkyRotate;

void main() {
    vUV = aUV;
    vec4 pos = uProj * uViewNoTranslation * uSkyRotate * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
