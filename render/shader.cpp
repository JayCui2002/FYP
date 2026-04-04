#include "shader.h"

#include <fstream>
#include <sstream>

static bool readTextFile(const std::string& path, std::string& out) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return !out.empty();
}

static GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    return shader;
}

GLuint CreateProgramFromSource(const char* vertexSource, const char* fragmentSource) {
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}

GLuint LoadShadersFromFile(const std::string& vertPath, const std::string& fragPath) {
    std::string vs;
    std::string fs;
    if (!readTextFile(vertPath, vs) || !readTextFile(fragPath, fs)) return 0;
    return CreateProgramFromSource(vs.c_str(), fs.c_str());
}
