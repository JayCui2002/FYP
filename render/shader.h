#pragma once

#include <string>

#include <glad/glad.h>

GLuint CreateProgramFromSource(const char* vertexSource, const char* fragmentSource);
GLuint LoadShadersFromFile(const std::string& vertPath, const std::string& fragPath);
