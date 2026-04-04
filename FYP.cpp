#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <winsock2.h>
#include <ws2tcpip.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include "render/shader.h"

struct SceneConfig {
    std::string modelPath;
    std::string skyboxAtlasPath;
    float streamInputFps = 10.0f;
    bool streamV2Autostart = false;
    std::string streamV2Python = "python";
    std::string streamV2Script = "streamdiffusionv2_bridge.py";
    std::string streamV2Args;
    int streamV2Port = 8765;
    float streamV2WaitTimeoutSec = 120.0f;
    int streamCaptureSide = 512;
};

struct GltfPrimitiveGpu {
    GLuint vao = 0;
    GLuint posVbo = 0;
    GLuint normalVbo = 0;
    GLuint uvVbo = 0;
    GLuint ebo = 0;
    GLuint baseColorTex = 0;
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    GLenum mode = GL_TRIANGLES;
    GLsizei vertexCount = 0;
    GLsizei indexCount = 0;
    bool hasIndices = false;
    bool hasBaseColorTexture = false;
    bool alphaBlend = false;
    bool doubleSided = false;
};

struct GltfModelGpu {
    std::vector<GltfPrimitiveGpu> primitives;
    glm::vec3 center = glm::vec3(0.0f);
    float radius = 1.0f;
};

struct CaptureFramebuffer {
    GLuint fbo = 0;
    GLuint colorTex = 0;
    GLuint depthRbo = 0;
    int side = 0;
};

struct ShadowFramebuffer {
    GLuint fbo = 0;
    GLuint depthTex = 0;
    int width = 0;
    int height = 0;
};

static int gWindowWidth = 1024;
static int gWindowHeight = 1024;
static bool gShowStylized = false;

struct StreamRequestHeader {
    uint32_t magic = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t payloadSize = 0;
    uint32_t seq = 0;
};

struct StreamResponseHeader {
    uint32_t magic = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t payloadSize = 0;
    uint32_t flags = 0;
    uint32_t seq = 0;
};

struct StreamPendingInput {
    std::vector<unsigned char> rgb;
    int side = 0;
    uint32_t seq = 0;
};

struct StreamSharedState {
    std::mutex mutex;
    std::deque<StreamPendingInput> pendingInputs;
    std::vector<unsigned char> latestOutput;
    int outputWidth = 0;
    int outputHeight = 0;
    uint32_t latestOutputSeq = 0;
    uint32_t lastSentSeq = 0;
    uint64_t capturedCount = 0;
    uint64_t enqueuedCount = 0;
    uint64_t droppedCount = 0;
    uint64_t sentCount = 0;
    uint64_t readyCount = 0;
    uint64_t newOutputCount = 0;
    bool hasNewOutput = false;
    bool stop = false;
};

static const uint32_t kStreamRequestMagic = 0x304D5246u;
static const uint32_t kStreamResponseMagic = 0x3054554Fu;
static const uint32_t kStreamResponseFlagStylizedReady = 1u;
static const uint32_t kStreamResponseFlagReuseLatest = 2u;
static constexpr size_t kMaxPendingStreamInputs = 32;
static constexpr size_t kStreamCaptureBackpressureThreshold = (kMaxPendingStreamInputs * 3u) / 4u;

static glm::vec3 gLookAt(0.0f, 0.0f, 0.0f);
static glm::vec3 gUp(0.0f, 1.0f, 0.0f);
static float gViewAzimuth = 0.0f;
static float gViewPolar = 0.0f;
static float gViewDistance = 3.0f;
static glm::vec3 gCameraPos(0.0f, 0.0f, 3.0f);
static glm::vec3 gLightPosition(-2.75f, 5.0f, 3.0f);
static glm::vec3 gLightIntensity(8.0f, 8.0f, 8.0f);
static float gLightCursorX = 0.0f;
static float gLightCursorY = 0.0f;
static float gDefaultViewDistance = 4.0f;
static float gModelFitScale = 1.0f;

static const char* kModelVertPath = "input/shader/model.vert";
static const char* kModelFragPath = "input/shader/model.frag";
static const char* kShadowVertPath = "input/shader/model_shadow.vert";
static const char* kShadowFragPath = "input/shader/model_shadow.frag";
static const char* kSkyVertPath = "input/shader/skybox.vert";
static const char* kSkyFragPath = "input/shader/skybox.frag";

static void drawGltfModel(const GltfModelGpu& model, GLuint modelProgram);

static void updateLightFromCursor() {
    glm::vec3 forward = glm::normalize(gLookAt - gCameraPos);
    glm::vec3 right = glm::cross(forward, gUp);
    if (glm::length(right) < 1e-4f) right = glm::vec3(1.0f, 0.0f, 0.0f);
    else right = glm::normalize(right);
    glm::vec3 up = glm::normalize(glm::cross(right, forward));
    const float scale = glm::max(1.5f, gViewDistance * 1.5f);
    const glm::vec3 anchor = gCameraPos + forward * gViewDistance;
    gLightPosition = anchor + right * (gLightCursorX * scale) + up * (gLightCursorY * scale);
}

static void updateCameraFromSpherical() {
    const float maxPolar = glm::radians(85.0f);
    if (gViewPolar > maxPolar) gViewPolar = maxPolar;
    if (gViewPolar < -maxPolar) gViewPolar = -maxPolar;
    if (gViewDistance < 0.8f) gViewDistance = 0.8f;
    if (gViewDistance > 30.0f) gViewDistance = 30.0f;

    gCameraPos.x = gViewDistance * std::cos(gViewPolar) * std::cos(gViewAzimuth);
    gCameraPos.y = gViewDistance * std::sin(gViewPolar);
    gCameraPos.z = gViewDistance * std::cos(gViewPolar) * std::sin(gViewAzimuth);
    updateLightFromCursor();
}

static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;
    if (gWindowWidth <= 0 || gWindowHeight <= 0) return;
    const float nx = static_cast<float>(xpos / static_cast<double>(gWindowWidth));
    const float ny = static_cast<float>(ypos / static_cast<double>(gWindowHeight));
    gLightCursorX = nx * 2.0f - 1.0f;
    gLightCursorY = 1.0f - ny * 2.0f;
    updateLightFromCursor();
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode;
    (void)mods;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GL_TRUE);
        return;
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        gShowStylized = !gShowStylized;
        return;
    }
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        gViewAzimuth = 0.0f;
        gViewPolar = 0.0f;
        gViewDistance = gDefaultViewDistance;
        updateCameraFromSpherical();
        return;
    }

    if (key == GLFW_KEY_LEFT) gViewAzimuth -= 0.05f;
    if (key == GLFW_KEY_RIGHT) gViewAzimuth += 0.05f;
    if (key == GLFW_KEY_UP) gViewPolar += 0.05f;
    if (key == GLFW_KEY_DOWN) gViewPolar -= 0.05f;
    if (key == GLFW_KEY_W) gViewDistance -= 0.2f;
    if (key == GLFW_KEY_S) gViewDistance += 0.2f;
    updateCameraFromSpherical();
}

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    (void)window;
    gWindowWidth = width;
    gWindowHeight = height;
    glViewport(0, 0, width, height);
}

static std::string trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) begin++;
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
    return s.substr(begin, end - begin);
}

static std::vector<std::string> splitArgsPreservingQuotes(const std::string& args) {
    std::vector<std::string> tokens;
    std::string current;
    bool inQuotes = false;
    for (char ch : args) {
        if (ch == '"') {
            inQuotes = !inQuotes;
            continue;
        }
        if (!inQuotes && std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

static int extractIntArgValue(const std::string& args, const std::string& key, int fallbackValue) {
    const std::vector<std::string> tokens = splitArgsPreservingQuotes(args);
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == key && i + 1 < tokens.size()) {
            try {
                return std::stoi(tokens[i + 1]);
            } catch (...) {
                return fallbackValue;
            }
        }
        const std::string prefix = key + "=";
        if (tokens[i].rfind(prefix, 0) == 0) {
            try {
                return std::stoi(tokens[i].substr(prefix.size()));
            } catch (...) {
                return fallbackValue;
            }
        }
    }
    return fallbackValue;
}

static bool loadSceneConfig(const std::string& filePath, SceneConfig& cfg) {
    std::ifstream in(filePath);
    if (!in.is_open()) return false;

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }

    cfg.modelPath = kv["model"];
    cfg.skyboxAtlasPath = kv["skybox_atlas"];
    if (kv.count("stream_input_fps") > 0) cfg.streamInputFps = std::stof(kv["stream_input_fps"]);
    if (kv.count("stream_v2_autostart") > 0) cfg.streamV2Autostart = (kv["stream_v2_autostart"] == "1" || kv["stream_v2_autostart"] == "true" || kv["stream_v2_autostart"] == "TRUE");
    if (kv.count("stream_v2_python") > 0) cfg.streamV2Python = kv["stream_v2_python"];
    if (kv.count("stream_v2_script") > 0) cfg.streamV2Script = kv["stream_v2_script"];
    if (kv.count("stream_v2_args") > 0) cfg.streamV2Args = kv["stream_v2_args"];
    if (kv.count("stream_v2_port") > 0) cfg.streamV2Port = std::stoi(kv["stream_v2_port"]);
    if (kv.count("stream_v2_wait_timeout_sec") > 0) cfg.streamV2WaitTimeoutSec = std::stof(kv["stream_v2_wait_timeout_sec"]);
    const int streamWidth = extractIntArgValue(cfg.streamV2Args, "--width", 512);
    const int streamHeight = extractIntArgValue(cfg.streamV2Args, "--height", 512);
    cfg.streamCaptureSide = std::max(64, std::min(streamWidth, streamHeight));
    return !cfg.modelPath.empty() && !cfg.skyboxAtlasPath.empty();
}

static std::string quoteArg(const std::string& s) {
    return "\"" + s + "\"";
}

static void launchStreamV2Bridge(const SceneConfig& cfg) {
    if (!cfg.streamV2Autostart) return;
    if (cfg.streamV2Python.empty() || cfg.streamV2Script.empty()) return;
    std::stringstream cmd;
    cmd << "start \"\" /B "
        << quoteArg(cfg.streamV2Python) << " "
        << quoteArg(cfg.streamV2Script)
        << " --port " << cfg.streamV2Port
        << " --fps " << cfg.streamInputFps;
    if (!cfg.streamV2Args.empty()) cmd << " " << cfg.streamV2Args;
    std::system(cmd.str().c_str());
}

static bool sendAll(SOCKET sock, const unsigned char* data, int size) {
    int sent = 0;
    while (sent < size) {
        int rc = send(sock, reinterpret_cast<const char*>(data) + sent, size - sent, 0);
        if (rc == SOCKET_ERROR || rc == 0) return false;
        sent += rc;
    }
    return true;
}

static bool recvAll(SOCKET sock, unsigned char* data, int size) {
    int received = 0;
    while (received < size) {
        int rc = recv(sock, reinterpret_cast<char*>(data) + received, size - received, 0);
        if (rc == SOCKET_ERROR || rc == 0) return false;
        received += rc;
    }
    return true;
}

static void stopStreamWorker(StreamSharedState* shared) {
    {
        std::lock_guard<std::mutex> lock(shared->mutex);
        shared->stop = true;
    }
}

static bool initCaptureFramebuffer(CaptureFramebuffer& capture, int side) {
    if (side <= 0) return false;
    capture.side = side;
    glGenFramebuffers(1, &capture.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, capture.fbo);

    glGenTextures(1, &capture.colorTex);
    glBindTexture(GL_TEXTURE_2D, capture.colorTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, side, side, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, capture.colorTex, 0);

    glGenRenderbuffers(1, &capture.depthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, capture.depthRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, side, side);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, capture.depthRbo);

    const bool ok = (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return ok;
}

static void cleanupCaptureFramebuffer(CaptureFramebuffer& capture) {
    if (capture.depthRbo != 0) glDeleteRenderbuffers(1, &capture.depthRbo);
    if (capture.colorTex != 0) glDeleteTextures(1, &capture.colorTex);
    if (capture.fbo != 0) glDeleteFramebuffers(1, &capture.fbo);
    capture = {};
}

static bool initShadowFramebuffer(ShadowFramebuffer& shadow, int width, int height) {
    if (width <= 0 || height <= 0) return false;
    shadow.width = width;
    shadow.height = height;
    glGenFramebuffers(1, &shadow.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, shadow.fbo);

    glGenTextures(1, &shadow.depthTex);
    glBindTexture(GL_TEXTURE_2D, shadow.depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const float borderColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow.depthTex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);

    const bool ok = (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return ok;
}

static void cleanupShadowFramebuffer(ShadowFramebuffer& shadow) {
    if (shadow.depthTex != 0) glDeleteTextures(1, &shadow.depthTex);
    if (shadow.fbo != 0) glDeleteFramebuffers(1, &shadow.fbo);
    shadow = {};
}

static glm::mat4 computeLightSpaceMatrix(const GltfModelGpu& modelGpu, const glm::mat4& modelMatrix, const glm::vec3& lightPosition) {
    const glm::vec3 sceneCenter = glm::vec3(modelMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    glm::vec3 lightDir = sceneCenter - lightPosition;
    if (glm::length(lightDir) < 1e-4f) lightDir = glm::vec3(0.0f, 0.0f, -1.0f);
    else lightDir = glm::normalize(lightDir);
    glm::vec3 lightUp = (std::abs(glm::dot(lightDir, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.98f)
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::mat4 lightView = glm::lookAt(lightPosition, sceneCenter, lightUp);
    const float sceneRadius = glm::max(1.5f, modelGpu.radius * gModelFitScale * 2.5f);
    const float lightDistance = glm::max(0.1f, glm::distance(lightPosition, sceneCenter));
    const float orthoSize = glm::max(sceneRadius * 2.0f, lightDistance * 0.75f);
    const float nearPlane = glm::max(0.1f, lightDistance - sceneRadius * 4.0f);
    const float farPlane = lightDistance + sceneRadius * 4.0f;
    const glm::mat4 lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, nearPlane, farPlane);
    return lightProjection * lightView;
}

static void renderShadowMap(
    const GltfModelGpu& modelGpu,
    GLuint shadowProgram,
    const ShadowFramebuffer& shadowFramebuffer,
    const glm::mat4& modelMatrix,
    const glm::mat4& lightSpaceMatrix
) {
    glViewport(0, 0, shadowFramebuffer.width, shadowFramebuffer.height);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFramebuffer.fbo);
    glClear(GL_DEPTH_BUFFER_BIT);
    glUseProgram(shadowProgram);
    glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
    glUniformMatrix4fv(glGetUniformLocation(shadowProgram, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
    drawGltfModel(modelGpu, shadowProgram);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static bool captureFrameFromFramebuffer(const CaptureFramebuffer& capture, std::vector<unsigned char>& outRgb, int& outSide) {
    if (capture.fbo == 0 || capture.side <= 0) return false;

    std::vector<unsigned char> pixels(static_cast<size_t>(capture.side) * static_cast<size_t>(capture.side) * 3u);
    std::vector<unsigned char> flipped(static_cast<size_t>(capture.side) * static_cast<size_t>(capture.side) * 3u);

    glBindFramebuffer(GL_FRAMEBUFFER, capture.fbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, capture.side, capture.side, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    const size_t rowBytes = static_cast<size_t>(capture.side) * 3u;
    for (int y = 0; y < capture.side; ++y) {
        size_t src = static_cast<size_t>(capture.side - 1 - y) * rowBytes;
        size_t dst = static_cast<size_t>(y) * rowBytes;
        std::copy(pixels.begin() + static_cast<long long>(src),
                  pixels.begin() + static_cast<long long>(src + rowBytes),
                  flipped.begin() + static_cast<long long>(dst));
    }
    outRgb = std::move(flipped);
    outSide = capture.side;
    return true;
}

static void updateStylizedTextureFromMemory(const std::vector<unsigned char>& rgb, int width, int height, GLuint texture) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
}

static void renderSceneToCurrentFramebuffer(
    const GltfModelGpu& modelGpu,
    GLuint modelProgram,
    GLuint skyProgram,
    GLuint skyVAO,
    GLuint skyAtlasTex,
    GLuint shadowMapTex,
    const glm::mat4& modelMatrix,
    const glm::mat4& view,
    const glm::mat4& lightSpaceMatrix,
    const glm::vec3& lightPosition,
    const glm::vec3& lightIntensity,
    int viewportWidth,
    int viewportHeight
) {
    glViewport(0, 0, viewportWidth, viewportHeight);
    glClearColor(0.08f, 0.1f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const float aspect = (viewportHeight != 0) ? static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight) : 1.0f;
    const glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
    const glm::mat4 rotateSky = glm::mat4(1.0f);

    glUseProgram(modelProgram);
    glUniform1i(glGetUniformLocation(modelProgram, "uBaseColorTex"), 0);
    glUniform1i(glGetUniformLocation(modelProgram, "shadowMap"), 1);
    const glm::mat4 mvp = proj * view * modelMatrix;
    glUniformMatrix4fv(glGetUniformLocation(modelProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniformMatrix4fv(glGetUniformLocation(modelProgram, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(glGetUniformLocation(modelProgram, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
    glUniform3fv(glGetUniformLocation(modelProgram, "lightPosition"), 1, glm::value_ptr(lightPosition));
    glUniform3fv(glGetUniformLocation(modelProgram, "lightIntensity"), 1, glm::value_ptr(lightIntensity));
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, shadowMapTex);
    drawGltfModel(modelGpu, modelProgram);
    glActiveTexture(GL_TEXTURE0);

    glDisable(GL_CULL_FACE);
    glDepthFunc(GL_LEQUAL);
    glUseProgram(skyProgram);
    glm::mat4 viewNoTranslation = glm::mat4(glm::mat3(view));
    glUniformMatrix4fv(glGetUniformLocation(skyProgram, "uViewNoTranslation"), 1, GL_FALSE, glm::value_ptr(viewNoTranslation));
    glUniformMatrix4fv(glGetUniformLocation(skyProgram, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniformMatrix4fv(glGetUniformLocation(skyProgram, "uSkyRotate"), 1, GL_FALSE, glm::value_ptr(rotateSky));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, skyAtlasTex);
    glBindVertexArray(skyVAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
    glBindVertexArray(0);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
}

static SOCKET connectToStreamV2(const SceneConfig& cfg) {
    WSADATA wsaData{};
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return INVALID_SOCKET;
    const auto timeout = std::chrono::duration<float>((cfg.streamV2WaitTimeoutSec > 0.0f) ? cfg.streamV2WaitTimeoutSec : 120.0f);
    const auto begin = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - begin < timeout) {
        SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == INVALID_SOCKET) break;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<u_short>(cfg.streamV2Port));
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            DWORD timeoutMs = 300000;
            setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeoutMs), sizeof(timeoutMs));
            setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeoutMs), sizeof(timeoutMs));
            return sock;
        }
        closesocket(sock);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    WSACleanup();
    return INVALID_SOCKET;
}

static void streamSendLoop(SOCKET sock, StreamSharedState* shared) {
    while (true) {
        std::vector<unsigned char> input;
        int side = 0;
        uint32_t seq = 0;
        {
            std::lock_guard<std::mutex> lock(shared->mutex);
            if (shared->stop) break;
            if (!shared->pendingInputs.empty()) {
                StreamPendingInput pending = std::move(shared->pendingInputs.front());
                shared->pendingInputs.pop_front();
                input = std::move(pending.rgb);
                side = pending.side;
                seq = pending.seq;
            }
        }
        if (input.empty() || side <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        StreamRequestHeader request{};
        request.magic = kStreamRequestMagic;
        request.width = static_cast<uint32_t>(side);
        request.height = static_cast<uint32_t>(side);
        request.payloadSize = static_cast<uint32_t>(input.size());
        request.seq = seq;
        if (!sendAll(sock, reinterpret_cast<const unsigned char*>(&request), sizeof(request))) {
            stopStreamWorker(shared);
            return;
        }
        if (!sendAll(sock, input.data(), static_cast<int>(input.size()))) {
            stopStreamWorker(shared);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(shared->mutex);
            shared->lastSentSeq = seq;
            ++shared->sentCount;
        }
    }
}

static void streamReceiveLoop(SOCKET sock, StreamSharedState* shared) {
    while (true) {
        {
            std::lock_guard<std::mutex> lock(shared->mutex);
            if (shared->stop) break;
        }
        StreamResponseHeader response{};
        if (!recvAll(sock, reinterpret_cast<unsigned char*>(&response), sizeof(response))) {
            stopStreamWorker(shared);
            return;
        }
        if (response.magic != kStreamResponseMagic) {
            stopStreamWorker(shared);
            return;
        }
        std::vector<unsigned char> output(response.payloadSize);
        if (!recvAll(sock, output.data(), static_cast<int>(output.size()))) {
            stopStreamWorker(shared);
            return;
        }
        if ((response.flags & kStreamResponseFlagStylizedReady) == 0u) continue;
        {
            std::lock_guard<std::mutex> lock(shared->mutex);
            ++shared->readyCount;
            const bool hasPayload = response.payloadSize > 0u;
            const bool reuseLatest = (response.flags & kStreamResponseFlagReuseLatest) != 0u;
            if (hasPayload) {
                if (response.seq < shared->latestOutputSeq) continue;
                shared->latestOutput = std::move(output);
                shared->outputWidth = static_cast<int>(response.width);
                shared->outputHeight = static_cast<int>(response.height);
                shared->latestOutputSeq = response.seq;
                ++shared->newOutputCount;
                shared->hasNewOutput = true;
            }
        }
    }
}

static bool copyAccessorVec3(const tinygltf::Model& model, int accessorIndex, std::vector<float>& outData) {
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size())) return false;
    const tinygltf::Accessor& accessor = model.accessors[static_cast<size_t>(accessorIndex)];
    if (accessor.type != TINYGLTF_TYPE_VEC3 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) return false;
    const tinygltf::BufferView& bv = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    const tinygltf::Buffer& buf = model.buffers[static_cast<size_t>(bv.buffer)];
    const unsigned char* base = buf.data.data() + bv.byteOffset + accessor.byteOffset;
    int stride = accessor.ByteStride(bv);
    if (stride <= 0) stride = static_cast<int>(sizeof(float) * 3);

    outData.resize(accessor.count * 3);
    for (size_t i = 0; i < accessor.count; ++i) {
        const float* src = reinterpret_cast<const float*>(base + i * static_cast<size_t>(stride));
        outData[i * 3 + 0] = src[0];
        outData[i * 3 + 1] = src[1];
        outData[i * 3 + 2] = src[2];
    }
    return true;
}

static bool copyAccessorVec2(const tinygltf::Model& model, int accessorIndex, std::vector<float>& outData) {
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size())) return false;
    const tinygltf::Accessor& accessor = model.accessors[static_cast<size_t>(accessorIndex)];
    if (accessor.type != TINYGLTF_TYPE_VEC2 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) return false;
    const tinygltf::BufferView& bv = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    const tinygltf::Buffer& buf = model.buffers[static_cast<size_t>(bv.buffer)];
    const unsigned char* base = buf.data.data() + bv.byteOffset + accessor.byteOffset;
    int stride = accessor.ByteStride(bv);
    if (stride <= 0) stride = static_cast<int>(sizeof(float) * 2);

    outData.resize(accessor.count * 2);
    for (size_t i = 0; i < accessor.count; ++i) {
        const float* src = reinterpret_cast<const float*>(base + i * static_cast<size_t>(stride));
        outData[i * 2 + 0] = src[0];
        outData[i * 2 + 1] = src[1];
    }
    return true;
}

static bool copyIndicesAsUint(const tinygltf::Model& model, int accessorIndex, std::vector<unsigned int>& outIndices) {
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size())) return false;
    const tinygltf::Accessor& accessor = model.accessors[static_cast<size_t>(accessorIndex)];
    if (accessor.type != TINYGLTF_TYPE_SCALAR) return false;
    const tinygltf::BufferView& bv = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    const tinygltf::Buffer& buf = model.buffers[static_cast<size_t>(bv.buffer)];
    const unsigned char* base = buf.data.data() + bv.byteOffset + accessor.byteOffset;
    int stride = accessor.ByteStride(bv);
    if (stride <= 0) {
        if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) stride = 1;
        else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) stride = 2;
        else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) stride = 4;
    }

    outIndices.resize(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        const unsigned char* p = base + i * static_cast<size_t>(stride);
        if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) outIndices[i] = *reinterpret_cast<const uint8_t*>(p);
        else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) outIndices[i] = *reinterpret_cast<const uint16_t*>(p);
        else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) outIndices[i] = *reinterpret_cast<const uint32_t*>(p);
        else return false;
    }
    return true;
}

static GLuint uploadTexture2D(const unsigned char* pixels, int width, int height, int components) {
    if (!pixels || width <= 0 || height <= 0 || components <= 0) return 0;

    GLenum format = GL_RGB;
    if (components == 1) format = GL_RED;
    else if (components == 2) format = GL_RG;
    else if (components == 4) format = GL_RGBA;

    GLuint texId = 0;
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        format,
        width,
        height,
        0,
        format,
        GL_UNSIGNED_BYTE,
        pixels
    );
    return texId;
}

static GLuint loadGltfBaseColorTexture(
    const tinygltf::Model& model,
    int materialIndex,
    const std::filesystem::path& modelDir
) {
    if (materialIndex < 0 || materialIndex >= static_cast<int>(model.materials.size())) return 0;
    const tinygltf::Material& material = model.materials[static_cast<size_t>(materialIndex)];
    const int textureIndex = material.pbrMetallicRoughness.baseColorTexture.index;
    if (textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size())) return 0;
    const tinygltf::Texture& texture = model.textures[static_cast<size_t>(textureIndex)];
    const int imageIndex = texture.source;
    if (imageIndex < 0 || imageIndex >= static_cast<int>(model.images.size())) return 0;
    const tinygltf::Image& image = model.images[static_cast<size_t>(imageIndex)];
    if (!image.image.empty() && image.width > 0 && image.height > 0) {
        return uploadTexture2D(image.image.data(), image.width, image.height, image.component);
    }

    if (!image.uri.empty()) {
        const std::filesystem::path texturePath = modelDir / std::filesystem::path(image.uri);
        int width = 0;
        int height = 0;
        int components = 0;
        stbi_set_flip_vertically_on_load(false);
        unsigned char* pixels = stbi_load(texturePath.string().c_str(), &width, &height, &components, 0);
        if (pixels) {
            const GLuint texId = uploadTexture2D(pixels, width, height, components);
            stbi_image_free(pixels);
            return texId;
        }
        (void)texturePath;
        (void)material;
    }

    return 0;
}

static bool loadGltfModel(const std::string& path, GltfModelGpu& outModel) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;

    const bool isGlb = path.size() >= 4 &&
        (path.substr(path.size() - 4) == ".glb" || path.substr(path.size() - 4) == ".GLB");
    bool ok = isGlb ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                    : loader.LoadASCIIFromFile(&model, &err, &warn, path);
    (void)warn;
    (void)err;
    if (!ok) return false;

    const std::filesystem::path modelDir = std::filesystem::path(path).parent_path();
    outModel.primitives.clear();
    bool hasBounds = false;
    glm::vec3 minP(0.0f), maxP(0.0f);
    size_t primitiveIndex = 0;

    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            auto itPos = primitive.attributes.find("POSITION");
            if (itPos == primitive.attributes.end()) continue;

            std::vector<float> positions;
            if (!copyAccessorVec3(model, itPos->second, positions) || positions.empty()) continue;

            for (size_t i = 0; i < positions.size(); i += 3) {
                glm::vec3 p(positions[i], positions[i + 1], positions[i + 2]);
                if (!hasBounds) { minP = p; maxP = p; hasBounds = true; }
                else { minP = glm::min(minP, p); maxP = glm::max(maxP, p); }
            }

            std::vector<float> normals;
            auto itNormal = primitive.attributes.find("NORMAL");
            if (itNormal != primitive.attributes.end()) {
                if (!copyAccessorVec3(model, itNormal->second, normals) || normals.size() != positions.size()) {
                    normals.assign(positions.size(), 0.0f);
                }
            } else {
                normals.assign(positions.size(), 0.0f);
            }
            for (size_t i = 0; i < normals.size(); i += 3) {
                if (normals[i] == 0.0f && normals[i + 1] == 0.0f && normals[i + 2] == 0.0f) normals[i + 1] = 1.0f;
            }

            std::vector<float> uvs;
            auto itUv = primitive.attributes.find("TEXCOORD_0");
            if (itUv != primitive.attributes.end()) {
                if (!copyAccessorVec2(model, itUv->second, uvs) || (uvs.size() / 2u) != (positions.size() / 3u)) {
                    uvs.clear();
                }
            }

            GltfPrimitiveGpu gpu{};
            gpu.mode = (primitive.mode >= 0) ? static_cast<GLenum>(primitive.mode) : GL_TRIANGLES;
            if (primitive.material >= 0 && primitive.material < static_cast<int>(model.materials.size())) {
                const tinygltf::Material& material = model.materials[static_cast<size_t>(primitive.material)];
                gpu.doubleSided = material.doubleSided;
                gpu.alphaBlend = (material.alphaMode == "BLEND");
                const auto& factor = material.pbrMetallicRoughness.baseColorFactor;
                if (factor.size() == 4) {
                    gpu.baseColorFactor = glm::vec4(
                        static_cast<float>(factor[0]),
                        static_cast<float>(factor[1]),
                        static_cast<float>(factor[2]),
                        static_cast<float>(factor[3])
                    );
                }
            }

            glGenVertexArrays(1, &gpu.vao);
            glBindVertexArray(gpu.vao);

            glGenBuffers(1, &gpu.posVbo);
            glBindBuffer(GL_ARRAY_BUFFER, gpu.posVbo);
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(positions.size() * sizeof(float)), positions.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0));

            glGenBuffers(1, &gpu.normalVbo);
            glBindBuffer(GL_ARRAY_BUFFER, gpu.normalVbo);
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(normals.size() * sizeof(float)), normals.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0));

            if (!uvs.empty()) {
                glGenBuffers(1, &gpu.uvVbo);
                glBindBuffer(GL_ARRAY_BUFFER, gpu.uvVbo);
                glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(uvs.size() * sizeof(float)), uvs.data(), GL_STATIC_DRAW);
                glEnableVertexAttribArray(2);
                glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0));
                gpu.baseColorTex = loadGltfBaseColorTexture(model, primitive.material, modelDir);
                gpu.hasBaseColorTexture = (gpu.baseColorTex != 0);
            }

            std::string materialName = "<none>";
            if (primitive.material >= 0 && primitive.material < static_cast<int>(model.materials.size())) {
                materialName = model.materials[static_cast<size_t>(primitive.material)].name;
            }
            ++primitiveIndex;

            if (primitive.indices >= 0) {
                std::vector<unsigned int> indices;
                if (copyIndicesAsUint(model, primitive.indices, indices) && !indices.empty()) {
                    glGenBuffers(1, &gpu.ebo);
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu.ebo);
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned int)), indices.data(), GL_STATIC_DRAW);
                    gpu.hasIndices = true;
                    gpu.indexCount = static_cast<GLsizei>(indices.size());
                }
            }
            if (!gpu.hasIndices) gpu.vertexCount = static_cast<GLsizei>(positions.size() / 3);

            glBindVertexArray(0);
            outModel.primitives.push_back(gpu);
        }
    }

    if (hasBounds) {
        outModel.center = (minP + maxP) * 0.5f;
        outModel.radius = glm::max(0.001f, glm::length(maxP - minP) * 0.5f);
    }
    return !outModel.primitives.empty();
}

static void drawGltfModel(const GltfModelGpu& model, GLuint modelProgram) {
    for (const auto& primitive : model.primitives) {
        glUniform4fv(glGetUniformLocation(modelProgram, "uBaseColorFactor"), 1, glm::value_ptr(primitive.baseColorFactor));
        glUniform1i(glGetUniformLocation(modelProgram, "uUseBaseColorTex"), primitive.hasBaseColorTexture ? 1 : 0);
        if (primitive.alphaBlend) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        } else {
            glDisable(GL_BLEND);
        }
        if (primitive.doubleSided) glDisable(GL_CULL_FACE);
        else glEnable(GL_CULL_FACE);
        if (primitive.hasBaseColorTexture && primitive.baseColorTex != 0) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, primitive.baseColorTex);
        } else {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glBindVertexArray(primitive.vao);
        if (primitive.hasIndices) glDrawElements(primitive.mode, primitive.indexCount, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
        else glDrawArrays(primitive.mode, 0, primitive.vertexCount);
    }
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
}

static void cleanupGltfModel(GltfModelGpu& model) {
    for (auto& primitive : model.primitives) {
        if (primitive.ebo != 0) glDeleteBuffers(1, &primitive.ebo);
        if (primitive.uvVbo != 0) glDeleteBuffers(1, &primitive.uvVbo);
        if (primitive.posVbo != 0) glDeleteBuffers(1, &primitive.posVbo);
        if (primitive.normalVbo != 0) glDeleteBuffers(1, &primitive.normalVbo);
        if (primitive.baseColorTex != 0) glDeleteTextures(1, &primitive.baseColorTex);
        if (primitive.vao != 0) glDeleteVertexArrays(1, &primitive.vao);
    }
    model.primitives.clear();
}

static GLuint loadSkyAtlasTexture2D(const std::string& path) {
    int width = 0, height = 0, channels = 0;
    stbi_set_flip_vertically_on_load(false);
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    if (!data) return 0;

    GLuint texId = 0;
    glGenTextures(1, &texId);
    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLenum format = (channels == 4) ? GL_RGBA : (channels == 1 ? GL_RED : GL_RGB);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
    return texId;
}

int main() {
    SceneConfig config{};
    if (!loadSceneConfig("input/scene.cfg", config)) {
        return 1;
    }
    launchStreamV2Bridge(config);
    SOCKET streamSocket = connectToStreamV2(config);
    if (streamSocket == INVALID_SOCKET) {
        return 1;
    }
    if (!glfwInit()) {
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(gWindowWidth, gWindowHeight, "OUT FPS: 0 | NEW FPS: 0", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }
    glfwSetWindowAspectRatio(window, 1, 1);
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    GLuint modelProgram = LoadShadersFromFile(kModelVertPath, kModelFragPath);
    GLuint shadowProgram = LoadShadersFromFile(kShadowVertPath, kShadowFragPath);
    GLuint skyProgram = LoadShadersFromFile(kSkyVertPath, kSkyFragPath);
    const char* quadVs = "#version 330 core\nlayout(location=0) in vec2 aPos;\nlayout(location=1) in vec2 aUv;\nout vec2 vUv;\nvoid main(){vUv=aUv;gl_Position=vec4(aPos,0.0,1.0);}";
    const char* quadFs = "#version 330 core\nin vec2 vUv;\nout vec4 FragColor;\nuniform sampler2D uTex;\nvoid main(){FragColor=texture(uTex,vec2(vUv.x,1.0-vUv.y));}";
    GLuint quadProgram = CreateProgramFromSource(quadVs, quadFs);
    if (modelProgram == 0 || shadowProgram == 0 || skyProgram == 0 || quadProgram == 0) {
        if (shadowProgram != 0) glDeleteProgram(shadowProgram);
        if (quadProgram != 0) glDeleteProgram(quadProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    GltfModelGpu modelGpu;
    if (!loadGltfModel(config.modelPath, modelGpu)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    gModelFitScale = 1.2f / glm::max(0.001f, modelGpu.radius);
    gLookAt = glm::vec3(0.0f);
    gDefaultViewDistance = 4.0f;
    gViewDistance = gDefaultViewDistance;
    updateCameraFromSpherical();

    float skyboxVertexData[] = {
        -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
         1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f
    };
    float skyboxUvData[] = {
        0.25f, 0.666f, 0.5f, 0.666f, 0.5f, 0.333f, 0.25f, 0.333f,
        0.75f, 0.666f, 1.0f, 0.666f, 1.0f, 0.333f, 0.75f, 0.333f,
        0.0f, 0.666f, 0.25f, 0.666f, 0.25f, 0.333f, 0.0f, 0.333f,
        0.5f, 0.666f, 0.75f, 0.666f, 0.75f, 0.333f, 0.5f, 0.333f,
        0.25f, 0.333f, 0.5f, 0.333f, 0.5f, 0.0f, 0.25f, 0.0f,
        0.25f, 1.0f, 0.5f, 1.0f, 0.5f, 0.666f, 0.25f, 0.666f
    };
    unsigned int skyboxIndexData[] = {
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11,
        12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23
    };

    GLuint skyVAO = 0, skyPosVBO = 0, skyUvVBO = 0, skyEBO = 0;
    glGenVertexArrays(1, &skyVAO);
    glGenBuffers(1, &skyPosVBO);
    glGenBuffers(1, &skyUvVBO);
    glGenBuffers(1, &skyEBO);
    glBindVertexArray(skyVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyPosVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertexData), skyboxVertexData, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0));
    glBindBuffer(GL_ARRAY_BUFFER, skyUvVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxUvData), skyboxUvData, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<void*>(0));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, skyEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(skyboxIndexData), skyboxIndexData, GL_STATIC_DRAW);
    glBindVertexArray(0);

    GLuint skyAtlasTex = loadSkyAtlasTexture2D(config.skyboxAtlasPath);
    if (skyAtlasTex == 0) {
        cleanupGltfModel(modelGpu);
        glDeleteVertexArrays(1, &skyVAO);
        glDeleteBuffers(1, &skyPosVBO);
        glDeleteBuffers(1, &skyUvVBO);
        glDeleteBuffers(1, &skyEBO);
        glDeleteProgram(modelProgram);
        glDeleteProgram(shadowProgram);
        glDeleteProgram(skyProgram);
        glDeleteProgram(quadProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    float quadVertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };
    unsigned int quadIndices[] = {0, 1, 2, 0, 2, 3};
    GLuint quadVAO = 0, quadVBO = 0, quadEBO = 0, stylizedTex = 0;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glGenBuffers(1, &quadEBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * static_cast<GLsizei>(sizeof(float)), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * static_cast<GLsizei>(sizeof(float)), reinterpret_cast<void*>(2 * sizeof(float)));
    glBindVertexArray(0);
    glGenTextures(1, &stylizedTex);
    glUseProgram(quadProgram);
    glUniform1i(glGetUniformLocation(quadProgram, "uTex"), 0);

    glUseProgram(skyProgram);
    glUniform1i(glGetUniformLocation(skyProgram, "uSkyAtlas"), 0);

    CaptureFramebuffer captureFramebuffer{};
    ShadowFramebuffer shadowFramebuffer{};
    if (!initCaptureFramebuffer(captureFramebuffer, config.streamCaptureSide)) {
        cleanupGltfModel(modelGpu);
        glDeleteVertexArrays(1, &skyVAO);
        glDeleteBuffers(1, &skyPosVBO);
        glDeleteBuffers(1, &skyUvVBO);
        glDeleteBuffers(1, &skyEBO);
        glDeleteVertexArrays(1, &quadVAO);
        glDeleteBuffers(1, &quadVBO);
        glDeleteBuffers(1, &quadEBO);
        glDeleteTextures(1, &skyAtlasTex);
        glDeleteTextures(1, &stylizedTex);
        glDeleteProgram(modelProgram);
        glDeleteProgram(skyProgram);
        glDeleteProgram(quadProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    if (!initShadowFramebuffer(shadowFramebuffer, 1024, 1024)) {
        cleanupCaptureFramebuffer(captureFramebuffer);
        cleanupGltfModel(modelGpu);
        glDeleteVertexArrays(1, &skyVAO);
        glDeleteBuffers(1, &skyPosVBO);
        glDeleteBuffers(1, &skyUvVBO);
        glDeleteBuffers(1, &skyEBO);
        glDeleteVertexArrays(1, &quadVAO);
        glDeleteBuffers(1, &quadVBO);
        glDeleteBuffers(1, &quadEBO);
        glDeleteTextures(1, &skyAtlasTex);
        glDeleteTextures(1, &stylizedTex);
        glDeleteProgram(modelProgram);
        glDeleteProgram(shadowProgram);
        glDeleteProgram(skyProgram);
        glDeleteProgram(quadProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    auto lastCaptureTime = std::chrono::steady_clock::now();
    const float streamFps = (config.streamInputFps > 0.0f) ? config.streamInputFps : 10.0f;
    const std::chrono::duration<float> streamInterval(1.0f / streamFps);
    StreamSharedState sharedState{};
    uint32_t nextCaptureSeq = 1;
    std::thread streamSendThread(streamSendLoop, streamSocket, &sharedState);
    std::thread streamReceiveThread(streamReceiveLoop, streamSocket, &sharedState);
    bool hasStylizedFrame = false;
    int displayedOutputFps = 0;
    int displayedNewOutputFps = 0;
    uint64_t lastReadyCount = 0;
    uint64_t lastNewOutputCount = 0;
    auto outputFpsWindowStart = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
        int w = 1, h = 1;
        glfwGetFramebufferSize(window, &w, &h);
        glm::mat4 view = glm::lookAt(gCameraPos, gLookAt, gUp);
        glm::mat4 rotateModel = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        rotateModel = glm::rotate(rotateModel, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 modelMatrix = rotateModel *
            glm::scale(glm::mat4(1.0f), glm::vec3(gModelFitScale)) *
            glm::translate(glm::mat4(1.0f), -modelGpu.center);
        const glm::mat4 lightSpaceMatrix = computeLightSpaceMatrix(modelGpu, modelMatrix, gLightPosition);
        bool shadowRenderedThisFrame = false;

        auto nowCapture = std::chrono::steady_clock::now();
        if (nowCapture - lastCaptureTime >= streamInterval) {
            bool allowCapture = false;
            {
                std::lock_guard<std::mutex> lock(sharedState.mutex);
                allowCapture = sharedState.pendingInputs.size() < kStreamCaptureBackpressureThreshold;
            }
            if (allowCapture) {
                std::vector<unsigned char> captured;
                int captureSide = 0;
            renderShadowMap(modelGpu, shadowProgram, shadowFramebuffer, modelMatrix, lightSpaceMatrix);
            shadowRenderedThisFrame = true;
            glBindFramebuffer(GL_FRAMEBUFFER, captureFramebuffer.fbo);
            renderSceneToCurrentFramebuffer(
                modelGpu,
                modelProgram,
                skyProgram,
                skyVAO,
                skyAtlasTex,
                shadowFramebuffer.depthTex,
                modelMatrix,
                view,
                lightSpaceMatrix,
                gLightPosition,
                gLightIntensity,
                captureFramebuffer.side,
                captureFramebuffer.side
            );
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            if (captureFrameFromFramebuffer(captureFramebuffer, captured, captureSide)) {
                std::lock_guard<std::mutex> lock(sharedState.mutex);
                ++sharedState.capturedCount;
                if (sharedState.pendingInputs.size() < kMaxPendingStreamInputs) {
                    StreamPendingInput pending{};
                    pending.rgb = std::move(captured);
                    pending.side = captureSide;
                    pending.seq = nextCaptureSeq++;
                    sharedState.pendingInputs.push_back(std::move(pending));
                    ++sharedState.enqueuedCount;
                } else {
                    ++sharedState.droppedCount;
                }
            }
            }
            lastCaptureTime = nowCapture;
        }
        {
            std::lock_guard<std::mutex> lock(sharedState.mutex);
            if (sharedState.hasNewOutput) {
                updateStylizedTextureFromMemory(sharedState.latestOutput, sharedState.outputWidth, sharedState.outputHeight, stylizedTex);
                sharedState.hasNewOutput = false;
                hasStylizedFrame = true;
            }
        }
        const bool showStylizedFrame = gShowStylized && hasStylizedFrame;
        if (!showStylizedFrame) {
            if (!shadowRenderedThisFrame) {
                renderShadowMap(modelGpu, shadowProgram, shadowFramebuffer, modelMatrix, lightSpaceMatrix);
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            renderSceneToCurrentFramebuffer(
                modelGpu,
                modelProgram,
                skyProgram,
                skyVAO,
                skyAtlasTex,
                shadowFramebuffer.depthTex,
                modelMatrix,
                view,
                lightSpaceMatrix,
                gLightPosition,
                gLightIntensity,
                w,
                h
            );
        } else {
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            const int displaySide = std::min(w, h);
            const int viewportX = (w - displaySide) / 2;
            const int viewportY = (h - displaySide) / 2;
            glViewport(viewportX, viewportY, displaySide, displaySide);
            glUseProgram(quadProgram);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, stylizedTex);
            glBindVertexArray(quadVAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
            glBindVertexArray(0);
            glViewport(0, 0, w, h);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
        }

        const auto nowFps = std::chrono::steady_clock::now();
        const std::chrono::duration<float> fpsElapsed = nowFps - outputFpsWindowStart;
        if (fpsElapsed.count() >= 0.5f) {
            uint64_t readyCount = 0;
            uint64_t newOutputCount = 0;
            {
                std::lock_guard<std::mutex> lock(sharedState.mutex);
                readyCount = sharedState.readyCount;
                newOutputCount = sharedState.newOutputCount;
            }
            displayedOutputFps = static_cast<int>(std::lround(static_cast<float>(readyCount - lastReadyCount) / fpsElapsed.count()));
            displayedNewOutputFps = static_cast<int>(std::lround(static_cast<float>(newOutputCount - lastNewOutputCount) / fpsElapsed.count()));
            const std::string windowTitle =
                "OUT FPS: " + std::to_string(displayedOutputFps) +
                " | NEW FPS: " + std::to_string(displayedNewOutputFps);
            glfwSetWindowTitle(window, windowTitle.c_str());
            lastReadyCount = readyCount;
            lastNewOutputCount = newOutputCount;
            outputFpsWindowStart = nowFps;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    {
        std::lock_guard<std::mutex> lock(sharedState.mutex);
        sharedState.stop = true;
    }
    shutdown(streamSocket, SD_BOTH);
    if (streamSendThread.joinable()) streamSendThread.join();
    if (streamReceiveThread.joinable()) streamReceiveThread.join();
    closesocket(streamSocket);
    WSACleanup();

    cleanupGltfModel(modelGpu);
    cleanupCaptureFramebuffer(captureFramebuffer);
    cleanupShadowFramebuffer(shadowFramebuffer);
    glDeleteVertexArrays(1, &skyVAO);
    glDeleteBuffers(1, &skyPosVBO);
    glDeleteBuffers(1, &skyUvVBO);
    glDeleteBuffers(1, &skyEBO);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteBuffers(1, &quadEBO);
    glDeleteTextures(1, &skyAtlasTex);
    glDeleteTextures(1, &stylizedTex);
    glDeleteProgram(modelProgram);
    glDeleteProgram(shadowProgram);
    glDeleteProgram(skyProgram);
    glDeleteProgram(quadProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
