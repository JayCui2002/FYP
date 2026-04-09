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
#include <vector>

#include <winsock2.h>
#include <ws2tcpip.h>
#include <objidl.h>
#include <wincodec.h>

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
    std::string texturePath;
    float streamInputFps = 10.0f;
    bool streamV2Autostart = false;
    std::string streamV2Host = "127.0.0.1";
    std::string streamV2Python = "python";
    std::string streamV2Script = "streamdiffusionv2_bridge.py";
    std::string streamV2Args;
    int streamV2Port = 8765;
    float streamV2WaitTimeoutSec = 120.0f;
    int streamCaptureSide = 512;
    bool streamV2UseSuperres = false;
};

struct GltfPrimitiveGpu {
    GLuint vao = 0;
    GLuint posVbo = 0;
    GLuint normalVbo = 0;
    GLuint uvVbo = 0;
    GLuint ebo = 0;
    GLuint baseColorTex = 0;
    GLuint normalTex = 0;
    GLuint metallicRoughnessTex = 0;
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    float normalScale = 1.0f;
    GLenum mode = GL_TRIANGLES;
    GLsizei vertexCount = 0;
    GLsizei indexCount = 0;
    bool hasIndices = false;
    bool hasBaseColorTexture = false;
    bool hasNormalTexture = false;
    bool hasMetallicRoughnessTexture = false;
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

static int windowWidth = 1536;
static int windowHeight = 512;
static int originalPanelX = 0;
static int originalPanelY = 0;
static int originalPanelWidth = 0;
static int originalPanelHeight = 0;

struct StreamRequestHeader {
    uint32_t magic = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t payloadSize = 0;
    uint32_t seq = 0;
};

struct StreamResponseHeader {
    uint32_t magic = 0;
    uint32_t stylizedWidth = 0;
    uint32_t stylizedHeight = 0;
    uint32_t stylizedPayloadSize = 0;
    uint32_t superresWidth = 0;
    uint32_t superresHeight = 0;
    uint32_t superresPayloadSize = 0;
    uint32_t flags = 0;
    uint32_t seq = 0;
};

struct StreamPendingInput {
    std::vector<unsigned char> rgb;
    int side = 0;
    uint32_t seq = 0;
    std::chrono::steady_clock::time_point capturedAt{};
};

struct StreamSharedState {
    std::mutex mutex;
    std::deque<StreamPendingInput> pendingInputs;
    std::vector<unsigned char> latestStylizedOutput;
    int stylizedWidth = 0;
    int stylizedHeight = 0;
    std::vector<unsigned char> latestSuperresOutput;
    int superresWidth = 0;
    int superresHeight = 0;
    uint32_t latestOutputSeq = 0;
    uint64_t readyCount = 0;
    uint64_t newOutputCount = 0;
    bool hasNewOutput = false;
    bool stop = false;
};

static const uint32_t kStreamRequestMagic = 0x304D5246u;
static const uint32_t kStreamResponseMagic = 0x3054554Fu;
static const uint32_t kStreamResponseFlagStylizedReady = 1u;
static const uint32_t kStreamResponseFlagReuseLatest = 2u;
static constexpr size_t kMaxPendingStreamInputs = 4;
static constexpr size_t kStreamCaptureBackpressureThreshold = 2;
static constexpr auto kMaxPendingInputAge = std::chrono::seconds(2);
static constexpr float kStreamInputJpegQuality = 0.75f;
static constexpr float kDisplayBlendDurationSec = 0.12f;

static glm::vec3 lookat(0.0f, 0.0f, 0.0f);
static glm::vec3 up(0.0f, 1.0f, 0.0f);
static float viewAzimuth = 0.0f;
static float viewPolar = 0.0f;
static float viewDistance = 3.0f;
static glm::vec3 eye_center(0.0f, 0.0f, 3.0f);
static glm::vec3 lightPosition(-2.75f, 5.0f, 3.0f);
static glm::vec3 lightIntensity(8.0f, 8.0f, 8.0f);
static float lightCursorX = 0.0f;
static float lightCursorY = 0.0f;
static float defaultViewDistance = 4.0f;
static float modelFitScale = 1.0f;

static const char* kModelVertPath = "input/shader/model.vert";
static const char* kModelFragPath = "input/shader/model.frag";
static const char* kShadowVertPath = "input/shader/model_shadow.vert";
static const char* kShadowFragPath = "input/shader/model_shadow.frag";
static const char* kSkyVertPath = "input/shader/skybox.vert";
static const char* kSkyFragPath = "input/shader/skybox.frag";
static void drawGltfModel(const GltfModelGpu& model, GLuint modelProgram);

// Update light target
static void updateLightFromCursor() {
    glm::vec3 forward = glm::normalize(lookat - eye_center);
    glm::vec3 right = glm::cross(forward, up);
    if (glm::length(right) < 1e-4f) right = glm::vec3(1.0f, 0.0f, 0.0f);
    else right = glm::normalize(right);
    glm::vec3 up = glm::normalize(glm::cross(right, forward));
    const float scale = glm::max(1.5f, viewDistance * 1.5f);
    const glm::vec3 anchor = eye_center + forward * viewDistance;
    lightPosition = anchor + right * (lightCursorX * scale) + up * (lightCursorY * scale);
}

// Refresh camera pose
static void updateCameraFromSpherical() {
    const float maxPolar = glm::radians(85.0f);
    if (viewPolar > maxPolar) viewPolar = maxPolar;
    if (viewPolar < -maxPolar) viewPolar = -maxPolar;
    if (viewDistance < 0.8f) viewDistance = 0.8f;
    if (viewDistance > 30.0f) viewDistance = 30.0f;

    eye_center.x = viewDistance * std::cos(viewPolar) * std::cos(viewAzimuth);
    eye_center.y = viewDistance * std::sin(viewPolar);
    eye_center.z = viewDistance * std::cos(viewPolar) * std::sin(viewAzimuth);
    updateLightFromCursor();
}

// Drop stale frames
static void pruneStalePendingInputs(std::deque<StreamPendingInput>& pendingInputs, const std::chrono::steady_clock::time_point now) {
    while (!pendingInputs.empty() && now - pendingInputs.front().capturedAt > kMaxPendingInputAge) {
        pendingInputs.pop_front();
    }
}

// Handle mouse light
static void cursor_callback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;
    if (originalPanelWidth <= 0 || originalPanelHeight <= 0) return;
    if (xpos < static_cast<double>(originalPanelX) ||
        xpos > static_cast<double>(originalPanelX + originalPanelWidth) ||
        ypos < static_cast<double>(originalPanelY) ||
        ypos > static_cast<double>(originalPanelY + originalPanelHeight)) {
        return;
    }
    const float nx = static_cast<float>((xpos - static_cast<double>(originalPanelX)) / static_cast<double>(originalPanelWidth));
    const float ny = static_cast<float>((ypos - static_cast<double>(originalPanelY)) / static_cast<double>(originalPanelHeight));
    lightCursorX = nx * 2.0f - 1.0f;
    lightCursorY = 1.0f - ny * 2.0f;
    updateLightFromCursor();
}

// Handle keys
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode;
    (void)mods;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(window, GL_TRUE);
        return;
    }
    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        viewAzimuth = 0.0f;
        viewPolar = 0.0f;
        viewDistance = defaultViewDistance;
        updateCameraFromSpherical();
        return;
    }

    if (key == GLFW_KEY_LEFT) viewAzimuth -= 0.05f;
    if (key == GLFW_KEY_RIGHT) viewAzimuth += 0.05f;
    if (key == GLFW_KEY_UP) viewPolar += 0.05f;
    if (key == GLFW_KEY_DOWN) viewPolar -= 0.05f;
    if (key == GLFW_KEY_W) viewDistance -= 0.2f;
    if (key == GLFW_KEY_S) viewDistance += 0.2f;
    updateCameraFromSpherical();
}

// Track framebuffer size
static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    (void)window;
    windowWidth = width;
    windowHeight = height;
    glViewport(0, 0, width, height);
}

// Trim config text
static std::string trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) begin++;
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) end--;
    return s.substr(begin, end - begin);
}

// Split quoted args
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

// Check arg flag
static bool hasFlagArg(const std::string& args, const std::string& key) {
    const std::vector<std::string> tokens = splitArgsPreservingQuotes(args);
    for (const std::string& token : tokens) {
        if (token == key) return true;
    }
    return false;
}

// Read int arg
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

// Load scene config
static bool loadSceneConfig(const std::string& filePath, SceneConfig& cfg) {
    std::ifstream in(filePath);
    if (!in.is_open()) return false;

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key == "model") {
            cfg.modelPath = value;
        } else if (key == "skybox_atlas") {
            cfg.skyboxAtlasPath = value;
        } else if (key == "texture") {
            cfg.texturePath = value;
        } else if (key == "stream_input_fps") {
            cfg.streamInputFps = std::stof(value);
        } else if (key == "stream_v2_autostart") {
            cfg.streamV2Autostart = (value == "1" || value == "true" || value == "TRUE");
        } else if (key == "stream_v2_python") {
            cfg.streamV2Python = value;
        } else if (key == "stream_v2_script") {
            cfg.streamV2Script = value;
        } else if (key == "stream_v2_host") {
            cfg.streamV2Host = value;
        } else if (key == "stream_v2_args") {
            cfg.streamV2Args = value;
        } else if (key == "stream_v2_port") {
            cfg.streamV2Port = std::stoi(value);
        } else if (key == "stream_v2_wait_timeout_sec") {
            cfg.streamV2WaitTimeoutSec = std::stof(value);
        }
    }

    cfg.streamV2UseSuperres = hasFlagArg(cfg.streamV2Args, "--sr");
    const int streamWidth = extractIntArgValue(cfg.streamV2Args, "--width", 512);
    const int streamHeight = extractIntArgValue(cfg.streamV2Args, "--height", 512);
    cfg.streamCaptureSide = std::max(64, std::min(streamWidth, streamHeight));
    return !cfg.modelPath.empty() && !cfg.skyboxAtlasPath.empty();
}

// Quote shell arg
static std::string quoteArg(const std::string& s) {
    return "\"" + s + "\"";
}

// Launch bridge process
static void launchStreamV2Bridge(const SceneConfig& cfg) {
    if (!cfg.streamV2Autostart) return;
    if (cfg.streamV2Python.empty() || cfg.streamV2Script.empty()) return;
    std::stringstream cmd;
    cmd << "start \"\" /B "
        << quoteArg(cfg.streamV2Python) << " "
        << quoteArg(cfg.streamV2Script)
        << " --host " << quoteArg(cfg.streamV2Host)
        << " --port " << cfg.streamV2Port
        << " --fps " << cfg.streamInputFps;
    if (!cfg.streamV2Args.empty()) cmd << " " << cfg.streamV2Args;
    std::system(cmd.str().c_str());
}

// Send socket bytes
static bool sendAll(SOCKET sock, const unsigned char* data, int size) {
    int sent = 0;
    while (sent < size) {
        int rc = send(sock, reinterpret_cast<const char*>(data) + sent, size - sent, 0);
        if (rc == SOCKET_ERROR || rc == 0) return false;
        sent += rc;
    }
    return true;
}

// Receive socket bytes
static bool recvAll(SOCKET sock, unsigned char* data, int size) {
    int received = 0;
    while (received < size) {
        int rc = recv(sock, reinterpret_cast<char*>(data) + received, size - received, 0);
        if (rc == SOCKET_ERROR || rc == 0) return false;
        received += rc;
    }
    return true;
}

// Release COM pointer
template <typename T>
static void releaseCom(T*& ptr) {
    if (ptr != nullptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

// Encode input jpeg
static bool encodeJpegRgb(const unsigned char* rgb, int width, int height, std::vector<unsigned char>& outJpeg) {
    outJpeg.clear();
    if (rgb == nullptr || width <= 0 || height <= 0) return false;

    IStream* stream = nullptr;
    IWICImagingFactory* factory = nullptr;
    IWICBitmapEncoder* encoder = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IPropertyBag2* props = nullptr;
    bool ok = false;
    WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
    STATSTG stat{};
    LARGE_INTEGER origin{};
    ULONG readBytes = 0;

    std::vector<unsigned char> bgr(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u);
    for (size_t i = 0; i < bgr.size(); i += 3u) {
        bgr[i + 0] = rgb[i + 2];
        bgr[i + 1] = rgb[i + 1];
        bgr[i + 2] = rgb[i + 0];
    }

    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))) goto cleanup;
    if (FAILED(CreateStreamOnHGlobal(nullptr, TRUE, &stream))) goto cleanup;
    if (FAILED(factory->CreateEncoder(GUID_ContainerFormatJpeg, nullptr, &encoder))) goto cleanup;
    if (FAILED(encoder->Initialize(stream, WICBitmapEncoderNoCache))) goto cleanup;
    if (FAILED(encoder->CreateNewFrame(&frame, &props))) goto cleanup;
    if (props != nullptr) {
        PROPBAG2 option{};
        option.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
        VARIANT value{};
        VariantInit(&value);
        value.vt = VT_R4;
        value.fltVal = kStreamInputJpegQuality;
        props->Write(1, &option, &value);
        VariantClear(&value);
    }
    if (FAILED(frame->Initialize(props))) goto cleanup;
    if (FAILED(frame->SetSize(static_cast<UINT>(width), static_cast<UINT>(height)))) goto cleanup;
    if (FAILED(frame->SetPixelFormat(&format))) goto cleanup;
    if (FAILED(frame->WritePixels(static_cast<UINT>(height), static_cast<UINT>(width * 3), static_cast<UINT>(bgr.size()), bgr.data()))) goto cleanup;
    if (FAILED(frame->Commit())) goto cleanup;
    if (FAILED(encoder->Commit())) goto cleanup;

    if (FAILED(stream->Stat(&stat, STATFLAG_NONAME))) goto cleanup;
    if (stat.cbSize.QuadPart <= 0) goto cleanup;
    outJpeg.resize(static_cast<size_t>(stat.cbSize.QuadPart));
    if (FAILED(stream->Seek(origin, STREAM_SEEK_SET, nullptr))) goto cleanup;
    if (FAILED(stream->Read(outJpeg.data(), static_cast<ULONG>(outJpeg.size()), &readBytes))) goto cleanup;
    ok = (readBytes == outJpeg.size());

cleanup:
    releaseCom(props);
    releaseCom(frame);
    releaseCom(encoder);
    releaseCom(factory);
    releaseCom(stream);
    if (!ok) outJpeg.clear();
    return ok;
}

// Stop stream worker
static void stopStreamWorker(StreamSharedState* shared) {
    {
        std::lock_guard<std::mutex> lock(shared->mutex);
        shared->stop = true;
    }
}

// Create capture target
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

// Release capture target
static void cleanupCaptureFramebuffer(CaptureFramebuffer& capture) {
    if (capture.depthRbo != 0) glDeleteRenderbuffers(1, &capture.depthRbo);
    if (capture.colorTex != 0) glDeleteTextures(1, &capture.colorTex);
    if (capture.fbo != 0) glDeleteFramebuffers(1, &capture.fbo);
    capture = {};
}

// Create shadow target
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

// Release shadow target
static void cleanupShadowFramebuffer(ShadowFramebuffer& shadow) {
    if (shadow.depthTex != 0) glDeleteTextures(1, &shadow.depthTex);
    if (shadow.fbo != 0) glDeleteFramebuffers(1, &shadow.fbo);
    shadow = {};
}

// Build light matrix
static glm::mat4 computeLightSpaceMatrix(const GltfModelGpu& modelGpu, const glm::mat4& modelMatrix, const glm::vec3& lightPosition) {
    const glm::vec3 sceneCenter = glm::vec3(modelMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    glm::vec3 lightDir = sceneCenter - lightPosition;
    if (glm::length(lightDir) < 1e-4f) lightDir = glm::vec3(0.0f, 0.0f, -1.0f);
    else lightDir = glm::normalize(lightDir);
    glm::vec3 lightUp = (std::abs(glm::dot(lightDir, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.98f)
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::mat4 lightView = glm::lookAt(lightPosition, sceneCenter, lightUp);
    const float sceneRadius = glm::max(1.5f, modelGpu.radius * modelFitScale * 2.5f);
    const float lightDistance = glm::max(0.1f, glm::distance(lightPosition, sceneCenter));
    const float orthoSize = glm::max(sceneRadius * 2.0f, lightDistance * 0.75f);
    const float nearPlane = glm::max(0.1f, lightDistance - sceneRadius * 4.0f);
    const float farPlane = lightDistance + sceneRadius * 4.0f;
    const glm::mat4 lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, nearPlane, farPlane);
    return lightProjection * lightView;
}

// Render shadow pass
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

// Read capture frame
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

// Upload output texture
static void updateStylizedTextureFromMemory(const std::vector<unsigned char>& rgb, int width, int height, GLuint texture) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
}

// Blend rgb frames
static std::vector<unsigned char> blendRgbFrames(const std::vector<unsigned char>& from, const std::vector<unsigned char>& to, float alpha) {
    if (from.size() != to.size()) return to;
    const float t = std::clamp(alpha, 0.0f, 1.0f);
    std::vector<unsigned char> blended(to.size());
    for (size_t i = 0; i < to.size(); ++i) {
        const float v = static_cast<float>(from[i]) * (1.0f - t) + static_cast<float>(to[i]) * t;
        blended[i] = static_cast<unsigned char>(std::clamp(std::lround(v), 0l, 255l));
    }
    return blended;
}

// Draw panel quad
static void drawTexturePanel(GLuint quadProgram, GLuint quadVAO, GLuint texture, int x, int y, int width, int height) {
    if (texture == 0 || width <= 0 || height <= 0) return;
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glViewport(x, y, width, height);
    glUseProgram(quadProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindVertexArray(quadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

// Render scene pass
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
    int viewportX,
    int viewportY,
    int viewportWidth,
    int viewportHeight,
    bool clearBuffers
) {
    glViewport(viewportX, viewportY, viewportWidth, viewportHeight);
    if (clearBuffers) {
        glClearColor(0.08f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    const float aspect = (viewportHeight != 0) ? static_cast<float>(viewportWidth) / static_cast<float>(viewportHeight) : 1.0f;
    const glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
    const glm::mat4 rotateSky = glm::mat4(1.0f);

    glUseProgram(modelProgram);
    glUniform1i(glGetUniformLocation(modelProgram, "uBaseColorTex"), 0);
    glUniform1i(glGetUniformLocation(modelProgram, "shadowMap"), 1);
    glUniform1i(glGetUniformLocation(modelProgram, "uNormalTex"), 2);
    glUniform1i(glGetUniformLocation(modelProgram, "uMetallicRoughnessTex"), 3);
    const glm::mat4 mvp = proj * view * modelMatrix;
    glUniformMatrix4fv(glGetUniformLocation(modelProgram, "MVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniformMatrix4fv(glGetUniformLocation(modelProgram, "modelMatrix"), 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(glGetUniformLocation(modelProgram, "lightSpaceMatrix"), 1, GL_FALSE, glm::value_ptr(lightSpaceMatrix));
    glUniform3fv(glGetUniformLocation(modelProgram, "lightPosition"), 1, glm::value_ptr(lightPosition));
    glUniform3fv(glGetUniformLocation(modelProgram, "lightIntensity"), 1, glm::value_ptr(lightIntensity));
    glUniform3fv(glGetUniformLocation(modelProgram, "cameraPosition"), 1, glm::value_ptr(eye_center));
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

// Connect stream socket
static SOCKET connectToStreamV2(const SceneConfig& cfg) {
    WSADATA wsaData{};
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return INVALID_SOCKET;
    const auto timeout = std::chrono::duration<float>((cfg.streamV2WaitTimeoutSec > 0.0f) ? cfg.streamV2WaitTimeoutSec : 120.0f);
    const auto begin = std::chrono::steady_clock::now();
    const std::string host = cfg.streamV2Host.empty() ? "127.0.0.1" : cfg.streamV2Host;
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo* resolved = nullptr;
    const std::string port = std::to_string(cfg.streamV2Port);
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &resolved) != 0 || resolved == nullptr) {
        WSACleanup();
        return INVALID_SOCKET;
    }
    while (std::chrono::steady_clock::now() - begin < timeout) {
        SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == INVALID_SOCKET) break;
        if (connect(sock, resolved->ai_addr, static_cast<int>(resolved->ai_addrlen)) == 0) {
            DWORD timeoutMs = 300000;
            setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeoutMs), sizeof(timeoutMs));
            setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeoutMs), sizeof(timeoutMs));
            freeaddrinfo(resolved);
            return sock;
        }
        closesocket(sock);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    freeaddrinfo(resolved);
    WSACleanup();
    return INVALID_SOCKET;
}

// Send stream frames
static void streamSendLoop(SOCKET sock, StreamSharedState* shared) {
    const HRESULT comResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool comInitialized = SUCCEEDED(comResult) || comResult == S_FALSE;
    while (true) {
        std::vector<unsigned char> input;
        std::vector<unsigned char> payload;
        int side = 0;
        uint32_t seq = 0;
        {
            std::lock_guard<std::mutex> lock(shared->mutex);
            if (shared->stop) break;
            const auto now = std::chrono::steady_clock::now();
            pruneStalePendingInputs(shared->pendingInputs, now);
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

        if (!comInitialized || !encodeJpegRgb(input.data(), side, side, payload)) {
            payload = std::move(input);
        }

        StreamRequestHeader request{};
        request.magic = kStreamRequestMagic;
        request.width = static_cast<uint32_t>(side);
        request.height = static_cast<uint32_t>(side);
        request.payloadSize = static_cast<uint32_t>(payload.size());
        request.seq = seq;
        if (!sendAll(sock, reinterpret_cast<const unsigned char*>(&request), sizeof(request))) {
            stopStreamWorker(shared);
            if (comInitialized) CoUninitialize();
            return;
        }
        if (!sendAll(sock, payload.data(), static_cast<int>(payload.size()))) {
            stopStreamWorker(shared);
            if (comInitialized) CoUninitialize();
            return;
        }
    }
    if (comInitialized) CoUninitialize();
}

// Receive stream frames
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
        std::vector<unsigned char> stylizedOutput(response.stylizedPayloadSize);
        if (!recvAll(sock, stylizedOutput.data(), static_cast<int>(stylizedOutput.size()))) {
            stopStreamWorker(shared);
            return;
        }
        std::vector<unsigned char> superresOutput(response.superresPayloadSize);
        if (!recvAll(sock, superresOutput.data(), static_cast<int>(superresOutput.size()))) {
            stopStreamWorker(shared);
            return;
        }
        if ((response.flags & kStreamResponseFlagStylizedReady) == 0u) continue;
        {
            std::lock_guard<std::mutex> lock(shared->mutex);
            ++shared->readyCount;
            const bool hasPayload = response.stylizedPayloadSize > 0u;
            if (hasPayload) {
                if (response.seq < shared->latestOutputSeq) continue;
                shared->latestStylizedOutput = std::move(stylizedOutput);
                shared->stylizedWidth = static_cast<int>(response.stylizedWidth);
                shared->stylizedHeight = static_cast<int>(response.stylizedHeight);
                shared->latestSuperresOutput = std::move(superresOutput);
                shared->superresWidth = static_cast<int>(response.superresWidth);
                shared->superresHeight = static_cast<int>(response.superresHeight);
                shared->latestOutputSeq = response.seq;
                ++shared->newOutputCount;
                shared->hasNewOutput = true;
            }
        }
    }
}

// Copy vec3 accessor
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

// Copy vec2 accessor
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

// Copy index data
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

// Upload texture data
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
    if (components == 1) {
        GLint swizzleMask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
    }
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

// Load texture file
static GLuint loadTexture2DFromFile(const std::string& path, GLint wrapS, GLint wrapT) {
    int width = 0;
    int height = 0;
    int components = 0;
    stbi_set_flip_vertically_on_load(false);
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &components, 0);
    if (!pixels) return 0;

    const GLuint texId = uploadTexture2D(pixels, width, height, components);
    stbi_image_free(pixels);
    if (texId == 0) return 0;

    glBindTexture(GL_TEXTURE_2D, texId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapT);
    return texId;
}

// Load gltf image
static GLuint loadGltfTextureImage(
    const tinygltf::Model& model,
    int textureIndex,
    const std::filesystem::path& modelDir,
    const std::filesystem::path& textureRootDir
) {
    if (textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size())) return 0;
    const tinygltf::Texture& texture = model.textures[static_cast<size_t>(textureIndex)];
    const int imageIndex = texture.source;
    if (imageIndex < 0 || imageIndex >= static_cast<int>(model.images.size())) return 0;
    const tinygltf::Image& image = model.images[static_cast<size_t>(imageIndex)];
    if (!image.image.empty() && image.width > 0 && image.height > 0) {
        return uploadTexture2D(image.image.data(), image.width, image.height, image.component);
    }

    if (!image.uri.empty()) {
        const std::filesystem::path imagePath = std::filesystem::path(image.uri);
        std::vector<std::filesystem::path> candidates;
        candidates.push_back(modelDir / imagePath);
        if (!textureRootDir.empty()) {
            candidates.push_back(textureRootDir / imagePath);
            candidates.push_back(textureRootDir / imagePath.filename());
        }

        for (const auto& texturePath : candidates) {
            int width = 0;
            int height = 0;
            int components = 0;
            stbi_set_flip_vertically_on_load(false);
            unsigned char* pixels = stbi_load(texturePath.string().c_str(), &width, &height, &components, 0);
            if (!pixels) continue;

            const GLuint texId = uploadTexture2D(pixels, width, height, components);
            stbi_image_free(pixels);
            if (texId != 0) return texId;
        }
    }

    return 0;
}

// Load base color
static GLuint loadGltfBaseColorTexture(
    const tinygltf::Model& model,
    int materialIndex,
    const std::filesystem::path& modelDir,
    const std::filesystem::path& textureRootDir
) {
    if (materialIndex < 0 || materialIndex >= static_cast<int>(model.materials.size())) return 0;
    const tinygltf::Material& material = model.materials[static_cast<size_t>(materialIndex)];
    const int textureIndex = material.pbrMetallicRoughness.baseColorTexture.index;
    return loadGltfTextureImage(model, textureIndex, modelDir, textureRootDir);
}

// Load normal map
static GLuint loadGltfNormalTexture(
    const tinygltf::Model& model,
    int materialIndex,
    const std::filesystem::path& modelDir,
    const std::filesystem::path& textureRootDir
) {
    if (materialIndex < 0 || materialIndex >= static_cast<int>(model.materials.size())) return 0;
    const tinygltf::Material& material = model.materials[static_cast<size_t>(materialIndex)];
    return loadGltfTextureImage(model, material.normalTexture.index, modelDir, textureRootDir);
}

// Load roughness map
static GLuint loadGltfMetallicRoughnessTexture(
    const tinygltf::Model& model,
    int materialIndex,
    const std::filesystem::path& modelDir,
    const std::filesystem::path& textureRootDir
) {
    if (materialIndex < 0 || materialIndex >= static_cast<int>(model.materials.size())) return 0;
    const tinygltf::Material& material = model.materials[static_cast<size_t>(materialIndex)];
    return loadGltfTextureImage(model, material.pbrMetallicRoughness.metallicRoughnessTexture.index, modelDir, textureRootDir);
}

// Load gltf model
static bool loadGltfModel(const std::string& path, const std::string& textureRootPath, GltfModelGpu& outModel) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;

    const bool isGlb = path.size() >= 4 &&
        (path.substr(path.size() - 4) == ".glb" || path.substr(path.size() - 4) == ".GLB");
    bool ok = isGlb ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                    : loader.LoadASCIIFromFile(&model, &err, &warn, path);
    (void)warn;
    if (!ok) {
        return false;
    }

    const std::filesystem::path modelDir = std::filesystem::path(path).parent_path();
    const std::filesystem::path textureRootDir = textureRootPath.empty()
        ? std::filesystem::path()
        : std::filesystem::path(textureRootPath);
    outModel.primitives.clear();
    bool hasBounds = false;
    glm::vec3 minP(0.0f), maxP(0.0f);
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
                gpu.metallicFactor = static_cast<float>(material.pbrMetallicRoughness.metallicFactor);
                gpu.roughnessFactor = static_cast<float>(material.pbrMetallicRoughness.roughnessFactor);
                gpu.normalScale = static_cast<float>(material.normalTexture.scale);
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
                gpu.baseColorTex = loadGltfBaseColorTexture(model, primitive.material, modelDir, textureRootDir);
                gpu.hasBaseColorTexture = (gpu.baseColorTex != 0);
                gpu.normalTex = loadGltfNormalTexture(model, primitive.material, modelDir, textureRootDir);
                gpu.hasNormalTexture = (gpu.normalTex != 0);
                gpu.metallicRoughnessTex = loadGltfMetallicRoughnessTexture(model, primitive.material, modelDir, textureRootDir);
                gpu.hasMetallicRoughnessTexture = (gpu.metallicRoughnessTex != 0);
            }

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
    if (outModel.primitives.empty()) {
        return false;
    }
    return true;
}

// Draw gltf model
static void drawGltfModel(const GltfModelGpu& model, GLuint modelProgram) {
    for (const auto& primitive : model.primitives) {
        glUniform4fv(glGetUniformLocation(modelProgram, "uBaseColorFactor"), 1, glm::value_ptr(primitive.baseColorFactor));
        glUniform1i(glGetUniformLocation(modelProgram, "uUseBaseColorTex"), primitive.hasBaseColorTexture ? 1 : 0);
        glUniform1i(glGetUniformLocation(modelProgram, "uUseNormalTex"), primitive.hasNormalTexture ? 1 : 0);
        glUniform1i(glGetUniformLocation(modelProgram, "uUseMetallicRoughnessTex"), primitive.hasMetallicRoughnessTexture ? 1 : 0);
        glUniform1f(glGetUniformLocation(modelProgram, "uMetallicFactor"), primitive.metallicFactor);
        glUniform1f(glGetUniformLocation(modelProgram, "uRoughnessFactor"), primitive.roughnessFactor);
        glUniform1f(glGetUniformLocation(modelProgram, "uNormalScale"), primitive.normalScale);
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
        if (primitive.hasNormalTexture && primitive.normalTex != 0) {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, primitive.normalTex);
        } else {
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        if (primitive.hasMetallicRoughnessTexture && primitive.metallicRoughnessTex != 0) {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, primitive.metallicRoughnessTex);
        } else {
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glBindVertexArray(primitive.vao);
        if (primitive.hasIndices) glDrawElements(primitive.mode, primitive.indexCount, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
        else glDrawArrays(primitive.mode, 0, primitive.vertexCount);
    }
    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
}

// Release gltf model
static void cleanupGltfModel(GltfModelGpu& model) {
    for (auto& primitive : model.primitives) {
        if (primitive.ebo != 0) glDeleteBuffers(1, &primitive.ebo);
        if (primitive.uvVbo != 0) glDeleteBuffers(1, &primitive.uvVbo);
        if (primitive.posVbo != 0) glDeleteBuffers(1, &primitive.posVbo);
        if (primitive.normalVbo != 0) glDeleteBuffers(1, &primitive.normalVbo);
        if (primitive.baseColorTex != 0) glDeleteTextures(1, &primitive.baseColorTex);
        if (primitive.normalTex != 0) glDeleteTextures(1, &primitive.normalTex);
        if (primitive.metallicRoughnessTex != 0) glDeleteTextures(1, &primitive.metallicRoughnessTex);
        if (primitive.vao != 0) glDeleteVertexArrays(1, &primitive.vao);
    }
    model.primitives.clear();
}

// Load sky atlas
static GLuint loadSkyAtlasTexture2D(const std::string& path) {
    return loadTexture2DFromFile(path, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
}

// Run application
int main() {
    int exitCode = 0;
    bool glfwReady = false;
    bool glReady = false;
    SceneConfig config{};
    SOCKET streamSocket = INVALID_SOCKET;
    GLFWwindow* window = nullptr;
    GLuint modelProgram = 0;
    GLuint shadowProgram = 0;
    GLuint skyProgram = 0;
    GLuint quadProgram = 0;
    GLuint skyVAO = 0;
    GLuint skyPosVBO = 0;
    GLuint skyUvVBO = 0;
    GLuint skyEBO = 0;
    GLuint skyAtlasTex = 0;
    GLuint quadVAO = 0;
    GLuint quadVBO = 0;
    GLuint quadEBO = 0;
    GLuint stylizedTex = 0;
    GLuint superresTex = 0;
    GltfModelGpu modelGpu;
    CaptureFramebuffer captureFramebuffer{};
    ShadowFramebuffer shadowFramebuffer{};
    StreamSharedState sharedState{};
    std::thread streamSendThread;
    std::thread streamReceiveThread;

    do {
        if (!loadSceneConfig("input/scene.cfg", config)) {
            exitCode = 2;
            break;
        }
        launchStreamV2Bridge(config);
        streamSocket = connectToStreamV2(config);
        if (streamSocket == INVALID_SOCKET) {
            exitCode = 2;
            break;
        }
        if (!glfwInit()) {
            exitCode = 2;
            break;
        }
        glfwReady = true;

        int panelCount = config.streamV2UseSuperres ? 3 : 2;
        windowHeight = 512;
        windowWidth = windowHeight * panelCount;

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window = glfwCreateWindow(windowWidth, windowHeight, "OUT FPS: 0 | NEW FPS: 0", nullptr, nullptr);
        if (!window) {
            exitCode = 2;
            break;
        }
        glfwSetWindowAspectRatio(window, panelCount, 1);
        glfwMakeContextCurrent(window);
        glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
        glfwSetKeyCallback(window, key_callback);
        glfwSetCursorPosCallback(window, cursor_callback);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            exitCode = 2;
            break;
        }
        glReady = true;

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        modelProgram = LoadShadersFromFile(kModelVertPath, kModelFragPath);
        shadowProgram = LoadShadersFromFile(kShadowVertPath, kShadowFragPath);
        skyProgram = LoadShadersFromFile(kSkyVertPath, kSkyFragPath);
        const char* quadVs = "#version 330 core\nlayout(location=0) in vec2 aPos;\nlayout(location=1) in vec2 aUv;\nout vec2 vUv;\nvoid main(){vUv=aUv;gl_Position=vec4(aPos,0.0,1.0);}";
        const char* quadFs = "#version 330 core\nin vec2 vUv;\nout vec4 FragColor;\nuniform sampler2D uTex;\nvoid main(){FragColor=texture(uTex,vec2(vUv.x,1.0-vUv.y));}";
        quadProgram = CreateProgramFromSource(quadVs, quadFs);
        if (modelProgram == 0 || shadowProgram == 0 || skyProgram == 0 || quadProgram == 0) {
            exitCode = 2;
            break;
        }

        if (!loadGltfModel(config.modelPath, config.texturePath, modelGpu)) {
            exitCode = 2;
            break;
        }

        modelFitScale = 1.2f / glm::max(0.001f, modelGpu.radius);
        lookat = glm::vec3(0.0f);
        defaultViewDistance = 4.0f;
        viewDistance = defaultViewDistance;
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

        skyAtlasTex = loadSkyAtlasTexture2D(config.skyboxAtlasPath);
        if (skyAtlasTex == 0) {
            exitCode = 2;
            break;
        }

    float quadVertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };
    unsigned int quadIndices[] = {0, 1, 2, 0, 2, 3};
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
    glGenTextures(1, &superresTex);
    glUseProgram(quadProgram);
    glUniform1i(glGetUniformLocation(quadProgram, "uTex"), 0);

    glUseProgram(skyProgram);
    glUniform1i(glGetUniformLocation(skyProgram, "uSkyAtlas"), 0);

        if (!initCaptureFramebuffer(captureFramebuffer, config.streamCaptureSide)) {
            exitCode = 2;
            break;
        }
        if (!initShadowFramebuffer(shadowFramebuffer, 1024, 1024)) {
            exitCode = 2;
            break;
        }
    auto lastCaptureTime = std::chrono::steady_clock::now();
    const float streamFps = (config.streamInputFps > 0.0f) ? config.streamInputFps : 10.0f;
    const std::chrono::duration<float> streamInterval(1.0f / streamFps);
    uint32_t nextCaptureSeq = 1;
    streamSendThread = std::thread(streamSendLoop, streamSocket, &sharedState);
    streamReceiveThread = std::thread(streamReceiveLoop, streamSocket, &sharedState);
    bool hasStylizedFrame = false;
    bool hasSuperresFrame = false;
    std::vector<unsigned char> displayedStylizedFrame;
    int displayedStylizedWidth = 0;
    int displayedStylizedHeight = 0;
    std::vector<unsigned char> displayedSuperresFrame;
    int displayedSuperresWidth = 0;
    int displayedSuperresHeight = 0;
    std::vector<unsigned char> blendFromStylizedFrame;
    std::vector<unsigned char> blendToStylizedFrame;
    int blendStylizedWidth = 0;
    int blendStylizedHeight = 0;
    std::vector<unsigned char> blendFromSuperresFrame;
    std::vector<unsigned char> blendToSuperresFrame;
    int blendSuperresWidth = 0;
    int blendSuperresHeight = 0;
    bool stylizedBlendActive = false;
    bool superresBlendActive = false;
    auto stylizedBlendStart = std::chrono::steady_clock::now();
    auto superresBlendStart = std::chrono::steady_clock::now();
    int activePanelCount = panelCount;
    int displayedOutputFps = 0;
    int displayedNewOutputFps = 0;
    uint64_t lastReadyCount = 0;
    uint64_t lastNewOutputCount = 0;
    auto outputFpsWindowStart = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
        int w = 1, h = 1;
        glfwGetFramebufferSize(window, &w, &h);
        glm::mat4 view = glm::lookAt(eye_center, lookat, up);
        glm::mat4 rotateModel = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        rotateModel = glm::rotate(rotateModel, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 modelMatrix = rotateModel *
            glm::scale(glm::mat4(1.0f), glm::vec3(modelFitScale)) *
            glm::translate(glm::mat4(1.0f), -modelGpu.center);
        const glm::mat4 lightSpaceMatrix = computeLightSpaceMatrix(modelGpu, modelMatrix, lightPosition);
        bool shadowRenderedThisFrame = false;

        auto nowCapture = std::chrono::steady_clock::now();
        if (nowCapture - lastCaptureTime >= streamInterval) {
            bool allowCapture = false;
            {
                std::lock_guard<std::mutex> lock(sharedState.mutex);
                pruneStalePendingInputs(sharedState.pendingInputs, nowCapture);
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
                lightPosition,
                lightIntensity,
                0,
                0,
                captureFramebuffer.side,
                captureFramebuffer.side,
                true
            );
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            if (captureFrameFromFramebuffer(captureFramebuffer, captured, captureSide)) {
                std::lock_guard<std::mutex> lock(sharedState.mutex);
                const auto capturedAt = std::chrono::steady_clock::now();
                pruneStalePendingInputs(sharedState.pendingInputs, capturedAt);
                StreamPendingInput pending{};
                pending.rgb = std::move(captured);
                pending.side = captureSide;
                pending.seq = nextCaptureSeq++;
                pending.capturedAt = capturedAt;
                if (sharedState.pendingInputs.size() >= kMaxPendingStreamInputs) {
                    sharedState.pendingInputs.pop_front();
                }
                sharedState.pendingInputs.push_back(std::move(pending));
            }
            }
            lastCaptureTime = nowCapture;
        }
        {
            std::vector<unsigned char> newStylizedFrame;
            int newStylizedWidth = 0;
            int newStylizedHeight = 0;
            std::vector<unsigned char> newSuperresFrame;
            int newSuperresWidth = 0;
            int newSuperresHeight = 0;
            bool gotNewOutput = false;
            {
                std::lock_guard<std::mutex> lock(sharedState.mutex);
                if (sharedState.hasNewOutput) {
                    newStylizedFrame = sharedState.latestStylizedOutput;
                    newStylizedWidth = sharedState.stylizedWidth;
                    newStylizedHeight = sharedState.stylizedHeight;
                    newSuperresFrame = sharedState.latestSuperresOutput;
                    newSuperresWidth = sharedState.superresWidth;
                    newSuperresHeight = sharedState.superresHeight;
                    sharedState.hasNewOutput = false;
                    gotNewOutput = true;
                }
            }
            if (gotNewOutput) {
                if (!displayedStylizedFrame.empty() &&
                    displayedStylizedWidth == newStylizedWidth &&
                    displayedStylizedHeight == newStylizedHeight) {
                    blendFromStylizedFrame = displayedStylizedFrame;
                    blendToStylizedFrame = std::move(newStylizedFrame);
                    blendStylizedWidth = newStylizedWidth;
                    blendStylizedHeight = newStylizedHeight;
                    stylizedBlendStart = std::chrono::steady_clock::now();
                    stylizedBlendActive = true;
                } else {
                    displayedStylizedFrame = std::move(newStylizedFrame);
                    displayedStylizedWidth = newStylizedWidth;
                    displayedStylizedHeight = newStylizedHeight;
                    updateStylizedTextureFromMemory(displayedStylizedFrame, displayedStylizedWidth, displayedStylizedHeight, stylizedTex);
                    stylizedBlendActive = false;
                }
                hasStylizedFrame = true;
                if (!newSuperresFrame.empty()) {
                    if (!displayedSuperresFrame.empty() &&
                        displayedSuperresWidth == newSuperresWidth &&
                        displayedSuperresHeight == newSuperresHeight) {
                        blendFromSuperresFrame = displayedSuperresFrame;
                        blendToSuperresFrame = std::move(newSuperresFrame);
                        blendSuperresWidth = newSuperresWidth;
                        blendSuperresHeight = newSuperresHeight;
                        superresBlendStart = std::chrono::steady_clock::now();
                        superresBlendActive = true;
                    } else {
                        displayedSuperresFrame = std::move(newSuperresFrame);
                        displayedSuperresWidth = newSuperresWidth;
                        displayedSuperresHeight = newSuperresHeight;
                        updateStylizedTextureFromMemory(displayedSuperresFrame, displayedSuperresWidth, displayedSuperresHeight, superresTex);
                        superresBlendActive = false;
                    }
                    hasSuperresFrame = true;
                }
            }
            if (stylizedBlendActive) {
                const float alpha = std::chrono::duration<float>(std::chrono::steady_clock::now() - stylizedBlendStart).count() / kDisplayBlendDurationSec;
                if (alpha >= 1.0f) {
                    displayedStylizedFrame = std::move(blendToStylizedFrame);
                    displayedStylizedWidth = blendStylizedWidth;
                    displayedStylizedHeight = blendStylizedHeight;
                    updateStylizedTextureFromMemory(displayedStylizedFrame, displayedStylizedWidth, displayedStylizedHeight, stylizedTex);
                    stylizedBlendActive = false;
                } else {
                    std::vector<unsigned char> blended = blendRgbFrames(blendFromStylizedFrame, blendToStylizedFrame, alpha);
                    updateStylizedTextureFromMemory(blended, blendStylizedWidth, blendStylizedHeight, stylizedTex);
                }
            }
            if (superresBlendActive) {
                const float alpha = std::chrono::duration<float>(std::chrono::steady_clock::now() - superresBlendStart).count() / kDisplayBlendDurationSec;
                if (alpha >= 1.0f) {
                    displayedSuperresFrame = std::move(blendToSuperresFrame);
                    displayedSuperresWidth = blendSuperresWidth;
                    displayedSuperresHeight = blendSuperresHeight;
                    updateStylizedTextureFromMemory(displayedSuperresFrame, displayedSuperresWidth, displayedSuperresHeight, superresTex);
                    superresBlendActive = false;
                } else {
                    std::vector<unsigned char> blended = blendRgbFrames(blendFromSuperresFrame, blendToSuperresFrame, alpha);
                    updateStylizedTextureFromMemory(blended, blendSuperresWidth, blendSuperresHeight, superresTex);
                }
            }
        }
        const int desiredPanelCount = (config.streamV2UseSuperres || hasSuperresFrame) ? 3 : 2;
        if (desiredPanelCount != activePanelCount) {
            activePanelCount = desiredPanelCount;
            glfwSetWindowAspectRatio(window, activePanelCount, 1);
            glfwSetWindowSize(window, windowHeight * activePanelCount, windowHeight);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, w, h);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!shadowRenderedThisFrame) {
            renderShadowMap(modelGpu, shadowProgram, shadowFramebuffer, modelMatrix, lightSpaceMatrix);
        }

        const int panelSide = std::max(1, std::min(h, w / activePanelCount));
        const int totalPanelsWidth = panelSide * activePanelCount;
        const int panelOffsetX = (w - totalPanelsWidth) / 2;
        const int panelY = (h - panelSide) / 2;
        const auto panelX = [&](int index) { return panelOffsetX + index * panelSide; };

        const int originalX = panelX(0);
        originalPanelX = originalX;
        originalPanelY = panelY;
        originalPanelWidth = panelSide;
        originalPanelHeight = panelSide;
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
            lightPosition,
            lightIntensity,
            originalX,
            panelY,
            panelSide,
            panelSide,
            false
        );

        const int stylizedX = panelX(1);
        if (hasStylizedFrame) {
            drawTexturePanel(quadProgram, quadVAO, stylizedTex, stylizedX, panelY, panelSide, panelSide);
        }

        if (activePanelCount == 3) {
            const int superresX = panelX(2);
            if (hasSuperresFrame) {
                drawTexturePanel(quadProgram, quadVAO, superresTex, superresX, panelY, panelSide, panelSide);
            }
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
    } while (false);

    if (streamSendThread.joinable() || streamReceiveThread.joinable()) {
        std::lock_guard<std::mutex> lock(sharedState.mutex);
        sharedState.stop = true;
    }
    if (streamSocket != INVALID_SOCKET) shutdown(streamSocket, SD_BOTH);
    if (streamSendThread.joinable()) streamSendThread.join();
    if (streamReceiveThread.joinable()) streamReceiveThread.join();
    if (streamSocket != INVALID_SOCKET) {
        closesocket(streamSocket);
        WSACleanup();
    }

    if (glReady) {
        cleanupGltfModel(modelGpu);
        cleanupCaptureFramebuffer(captureFramebuffer);
        cleanupShadowFramebuffer(shadowFramebuffer);
        if (skyVAO != 0) glDeleteVertexArrays(1, &skyVAO);
        if (skyPosVBO != 0) glDeleteBuffers(1, &skyPosVBO);
        if (skyUvVBO != 0) glDeleteBuffers(1, &skyUvVBO);
        if (skyEBO != 0) glDeleteBuffers(1, &skyEBO);
        if (quadVAO != 0) glDeleteVertexArrays(1, &quadVAO);
        if (quadVBO != 0) glDeleteBuffers(1, &quadVBO);
        if (quadEBO != 0) glDeleteBuffers(1, &quadEBO);
        if (skyAtlasTex != 0) glDeleteTextures(1, &skyAtlasTex);
        if (stylizedTex != 0) glDeleteTextures(1, &stylizedTex);
        if (superresTex != 0) glDeleteTextures(1, &superresTex);
        if (modelProgram != 0) glDeleteProgram(modelProgram);
        if (shadowProgram != 0) glDeleteProgram(shadowProgram);
        if (skyProgram != 0) glDeleteProgram(skyProgram);
        if (quadProgram != 0) glDeleteProgram(quadProgram);
    }
    if (window != nullptr) glfwDestroyWindow(window);
    if (glfwReady) glfwTerminate();
    return exitCode;
}
