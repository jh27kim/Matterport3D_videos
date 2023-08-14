// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "WindowlessContext.h"

#ifdef __APPLE__
#include <Magnum/Platform/WindowlessCglApplication.h>
#elif __linux__
#include <Magnum/Platform/GLContext.h>

#ifdef __ESP_USE_EGL__
#include <glad/glad_egl.h>
#else
#include <Magnum/Platform/WindowlessGlxApplication.h>
#endif

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#elif __win32__
#include <Magnum/Platform/WindowlessWglApplication.h>
#endif

#include <Magnum/Platform/GLContext.h>

using namespace Magnum;

// Based on code from:
// https://devblogs.nvidia.com/parallelforall/egl-eye-opengl-visualization-without-x-server/
// https://github.com/facebookresearch/House3D/blob/master/renderer/gl/glContext.cc

namespace esp {
namespace gfx {

#ifdef __linux__

namespace {

struct ESPContext {
  virtual void makeCurrent() = 0;
  virtual bool isValid() = 0;

  virtual ~ESPContext(){};

  ESP_SMART_POINTERS(ESPContext);
};

#ifdef __ESP_USE_EGL__
const int MAX_DEVICES = 128;

#define CHECK_EGL_ERROR()                             \
  do {                                                \
    EGLint err = eglGetError();                       \
    CHECK(err == EGL_SUCCESS) << "EGL error:" << err; \
  } while (0)

bool isNvidiaGpuReadable(int device) {
  const std::string dev = "/dev/nvidia" + std::to_string(device);
  const int retval = open(dev.c_str(), O_RDONLY);
  if (retval == -1) {
    return false;
  }
  close(retval);
  return true;
}

struct ESPEGLContext : ESPContext {
  ESPEGLContext(int device) : magnumGlContext_{NoCreate} {
    CHECK(gladLoadEGL()) << "Failed to load EGL";

    static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                           EGL_PBUFFER_BIT,
                                           EGL_BLUE_SIZE,
                                           8,
                                           EGL_GREEN_SIZE,
                                           8,
                                           EGL_RED_SIZE,
                                           8,
                                           EGL_DEPTH_SIZE,
                                           24,
                                           EGL_RENDERABLE_TYPE,
                                           EGL_OPENGL_BIT,
                                           EGL_NONE};

    // 1. Initialize EGL
    {
      EGLDeviceEXT eglDevices[MAX_DEVICES];
      EGLint numDevices;
      eglQueryDevicesEXT(MAX_DEVICES, eglDevices, &numDevices);
      CHECK_EGL_ERROR();

      CHECK(numDevices > 0) << "[EGL] No devices detected";
      LOG(INFO) << "[EGL] Detected " << numDevices << " EGL devices";

      int eglDevId;
      for (eglDevId = 0; eglDevId < numDevices; ++eglDevId) {
        EGLAttrib cudaDevNumber;

        if (eglQueryDeviceAttribEXT(eglDevices[eglDevId], EGL_CUDA_DEVICE_NV,
                                    &cudaDevNumber) == EGL_FALSE)
          continue;

        if (cudaDevNumber == device)
          break;
      }

      CHECK(eglDevId < numDevices)
          << "[EGL] Could not find an EGL device for CUDA device " << device;

      CHECK(isNvidiaGpuReadable(eglDevId))
          << "[EGL] EGL device " << eglDevId << ", CUDA device " << device
          << " is not readable";

      LOG(INFO) << "[EGL] Selected EGL device " << eglDevId
                << " for CUDA device " << device;
      display_ = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                          eglDevices[eglDevId], 0);
      CHECK_EGL_ERROR();
    }

    EGLint major, minor;
    EGLBoolean retval = eglInitialize(display_, &major, &minor);
    if (!retval) {
      LOG(ERROR) << "[EGL] Failed to initialize.";
    }
    CHECK_EGL_ERROR();

    LOG(INFO) << "[EGL] Version: " << eglQueryString(display_, EGL_VERSION);
    LOG(INFO) << "[EGL] Vendor: " << eglQueryString(display_, EGL_VENDOR);

    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglConfig;
    retval =
        eglChooseConfig(display_, configAttribs, &eglConfig, 1, &numConfigs);
    if (numConfigs != 1) {
      LOG(ERROR)
          << "[EGL] Cannot create EGL config. Your driver may not support EGL.";
    }
    CHECK_EGL_ERROR();

    // 3. Bind the API
    retval = eglBindAPI(EGL_OPENGL_API);
    if (!retval) {
      LOG(ERROR) << "[EGL] failed to bind OpenGL API";
    }
    CHECK_EGL_ERROR();

    // 4. Create a context
    context_ = eglCreateContext(display_, eglConfig, EGL_NO_CONTEXT, NULL);
    CHECK_EGL_ERROR();

    // 5. Make context current and create Magnum context
    makeCurrent();

    CHECK(magnumGlContext_.tryCreate())
        << "[EGL] Failed to create OpenGL context";
    isValid_ = true;
  };

  void makeCurrent() {
    EGLBoolean retval =
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, context_);
    if (!retval) {
      LOG(ERROR) << "[EGL] Failed to make EGL context current";
    }
    CHECK_EGL_ERROR();
  };

  bool isValid() { return isValid_; };

  ~ESPEGLContext() {
    eglDestroyContext(display_, context_);
    eglTerminate(display_);
  }

 private:
  EGLDisplay display_;
  EGLContext context_;
  Platform::GLContext magnumGlContext_;
  bool isValid_ = false;

  ESP_SMART_POINTERS(ESPEGLContext);
};

#else  // __ESP_USE_EGL__ not defined

struct ESPGLXContext : ESPContext {
  ESPGLXContext()
      : glxCtx_{Platform::WindowlessGlxContext::Configuration()},
        magnumGlContext_{NoCreate} {
    CHECK(glxCtx_.isCreated())
        << "[GLX] Failed to created headless glX context";

    makeCurrent();

    CHECK(magnumGlContext_.tryCreate())
        << "[GLX] Failed to create OpenGL Context";
    isValid_ = true;
  };

  void makeCurrent() { glxCtx_.makeCurrent(); };
  bool isValid() { return isValid_; };

 private:
  Platform::WindowlessGlxContext glxCtx_;
  Platform::GLContext magnumGlContext_;
  bool isValid_ = false;

  ESP_SMART_POINTERS(ESPGLXContext);
};

#endif

};  // namespace

struct WindowlessContext::Impl {
  Impl(int device) {
#ifdef __ESP_USE_EGL__
    glContext_ = ESPEGLContext::create_unique(device);
#else
    CHECK_EQ(device, 0)
        << "glX context does not support multiple GPUs. Please compile with "
           "BUILD_GUI_VIEWERS=0 for multi-gpu support via EGL";
    CHECK(std::getenv("DISPLAY") != nullptr)
        << "DISPLAY not detected. For headless systems, compile with "
           "--headless for EGL support";

    glContext_ = ESPGLXContext::create_unique();
#endif

    makeCurrent();
  }

  ~Impl() { LOG(INFO) << "Deconstructing GL context"; }

  void makeCurrent() { glContext_->makeCurrent(); }

  ESPContext::uptr glContext_ = nullptr;
};

#else  // not __linux__

struct WindowlessContext::Impl {
  Impl(int device) : glContext_({}), magnumGlContext_(NoCreate) {
    glContext_.makeCurrent();
    if (!magnumGlContext_.tryCreate()) {
      LOG(ERROR) << "Failed to create GL context";
    }
  }

  ~Impl() { LOG(INFO) << "Deconstructing GL context"; }

  void makeCurrent() { glContext_.makeCurrent(); }

  Platform::WindowlessGLContext glContext_;
  Platform::GLContext magnumGlContext_;
};

#endif

WindowlessContext::WindowlessContext(int device /* = 0 */)
    : pimpl_(spimpl::make_unique_impl<Impl>(device)) {}

void WindowlessContext::makeCurrent() {
  pimpl_->makeCurrent();
}

}  // namespace gfx
}  // namespace esp
