// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include "esp/core/esp.h"

namespace esp {
namespace assets {

//! Raw mesh data storage
struct MeshData {
  //! Vertex positions
  std::vector<vec3f> vbo;
  //! Vertex normals
  std::vector<vec3f> nbo;
  //! Texture coordinates
  std::vector<vec2f> tbo;
  //! Vertex colors
  std::vector<vec3f> cbo;
  //! Index buffer
  std::vector<uint32_t> ibo;
};

}  // namespace assets
}  // namespace esp
