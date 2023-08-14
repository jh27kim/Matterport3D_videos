// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "PTexMeshShader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <iostream>

#include <Corrade/Containers/Reference.h>
#include <Magnum/GL/BufferTextureFormat.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Version.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>

#include "esp/assets/PTexMeshData.h"
#include "esp/core/esp.h"
#include "esp/io/io.h"

using namespace Magnum;

namespace esp {
namespace gfx {

const static std::string PTEX_SHADER_VS = R"(
layout(location = 0) in vec4 position;
uniform mat4 MVP;

void main() {
  gl_Position = MVP * position;
}
)";

const static std::string PTEX_SHADER_GS = R"(
layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;

out vec2 uv;

void main() {
  gl_PrimitiveID = gl_PrimitiveIDIn;

  uv = vec2(1.0, 0.0);
  gl_Position = gl_in[1].gl_Position;
  EmitVertex();

  uv = vec2(0.0, 0.0);
  gl_Position = gl_in[0].gl_Position;
  EmitVertex();

  uv = vec2(1.0, 1.0);
  gl_Position = gl_in[2].gl_Position;
  EmitVertex();

  uv = vec2(0.0, 1.0);
  gl_Position = gl_in[3].gl_Position;
  EmitVertex();

  EndPrimitive();
}
)";

const static std::string PTEX_SHADER_FS = R"(
uniform int tileSize;
uniform int widthInTiles;
uniform samplerBuffer meshAdjFaces;

ivec2 FaceToAtlasPos(int faceID, int tileSize) {
  ivec2 tilePos;
  tilePos.y = faceID / widthInTiles;
  tilePos.x = faceID - (tilePos.y * widthInTiles);
  return tilePos * tileSize;
}

// rotate UVs into neighbouring face frame
// rot = number of 90 degree anti-clockwise rotations
ivec2 RotateUVs(ivec2 p, int rot, int size) {
  switch (rot) {
    case 0:
      return p;
    case 1:
      return ivec2(p.y, (size - 1) - p.x);
    case 2:
      return ivec2((size - 1) - p.x, (size - 1) - p.y);
    case 3:
      return ivec2((size - 1) - p.y, p.x);
  }
}

const uint ROTATION_SHIFT = 30;
const uint FACE_MASK = 0x3FFFFFFF;

int GetAdjFace(int face, int edge, out int rot) {
  // uint data = meshAdjFaces[face * 4 + edge];
  uint data = uint(texelFetch(meshAdjFaces, face * 4 + edge));
  rot = int(data >> ROTATION_SHIFT);
  return int(data & FACE_MASK);
}

bool IsValid(int adjFace) {
  return adjFace != FACE_MASK;
}

// fetch texel from atlas
// p is integer tile coordinate
// is p is outside tile, fetches from correct adjacent tile, taking account of
// rotation
int indexAdjacentFaces(int faceID, inout ivec2 p, int tsize) {
  int rot;
  // edge 0
  if (p.y < 0) {
    int adjFace = GetAdjFace(faceID, 0, rot);

    if (IsValid(adjFace)) {
      p.y += tsize;
      if (p.x > tsize - 1) {
        p.x -= tsize;
        p = RotateUVs(p, rot, tsize);

        adjFace = GetAdjFace(adjFace, (1 - rot) & 3, rot);
        if (IsValid(adjFace)) {
          p = RotateUVs(p, rot, tsize);
          return adjFace;
        }
      } else if (p.x < 0) {
        p.x += tsize;
        p = RotateUVs(p, rot, tsize);

        adjFace = GetAdjFace(adjFace, (3 - rot) & 3, rot);
        if (IsValid(adjFace)) {
          p = RotateUVs(p, rot, tsize);
          return adjFace;
        }
      } else {
        p = RotateUVs(p, rot, tsize);
        return adjFace;
      }
    }
  }
  // edge 2
  else if (p.y > tsize - 1) {
    int adjFace = GetAdjFace(faceID, 2, rot);

    if (IsValid(adjFace)) {
      p.y -= tsize;
      if (p.x > tsize - 1) {
        p.x -= tsize;
        p = RotateUVs(p, rot, tsize);

        adjFace = GetAdjFace(adjFace, (1 - rot) & 3, rot);
        if (IsValid(adjFace)) {
          p = RotateUVs(p, rot, tsize);
          return adjFace;
        }
      } else if (p.x < 0) {
        p.x += tsize;
        p = RotateUVs(p, rot, tsize);

        adjFace = GetAdjFace(adjFace, (3 - rot) & 3, rot);
        if (IsValid(adjFace)) {
          p = RotateUVs(p, rot, tsize);
          return adjFace;
        }
      } else {
        p = RotateUVs(p, rot, tsize);
        return adjFace;
      }
    }
  } else {
    // edge 3
    if (p.x < 0) {
      int adjFace = GetAdjFace(faceID, 3, rot);
      if (IsValid(adjFace)) {
        p.x += tsize;
        p = RotateUVs(p, rot, tsize);
        return adjFace;
      }
    }
    // edge 1
    else if (p.x > tsize - 1) {
      int adjFace = GetAdjFace(faceID, 1, rot);
      if (IsValid(adjFace)) {
        p.x -= tsize;
        p = RotateUVs(p, rot, tsize);
        return adjFace;
      }
    }
  }

  return faceID;
}

// load texel from atlas, handling adjacent faces
vec4 texelFetchAtlasAdj(sampler2D tex, int faceID, ivec2 p, int level) {
  int tsize = tileSize >> level;

  // fetch from adjacent face if necessary
  faceID = indexAdjacentFaces(faceID, p, tsize);

  // clamp to tile edge
  p = clamp(p, ivec2(0, 0), ivec2(tsize - 1, tsize - 1));

  ivec2 atlasPos = FaceToAtlasPos(faceID, tsize);
  return texelFetch(tex, atlasPos + p, level);
}

// fetch with bilinear filtering
vec4 textureAtlas(sampler2D tex, int faceID, vec2 p) {
  int level = 0;
  p -= 0.5;
  ivec2 i = ivec2(floor(p));
  vec2 f = p - vec2(i);
  return mix(
      mix(texelFetchAtlasAdj(tex, faceID, ivec2(i), level),
          texelFetchAtlasAdj(tex, faceID, ivec2(i.x + 1, i.y), level), f.x),
      mix(texelFetchAtlasAdj(tex, faceID, ivec2(i.x, i.y + 1), level),
          texelFetchAtlasAdj(tex, faceID, ivec2(i.x + 1, i.y + 1), level), f.x),
      f.y);
}

layout(location = 0) out vec4 FragColor;
uniform sampler2D atlasTex;

uniform float exposure;

in vec2 uv;

void main() {
  vec4 c = textureAtlas(atlasTex, gl_PrimitiveID, uv * tileSize) * exposure;
  // c = vec4(1.0f, 1.0f, 1.0f, 1.0f);
  FragColor = vec4(c.xyz, 1.0f);
}
)";

PTexMeshShader::PTexMeshShader() {
  MAGNUM_ASSERT_GL_VERSION_SUPPORTED(GL::Version::GL410);

  GL::Shader vert{GL::Version::GL410, GL::Shader::Type::Vertex};
  GL::Shader geom{GL::Version::GL410, GL::Shader::Type::Geometry};
  GL::Shader frag{GL::Version::GL410, GL::Shader::Type::Fragment};

  vert.addSource(PTEX_SHADER_VS);
  geom.addSource(PTEX_SHADER_GS);
  frag.addSource(PTEX_SHADER_FS);

  CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, geom, frag}));

  attachShaders({vert, geom, frag});

  CORRADE_INTERNAL_ASSERT_OUTPUT(link());
}

}  // namespace gfx
}  // namespace esp
