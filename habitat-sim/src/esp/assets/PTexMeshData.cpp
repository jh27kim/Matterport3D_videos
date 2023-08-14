// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "PTexMeshData.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <Magnum/GL/BufferTextureFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>

#include "esp/core/esp.h"
#include "esp/gfx/PTexMeshShader.h"
#include "esp/io/io.h"
#include "esp/io/json.h"

#ifdef __unix__
#define MAP_PP MAP_PRIVATE | MAP_POPULATE
#else
#define MAP_PP MAP_PRIVATE
#endif

static constexpr int ROTATION_SHIFT = 30;
static constexpr int FACE_MASK = 0x3FFFFFFF;

namespace esp {
namespace assets {

void PTexMeshData::load(const std::string& meshFile,
                        const std::string& atlasFolder) {
  ASSERT(io::exists(meshFile));
  ASSERT(io::exists(atlasFolder));

  // Parse parameters
  const auto& paramsFile = atlasFolder + "/parameters.json";
  ASSERT(io::exists(paramsFile));
  const io::JsonDocument json = io::parseJsonFile(paramsFile);
  splitSize_ = json["splitSize"].GetDouble();
  tileSize_ = json["tileSize"].GetInt();
  atlasFolder_ = atlasFolder;

  loadMeshData(meshFile);
}

float PTexMeshData::exposure() const {
  return exposure_;
}

void PTexMeshData::setExposure(const float& val) {
  exposure_ = val;
}

const std::vector<PTexMeshData::MeshData>& PTexMeshData::meshes() const {
  return submeshes_;
}

std::string PTexMeshData::atlasFolder() const {
  return atlasFolder_;
}

std::vector<PTexMeshData::MeshData> splitMesh(
    const PTexMeshData::MeshData& mesh,
    const float splitSize) {
  std::vector<uint32_t> verts;
  verts.resize(mesh.vbo.size());

  auto Part1By2 = [](uint64_t x) {
    x &= 0x1fffff;  // mask off lower 21 bits
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8)) & 0x100f00f00f00f00f;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3;
    x = (x | (x << 2)) & 0x1249249249249249;
    return x;
  };

  auto EncodeMorton3 = [&Part1By2](const vec3i& v) {
    return (Part1By2(v(2)) << 2) + (Part1By2(v(1)) << 1) + Part1By2(v(0));
  };

  box3f boundingBox;

  for (size_t i = 0; i < mesh.vbo.size(); i++) {
    boundingBox.extend(mesh.vbo[i].head<3>());
  }

// calculate vertex grid position and code
#pragma omp parallel for
  for (size_t i = 0; i < mesh.vbo.size(); i++) {
    const vec3f p = mesh.vbo[i].head<3>();
    vec3f pi = (p - boundingBox.min()) / splitSize;
    verts[i] = EncodeMorton3(pi.cast<int>());
  }

  // data structure for sorting faces
  struct SortFace {
    uint32_t index[4];
    uint32_t code;
    size_t originalFace;
  };

  // fill per-face data structures (including codes)
  size_t numFaces = mesh.ibo.size() / 4;
  std::vector<SortFace> faces;
  faces.resize(numFaces);

#pragma omp parallel for
  for (size_t i = 0; i < numFaces; i++) {
    faces[i].originalFace = i;
    faces[i].code = std::numeric_limits<uint32_t>::max();
    for (int j = 0; j < 4; j++) {
      faces[i].index[j] = mesh.ibo[i * 4 + j];

      // face code is minimum of referenced vertices codes
      faces[i].code = std::min(faces[i].code, verts[faces[i].index[j]]);
    }
  }

  // sort faces by code
  std::sort(faces.begin(), faces.end(),
            [](const SortFace& f1, const SortFace& f2) -> bool {
              return (f1.code < f2.code);
            });

  // find face chunk start indices
  std::vector<uint32_t> chunkStart;
  chunkStart.push_back(0);
  uint32_t prevCode = faces[0].code;
  for (size_t i = 1; i < faces.size(); i++) {
    if (faces[i].code != prevCode) {
      chunkStart.push_back(i);
      prevCode = faces[i].code;
    }
  }

  chunkStart.push_back(faces.size());
  size_t numChunks = chunkStart.size() - 1;

  size_t maxFaces = 0;
  for (size_t i = 0; i < numChunks; i++) {
    uint32_t chunkSize = chunkStart[i + 1] - chunkStart[i];
    if (chunkSize > maxFaces)
      maxFaces = chunkSize;
  }

  // create new mesh for each chunk of faces
  std::vector<PTexMeshData::MeshData> subMeshes;

  for (size_t i = 0; i < numChunks; i++) {
    subMeshes.emplace_back();
  }

#pragma omp parallel for
  for (size_t i = 0; i < numChunks; i++) {
    uint32_t chunkSize = chunkStart[i + 1] - chunkStart[i];

    std::vector<uint32_t> refdVerts;
    std::unordered_map<uint32_t, uint32_t> refdVertsMap;
    subMeshes[i].ibo.resize(chunkSize * 4);

    for (size_t j = 0; j < chunkSize; j++) {
      size_t faceIdx = chunkStart[i] + j;
      for (int k = 0; k < 4; k++) {
        uint32_t vertIndex = faces[faceIdx].index[k];
        uint32_t newIndex = 0;

        auto it = refdVertsMap.find(vertIndex);

        if (it == refdVertsMap.end()) {
          // vertex not found, add
          newIndex = refdVerts.size();
          refdVerts.push_back(vertIndex);
          refdVertsMap[vertIndex] = newIndex;
        } else {
          // found, use existing index
          newIndex = it->second;
        }
        subMeshes[i].ibo[j * 4 + k] = newIndex;
      }
    }

    // add referenced vertices to submesh
    subMeshes[i].vbo.resize(refdVerts.size());
    subMeshes[i].nbo.resize(refdVerts.size());
    for (size_t j = 0; j < refdVerts.size(); j++) {
      uint32_t index = refdVerts[j];
      subMeshes[i].vbo[j] = mesh.vbo[index];
      subMeshes[i].nbo[j] = mesh.nbo[index];
    }
  }

  return subMeshes;
}

void PTexMeshData::calculateAdjacency(const PTexMeshData::MeshData& mesh,
                                      std::vector<uint32_t>& adjFaces) {
  struct EdgeData {
    int face;
    int edge;
  };

  std::unordered_map<uint64_t, std::vector<EdgeData>> edgeMap;

  size_t numFaces = mesh.ibo.size() / 4;

  typedef std::unordered_map<uint64_t, std::vector<EdgeData>>::iterator
      EdgeIter;
  std::vector<EdgeIter> edgeIterators(numFaces * 4);

  // for each face
  for (int f = 0; f < numFaces; f++) {
    // for each edge
    for (int e = 0; e < 4; e++) {
      // add to edge to face map
      const int e_index = f * 4 + e;
      const uint32_t i0 = mesh.ibo[e_index];
      const uint32_t i1 = mesh.ibo[f * 4 + ((e + 1) % 4)];
      const uint64_t key =
          (uint64_t)std::min(i0, i1) << 32 | (uint32_t)std::max(i0, i1);

      const EdgeData edgeData{f, e};

      auto it = edgeMap.find(key);

      if (it == edgeMap.end()) {
        it = edgeMap.emplace(key, std::vector<EdgeData>()).first;
        it->second.reserve(4);
        it->second.push_back(edgeData);
      } else {
        it->second.push_back(edgeData);
      }

      edgeIterators[e_index] = it;
    }
  }

  adjFaces.resize(numFaces * 4);

  for (int f = 0; f < numFaces; f++) {
    for (int e = 0; e < 4; e++) {
      const int e_index = f * 4 + e;
      auto it = edgeIterators[e_index];
      const std::vector<EdgeData>& adj = it->second;

      // find adjacent face
      int adjFace = -1;
      for (size_t i = 0; i < adj.size(); i++) {
        if (adj[i].face != (int)f)
          adjFace = adj[i].face;
      }

      // find number of 90 degree rotation steps between faces
      int rot = 0;
      if (adj.size() == 2) {
        int edge0 = 0, edge1 = 0;
        if (adj[0].edge == e) {
          edge0 = adj[0].edge;
          edge1 = adj[1].edge;
        } else if (adj[1].edge == e) {
          edge0 = adj[1].edge;
          edge1 = adj[0].edge;
        }

        rot = (edge0 - edge1 + 2) & 3;
      }

      // pack adjacent face and rotation into 32-bit int
      adjFaces[f * 4 + e] = (rot << ROTATION_SHIFT) | (adjFace & FACE_MASK);
    }
  }
}

void PTexMeshData::loadMeshData(const std::string& meshFile) {
  PTexMeshData::MeshData originalMesh;
  parsePLY(meshFile, originalMesh);

  submeshes_.clear();
  if (splitSize_ > 0.0f) {
    std::cout << "Splitting mesh... ";
    submeshes_ = splitMesh(originalMesh, splitSize_);
    std::cout << "done" << std::endl;
  } else {
    submeshes_.emplace_back(std::move(originalMesh));
  }
}

void PTexMeshData::parsePLY(const std::string& filename,
                            PTexMeshData::MeshData& meshData) {
  std::vector<std::string> comments;
  std::vector<std::string> objInfo;

  std::string lastElement;
  std::string lastProperty;

  enum Properties { POSITION = 0, NORMAL, COLOR, NUM_PROPERTIES };

  size_t numVertices = 0;

  size_t positionDimensions = 0;
  size_t normalDimensions = 0;
  size_t colorDimensions = 0;

  std::vector<Properties> vertexLayout;

  size_t numFaces = 0;

  std::ifstream file(filename, std::ios::binary);

  // Header parsing
  {
    std::string line;

    while (std::getline(file, line)) {
      std::istringstream ls(line);
      std::string token;
      ls >> token;

      if (token == "ply" || token == "PLY" || token == "") {
        // Skip preamble line
        continue;
      } else if (token == "comment") {
        // Just store these incase
        comments.push_back(line.erase(0, 8));
      } else if (token == "format") {
        // We can only parse binary data, so check that's what it is
        std::string s;
        ls >> s;
        ASSERT(s == "binary_little_endian");
      } else if (token == "element") {
        std::string name;
        size_t size;
        ls >> name >> size;

        if (name == "vertex") {
          // Pull out the number of vertices
          numVertices = size;
        } else if (name == "face") {
          // Pull out number of faces
          numFaces = size;
          ASSERT(numFaces > 0);
        } else {
          ASSERT(false, "Can't parse element (%)", name);
        }

        // Keep track of what element we parsed last to associate the properties
        // that follow
        lastElement = name;
      } else if (token == "property") {
        std::string type, name;
        ls >> type;

        // Special parsing for list properties (e.g. faces)
        bool isList = false;

        if (type == "list") {
          isList = true;

          std::string countType;
          ls >> countType >> type;

          ASSERT(countType == "uchar" || countType == "uint8",
                 "Don't understand count type (%)", countType);

          ASSERT(type == "int", "Don't understand index type (%)", type);

          ASSERT(lastElement == "face",
                 "Only expecting list after face element, not after (%)",
                 lastElement);
        }

        ASSERT(type == "float" || type == "int" || type == "uchar" ||
                   type == "uint8",
               "Don't understand type (%)", type);

        ls >> name;

        // Collecting vertex property information
        if (lastElement == "vertex") {
          ASSERT(type != "int", "Don't support 32-bit integer properties");

          // Position information
          if (name == "x") {
            positionDimensions = 1;
            vertexLayout.push_back(Properties::POSITION);
            ASSERT(type == "float", "Don't support 8-bit integer positions");
          } else if (name == "y") {
            ASSERT(lastProperty == "x",
                   "Properties should follow x, y, z, (w) order");
            positionDimensions = 2;
          } else if (name == "z") {
            ASSERT(lastProperty == "y",
                   "Properties should follow x, y, z, (w) order");
            positionDimensions = 3;
          } else if (name == "w") {
            ASSERT(lastProperty == "z",
                   "Properties should follow x, y, z, (w) order");
            positionDimensions = 4;
          }

          // Normal information
          if (name == "nx") {
            normalDimensions = 1;
            vertexLayout.push_back(Properties::NORMAL);
            ASSERT(type == "float", "Don't support 8-bit integer normals");
          } else if (name == "ny") {
            ASSERT(lastProperty == "nx",
                   "Properties should follow nx, ny, nz order");
            normalDimensions = 2;
          } else if (name == "nz") {
            ASSERT(lastProperty == "ny",
                   "Properties should follow nx, ny, nz order");
            normalDimensions = 3;
          }

          // Color information
          if (name == "red") {
            colorDimensions = 1;
            vertexLayout.push_back(Properties::COLOR);
            ASSERT(type == "uchar" || type == "uint8",
                   "Don't support non-8-bit integer colors");
          } else if (name == "green") {
            ASSERT(lastProperty == "red",
                   "Properties should follow red, green, blue, (alpha) order");
            colorDimensions = 2;
          } else if (name == "blue") {
            ASSERT(lastProperty == "green",
                   "Properties should follow red, green, blue, (alpha) order");
            colorDimensions = 3;
          } else if (name == "alpha") {
            ASSERT(lastProperty == "blue",
                   "Properties should follow red, green, blue, (alpha) order");
            colorDimensions = 4;
          }
        } else if (lastElement == "face") {
          ASSERT(isList, "No idea what to do with properties following faces");
        } else {
          ASSERT(false, "No idea what to do with properties before elements");
        }

        lastProperty = name;
      } else if (token == "obj_info") {
        // Just store these incase
        objInfo.push_back(line.erase(0, 9));
      } else if (token == "end_header") {
        // Done reading!
        break;
      } else {
        // Something unrecognised
        ASSERT(false);
      }
    }

    // Check things make sense.
    ASSERT(numVertices > 0);
    ASSERT(positionDimensions > 0);
  }

  meshData.vbo.resize(numVertices, vec4f(0, 0, 0, 1));

  if (normalDimensions) {
    meshData.nbo.resize(numVertices, vec4f(0, 0, 0, 1));
  }

  if (colorDimensions) {
    meshData.cbo.resize(numVertices, vec4uc(0, 0, 0, 255));
  }

  // Can only be FLOAT32 or UINT8
  const size_t positionBytes = positionDimensions * sizeof(float);  // floats
  const size_t normalBytes = normalDimensions * sizeof(float);      // floats
  const size_t colorBytes = colorDimensions * sizeof(uint8_t);      // bytes

  const size_t vertexPacketSizeBytes = positionBytes + normalBytes + colorBytes;

  size_t positionOffsetBytes = 0;
  size_t normalOffsetBytes = 0;
  size_t colorOffsetBytes = 0;

  size_t offsetSoFarBytes = 0;

  for (size_t i = 0; i < vertexLayout.size(); i++) {
    if (vertexLayout[i] == Properties::POSITION) {
      positionOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += positionBytes;
    } else if (vertexLayout[i] == Properties::NORMAL) {
      normalOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += normalBytes;
    } else if (vertexLayout[i] == Properties::COLOR) {
      colorOffsetBytes = offsetSoFarBytes;
      offsetSoFarBytes += colorBytes;
    } else {
      ASSERT(false);
    }
  }

  // Close after parsing header and re-open memory mapped
  const size_t postHeader = file.tellg();

  file.close();

  const size_t fileSize = io::fileSize(filename);

  int fd = open(filename.c_str(), O_RDONLY, 0);
  void* mmappedData = mmap(NULL, fileSize, PROT_READ, MAP_PP, fd, 0);

  // Parse each vertex packet and unpack
  char* bytes = &(((char*)mmappedData)[postHeader]);

  for (size_t i = 0; i < numVertices; i++) {
    char* nextBytes = &bytes[vertexPacketSizeBytes * i];

    memcpy(meshData.vbo[i].data(), &nextBytes[positionOffsetBytes],
           positionBytes);

    if (normalDimensions)
      memcpy(meshData.nbo[i].data(), &nextBytes[normalOffsetBytes],
             normalBytes);

    if (colorDimensions)
      memcpy(meshData.cbo[i].data(), &nextBytes[colorOffsetBytes], colorBytes);
  }

  const size_t bytesSoFar = postHeader + vertexPacketSizeBytes * numVertices;

  bytes =
      &(((char*)mmappedData)[postHeader + vertexPacketSizeBytes * numVertices]);

  // Read first face to get number of indices;
  const uint8_t faceDimensions = *bytes;

  ASSERT(faceDimensions == 3 || faceDimensions == 4);

  const size_t countBytes = 1;
  const size_t faceBytes = faceDimensions * sizeof(uint32_t);  // uint32_t
  const size_t facePacketSizeBytes = countBytes + faceBytes;

  const size_t predictedFaces = (fileSize - bytesSoFar) / facePacketSizeBytes;

  // Not sure what to do here
  //    if(predictedFaces < numFaces)
  //    {
  //        std::cout << "Skipping " << numFaces - predictedFaces << " missing
  //        faces" << std::endl;
  //    }
  //    else if(numFaces < predictedFaces)
  //    {
  //        std::cout << "Ignoring " << predictedFaces - numFaces << " extra
  //        faces" << std::endl;
  //    }

  numFaces = std::min(numFaces, predictedFaces);

  meshData.ibo.resize(numFaces * faceDimensions);

  for (size_t i = 0; i < numFaces; i++) {
    char* nextBytes = &bytes[facePacketSizeBytes * i];

    memcpy(&meshData.ibo[i * faceDimensions], &nextBytes[countBytes],
           faceBytes);
  }

  munmap(mmappedData, fileSize);

  close(fd);
}

void PTexMeshData::uploadBuffersToGPU(bool forceReload) {
  if (forceReload) {
    buffersOnGPU_ = false;
  }
  if (buffersOnGPU_) {
    return;
  }

  for (int iMesh = 0; iMesh < submeshes_.size(); ++iMesh) {
    std::cout << "\rLoading mesh " << iMesh + 1 << "/" << submeshes_.size()
              << "... ";
    std::cout.flush();

    renderingBuffers_.emplace_back(
        std::make_unique<PTexMeshData::RenderingBuffer>());

    auto& currentMesh = renderingBuffers_.back();
    currentMesh->vbo.setData(submeshes_[iMesh].vbo,
                             Magnum::GL::BufferUsage::StaticDraw);
    currentMesh->ibo.setData(submeshes_[iMesh].ibo,
                             Magnum::GL::BufferUsage::StaticDraw);
  }
  std::cout << "... done" << std::endl;

  std::cout << "Calculating mesh adjacency... ";
  std::cout.flush();

  std::vector<std::vector<uint32_t>> adjFaces(submeshes_.size());

#pragma omp parallel for
  for (int iMesh = 0; iMesh < submeshes_.size(); ++iMesh) {
    calculateAdjacency(submeshes_[iMesh], adjFaces[iMesh]);
  }

  for (int iMesh = 0; iMesh < submeshes_.size(); ++iMesh) {
    auto& currentMesh = renderingBuffers_[iMesh];

    currentMesh->adjTex.setBuffer(Magnum::GL::BufferTextureFormat::R32UI,
                                  currentMesh->abo);
    currentMesh->abo.setData(adjFaces[iMesh],
                             Magnum::GL::BufferUsage::StaticDraw);
    currentMesh->mesh.setPrimitive(Magnum::GL::MeshPrimitive::LinesAdjacency)
        .setCount(currentMesh->ibo.size() / 2)
        .addVertexBuffer(currentMesh->vbo, 0, gfx::PTexMeshShader::Position{})
        .setIndexBuffer(currentMesh->ibo, 0,
                        Magnum::GL::MeshIndexType::UnsignedInt);
  }

  for (size_t iMesh = 0; iMesh < renderingBuffers_.size(); ++iMesh) {
    const std::string rgbFile =
        atlasFolder_ + "/" + std::to_string(iMesh) + "-color-ptex.rgb";
    if (!io::exists(rgbFile)) {
      ASSERT(false, "Can't find " + rgbFile);
    }
    std::cout << "\rLoading atlas " << iMesh + 1 << "/"
              << renderingBuffers_.size() << "... ";
    std::cout.flush();

    const size_t numBytes = io::fileSize(rgbFile);
    const int dim = static_cast<int>(std::sqrt(numBytes / 3));  // square
    int fd = open(rgbFile.c_str(), O_RDONLY, 0);
    void* data = mmap(NULL, numBytes, PROT_READ, MAP_PP, fd, 0);
    Magnum::Containers::ArrayView<const void> dataView{data, numBytes};
    Magnum::ImageView2D image(Magnum::PixelFormat::RGB8UI, {dim, dim},
                              dataView);
    renderingBuffers_[iMesh]
        ->tex.setWrapping(Magnum::GL::SamplerWrapping::ClampToEdge)
        .setMagnificationFilter(Magnum::GL::SamplerFilter::Linear)
        .setMinificationFilter(Magnum::GL::SamplerFilter::Linear)
        // .setStorage(1, GL::TextureFormat::RGB8UI, image.size())
        .setSubImage(0, {}, image);
    munmap(data, numBytes);
    close(fd);
  }
  std::cout << "... done" << std::endl;

  buffersOnGPU_ = true;
}

PTexMeshData::RenderingBuffer* PTexMeshData::getRenderingBuffer(int submeshID) {
  ASSERT(submeshID >= 0 && submeshID < renderingBuffers_.size());
  return renderingBuffers_[submeshID].get();
}

Magnum::GL::Mesh* PTexMeshData::getMagnumGLMesh(int submeshID) {
  ASSERT(submeshID >= 0 && submeshID < renderingBuffers_.size());
  return &(renderingBuffers_[submeshID]->mesh);
}

}  // namespace assets
}  // namespace esp
