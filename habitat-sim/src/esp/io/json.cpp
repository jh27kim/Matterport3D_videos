// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "esp/io/json.h"

#include <rapidjson/filereadstream.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "esp/core/esp.h"

namespace esp {
namespace io {

JsonDocument parseJsonFile(const std::string& file) {
  FILE* pFile = fopen(file.c_str(), "rb");
  char buffer[65536];
  rapidjson::FileReadStream is(pFile, buffer, sizeof(buffer));
  JsonDocument d;
  d.ParseStream<0, rapidjson::UTF8<>, rapidjson::FileReadStream>(is);
  fclose(pFile);

  if (d.HasParseError()) {
    LOG(ERROR) << "Parse error reading " << file << std::endl
               << "Error code " << d.GetParseError() << " at "
               << d.GetErrorOffset();
    throw std::runtime_error("JSON parse error");
  }
  return d;
}

JsonDocument parseJsonString(const std::string& jsonString) {
  JsonDocument d;
  d.Parse(jsonString.c_str());

  if (d.HasParseError()) {
    LOG(ERROR) << "Parse error parsing json string. Error code "
               << d.GetParseError() << " at " << d.GetErrorOffset();
    throw std::runtime_error("JSON parse error");
  }
  return d;
}

std::string jsonToString(const JsonDocument& d) {
  rapidjson::StringBuffer buffer{};
  rapidjson::Writer<rapidjson::StringBuffer> writer{buffer};
  d.Accept(writer);
  return buffer.GetString();
}

}  // namespace io
}  // namespace esp
