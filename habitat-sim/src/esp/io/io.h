// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <vector>

namespace esp {
namespace io {

bool exists(const std::string& file);

size_t fileSize(const std::string& file);

std::string removeExtension(const std::string& file);

std::string changeExtension(const std::string& file, const std::string& ext);

/** @brief Tokenize input string by any delimiter char in delimiterCharList.
 *
 * @param delimiterCharList string containing all delimiter chars
 * @param limit > 0 indicates maximum number of times delimiter is applied
 * @param mergeAdjacentDelimiters whether to merge adjacent delimiters
 * @return std::vector<std::string>> of tokens
 */
std::vector<std::string> tokenize(const std::string& string,
                                  const std::string& delimiterCharList,
                                  int limit = 0,
                                  bool mergeAdjacentDelimiters = false);

}  // namespace io
}  // namespace esp
