// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "esp/core/esp.h"
#include "esp/io/io.h"

using namespace esp::io;

TEST(IOTest, fileExistTest) {
  std::string file = "build/tests/IOTest";
  bool result = exists(file);
  EXPECT_TRUE(result);

  file = "Foo.bar";
  result = exists(file);
  EXPECT_FALSE(result);
}

TEST(IOTest, fileSizeTest) {
  std::string existingFile = "build/tests/IOTest";
  auto result = fileSize(existingFile);
  LOG(INFO) << "File size of " << existingFile << " is " << result;

  std::string nonexistingFile = "Foo.bar";
  result = fileSize(nonexistingFile);
  LOG(INFO) << "File size of " << nonexistingFile << " is " << result;
}

TEST(IOTest, fileRmExtTest) {
  std::string filename = "/foo/bar.jpeg";

  // rm extension
  std::string result = removeExtension(filename);
  EXPECT_EQ(result, "/foo/bar");
  EXPECT_EQ(filename, "/foo/bar.jpeg");

  std::string filenameNoExt = "/path/to/foobar";
  result = removeExtension(filenameNoExt);
  EXPECT_EQ(result, filenameNoExt);
}

TEST(IOTest, fileReplaceExtTest) {
  std::string filename = "/foo/bar.jpeg";

  // change extension
  std::string ext = ".png";
  std::string result = changeExtension(filename, ext);

  EXPECT_EQ(result, "/foo/bar.png");

  std::string filenameNoExt = "/path/to/foobar";
  result = changeExtension(filenameNoExt, ext);
  EXPECT_EQ(result, "/path/to/foobar.png");

  std::string cornerCase = "";
  result = changeExtension(cornerCase, ext);
  EXPECT_EQ(result, ".png");

  cornerCase = ".";
  result = changeExtension(cornerCase, ext);
  EXPECT_EQ(result, "..png");

  cornerCase = "..";
  result = changeExtension(cornerCase, ext);
  EXPECT_EQ(result, "...png");

  std::string cornerCaseExt = "png";  // no dot
  result = changeExtension(filename, cornerCaseExt);
  EXPECT_EQ(result, "/foo/bar.png");

  cornerCase = ".";
  result = changeExtension(cornerCase, cornerCaseExt);
  EXPECT_EQ(result, "..png");

  cornerCase = "..";
  result = changeExtension(cornerCase, cornerCaseExt);
  EXPECT_EQ(result, "...png");

  cornerCase = ".jpg";
  result = changeExtension(cornerCase, cornerCaseExt);
  EXPECT_EQ(result, ".jpg.png");
}

TEST(IOTest, tokenizeTest) {
  std::string file = ",a,|,bb|c";
  const auto& t1 = tokenize(file, ",");
  EXPECT_EQ((std::vector<std::string>{"", "a", "|", "bb|c"}), t1);
  const auto& t2 = tokenize(file, "|");
  EXPECT_EQ((std::vector<std::string>{",a,", ",bb", "c"}), t2);
  const auto& t3 = tokenize(file, ",|", 0, true);
  EXPECT_EQ((std::vector<std::string>{"", "a", "bb", "c"}), t3);
}
