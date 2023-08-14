// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "esp/agent/Agent.h"
#include "esp/assets/SceneLoader.h"
#include "esp/core/esp.h"
#include "esp/core/random.h"
#include "esp/nav/PathFinder.h"
#include "esp/scene/ObjectControls.h"
#include "esp/scene/SceneGraph.h"

using namespace esp;
using namespace esp::nav;

void printPathPoint(int run, int step, const vec3f& p, float distance) {
  LOG(INFO) << run << "," << step << "," << p[0] << "," << p[1] << "," << p[2]
            << "," << distance;
}

void testPathFinder(PathFinder& pf) {
  for (int i = 0; i < 100000; i++) {
    ShortestPath path;
    path.requestedStart = pf.getRandomNavigablePoint();
    path.requestedEnd = pf.getRandomNavigablePoint();
    const bool foundPath = pf.findPath(path);
    if (foundPath) {
      const float islandSize = pf.islandRadius(path.requestedStart);
      CHECK(islandSize > 0.0);
      for (int j = 0; j < path.points.size(); j++) {
        printPathPoint(i, j, path.points[j], path.geodesicDistance);
        CHECK(pf.islandRadius(path.points[j]) == islandSize);
      }
      CHECK(pf.islandRadius(path.requestedEnd) == islandSize);
      const vec3f& pathStart = path.points.front();
      const vec3f& pathEnd = path.points.back();
      const vec3f end = pf.tryStep(pathStart, pathEnd);
      LOG(INFO) << "tryStep initial end=" << pathEnd.transpose()
                << ", final end=" << end.transpose();
      CHECK(path.geodesicDistance < std::numeric_limits<float>::infinity());
    }
  }
}

TEST(NavTest, PathFinderLoadTest) {
  PathFinder pf;
  pf.loadNavMesh("test.navmesh");
  testPathFinder(pf);
}

void printRandomizedPathSet(PathFinder& pf) {
  core::Random random;
  ShortestPath path;
  path.requestedStart = pf.getRandomNavigablePoint();
  path.requestedEnd = pf.getRandomNavigablePoint();
  std::cout << "run,step,x,y,z,geodesicDistance" << std::endl;
  for (int i = 0; i < 100; i++) {
    const float r = 0.1;
    vec3f rv(random.uniform_float(-r, r), 0, random.uniform_float(-r, r));
    vec3f rv2(random.uniform_float(-r, r), 0, random.uniform_float(-r, r));
    path.requestedStart += rv;
    path.requestedEnd += rv2;
    const bool foundPath = pf.findPath(path);

    if (foundPath) {
      printPathPoint(i, 0, path.requestedStart, path.geodesicDistance);
      for (int j = 0; j < path.points.size(); j++) {
        printPathPoint(i, j + 1, path.points[j], path.geodesicDistance);
      }
      printPathPoint(i, path.points.size() + 1, path.requestedEnd,
                     path.geodesicDistance);
    } else {
      LOG(WARNING) << "Failed to find shortest path between start="
                   << path.requestedStart.transpose()
                   << " and end=" << path.requestedEnd.transpose();
    }
  }
}

TEST(NavTest, PathFinderTestCases) {
  PathFinder pf;
  pf.loadNavMesh("test.navmesh");
  ShortestPath testPath;
  testPath.requestedStart = vec3f(-6.493, 0.072, -3.292);
  testPath.requestedEnd = vec3f(-8.98, 0.072, -0.62);
  LOG(INFO) << "TEST";
  pf.findPath(testPath);
  CHECK(testPath.points.size() == 0);
  CHECK_EQ(testPath.geodesicDistance, std::numeric_limits<float>::infinity());

  testPath.requestedStart = pf.getRandomNavigablePoint();
  // Jitter the point just enough so that it isn't exactly the same
  testPath.requestedEnd = testPath.requestedStart + vec3f(0.01, 0.0, 0.01);
  pf.findPath(testPath);
  // There should be 2 points
  CHECK_EQ(testPath.points.size(), 2);
  // The geodesicDistance should be almost exactly the L2 dist
  CHECK_LE(std::abs(testPath.geodesicDistance -
                    (testPath.requestedStart - testPath.requestedEnd).norm()),
           0.001);
}

TEST(NavTest, BuildNavMeshFromMeshTest) {
  using namespace esp::assets;
  SceneLoader loader;
  const AssetInfo info = AssetInfo::fromPath("test.glb");
  const MeshData mesh = loader.load(info);
  NavMeshSettings bs;
  bs.setDefaults();
  PathFinder pf;
  pf.build(bs, mesh);
  testPathFinder(pf);
}
