// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "esp/bindings/OpaqueTypes.h"

#include "esp/agent/Agent.h"
#include "esp/core/esp.h"
#include "esp/nav/GreedyFollower.h"
#include "esp/nav/PathFinder.h"
#include "esp/scene/ObjectControls.h"

namespace py = pybind11;
using namespace py::literals;
using namespace esp;
using namespace esp::nav;

void initShortestPathBindings(py::module& m) {
  py::class_<HitRecord>(m, "HitRecord")
      .def(py::init())
      .def_readwrite("hit_pos", &HitRecord::hitPos)
      .def_readwrite("hit_normal", &HitRecord::hitNormal)
      .def_readwrite("hit_dist", &HitRecord::hitDist);

  py::class_<ShortestPath, ShortestPath::ptr>(m, "ShortestPath")
      .def(py::init(&ShortestPath::create<>))
      .def_readwrite("requested_start", &ShortestPath::requestedStart)
      .def_readwrite("requested_end", &ShortestPath::requestedEnd)
      .def_readwrite("points", &ShortestPath::points)
      .def_readwrite("geodesic_distance", &ShortestPath::geodesicDistance);

  py::class_<MultiGoalShortestPath, MultiGoalShortestPath::ptr>(
      m, "MultiGoalShortestPath")
      .def(py::init(&MultiGoalShortestPath::create<>))
      .def_readwrite("requested_start", &MultiGoalShortestPath::requestedStart)
      .def_readwrite("requested_ends", &MultiGoalShortestPath::requestedEnds)
      .def_readwrite("points", &MultiGoalShortestPath::points)
      .def_readwrite("geodesic_distance",
                     &MultiGoalShortestPath::geodesicDistance);

  py::class_<PathFinder, PathFinder::ptr>(m, "PathFinder")
      .def(py::init(&PathFinder::create<>))
      .def("get_random_navigable_point", &PathFinder::getRandomNavigablePoint)
      .def("find_path", py::overload_cast<ShortestPath&>(&PathFinder::findPath),
           "path"_a)
      .def("find_path",
           py::overload_cast<MultiGoalShortestPath&>(&PathFinder::findPath),
           "path"_a)
      .def("try_step", &PathFinder::tryStep, R"()", "start"_a, "end"_a)
      .def("island_radius", &PathFinder::islandRadius, R"()", "pt"_a)
      .def_property_readonly("is_loaded", &PathFinder::isLoaded)
      .def("load_nav_mesh", &PathFinder::loadNavMesh)
      .def("distance_to_closest_obstacle",
           &PathFinder::distanceToClosestObstacle,
           R"(Returns the distance to the closest obstacle.
           If this distance is greater than :py:attr:`max_search_radius`,
           :py:attr:`max_search_radius` is returned instead.)",
           "pt"_a, "max_search_radius"_a = 2.0)
      .def(
          "closest_obstacle_surface_point",
          &PathFinder::closestObstacleSurfacePoint,
          R"(Returns the hit_pos, hit_normal, and hit_dist of the surface point on the closest obstacle.
           If the returned hit_dist is equal to :py:attr:`max_search_radius`,
           no obstacle was found.)",
          "pt"_a, "max_search_radius"_a = 2.0)
      .def("is_navigable", &PathFinder::isNavigable,
           R"(Checks to see if the agent can stand at the specified point.
          To check navigability, the point is snapped to the nearest polygon and
          then the snapped point is compared to the original point.
          Any amount of x-z translation indicates that the given point is not navigable.
          The amount of y-translation allowed is specified by max_y_delta to account
          for slight differences in floor height)",
           "pt"_a, "max_y_delta"_a = 0.5);

  py::class_<GreedyGeodesicFollowerImpl, GreedyGeodesicFollowerImpl::ptr>(
      m, "GreedyGeodesicFollowerImpl")
      .def(py::init(
          &GreedyGeodesicFollowerImpl::create<
              PathFinder::ptr&, GreedyGeodesicFollowerImpl::MoveFn&,
              GreedyGeodesicFollowerImpl::MoveFn&,
              GreedyGeodesicFollowerImpl::MoveFn&, double, double, double>))
      .def("next_action_along",
           py::overload_cast<const vec3f&, const vec4f&, const vec3f&>(
               &GreedyGeodesicFollowerImpl::nextActionAlong),
           py::return_value_policy::move)
      .def("find_path",
           py::overload_cast<const vec3f&, const vec4f&, const vec3f&>(
               &GreedyGeodesicFollowerImpl::findPath),
           py::return_value_policy::move);

  py::enum_<GreedyGeodesicFollowerImpl::CODES>(m, "GreedyFollowerCodes")
      .value("ERROR", GreedyGeodesicFollowerImpl::CODES::ERROR)
      .value("STOP", GreedyGeodesicFollowerImpl::CODES::STOP)
      .value("FORWARD", GreedyGeodesicFollowerImpl::CODES::FORWARD)
      .value("LEFT", GreedyGeodesicFollowerImpl::CODES::LEFT)
      .value("RIGHT", GreedyGeodesicFollowerImpl::CODES::RIGHT);

  py::bind_vector<std::vector<GreedyGeodesicFollowerImpl::CODES>>(
      m, "VectorGreedyCodes");
}
