// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "PinholeCamera.h"

namespace esp {
namespace sensor {

PinholeCamera::PinholeCamera(sensor::SensorSpec::ptr spec)
    : sensor::Sensor(spec) {
  setProjectionParameters(spec);
}

PinholeCamera::PinholeCamera(sensor::SensorSpec::ptr spec,
                             scene::SceneNode& pinholeCameraNode)
    : PinholeCamera(spec) {
  // attatch the pinholecamera to the scene node, and immediately set the
  // transformation (in Sensor::attach)
  Sensor::attach(pinholeCameraNode);
}

void PinholeCamera::setProjectionParameters(SensorSpec::ptr spec) {
  ASSERT(spec != nullptr);
  width_ = spec_->resolution[1];
  height_ = spec_->resolution[0];
  near_ = std::atof(spec_->parameters.at("near").c_str());
  far_ = std::atof(spec_->parameters.at("far").c_str());
  hfov_ = std::atof(spec_->parameters.at("hfov").c_str());
}

void PinholeCamera::setProjectionMatrix(gfx::RenderCamera& targetCamera) {
  targetCamera.setProjectionMatrix(width_, height_, near_, far_, hfov_);
}

}  // namespace sensor
}  // namespace esp
