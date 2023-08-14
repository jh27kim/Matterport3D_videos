// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "RenderCamera.h"

using namespace Magnum;

namespace esp {
namespace gfx {

RenderCamera::RenderCamera()
    : scene::AttachedObject(scene::AttachedObjectType::CAMERA) {}

RenderCamera::RenderCamera(scene::SceneNode& node) : RenderCamera() {
  // has to call the "attach" from the subclass
  attach(node);
}

RenderCamera::RenderCamera(scene::SceneNode& node,
                           const vec3f& eye,
                           const vec3f& target,
                           const vec3f& up)
    : RenderCamera(node) {
  // once it is attached, set the transformation
  setTransformation(eye, target, up);
}

void RenderCamera::attach(scene::SceneNode& node) {
  AttachedObject::attach(node);
  // "create and forget": magnum will handle the memory
  camera_ = new MagnumCamera(node);
}

void RenderCamera::detach() {
  AttachedObject::detach();
  // no need to free the camera_ since magnum will handle it
}

void RenderCamera::setProjectionMatrix(int width,
                                       int height,
                                       float znear,
                                       float zfar,
                                       float hfov) {
  ASSERT(isValid());
  const float aspectRatio = static_cast<float>(width) / height;
  camera_->setAspectRatioPolicy(SceneGraph::AspectRatioPolicy::NotPreserved)
      .setProjectionMatrix(
          Matrix4::perspectiveProjection(Deg{hfov}, aspectRatio, znear, zfar))
      .setViewport(Magnum::Vector2i(width, height));
}

mat4f RenderCamera::getProjectionMatrix() {
  ASSERT(isValid());
  return Eigen::Map<mat4f>(camera_->projectionMatrix().data());
}

mat4f RenderCamera::getCameraMatrix() {
  ASSERT(isValid());
  return Eigen::Map<mat4f>(camera_->cameraMatrix().data());
}

MagnumCamera& RenderCamera::getMagnumCamera() {
  ASSERT(isValid());
  return *camera_;
}

void RenderCamera::draw(MagnumDrawableGroup& drawables) {
  ASSERT(isValid());
  camera_->draw(drawables);
}

}  // namespace gfx
}  // namespace esp
