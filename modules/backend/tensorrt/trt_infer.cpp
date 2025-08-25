/* ==================================================================
* Copyright (c) 2024, LiWuHen.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an
 BASIS
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ===================================================================
*/

#include "trt_infer.h"

namespace hpc {
namespace appinfer {

TrtInfer::TrtInfer() {}

TrtInfer::~TrtInfer() {}

/**
 * @description: init．
 */
bool TrtInfer::Init() {

  if ((TrtVersion)parsemsgs_->trt_version_ == TrtVersion::TRT_LOWER_VERSION) {
    trt_lower_version_infer_ = createObject<TrtLowerVersionInfer>("TrtLowerVersionInfer");
    trt_lower_version_infer_->SetParam(parsemsgs_);
    trt_lower_version_infer_->Init();
  } else if ((TrtVersion)parsemsgs_->trt_version_ == TrtVersion::TRT_HIGH_VERSION) {
    trt_high_version_infer_  = createObject<TrtHighVersionInfer>("TrtHighVersionInfer");
    trt_high_version_infer_->SetParam(parsemsgs_);
    trt_high_version_infer_->Init();
  } else {
    GLOG_ERROR("[Init]: Trt infer module init failed ");
    return false;
  }

  GLOG_INFO("[Init]: Trt infer module init ");
  return true;
}

/**
 * @description: The inference algorithm handles threads．
 */
bool TrtInfer::RunStart() {
  GLOG_INFO("[RunStart]: Trt infer module start ");
  return true;
}

/**
 * @description: Thread stop．
 */
bool TrtInfer::RunStop() {
  GLOG_INFO("[RunStop]: Trt infer module stop ");
  return true;
}

/**
 * @description: Software function stops．
 */
bool TrtInfer::RunRelease() {
  GLOG_INFO("[RunRelease]: TrtInfer module release ");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool TrtInfer::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  } else {
    this->parsemsgs_ = nullptr;
    GLOG_ERROR("[SetParam]: TrtInfer module set param failed ");
    return false;
  }

  GLOG_INFO("[SetParam]: Trt infer module set param ");
  return true;
}

/**
 * @description: Module resource release.
 */
bool TrtInfer::DataResourceRelease() {}

/**
 * @description: Inference.
 */
bool TrtInfer::Inference(float* output_img_device) {
  if ((TrtVersion)parsemsgs_->trt_version_ == TrtVersion::TRT_LOWER_VERSION) {
    trt_lower_version_infer_->Inference(output_img_device);
  } else if ((TrtVersion)parsemsgs_->trt_version_ == TrtVersion::TRT_HIGH_VERSION) {
    trt_high_version_infer_->Inference(output_img_device);
  } else {
    GLOG_ERROR("[Inference]: Trt infer module inference failed ");
    return false;
  }

  return true;
}

/**
 * @description: Get output buffer.
 */
const std::vector<float*>& TrtInfer::GetOutputBuffer() const {
  if ((TrtVersion)parsemsgs_->trt_version_ == TrtVersion::TRT_LOWER_VERSION) {
    return trt_lower_version_infer_->output_buffers_;
  } else if ((TrtVersion)parsemsgs_->trt_version_ == TrtVersion::TRT_HIGH_VERSION) {
    return trt_high_version_infer_->output_buffers_;
  }
}

/**
 * @description: Cpu and gpu memory free.
 */
bool TrtInfer::MemFree() {
  // free memory
  if ((TrtVersion)parsemsgs_->trt_version_ == TrtVersion::TRT_LOWER_VERSION) {
    trt_lower_version_infer_->MemFree();
  } else if ((TrtVersion)parsemsgs_->trt_version_ == TrtVersion::TRT_HIGH_VERSION) {
    trt_high_version_infer_->MemFree();
  } else {
    GLOG_ERROR("[MemFree]: Trt infer module mem free failed ");
    return false;
  }
  return true;
}

}  // namespace appinfer
}  // namespace hpc
