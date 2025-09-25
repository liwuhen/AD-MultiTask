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

#ifndef APP_MULTITASK_TRT_INFER_H__
#define APP_MULTITASK_TRT_INFER_H__

#include <glog/logging.h>
#include <math.h>
#include <stdio.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "common.hpp"
#include "enum_msg.h"
#include "glog_msg.h"
#include "backend_module.h"
#include "parseconfig.h"
#include "std_buffer.h"
#include "std_cmake.h"
#include "task_struct.hpp"
#include "utils.hpp"
#include "file.hpp"
#include "calibrator.hpp"
#include "function_registry.hpp"
#include "class_factory.h"
#include "trt_lower_version_registry.h"
#include "trt_high_version_registry.h"

/**
 * @namespace hpc::appinfer
 * @brief hpc::appinfer
 */
namespace hpc {
namespace appinfer {

using namespace std;
using namespace hpc::common;

/**
 * @class TrtInfer.
 * @brief Trt model infer.
 */
class TrtInfer : public BackendModuleBase {
 public:
  TrtInfer();
  ~TrtInfer();

  /**
   * @brief     init．
   * @param[in] void．
   * @return    bool.
   */
  bool Init() override;

  /**
   * @brief     The inference algorithm handles threads．
   * @param[in] void．
   * @return    bool.
   */
  bool RunStart() override;

  /**
   * @brief     Thread stop．
   * @param[in] void．
   * @return    bool.
   */
  bool RunStop() override;

  /**
   * @brief     Software function stops．
   * @param[in] void．
   * @return    bool.
   */
  bool RunRelease() override;

  /**
   * @brief     Cpu and gpu memory free.
   * @param[in] void．
   * @return    bool.
   */
  bool MemFree() override;

  /**
   * @brief     Inference.
   * @param[in] float*.
   * @return    bool.
   */
  bool Inference(float* output_img_device) override;

  /**
   * @brief     Configuration parameters.
   * @param[in] shared_ptr<ParseMsgs>&.
   * @return    bool.
   */
  bool SetParam(shared_ptr<ParseMsgs>& parse_msgs) override;

  /**
   * @brief     Get output buffer.
   * @param[in] ．
   * @return    const std::vector<float*>&.
   */
  const std::vector<float*>& GetOutputBuffer() const override;

 private:
  /**
   * @brief     Module resource release.
   * @param[in] void．
   * @return    bool.
   */
  bool DataResourceRelease();

  /**
   * @brief     Generate smart pointer for nvidia function.
   * @param[in] _T．
   * @return    _T.
   */
  template <typename _T>
  shared_ptr<_T> make_nvshared(_T* ptr) {
    return shared_ptr<_T>(ptr, [](_T* p) { p->destroy(); });
  }

 private:
  std::shared_ptr<ParseMsgs> parsemsgs_;
  std::shared_ptr<BackendModuleBase> backend_trt_infer_;
};

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_MULTITASK_TRT_INFER_H__
