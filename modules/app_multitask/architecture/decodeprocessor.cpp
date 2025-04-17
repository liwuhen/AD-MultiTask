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

#include "decodeprocessor.h"

namespace hpc {

namespace appinfer {

DecodeProcessor::DecodeProcessor() {}

DecodeProcessor::~DecodeProcessor() {}

/**
 * @description: init．
 */
bool DecodeProcessor::Init() {
  GLOG_INFO("[Init]: DecodeProcessor module init ");
  return true;
}

/**
 * @brief The inference algorithm handles threads．
 */
bool DecodeProcessor::RunStart() {
  GLOG_INFO("[RunStart]: DecodeProcessor module start ");
  return true;
}

/**
 * @description: Thread stop．
 */
bool DecodeProcessor::RunStop() {
  GLOG_INFO("[RunStop]: DecodeProcessor module stop ");
  return true;
}

/**
 * @description: Software function stops．
 */
bool DecodeProcessor::RunRelease() {
  GLOG_INFO("[RunRelease]: DecodeProcessor module release ");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool DecodeProcessor::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  } else {
    this->parsemsgs_ = nullptr;
    GLOG_ERROR("[SetParam]: DecodeProcessor module set param failed ");
    return false;
  }
  imgshape_["dst"] = make_pair(parsemsgs_->dst_img_h_, parsemsgs_->dst_img_w_);

  GLOG_INFO("[SetParam]: DecodeProcessor module set param ");
  return true;
}

/**
 * @description: Module resource release.
 */
bool DecodeProcessor::DataResourceRelease() {}

/**
 * @description: Inference
 */
bool DecodeProcessor::Inference(std::vector<float*>& predict,
    InfertMsg& infer_msg,
    std::vector<InfertMsg>& callbackMsg,
    std::shared_ptr<InferMsgQue>& bboxQueue) {
  imgshape_["src"] = make_pair(infer_msg.height, infer_msg.width);

  MultiTaskMsg multitask_result;
  Decode(predict, infer_msg, multitask_result);

  InfertMsg msg;
  msg = infer_msg;
  for (auto& box : multitask_result.box_result) {
    msg.bboxes.emplace_back(box);
  }
  bboxQueue->Push(msg);
  callbackMsg.emplace_back(msg);

  VisualizationMultiTask(false, infer_msg.image, infer_msg.index, multitask_result);

  return true;
}

/**
 * @description: Visualization
 */
void DecodeProcessor::VisualizationDet(cv::Mat& img, 
    vector<Box>& results) {
  
  for (auto& box : results) {
    cv::Scalar color;
    tie(color[0], color[1], color[2]) = random_color(box.label);

    // TODO: 根据配置文件，增加数据集选择功能
    auto name = bdd1ooklabels[box.label];
    auto caption = cv::format("%s %.2f", name, box.confidence);
    cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 1);
    cv::putText(img, caption, cv::Point(box.left, box.top - 3), 0, 0.5, color, 1, 16);
  }

}

/**
 * @description: Visualization seg
 */
void DecodeProcessor::VisualizationSeg(cv::Mat& img, 
  SegTask segmode, vector<uint8_t>& mask) {
    
  auto pimage  = img.ptr<cv::Vec3b>(0);
  int img_size = img.cols * img.rows;

  for ( int ind = 0; ind < img_size; ind++ ) {

    float foreground = (mask[ind] == 0) ? 0.0f : 0.5f;
    float background = 1 - foreground;

    if ( mask[ind] == 1 ) {
      for ( int pixelchannl = 0; pixelchannl < 3; ++pixelchannl) {
        float value;
        if ( segmode == SegTask::SEG_DRIVABLE ) {
          value = pimage[ind][pixelchannl] * background + foreground * selectColor[mask[ind]][pixelchannl];
        } else if ( segmode == SegTask::SEG_LANE ) {
          value = pimage[ind][pixelchannl] * background + foreground * selectColor[mask[ind] + 1][pixelchannl];
        }
        pimage[ind][pixelchannl] = static_cast<uchar>(std::min((int)value, 255));
      }
    }
  }
}

/**
 * @description: Visualization multi task
 */
void DecodeProcessor::VisualizationMultiTask(bool real_time,
    cv::Mat& img, int64_t timestamp, MultiTaskMsg& multitask_result) {
  
  // od vis
  VisualizationDet(img, multitask_result.box_result);
  // seg drivable vis
  VisualizationSeg(img, SegTask::SEG_DRIVABLE, multitask_result.seg_drivable);
  // seg lane vis
  VisualizationSeg(img, SegTask::SEG_LANE, multitask_result.seg_lane);

  if (real_time) {
    cv::imshow("Live Video", img);
    // 按 'q' 键退出
    if (cv::waitKey(30) >= 0) {
      return;
    }
  } else {
    std::string path = parsemsgs_->save_img_ + "/img_" + std::to_string(timestamp) + ".jpg";
    cv::imwrite(path, img);
  }
}

/**
 * @description: Bbox mapping to original map scale.
 */
void DecodeProcessor::ScaleBoxes(vector<Box>& box_result) {
  float gain  = min(imgshape_["dst"].first / static_cast<float>(imgshape_["src"].first),\
                imgshape_["dst"].second / static_cast<float>(imgshape_["src"].second));
  float pad[] = {(imgshape_["dst"].second - imgshape_["src"].second * gain) * 0.5, \
                (imgshape_["dst"].first - imgshape_["src"].first * gain) * 0.5};
  for (int index = 0; index < box_result.size(); index++) {
    box_result[index].left   = clamp((box_result[index].left - pad[0]) / gain, 0.0f, \
                               static_cast<float>(imgshape_["src"].second));
    box_result[index].right  = clamp((box_result[index].right - pad[0]) / gain, 0.0f, \
                               static_cast<float>(imgshape_["src"].second));
    box_result[index].top    = clamp((box_result[index].top - pad[1]) / gain, 0.0f, \
                               static_cast<float>(imgshape_["src"].first));
    box_result[index].bottom = clamp((box_result[index].bottom - pad[1]) / gain, 0.0f, \
                               static_cast<float>(imgshape_["src"].first));
  }
}

/**
 * @description: Cpu decode．
 */
void DecodeProcessor::Decode(std::vector<float*>& predict,
    InfertMsg& infer_msg, MultiTaskMsg& multitask_result) {

  auto postprocess = Registry::getInstance()->getRegisterFunc<InfertMsg&, MultiTaskMsg&,
                      std::vector<float*>&, std::shared_ptr<ParseMsgs>&>(parsemsgs_->postprocess_type_);
  postprocess(infer_msg, multitask_result, predict, parsemsgs_);
}


}  // namespace appinfer
}  // namespace hpc
