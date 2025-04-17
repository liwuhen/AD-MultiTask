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
#ifndef APP_MULTITASK_POSTPROCESS_REGISTRY_H__
#define APP_MULTITASK_POSTPROCESS_REGISTRY_H__

#include <opencv2/opencv.hpp>
#include "logger.h"
#include "parseconfig.h"
#include "nms_registry.hpp"

namespace hpc {

namespace appinfer {

/**
 * @description: A-YOLOM det cpu.
 */
inline void AYoloMDetectCpuAchorFree(
    InfertMsg& infer_msg,
    std::vector<Box>& box_result,
    std::vector<float*>& predict,
    std::shared_ptr<ParseMsgs>& parsemsgs) {
    
    vector<Box> boxes;
    int num_classes = parsemsgs->det_predict_dim_[0][2] - 4;
    for (int i = 0; i < parsemsgs->det_predict_dim_[0][1]; ++i)
    {
        float* pitem  = predict[2] + i * parsemsgs->det_predict_dim_[0][2];
        float* pclass = pitem + 4;

        int label  = std::max_element(pclass, pclass + num_classes) - pclass;
        float prob = pclass[label];
        float confidence = prob;    // anchor free
        if (confidence < parsemsgs->obj_threshold_) continue;

        float cx     = pitem[0];
        float cy     = pitem[1];
        float width  = pitem[2];
        float height = pitem[3];
        float left   = cx - width  * 0.5;
        float top    = cy - height * 0.5;
        float right  = cx + width  * 0.5;
        float bottom = cy + height * 0.5;

        // 输入图像层级模型预测框 ==> 映射回原图上尺寸
        float image_left   = infer_msg.affineMatrix_inv(0, 0) * (left   - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_top    = infer_msg.affineMatrix_inv(1, 1) * (top    - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);
        float image_right  = infer_msg.affineMatrix_inv(0, 0) * (right  - infer_msg.affineVec(0)) \
                            + infer_msg.affineMatrix_inv(0, 2);
        float image_bottom = infer_msg.affineMatrix_inv(1, 1) * (bottom - infer_msg.affineVec(1)) \
                            + infer_msg.affineMatrix_inv(1, 2);

        if ( image_left < 0 || image_top< 0 ) {
            continue;
        }

        boxes.emplace_back(image_left, image_top, image_right, image_bottom, confidence, label);
    }

    auto nms = Registry::getInstance()->getRegisterFunc<float,
                std::vector<Box>&, std::vector<Box>&>(parsemsgs->nms_type_);

    nms(parsemsgs->nms_threshold_, boxes, box_result);
}

/**
 * @description: A-YOLOM seg cpu.
 */
inline void AYoloMSegCpuAchorFree(
    InfertMsg& infer_msg,
    std::vector<uint8_t>& seg_lane,
    std::vector<uint8_t>& seg_drivable,
    std::vector<float*>& predict,
    std::shared_ptr<ParseMsgs>& parsemsgs) {
    
    // seg drivable && lane
    // TODO: CUDA 
    auto seg_drivable_data = reinterpret_cast<uint32_t*>(predict[0]);
    auto seg_lane_data     = reinterpret_cast<uint32_t*>(predict[1]);
    seg_lane.resize((infer_msg.height * infer_msg.width) - 1, 0);
    seg_drivable.resize((infer_msg.height * infer_msg.width) -1, 0);
    for ( int ind_h = 0; ind_h < infer_msg.height; ind_h++ ) {
        for ( int ind_w = 0; ind_w < infer_msg.width; ind_w++ ) {
            float dst_index_x = infer_msg.affineMatrix(0, 0) * ind_w  + infer_msg.affineMatrix(0, 2);
            float dst_index_y = infer_msg.affineMatrix(1, 1) * ind_h  + infer_msg.affineMatrix(1, 2);
            int src_index     = ind_h * infer_msg.width + ind_w;
            int dst_index     = round(dst_index_y) * parsemsgs->segda_predict_dim_[0][3] + round(dst_index_x);
            int drivable_int  = seg_drivable_data[dst_index];
            int lane_int      = seg_lane_data[dst_index];
            if ( drivable_int == 1 ) {
                seg_drivable[src_index] = uint8_t(drivable_int);
            }

            if ( lane_int == 1 ) {
                seg_lane[src_index] = uint8_t(lane_int);
            } 
        }
    }
}

/**
 * @description: A-YOLOM cpu postprocess anchor free.
 */
inline void PostprocessAYoloMCpuAchorFree(
    InfertMsg& infer_msg,
    MultiTaskMsg& multitask_result,
    std::vector<float*>& predict,
    std::shared_ptr<ParseMsgs>& parsemsgs) {

    // bbox decode 
    AYoloMDetectCpuAchorFree(infer_msg, multitask_result.box_result, predict, parsemsgs);

    // seg decode
    AYoloMSegCpuAchorFree(infer_msg, multitask_result.seg_lane, multitask_result.seg_drivable, predict, parsemsgs);
}


// 全局自动注册
REGISTER_CALIBRATOR_FUNC("post_a_yolom_cpu_anchorfree", PostprocessAYoloMCpuAchorFree);

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_MULTITASK_POSTPROCESS_REGISTRY_H__
