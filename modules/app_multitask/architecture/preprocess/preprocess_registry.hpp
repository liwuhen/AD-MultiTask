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
#ifndef APP_MULTITASK_PREPROCESS_REGISTRY_H__
#define APP_MULTITASK_PREPROCESS_REGISTRY_H__

#include <opencv2/opencv.hpp>
#include "logger.h"
#include "parseconfig.h"
#include "warpaffine.hpp"
#include "function_registry.hpp"

namespace hpc {

namespace appinfer {

/**
 * @description: AffineMatrix.
 */
inline void CalAffineMatrix(cv::Mat& image,
    InfertMsg& input_msg,
    std::shared_ptr<ParseMsgs>& parsemsgs) {
    float scale_x = parsemsgs->dst_img_w_ / static_cast<float>(image.cols);
    float scale_y = parsemsgs->dst_img_h_ / static_cast<float>(image.rows);
    float scale   = min(scale_x, scale_y);

    input_msg.affineMatrix(0, 0) = scale;
    input_msg.affineMatrix(1, 1) = scale;
    input_msg.affineMatrix(0, 2) = -scale * image.cols * 0.5 + parsemsgs->dst_img_w_ * 0.5 + scale * 0.5 - 0.5;
    input_msg.affineMatrix(1, 2) = -scale * image.rows * 0.5 + parsemsgs->dst_img_h_ * 0.5 + scale * 0.5 - 0.5;

    input_msg.affineMatrix_cv = (cv::Mat_<float>(2, 3) << scale, 0.0,
        -scale * image.cols * 0.5 + parsemsgs->dst_img_w_ * 0.5 + scale * 0.5 - 0.5,
                                    0.0, scale,
        -scale * image.rows * 0.5 + parsemsgs->dst_img_h_ * 0.5 + scale * 0.5 - 0.5);

    // Compute inverse
    cv::invertAffineTransform(input_msg.affineMatrix_cv, input_msg.affineMatrix_inv_cv);
    input_msg.affineMatrix_inv = input_msg.affineMatrix.inverse();
    
}

/**
 * @description: A-YOLOM cpu calib.
 */
inline void PreprocessAYoloMCpuCalib(int current,
    int count,
    float* input_data_host,
    const std::vector<std::string>& files,
    std::shared_ptr<ParseMsgs>& parsemsgs) {

    GLOG_INFO("Calibrator Preprocess: "<<count<<" / "<<current);

    for ( auto& img : files ) {
        auto image = cv::imread(img);
        InfertMsg input_msg;
        CalAffineMatrix(image, input_msg, parsemsgs);
        cv::Mat input_image(parsemsgs->dst_img_h_, parsemsgs->dst_img_w_, CV_8UC3);
        // 对图像做平移缩放旋转变换，可逆
        cv::warpAffine(image, input_image, input_msg.affineMatrix_cv, input_image.size(), \
            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));

        int image_area = input_image.cols * input_image.rows;
        unsigned char* pimage = input_image.data;

        // HWC to CHW | BGR to RGB | 255
        float* phost_b = input_data_host + image_area * 0;
        float* phost_g = input_data_host + image_area * 1;
        float* phost_r = input_data_host + image_area * 2;
        for (int i = 0; i < image_area; ++i, pimage += 3) {
            // 注意这里的顺序 rgb 调换了
            *phost_r++ = pimage[0] / 255.0f;
            *phost_g++ = pimage[1] / 255.0f;
            *phost_b++ = pimage[2] / 255.0f;
        }
        input_data_host += image_area * 3;
    }
}

/**
 * @description: A-YOLOM gpu calib.
 */
inline void PreprocessAYoloMGpuCalib(int current,
    int count,
    float* dstimg,
    const std::vector<std::string>& files,
    std::shared_ptr<ParseMsgs>& parsemsgs) {

    GLOG_INFO("Calibrator Preprocess: "<<count<<" / "<<current);

    uint8_t* input_data_device;
    checkRuntime(cudaMalloc(&input_data_device, parsemsgs->srcimg_size_));

    for ( auto& img : files ) {

        auto image = cv::imread(img);
        InfertMsg input_msg;
        input_msg.image    = image.clone();
        input_msg.img_size = image.cols * image.rows * 3;
        input_msg.width    = image.cols;
        input_msg.height   = image.rows;
        CalAffineMatrix(image, input_msg, parsemsgs);

        checkRuntime(cudaMemcpy(input_data_device, input_msg.image.data,\
            input_msg.img_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
        warp_affine_bilinear(input_data_device, parsemsgs->batchsizes_,\
            input_msg, dstimg, parsemsgs->dst_img_w_, parsemsgs->dst_img_h_,\
            114, nullptr, AppTask::A_YOLOM_MODE);

        dstimg += input_msg.img_size;

    }
    checkRuntime(cudaFree(input_data_device));
}

/**
 * @description: A-YOLOM Cpu.
 */
inline void PreprocessAYoloMCpu(
    InfertMsg& input_msg,
    float* input_data_host,
    std::shared_ptr<ParseMsgs>& parsemsgs) {

    cv::Mat input_image(parsemsgs->dst_img_h_, parsemsgs->dst_img_w_, CV_8UC3);
    // 对图像做平移缩放旋转变换，可逆
    cv::warpAffine(input_msg.image, input_image, input_msg.affineMatrix_cv, input_image.size(), \
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));

    int image_area = input_image.cols * input_image.rows;
    unsigned char* pimage = input_image.data;

    // HWC to CHW | BGR to RGB | 255
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3) {
        // 注意这里的顺序 rgb 调换了
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }
}

/**
 * @description: A-YOLOM Gpu.
 */
inline void PreprocessAYoloMGpu(
    InfertMsg& input_msg,
    float* dstimg,
    uint8_t* input_data_device,
    std::shared_ptr<ParseMsgs>& parsemsgs) {

    warp_affine_bilinear(input_data_device, parsemsgs->batchsizes_,\
        input_msg, dstimg, parsemsgs->dst_img_w_, parsemsgs->dst_img_h_,\
        114, nullptr, AppTask::A_YOLOM_MODE);
}


// 全局自动注册
REGISTER_CALIBRATOR_FUNC("pre_a_yolom_cpu_calib", PreprocessAYoloMCpuCalib);
REGISTER_CALIBRATOR_FUNC("pre_a_yolom_gpu_calib", PreprocessAYoloMGpuCalib);
REGISTER_CALIBRATOR_FUNC("pre_a_yolom_cpu", PreprocessAYoloMCpu);
REGISTER_CALIBRATOR_FUNC("pre_a_yolom_gpu", PreprocessAYoloMGpu);

}  // namespace appinfer
}  // namespace hpc

#endif  // APP_MULTITASK_PREPROCESS_REGISTRY_H__
