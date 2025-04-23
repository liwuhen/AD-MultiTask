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

#include <cuda_runtime.h>
#include "gpu_seg.hpp"
#include "utils.hpp"

using namespace hpc::appinfer;

static __global__ void seg_decode_kernel(
    uint32_t* seg_drivable_data,
    uint32_t* seg_lane_data,
    uint8_t* seg_lane,
    uint8_t* seg_drivable,
    int height,
    int width,
    Eigen::Matrix3f& affine_matrix,
    int seg_width,
    int seg_height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate source index
    int src_index = y * width + x;
    
    // Calculate destination coordinates using affine transformation
    float dst_x = affine_matrix(0, 0) * x + affine_matrix(0, 2);
    float dst_y = affine_matrix(1, 1) * y + affine_matrix(1, 2);
    
    // Round to nearest integer for indexing
    int dst_x_int = roundf(dst_x);
    int dst_y_int = roundf(dst_y);

    // Add boundary check for destination coordinates
    if (dst_x_int < 0 || dst_x_int >= seg_width || 
        dst_y_int < 0 || dst_y_int >= seg_height) return;
    
    // Calculate destination index
    int dst_index = dst_y_int * seg_width + dst_x_int;
    
    // Get segmentation values
    int drivable_int = seg_drivable_data[dst_index];
    int lane_int = seg_lane_data[dst_index];
    
    // Write results
    if (drivable_int == 1) {
        seg_drivable[src_index] = uint8_t(drivable_int);
    }
    
    if (lane_int == 1) {
        seg_lane[src_index] = uint8_t(lane_int);
    }
}

void SemanticSeg(
    InfertMsg& infer_msg,
    std::vector<float*>& predict,
    std::vector<uint8_t*>& seg_data_device,
    std::shared_ptr<ParseMsgs>& parsemsgs) {
    
    // Get segmentation data
    auto seg_drivable_data = reinterpret_cast<uint32_t*>(predict[0]);
    auto seg_lane_data = reinterpret_cast<uint32_t*>(predict[1]);

    // Set up grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((infer_msg.width + block.x - 1) / block.x,
              (infer_msg.height + block.y - 1) / block.y);
    
    // Launch kernel
    seg_decode_kernel<<<grid, block, 0, nullptr>>>(
        seg_drivable_data,
        seg_lane_data,
        seg_data_device[0],
        seg_data_device[1],
        infer_msg.height,
        infer_msg.width,
        infer_msg.affineMatrix,
        parsemsgs->segda_predict_dim_[0][3],
        parsemsgs->segda_predict_dim_[0][2]
    );
}
