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
#include <stdio.h>
#include <iostream>
#include "gpu_decode.hpp"

static __device__ void affine_project(Eigen::Matrix3f& matrix, float x, float y, float* ox, float* oy) {
  *ox = matrix(0, 0) * x + matrix(0, 2);
  *oy = matrix(1, 1) * y + matrix(1, 2);
}

static __global__ void decode_kernel(  // 每次处理一个框，每个线程并行处理。
    float* predict, int num_bboxes, int num_classes, int bbox_dim, float confidence_threshold,
    Eigen::Matrix3f& invert_affine_matrix, float* parray, int max_objects, int decode_bbox_dim) {

  volatile int position = (blockDim.x * blockIdx.x + threadIdx.x);  // 1D线程索引
  if (position >= num_bboxes) return;

  // vscode调试控制台查看指针内容 *(@global float(*)[10])pitem表示查看前10个地址内容
  // 核函数地址，变量可以存储在寄存器中或local, shared, const 或者 global的内存
  // predict：85（5分别对应的是cx,cy,w,h,obj_conf，分别代表的含义是边界框中心点坐标、宽高、边界框内包含物体的置信度。80对应的是COCO数据中的80个类别。）
  float* pitem = predict + bbox_dim * position;  // pitem指向每个框的初始地址（每一行的首地址）

  float* class_confidence = pitem + 4;     // classification 对应的首地址（第一个类别）
  float confidence = *class_confidence++;  // 第一个类别地址对应的预测的类别概率值
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {  // 查找预测的80类别概率值中最大的类
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }

  if (confidence < confidence_threshold) return;

  // volatile 编译器对访问该变量的代码就不再进行优化，从而可以提供对特殊地址的稳定访问
  // 这里的parray是一段连续的内存空间数组， atomicAdd(parray,
  // 1)的功能是parray中每个元素都同时加1，并返回old_value。
  int index = 0;
  index = atomicAdd(parray, 1);  // 计算(old + val)， 函数将返回old 地址对应的新值;
                                 // 所有线程在这里会阻塞，在共享内存parray中进行计算。

  if (index >= max_objects) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width  = *pitem++;
  float height = *pitem++;
  float left   = cx - width  * 0.5f;
  float top    = cy - height * 0.5f;
  float right  = cx + width  * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left,  top,    &left,  &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  // left, top, right, bottom, confidence, class, keepflag
  // 这里的+1表示存放的bbox的数量count的信息。
  float* pout_item = parray + 1 + index * decode_bbox_dim;  // pout_item是output_device的每一个box信息的首地址。存在parray对应的内存地址空间中。
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;  // 1 = keep, 0 = ignore

}

void decode_kernel_invoker(float* predict, int num_bboxes,
                           int num_classes, int bbox_dim,
                           float confidence_threshold, int max_objects,
                           Eigen::Matrix3f& invert_affine_matrix, float* parray,
                           int decode_bbox_dim, cudaStream_t stream) {

  // num_bboxes的数量框，则需要num_bboxes的线程数（一线程处理一个框）。
  auto block = num_bboxes > 512 ? 512 : num_bboxes;
  auto grid  = (num_bboxes + block - 1) / block;  // +block -1是向上取整数，保证系统设定的线程数大于程序的需要。

  decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, bbox_dim,
                                            confidence_threshold, invert_affine_matrix,
                                            parray, max_objects, decode_bbox_dim);
}
