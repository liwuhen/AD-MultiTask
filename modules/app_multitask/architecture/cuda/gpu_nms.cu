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
#include "gpu_nms.hpp"

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom) {
  double eps = 1e-7;
  float cleft   = max(aleft, bleft);
  float ctop    = max(atop, btop);
  float cright  = min(aright, bright);
  float cbottom = min(abottom, bbottom);

  float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
  if (c_area == 0.0f) return 0.0f;

  float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
  float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
  return c_area / (a_area + b_area - c_area + eps);
}

static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold,
                                       int decode_bbox_dim) {
  // 如果测mAP的性能的时候， 只能采用cpu nms
  // 如果是日常推理， 则可以使用这个gpu nms
  int position = (blockDim.x * blockIdx.x + threadIdx.x);
  int count    = min(static_cast<int>(*bboxes), max_objects);  // *bboxes表示首地址的第一个元素。 count是bbox的数量。
  if (position >= count) return;

  // left, top, right, bottom, confidence, class, keepflag
  float* pcurrent = bboxes + 1 + position * decode_bbox_dim;  // +1是因为bboxes中第一个元素是记录bbox的数量的标识。
  for (int i = 0; i < count; ++i) {
    float* pitem = bboxes + 1 + i * decode_bbox_dim;
    if (i == position || pcurrent[5] != pitem[5]) continue;  // 剔除自己与类别不一致的框

    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && i < position) continue;

      float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                          pitem[2], pitem[3]);

      if (iou >= threshold) {
        pcurrent[6] = 0;  // 1=keep, 0=ignore
        return;
      }
    }
  }
}

void fast_nms_kernel_invoker(float* bboxes,
                             float nms_threshold, int max_objects,
                             int decode_bbox_dim, cudaStream_t stream) {
  auto block = max_objects > 512 ? 512 : max_objects;  // 每一个图像最多的框数量:max_objects
  auto grid  = (max_objects + block - 1) / block;
  fast_nms_kernel<<<grid, block, 0, stream>>>(bboxes, max_objects, nms_threshold, decode_bbox_dim);  // parray表示解码后的bbox
}
