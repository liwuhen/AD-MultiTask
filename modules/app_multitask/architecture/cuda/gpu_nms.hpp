#ifndef APP_MULTITASK_GPU_NMS_H__
#define APP_MULTITASK_GPU_NMS_H__

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "task_struct.hpp"
#include "parseconfig.h"


using namespace hpc::common;

void fast_nms_kernel_invoker(float* bboxes,
                             float nms_threshold, int max_objects,
                             int decode_bbox_dim, cudaStream_t stream);

#endif // APP_MULTITASK_GPU_NMS_H__ 