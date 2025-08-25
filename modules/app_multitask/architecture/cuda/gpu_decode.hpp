#ifndef APP_MULTITASK_GPU_DECODE_H__
#define APP_MULTITASK_GPU_DECODE_H__

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "task_struct.hpp"
#include "parseconfig.h"


using namespace hpc::common;

void decode_kernel_invoker(float* predict, int num_bboxes,
                           int num_classes, int bbox_dim,
                           float confidence_threshold, int max_objects,
                           Eigen::Matrix3f& invert_affine_matrix, float* parray,
                           int decode_bbox_dim, cudaStream_t stream);

#endif // APP_MULTITASK_GPU_DECODE_H__
