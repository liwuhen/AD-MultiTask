#ifndef APP_MULTITASK_GPU_SEG_H__
#define APP_MULTITASK_GPU_SEG_H__

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "task_struct.hpp"
#include "parseconfig.h"


using namespace hpc::common;

void SemanticSeg(
    InfertMsg& infer_msg,
    std::vector<float*>& predict,
    std::vector<uint8_t*>& seg_data_device,
    std::shared_ptr<ParseMsgs>& parsemsgs);

#endif // APP_MULTITASK_GPU_SEG_H__ 