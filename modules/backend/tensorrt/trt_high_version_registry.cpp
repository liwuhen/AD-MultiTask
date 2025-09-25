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

#include "trt_high_version_registry.h"

namespace hpc {
namespace appinfer {

TrtHighVersionInfer::TrtHighVersionInfer() {}

TrtHighVersionInfer::~TrtHighVersionInfer() {}

/**
 * @description: init．
 */
bool TrtHighVersionInfer::Init() {

  // onnx model loading, trt model generation
  if (!isFileExists_stat(parsemsgs_->trt_path_)) {
    GLOG_ERROR("Trt model does not exist. ");
    if ( !BuildModel() ) {
      GLOG_ERROR("Build trt model failed. ");
      return false;
    }
  }

  // parse trt model
  if (!ParseModel()) {
    GLOG_ERROR("Parse trt model failed. ");
    return false;
  }

  GLOG_INFO("[Init]: Trt high version infer module init ");
  return true;
}

/**
 * @description: The inference algorithm handles threads．
 */
bool TrtHighVersionInfer::RunStart() {
  GLOG_INFO("[RunStart]: Trt high version infer module start ");
  return true;
}

/**
 * @description: Thread stop．
 */
bool TrtHighVersionInfer::RunStop() {
  GLOG_INFO("[RunStop]: Trt high version infer module stop ");
  return true;
}

/**
 * @description: Software function stops．
 */
bool TrtHighVersionInfer::RunRelease() {
  GLOG_INFO("[RunRelease]: TrtHighVersionInfer module release ");
  return true;
}

/**
 * @description: Configuration parameters.
 */
bool TrtHighVersionInfer::SetParam(shared_ptr<ParseMsgs>& parse_msgs) {
  if (parse_msgs != nullptr) {
    this->parsemsgs_ = parse_msgs;
  } else {
    this->parsemsgs_ = nullptr;
    GLOG_ERROR("[SetParam]: TrtHighVersionInfer module set param failed ");
    return false;
  }

  GLOG_INFO("[SetParam]: Trt high version infer module set param ");
  return true;
}

/**
 * @description: Module resource release.
 */
bool TrtHighVersionInfer::DataResourceRelease() {}

/**
 * @description: Inference.
 */
bool TrtHighVersionInfer::Inference(float* output_img_device) {
  checkRuntime(cudaMemcpy(input_buffers_[engine_name_size_[binding_names_["input"][0]].first],\
      output_img_device, parsemsgs_->dstimg_size_ * sizeof(float), cudaMemcpyDeviceToDevice));

  bool success = execution_context_->enqueueV3(stream_);
  if (!success) {
    GLOG_ERROR(" Inference failed ");
    return false;
  }

  if ( static_cast<DeviceMode>(parsemsgs_->postprocess_mode_) == DeviceMode::CPU_MODE ) {
    for (int index = 0; index < parsemsgs_->branch_num_; index++) {
      checkRuntime(cudaMemcpyAsync(output_buffers_[index], input_buffers_[engine_name_size_[binding_names_["output"][index]].first],\
        sizeof(float) * engine_name_size_[binding_names_["output"][index]].second, cudaMemcpyDeviceToHost, stream_));
    }
  } else if ( static_cast<DeviceMode>(parsemsgs_->postprocess_mode_) == DeviceMode::GPU_MODE ) {
    for (int index = 0; index < parsemsgs_->branch_num_; index++) {
      checkRuntime(cudaMemcpyAsync(output_buffers_[index], input_buffers_[engine_name_size_[binding_names_["output"][index]].first],\
        sizeof(float) * engine_name_size_[binding_names_["output"][index]].second, cudaMemcpyDeviceToDevice, stream_));
    }
  } else {
    GLOG_INFO("[Inference]: NPU mode.");
  }

  checkRuntime(cudaStreamSynchronize(stream_));
  return true;
}

/**
 * @description: Build trt model from onnx.
 */
bool TrtHighVersionInfer::BuildModel() {
  GLOG_INFO("=====> Build TensorRT Engine <===== ");

  // Configure builder, config and network.
  auto builder = make_nvshared(createInferBuilder(gLogger));
  if (!builder) {
    GLOG_ERROR("Can not create builder. ");
    return false;
  }

  auto config = make_nvshared(builder->createBuilderConfig());
  if (!config) {
    GLOG_ERROR("Can not create config. ");
    return false;
  }

  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = make_nvshared(builder->createNetworkV2(explicitBatch));
  if (!network) {
    GLOG_ERROR("Can not create network. ");
    return false;
  }

  auto parser = make_nvshared(nvonnxparser::createParser(*network, gLogger));
  if (!parser->parseFromFile(parsemsgs_->onnx_path_.c_str(), 1)) {
    GLOG_ERROR("Failed to parse " << parsemsgs_->onnx_path_.c_str());
    return false;
  }


  auto profile      = builder->createOptimizationProfile();
  auto input_tensor = network->getInput(0);
  auto input_dims   = input_tensor->getDimensions();
  int max_workspace_size = 1 << 30;

  // Configure model precision
  switch ((ModelACC)parsemsgs_->model_acc_) {
    case ModelACC::MODEL_FLOAT32:
      if (builder->platformHasTf32()) {
        config->setFlag(BuilderFlag::kTF32);
      }
      break;
    case ModelACC::MODEL_FLOAT16:
      if (builder->platformHasFastFp16()) {
        config->setFlag(BuilderFlag::kFP16);
      } else { GLOG_ERROR("Platform not have fast fp16 support. "); }
      break;
    case ModelACC::MODEL_INT8:
      if (builder->platformHasFastInt8()) {
        config->setFlag(BuilderFlag::kINT8);
      } else { GLOG_ERROR("Platform not have fast int8 support. "); }
      break;
    default:
      break;
  }
  GLOG_INFO("Build model acc[0-fp32, 1-fp16, 2-int8]:  " << parsemsgs_->model_acc_);

  // Configure qat quantize
  if ( parsemsgs_->quantize_flag_ && (ModelACC)parsemsgs_->model_acc_ == ModelACC::MODEL_INT8 ) {
    GLOG_INFO("Build qat model. ");

    auto preprocess = Registry::getInstance()->getRegisterFunc<int,
                      int, float*, const std::vector<std::string>&,
                      std::shared_ptr<ParseMsgs>&>(parsemsgs_->calib_preprocess_type_);

    // Configure int8 calibration data reading tool
    hasEntropyCalibrator_ = false;
    std::vector<std::string> calib_files;
    std::vector<uint8_t> calib_data;
    if (!parsemsgs_->quantize_data_.empty()) {
      LoadCalibDataFile(parsemsgs_->quantize_data_, calib_files);
      if (calib_files.empty()) {
        GLOG_ERROR("Calibrator data file is empty. ");
      }
    }

    if (!parsemsgs_->calib_table_path_.empty()) {
      if (isFileExists_stat(parsemsgs_->calib_table_path_)) {
        calib_data = LoadFile(parsemsgs_->calib_table_path_);
        if (calib_data.empty()) {
          GLOG_ERROR("entropyCalibratorFile is exit, but file is empty. ");
          return false;
        }
        hasEntropyCalibrator_ = true;
      }
    }

    // Configure int8 calibrator tool
    auto calibratorDims = input_dims;
        calibratorDims.d[0] = parsemsgs_->calib_batchsize_;
    calib_ = createObject<Int8EntropyCalibrator>("Int8EntropyCalibrator");
    if (hasEntropyCalibrator_) {
      GLOG_INFO("Using entropy calibrator data:  " <<calib_data.size());
      calib_->Init(
          calib_data,      // calibration data
          calibratorDims,  // model input dimensions
          preprocess,      // preprocess
          parsemsgs_       // model parameter structure
      );
    } else {
      GLOG_INFO("Using calibrator image: " <<calib_files.size());
      calib_->Init(
          calib_files,     // calibration files datasets
          calibratorDims,  // model input dimensions
          preprocess,      // preprocess
          parsemsgs_       // model parameter structure
      );
    }
    config->setInt8Calibrator(calib_.get());
    calib_->MemFree();
  }

  // TensorRt info
  {
    GLOG_INFO("Input shape is " <<join_dims(vector<int>(input_dims.d, \
        input_dims.d + input_dims.nbDims)).c_str());
    GLOG_INFO("Set max batch size = " <<parsemsgs_->max_batchsize_);
    GLOG_INFO("Set max workspace size = " <<max_workspace_size / 1024.0f / 1024.0f<<"MB");

    int net_num_input = network->getNbInputs();
    GLOG_INFO("Network has "<<net_num_input<<" inputs:");
    std::vector<std::string> input_names(net_num_input);
    for ( int i = 0; i < net_num_input; ++i ) {
      auto tensor   = network->getInput(i);
      auto dims     = tensor->getDimensions();
      auto dims_str = join_dims(vector<int>(dims.d, dims.d+dims.nbDims));
      GLOG_INFO("      "<<i<<".["<<tensor->getName()<<"]"<<" shape is "<<dims_str.c_str());

      input_names[i] = tensor->getName();
    }

    int net_num_output = network->getNbOutputs();
    GLOG_INFO("Network has "<<net_num_output<<" outputs:");
    for ( int i = 0; i < net_num_output; ++i ) {
      auto tensor   = network->getOutput(i);
      auto dims     = tensor->getDimensions();
      auto dims_str = join_dims(vector<int>(dims.d, dims.d+dims.nbDims));
      GLOG_INFO("      "<<i<<".["<<tensor->getName()<<"]"<<" shape is "<<dims_str.c_str());
    }
  }

  builder->setMaxBatchSize(parsemsgs_->max_batchsize_);
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, max_workspace_size);

  // Configure dynamic shape
  if ( (BatchMode)parsemsgs_->batch_mode_ == BatchMode::DYNAMIC_MODE ) {
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = parsemsgs_->max_batchsize_;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);
  }

  GLOG_INFO("Infer batch mode: [0-static, 1-dynamic]:  " << parsemsgs_->batch_mode_);

  auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
  if (engine == nullptr) {
    GLOG_ERROR("Build engine failed. ");
    return false;
  }

  // 将模型序列化，并储存为文件
  auto seridata = make_nvshared(engine->serialize());
  save_file(parsemsgs_->trt_path_, seridata->data(), seridata->size());
  GLOG_INFO("Build engine success.  ");
  return true;
}

/**
 * @description: Parse model.
 */
bool TrtHighVersionInfer::ParseModel() {
  GLOG_INFO("=====> Begin Deserialize Engine <===== ");
  checkRuntime(cudaStreamCreate(&stream_));
  auto engine_data = LoadFile(parsemsgs_->trt_path_);
  if (engine_data.empty()) {
    GLOG_ERROR("Build engine failed  " << parsemsgs_->trt_path_);
    return false;
  }

  auto runtime = unique_ptr<IRuntime, NvInferDeleter>(createInferRuntime(gLogger));
  auto engine  = unique_ptr<ICudaEngine, NvInferDeleter>(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr));
  if (engine == nullptr) {
    GLOG_ERROR("Deserialize cuda engine failed! ");
    runtime->destroy();
    return false;
  }

  execution_context_ = std::unique_ptr<IExecutionContext, NvInferDeleter>(engine->createExecutionContext());
  if (execution_context_ == nullptr) {
    GLOG_ERROR("Failed to create context! ");
    return false;
  }

  int nb_bindings = engine->getNbIOTensors();
  int in_size = 0, out_size = 0;
  for ( int i = 0; i < nb_bindings; i++ ) {
    size_t size(1);
    auto name = engine->getIOTensorName(i);
    auto infer_dim = engine->getTensorShape(name);
    bool interface_type = (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);

    switch ( interface_type ) {
      case false:
        out_size++;
        binding_names_["output"].push_back(std::string(name));
        break;
      case true:
        in_size++;
        binding_names_["input"].push_back(std::string(name));

        // 明确当前推理时，使用的数据输入大小
        infer_dim.d[0] = parsemsgs_->batchsizes_;
        execution_context_->setInputShape(name, infer_dim);

        break;
    }

    for (int j = 0; j < infer_dim.nbDims; j++) {
      size *= infer_dim.d[j];
    }

    in_out_size_["input"]  = in_size;
    in_out_size_["output"] = out_size;

    engine_name_size_.emplace(std::string(name), make_pair(i, size));
  }

  // allocator memcory
  if ( !MemAllocator() ) {
    GLOG_ERROR("Memory allocator failed. ");
    return false;
  }

  // 输入和输出传递 TensorRT 缓冲区
  for ( int32_t i = 0; i < nb_bindings; i++ )
  {
    auto name = engine->getIOTensorName(i);
    execution_context_->setTensorAddress(name, input_buffers_[engine_name_size_[name].first]);
  }

  GLOG_INFO("[Model]: " + std::string(MODEL_FLAG));

  return true;
}

/**
 * @description: Memory allocator.
 */
bool TrtHighVersionInfer::MemAllocator() {
  GLOG_INFO("Begin allocator memory ");

  input_buffers_.resize(in_out_size_["input"] + in_out_size_["output"]);
  output_buffers_.resize(in_out_size_["output"]);  //

  // Allocate model input memory
  for (int i = 0; i < in_out_size_["input"]; i++) {
    checkRuntime(cudaMalloc(&input_buffers_[i], sizeof(float) * engine_name_size_[binding_names_["input"][i]].second));
  }

  // Allocating memory for output data.（host）
  for (int i = 0; i < in_out_size_["output"]; i++) {

    auto out_node_size = engine_name_size_[binding_names_["output"][i]].second;
    if ( static_cast<DeviceMode>(parsemsgs_->postprocess_mode_) == DeviceMode::CPU_MODE ) {
      checkRuntime(cudaMallocHost(&output_buffers_[i], sizeof(float) * out_node_size));
    } else if ( static_cast<DeviceMode>(parsemsgs_->postprocess_mode_) == DeviceMode::GPU_MODE ) {
      checkRuntime(cudaMalloc(&output_buffers_[i], sizeof(float) * out_node_size));
    } else {
      GLOG_INFO("[MemAllocator]: NPU mode.");
    }

    int out_index = in_out_size_["input"] + i;
    checkRuntime(cudaMalloc(&input_buffers_[out_index], sizeof(float) * out_node_size));

  }

  GLOG_INFO("Memory allocator done ");
  return true;
}

/**
 * @description: Cpu and gpu memory free.
 */
bool TrtHighVersionInfer::MemFree() {
  // free memory
  checkRuntime(cudaStreamDestroy(stream_));
  stream_ = nullptr;

  for (int out_index = 0; out_index < in_out_size_["output"]; out_index++) {
    if ( static_cast<DeviceMode>(parsemsgs_->postprocess_mode_) == DeviceMode::CPU_MODE ) {
      checkRuntime(cudaFreeHost(output_buffers_[out_index]));
    } else if ( static_cast<DeviceMode>(parsemsgs_->postprocess_mode_) == DeviceMode::GPU_MODE ) {
      checkRuntime(cudaFree(output_buffers_[out_index]));
    } else {
      GLOG_INFO("[MemFree]: NPU mode.");
    }
    output_buffers_[out_index] = nullptr;
  }

  for (int index = 0; index < in_out_size_["input"] + in_out_size_["output"]; index++) {
    checkRuntime(cudaFree(input_buffers_[index]));
    input_buffers_[index] = nullptr;
  }

  return true;
}

/**
 * @description: Get output buffer.
 */
const std::vector<float*>& TrtHighVersionInfer::GetOutputBuffer() const {
  return output_buffers_;
}

/**
 * @description: Load file.
 */
std::vector<uint8_t> TrtHighVersionInfer::LoadFile(const string& file) {
  ifstream in(file, ios::in | ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, ios::end);
  size_t length = in.tellg();
  vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, ios::beg);
    data.resize(length);

    in.read(reinterpret_cast<char*>(&data[0]), length);
  }
  in.close();
  return data;
}

/**
 * @description: Load image file.
 */
void TrtHighVersionInfer::LoadCalibDataFile(const std::string& path,
    std::vector<string>& data) {
  // 支持的图像扩展名
  static const std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};

  try {
    // RAII 管理目录资源
    FileSystem::DirectoryHandle dirHandle(path);

    // 清空输入向量并预估容量
    data.clear();

    struct dirent* entry;
    while ((entry = readdir(dirHandle.get())) != nullptr) {
        // 跳过 "." 和 ".." 目录
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            std::string filename = entry->d_name;
            std::string fullPath = path + (path.back() != '/' ? "/" : "") + filename;

            // 获取文件扩展名并转为小写
            std::string ext;
            size_t pos = filename.find_last_of('.');
            if (pos != std::string::npos) {
                ext = filename.substr(pos);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            }

            // 检查是否是图像文件
            if (!ext.empty() && std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                data.push_back(fullPath);
            }
        }
    }
  } catch (const std::runtime_error& e) {
    GLOG_ERROR(" Filesystem error:  "<< e.what());
    throw;  // 重新抛出，让调用者处理
  } catch (const std::exception& e) {
    GLOG_ERROR(" Error:  "<< e.what());
    throw;
  }

}

REGISTER_CLASS("TrtHighVersionInfer", TrtHighVersionInfer);

}  // namespace appinfer
}  // namespace hpc
