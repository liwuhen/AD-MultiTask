<div align="center">

<img src="./docs/images" width="500" height="140">

<h2 align="center">AI model deployment based on NVIDIA and Qualcomm platforms</h2>


[<span style="font-size:20px;">**Architecture**</span>](./docs/framework.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[<span style="font-size:20px;">**Documentation**</span>](https://liwuhen.cn/CVDeploy-2D)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[<span style="font-size:20px;">**Blog**</span>](https://www.zhihu.com/column/c_1839603173800697856)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[<span style="font-size:20px;">**Roadmap**</span>](./docs/roadmap.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[<span style="font-size:20px;">**Slack**</span>](https://app.slack.com/client/T07U5CEEXCP/C07UKUA9TCJ)


---

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)
![ARM Linux](https://img.shields.io/badge/ARM_Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![Qualcomm](https://img.shields.io/badge/Qualcomm-3253DC?style=for-the-badge&logo=qualcomm&logoColor=white)
![Parallel Computing](https://img.shields.io/badge/Parallel-Computing-orange?style=for-the-badge)
![HPC](https://img.shields.io/badge/HPC-High%20Performance%20Computing-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMiAxN3YtMmgtM3YtM2gydi0yaDJ2LTJoLTR2N2gtN3YtN0g4djhoLTNWM0gzdjE4aDE4di00eiIvPjwvc3ZnPg==)
![Performance](https://img.shields.io/badge/Performance-Optimized-red?style=for-the-badge)
![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

This repository primarily provides inference capabilities for multi-task networks in both 2D and 3D. It includes packaged libraries to support daily development, integration, testing, and inference. The framework implements multithreading, the singleton pattern, and producer-consumer patterns. It also supports cache log analysis.
</div>

# ![third-party](https://img.shields.io/badge/third-party-blue) Third-party Libraries

|Libraries|Eigen|Gflags|Glog|Yaml-cpp|Cuda|Cudnn|Tensorrt|Opencv|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Version|3.4|2.2.2|0.6.0|0.8.0|11.4|8.4|8.4|3.4.5|

# Getting Started
Visit our documentation to learn more.
- [Installation](./docs/hpcdoc/source/getting_started/installation.md)
- [Quickstart](./docs/hpcdoc/source/getting_started/Quickstart.md)
- [Supported Models](./docs/hpcdoc/source/algorithm/Supported_Models.md)
- [Supported Object Tracking](./docs/hpcdoc/source/algorithm/Supported_Object_Tracking.md)

# Performances
- Dataset: 
    - BDD100K
        > The validation dataset is BDD100K, which contains 70000 training samples and 10000 val samples. All models in the table were trained on the full BDD100K dataset.
    - nuscenes
        > The validation dataset is nuscenes-mini. All models in the table were trained on the full nuscenes dataset.
- Model: The deployed model is the 's' version of the YOLO multi-task network series.
- Quantize: Quantization was performed using NVIDIA's Post-Training Quantization (PTQ) method.

|Model|Platform|Resolution|mAP50-95(fp32)|mAP50(fp32)|mAP50-95(fp16)|mAP50(fp16)|mAP50-95(int8)|mAP50(int8)|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|[YoloP](https://drive.google.com/drive/folders/1_0YjElSSMCbeTdD2FUbJE6zIHsHhynug)|RTX4060/orin x|640x640|-|-|-|-|-|-|-|
|[A-YOLOM](https://drive.google.com/drive/folders/1_0YjElSSMCbeTdD2FUbJE6zIHsHhynug)|RTX4060/orin x|480x640|-|-|-|-|-|-|-|

# ![Contribute](https://img.shields.io/badge/how%20to%20contribute-project-brightgreen) Contributing
Welcome users to participate in these projects. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for the contributing guideline.We encourage you to join the effort and contribute feedback, ideas, and code. You can participate in Working Groups, Working Groups have most of their discussions on [Slack](https://app.slack.com/client/T07U5CEEXCP/C07UKUA9TCJ) or QQ (938558640).

# References
- [YoloP: https://github.com/hustvl/YOLOP](https://github.com/hustvl/YOLOP)
- [A-YOLOM: https://github.com/JiayuanWang-JW/YOLOv8-multi-task](https://github.com/JiayuanWang-JW/YOLOv8-multi-task)
- [Setup Environment: https://zhuanlan.zhihu.com/p/818205320](https://zhuanlan.zhihu.com/p/818205320)
