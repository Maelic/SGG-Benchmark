# Scene Graph Benchmark in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-2.2.1-%237732a8)
[![arXiv](https://img.shields.io/badge/arXiv-2405.16116-b31b1b.svg)](https://arxiv.org/abs/2405.16116)

## :warning: We are looking for contributors to add the task of SGG directly to the [ultralytics codebase](https://github.com/ultralytics/ultralytics)! If you are interested, please contact me at [teoneau@gmail.com](mailto:teoneau@gmail.com)! :warning:

## :rocket: [REAL-TIME SCENE GRAPH GENERATION](https://arxiv.org/abs/2405.16116) :rocket:

Previous work (PE-NET model) | Our REACT model for Real-Time SGG
:-: | :-:
<video src='https://github.com/user-attachments/assets/1e580ecc-6a31-409c-82b5-4488aadaf815' width=480/> | <video src='https://github.com/user-attachments/assets/6dfc22de-176a-4d50-9e3a-e91d8df76777' width=480/>


Our latest paper [REACT: Real-time Efficiency and Accuracy Compromise for Tradeoffs in Scene
Graph Generation](https://arxiv.org/abs/2405.16116) is finally available! Please have a look if you're interested! We dive into current bottlenecks of SGG models for real-time constraints and propose a simple yet very efficient implementation using YOLOV8. Weights are available [here](MODEL_ZOO.md).
Here is a snapshot of the main results:

<p align="center">
<img src="https://github.com/user-attachments/assets/5335b285-e54b-4d79-88f1-5f4a4ef6aab4" alt="intro_img" width="1080"/>
</p>

## Background

This implementation is a new benchmark for the task of Scene Graph Generation, based on a fork of the [SGG Benchmark by Kaihua Tang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). The implementation by Kaihua is a good starting point however it is very outdated and is missing a lot of new development for the task.
My goal with this new codebase is to provide an up-to-date and easy-to-run implementation of common approaches in the field of Scene Graph Generation. 
This codebase also focuses on real-time and real-world usage of Scene Graph Generation with dedicated dataset tools and a large choice of object detection backbones.
This codebase is actually a work-in-progress, do not expect everything to work properly on the first run. If you find any bugs, please feel free to post an issue or contribute with a PR.

## Recent Updates

- [X] 26.05.2025: I have added some explanation for two new metrics: InformativeRecall@K and Recall@K Relative. InformativeRecall@K is defined in [Mining Informativeness in Scene Graphs](https://www.sciencedirect.com/science/article/pii/S016786552500008X) and can help to measure the pertinence and robustness of models for real-world applications. Please check the [METRICS.md](METRICS.md) file for more information.
- [X] 26.05.2025: The codebase now supports also YOLOV12, see [configs/VG150/react_yolov12m.yaml](configs/VG150/react_yolov12m.yaml).
- [X] 04.12.2024: Official release of the REACT model weights for VG150, please see [MODEL_ZOO.md](MODEL_ZOO.md)
- [X] 03.12.2024: Official release of the [REACT model](https://arxiv.org/abs/2405.16116)
- [X] 23.05.2024: Added support for Hyperparameters Tuning with the RayTune library, please check it out: [Hyperparameters Tuning](#hyperparameters-tuning)
- [X] 23.05.2024: Added support for the YOLOV10 backbone and SQUAT relation head!
- [X] 28.05.2024: Official release of our [Real-Time Scene Graph Generation](https://arxiv.org/abs/2405.16116) implementation.
- [X] 23.05.2024: Added support for the [YOLO-World](https://www.yoloworld.cc/) backbone for Open-Vocabulary object detection!
- [X] 10.05.2024: Added support for the [PSG Dataset](https://github.com/Jingkang50/OpenPSG)
- [X] 03.04.2024: Added support for the IETrans method for data augmentation on the Visual Genome dataset, please check it out! [IETrans](./process_data/data_augmentation/README.md).
- [X] 03.04.2024: Update the demo, now working with any models, check [DEMO.md](./demo/).
- [X] 01.04.2024: Added support for Wandb for better visualization during training, tutorial coming soon.

## Contents

1. [Overview](#Overview)
2. [Install the Requirements](INSTALL.md)
3. [Prepare the Dataset](DATASET.md)
4. [Simple Webcam Demo](#demo)
5. [Supported Models](#supported-models)
6. [Metrics and Results for our Toolkit](METRICS.md)
    - [Explanation of R@K, mR@K, zR@K, ng-R@K, ng-mR@K, ng-zR@K, A@K, S2G](METRICS.md#explanation-of-our-metrics)
    - [Output Format](METRICS.md#output-format-of-our-code)
    - [Reported Results](METRICS.md#reported-results)
7. [Training on Scene Graph Generation](#perform-training-on-scene-graph-generation)
8. [Hyperparameters Tuning](#hyperparameters-tuning)
9. [Evaluation on Scene Graph Generation](#Evaluation)
<!-- 9. [**Detect Scene Graphs on Your Custom Images** :star2:](#SGDet-on-custom-images) -->
<!-- 10. [**Visualize Detected Scene Graphs of Custom Images** :star2:](#Visualize-Detected-SGs-of-Custom-Images) -->
10. [Other Options that May Improve the SGG](#other-options-that-may-improve-the-SGG)
<!-- 11. [Tips and Tricks for TDE on any Unbiased Task](#tips-and-Tricks-for-any-unbiased-taskX-from-biased-training) -->
11. [Frequently Asked Questions](#frequently-asked-questions)
12. [Citations](#Citations)

## Overview

Note from [Kaihua Tang](https://github.com/KaihuaTang), I keep it for reference:

" This project aims to build a new CODEBASE of Scene Graph Generation (SGG), and it is also a Pytorch implementation of the paper [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949). The previous widely adopted SGG codebase [neural-motifs](https://github.com/rowanz/neural-motifs) is detached from the recent development of Faster/Mask R-CNN. Therefore, I decided to build a scene graph benchmark on top of the well-known [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) project and define relationship prediction as an additional roi_head. By the way, thanks to their elegant framework, this codebase is much more novice-friendly and easier to read/modify for your own projects than previous neural-motifs framework (at least I hope so). It is a pity that when I was working on this project, the [detectron2](https://github.com/facebookresearch/detectron2) had not been released, but I think we can consider [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) as a more stable version with less bugs, hahahaha. I also introduce all the old and new metrics used in SGG, and clarify two common misunderstandings in SGG metrics in [METRICS.md](METRICS.md), which cause abnormal results in some papers. "

<!-- ### Benefit from the up-to-date Faster R-CNN in [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), this codebase achieves new state-of-the-art Recall@k on SGCls & SGGen (by 2020.2.16) through the reimplemented VCTree using two 1080ti GPUs and batch size 8:

Models | SGGen R@20 | SGGen R@50 | SGGen R@100 | SGCls R@20 | SGCls R@50 | SGCls R@100 | PredCls R@20 | PredCls R@50 | PredCls R@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
VCTree | 24.53 | 31.93 | 36.21 | 42.77 | 46.67 | 47.64 | 59.02 | 65.42 | 67.18

Note that all results of VCTree should be better than what we reported in [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), because we optimized the tree construction network after the publication.

### The illustration of the Unbiased SGG from 'Unbiased Scene Graph Generation from Biased Training'

![alt text](demo/teaser_figure.png "from 'Unbiased Scene Graph Generation from Biased Training'") -->

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions regarding dataset preprocessing.

## DEMO

You can [download a pre-train model](MODEL_ZOO.md) or [train your own model](#perform-training-on-scene-graph-generation) and run my off-the-shelf demo!

You can use the [SGDET_on_custom_images.ipynb](demo/SGDET_on_custom_images.ipynb) notebook to visualize detections on images.

I also made a demo code to try SGDET with your webcam in the [demo folder](./demo/README.md), feel free to have a look!

## Supported Models

Scene Graph Generation approaches can be categorized between one-stage and two-stage approaches:
1. **Two-stages approaches** are the original implementation of SGG. It decouples the training process into (1) training an object detection backbone and (2) using bounding box proposals and image features from the backbone to train a relation prediction model.
2. **One-stage approaches** are learning both the object and relation features in the same learning stage. This codebase focuses on the first category, two-stage approaches.

### Object Detection Backbones

We proposed different object detection backbones that can be plugged with any relation prediction head, depending on the use case.

:rocket: NEW! No need to train a backbone anymore, we support Yolo-World for fast and easy open-vocabulary inference. Please check it out!

- [x] [YOLOV10](https://docs.ultralytics.com/models/yolov10/): New end-to-end yolo architecture for SOTA real-time object detection.
- [x] [YOLOV8-World](https://docs.ultralytics.com/models/yolo-world/): SOTA in real-time open-vocabulary object detection!
- [x] [YOLOV9](https://docs.ultralytics.com/models/yolov9/): SOTA in real-time object detection.
- [x] [YOLOV8](https://docs.ultralytics.com/models/yolov8/): SOTA in real-time object detection.
- [x] Faster-RCNN: This is the original backbone used in most SGG approaches. It is based on a ResNeXt-101 feature extractor and an RPN for regression and classification. See [the original paper for reference](https://arxiv.org/pdf/1506.01497.pdf). Performance is 38.52/26.35/28.14 mAp on VG train/val/test set respectively. You can find the original pretrained model by Kaihua [here](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw).

### Relation Heads

We try to compiled the main approaches for relation modeling in this codebase:

- [x] SQUAT: [Devil's on the Edges: Selective Quad Attention for Scene Graph Generation](https://arxiv.org/abs/2304.03495), thanks to the [official implementation by authors](https://github.com/hesedjds/SQUAT)

- [x] PE-NET: [Prototype-based Embedding Network for Scene Graph Generation](https://arxiv.org/abs/2303.07096), thanks to the [official implementation by authors](https://github.com/VL-Group/PENET)

- [x] SHA-GCL: [Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation in Pytorch](https://arxiv.org/abs/2203.09811), thanks to the [official implementation by authors](https://github.com/dongxingning/SHA-GCL-for-SGG)

- [x] GPS-NET: [GPS-Net: Graph Property Sensing Network for Scene Graph Generation
](https://arxiv.org/abs/2003.12962), thanks to the [official implementation by authors](https://github.com/siml3/GPS-Net)

- [x] VCTree: [Learning to Compose Dynamic Tree Structures for Visual Contexts](https://arxiv.org/abs/1812.01880), thanks to the [implementation by Kaihua](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

- [x] Neural-Motifs: [Neural Motifs: Scene Graph Parsing with Global Context](https://arxiv.org/abs/1711.06640), thanks to the [implementation by Kaihua](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

- [x] IMP: [Scene Graph Generation by Iterative Message Passing](https://arxiv.org/abs/1701.02426), thanks to the [implementation by Kaihua](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

### Debiasing methods

On top of relation heads, several debiasing methods have been proposed through the years with the aim of increasing the accuracy of baseline models in the prediction of tail classes.

- [x] Hierarchical: [Hierarchical Relationships: A New Perspective to Enhance Scene Graph Generation](https://arxiv.org/abs/2303.06842), thanks to the [implementation by authors](https://github.com/zzjun725/Scene-Graph-Benchmark.pytorch)

- [x] Causal: [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949), thanks to the [implementation by authors](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)

### Data Augmentation methods

Due to severe biases in datasets, the task of Scene Graph Generation as also been tackled through data-centring approaches.

- [x] IETrans: [Fine-Grained Scene Graph Generation with Data Transfer](https://arxiv.org/abs/2203.11654), custom implementation based on the one [by Zijian Zhou](https://github.com/franciszzj/HiLo/tree/main/tools/data_prepare)

### Model ZOO

We provide some of the pre-trained weights for evaluation or usage in downstream tasks, please see [MODEL_ZOO.md](MODEL_ZOO.md).

## Metrics and Results **(IMPORTANT)**
Explanation of metrics in our toolkit and reported results are given in [METRICS.md](METRICS.md)

<!-- ## Alternate links

Since OneDrive links might be broken in mainland China, we also provide the following alternate links for all the pretrained models and dataset annotations using BaiduNetDisk: 

Link：[https://pan.baidu.com/s/1oyPQBDHXMQ5Tsl0jy5OzgA](https://pan.baidu.com/s/1oyPQBDHXMQ5Tsl0jy5OzgA)
Extraction code：1234 -->
## YOLOV8/9/10/11/World Pre-training

If you want to use YoloV8/9/10/11 or Yolo-World as a backbone instead of Faster-RCNN, you need to first train a model using the official [ultralytics implementation](https://github.com/ultralytics/ultralytics). To help you with that, I have created a [dedicated notebook](process_data/convert_to_yolo.ipynb) to generate annotations in YOLO format from a .h5 file (SGG format). 
Once you have a model, you can modify [this config file](configs/VG150/e2e_relation_yolov8m.yaml) and change the path `PRETRAINED_DETECTOR_CKPT` to your model weights. Please note that you will also need to change the variable `SIZE` and `OUT_CHANNELS` accordingly if you use another variant of YOLO (nano, small or large for instance). 
For training an SGG model with YOLO as a backbone, you need to modify the `META_ARCHITECTURE` variable in the same config file to `GeneralizedYOLO`. You can then follow the standard procedure for PREDCLS, SGCLS or SGDET training below.

## Faster R-CNN pre-training (legacy)

:warning: Faster-RCNN pre-training is not officially supported anymore in this codebase, please use a YOLO backbone instead (see above). Using `detector_pretrain_net.py` will NOT WORK with a YOLO backbone.

The following command can be used to train your own Faster R-CNN model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_EPOCH 20 MODEL.RELATION_ON False OUTPUT_DIR ./checkpoints/pretrained_faster_rcnn SOLVER.PRE_VAL False
```
where ```CUDA_VISIBLE_DEVICES``` and ```--nproc_per_node``` represent the id of GPUs and number of GPUs you use, ```--config-file``` means the config we use, where you can change other parameters. ```SOLVER.IMS_PER_BATCH``` and ```TEST.IMS_PER_BATCH``` are the training and testing batch size respectively, ```DTYPE "float16"``` enables Automatic Mixed Precision, ```OUTPUT_DIR``` is the output directory to save checkpoints and log (considering `/home/username/checkpoints/pretrained_faster_rcnn`), ```SOLVER.PRE_VAL``` means whether we conduct validation before training or not.
 


## Perform training on Scene Graph Generation

There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use the argument ```--task``` to select the protocols. 

For **Predicate Classification (PredCls)**, we need to set:
``` bash
--task predcls
```
For **Scene Graph Classification (SGCls)**: :warning: SGCls mode is currently LEGACY and NOT SUPPORTED anymore for any YOLO-based model, please find the reason why [in this issue](https://github.com/Maelic/SGG-Benchmark/issues/45).
``` bash
--task sgcls
```
For **Scene Graph Detection (SGDet)**:
``` bash
--task sgdet
```

### Predefined Models
We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```. To select our predefined models, you can use ```MODEL.ROI_RELATION_HEAD.PREDICTOR```.

For [REACT](https://arxiv.org/abs/2405.16116v2) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor
```

For [PE-NET](https://arxiv.org/abs/2303.07096) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork
```

For [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
```
For [Iterative-Message-Passing(IMP)](https://arxiv.org/abs/1701.02426) Model (Note that SOLVER.BASE_LR should be changed to 0.001 in SGCls, or the model won't converge):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR IMPPredictor
```
For [VCTree](https://arxiv.org/abs/1812.01880) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor
```
For Transformer Model (Note that Transformer Model needs to change SOLVER.BASE_LR to 0.001, SOLVER.SCHEDULE.TYPE to WarmupMultiStepLR, SOLVER.MAX_ITER to 16000, SOLVER.IMS_PER_BATCH to 16, SOLVER.STEPS to (10000, 16000).), which is provided by [Jiaxin Shi](https://github.com/shijx12):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor
```
For [Unbiased-Causal-TDE](https://arxiv.org/abs/2002.11949) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor
```

The default settings are under ```configs/e2e_relation_X_101_32_8_FPN_1x.yaml``` and ```sgg_benchmark/config/defaults.py```. The priority is ```command > yaml > defaults.py```

### Customize Your Own Model
If you want to customize your own model, you can refer ```sgg_benchmark/modeling/roi_heads/relation_head/model_XXXXX.py``` and ```sgg_benchmark/modeling/roi_heads/relation_head/utils_XXXXX.py```. You also need to add the corresponding nn.Module in ```sgg_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py```. Sometimes you may also need to change the inputs & outputs of the module through ```sgg_benchmark/modeling/roi_heads/relation_head/relation_head.py```.

### The Causal TDE on [Unbiased Scene Graph Generation from Biased Training](https://arxiv.org/abs/2002.11949)
As to the Unbiased-Causal-TDE, there are some additional parameters you need to know. ```MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE``` is used to select the causal effect analysis type during inference(test), where "none" is original likelihood, "TDE" is total direct effect, "NIE" is natural indirect effect, "TE" is total effect. ```MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE``` has two choice "sum" or "gate". Since Unbiased Causal TDE Analysis is model-agnostic, we support [Neural-MOTIFS](https://arxiv.org/abs/1711.06640), [VCTree](https://arxiv.org/abs/1812.01880) and [VTransE](https://arxiv.org/abs/1702.08319). ```MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER``` is used to select these models for Unbiased Causal Analysis, which has three choices: motifs, vctree, vtranse.

Note that during training, we always set ```MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE``` to be 'none', because causal effect analysis is only applicable to the inference/test phase.

### Examples of the Training Command

**NEW: I replaced the training by iteration (steps) with training by epochs (iteration on the whole dataset), controlling the training loop by iteration is still possible but it's made easier by epochs imo, you can try with the argument `SOLVER.MAX_EPOCH` (see below)** 

By default, only the last checkpoint will be saved which is not very efficient. You can choose to save only the best checkpoint instead with the argument ```--save-best```.
Training Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --task predcls --save-best --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_EPOCH 20 MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-exmp
```
where ```MODEL.PRETRAINED_DETECTOR_CKPT``` is the pretrained Faster R-CNN model you want to load, ```OUTPUT_DIR``` is the output directory used to save checkpoints and the log. Since we use the ```WarmupReduceLROnPlateau``` as the learning scheduler for SGG, ```SOLVER.STEPS``` is not required anymore.

Training Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10026 --nproc_per_node=2 tools/relation_train_net.py --task sgcls --save-best  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_EPOCH 20 MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/causal-motifs-sgcls-exmp
```

## Hyperparameters Tuning

Required library:
```pip install ray[data,train,tune] optuna tensorboard```

We provide a training loop for hyperparameters tuning in [hyper_param_tuning.py](tools/hyper_param_tuning.py). This script uses the [RayTune](https://docs.ray.io/en/latest/tune/index.html) library for efficient hyperparameters search. You can define a ```search_space``` object with different values related to the optimizer (AdamW and SGD supported for now) or directly customize the model structure with model parameters (for instance Linear layers dimensions or MLP dimensions etc). The ```ASHAScheduler``` scheduler is used for the early stopping of bad trials. The default value to optimize is the overall loss but this can be customize to specific loss values or standard metrics such as ```mean_recall```.

To launch the script, do as follow:

```
CUDA_VISIBLE_DEVICES=0 python tools/hyper_param_tuning.py --save-best --task sgdet --config-file "./configs/IndoorVG/e2e_relation_yolov10.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/penet-yolov10m SOLVER.IMS_PER_BATCH 8
```

The config and OUTPUT_DIR paths need to be absolute to allow faster loading. A lot of terminal outputs are disabled by default during tuning, using the ```cfg.VERBOSE``` variable.

To watch the results with tensorboardX: 
```
tensorboard --logdir=./ray_results/train_relation_net_2024-06-23_15-28-01
```

## Evaluation

### Examples of the Test Command
Test Example 1 : (PreCls, Motif Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```

Test Example 2 : (SGCls, Causal, **TDE**, SUM Fusion, MOTIFS Model)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10028 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgcls-exmp OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgcls-exmp
```

<!-- ## SGDet on Custom Images
Note that evaluation on custum images is only applicable for SGDet model, because PredCls and SGCls model requires additional ground-truth bounding boxes information. To detect scene graphs into a json file on your own images, you need to turn on the switch TEST.CUSTUM_EVAL and give a folder path (or a json file containing a list of image paths) that contains the custom images to TEST.CUSTUM_PATH. Only JPG files are allowed. The output will be saved as custom_prediction.json in the given DETECTED_SGG_DIR.

Test Example 1 : (SGDet, **Causal TDE**, MOTIFS Model, SUM Fusion) [(checkpoint)](https://1drv.ms/u/s!AmRLLNf6bzcir9x7OYb6sKBlzoXuYA?e=s3Y602)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgdet OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/kaihua/checkpoints/custom_images DETECTED_SGG_DIR /home/kaihua/checkpoints/your_output_path
```

Test Example 2 : (SGDet, **Original**, MOTIFS Model, SUM Fusion) [(same checkpoint)](https://1drv.ms/u/s!AmRLLNf6bzcir9x7OYb6sKBlzoXuYA?e=s3Y602)
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgdet OUTPUT_DIR /home/kaihua/checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/kaihua/checkpoints/custom_images DETECTED_SGG_DIR /home/kaihua/checkpoints/your_output_path
```

The output is a json file. For each image, the scene graph information is saved as a dictionary containing bbox(sorted), bbox_labels(sorted), bbox_scores(sorted), rel_pairs(sorted), rel_labels(sorted), rel_scores(sorted), rel_all_scores(sorted), where the last rel_all_scores give all 51 predicates probability for each pair of objects. The dataset information is saved as custom_data_info.json in the same DETECTED_SGG_DIR. -->


## Other Options that May Improve the SGG

- For some models (not all), turning on or turning off ```MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS``` will affect the performance of predicate prediction, e.g., turning it off will improve VCTree PredCls but not the corresponding SGCls and SGGen. For the reported results of VCTree, we simply turn it on for all three protocols like other models.

- For some models (not all), a crazy fusion proposed by [Learning to Count Object](https://arxiv.org/abs/1802.05766) will significantly improves the results, which looks like ```f(x1, x2) = ReLU(x1 + x2) - (x1 - x2)**2```. It can be used to combine the subject and object features in ```roi_heads/relation_head/roi_relation_predictors.py```. For now, most of our model just concatenate them as ```torch.cat((head_rep, tail_rep), dim=-1)```.

- Not to mention the hidden dimensions in the models, e.g., ```MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM```. Due to the limited time, we didn't fully explore all the settings in this project, I won't be surprised if you improve our results by simply changing one of our hyper-parameters

<!-- ## Tips and Tricks for any Unbiased TaskX from Biased Training

The counterfactual inference is not only applicable to SGG. Actually, my collegue [Yulei](https://github.com/yuleiniu) found that counterfactual causal inference also has significant potential in [unbiased VQA](https://arxiv.org/abs/2006.04315). We believe such an counterfactual inference can also be applied to lots of reasoning tasks with significant bias. It basically just runs the model two times (one for original output, another for the intervened output), and the later one gets the biased prior that should be subtracted from the final prediction. But there are three tips you need to bear in mind:
- The most important things is always the causal graph. You need to find the correct causal graph with an identifiable branch that causes the biased predictions. If the causal graph is incorrect, the rest would be meaningless. Note that causal graph is not the summarization of the existing network (but the guidance to build networks), you should modify your network based on causal graph, but not vise versa. 
- For those nodes having multiple input branches in the causal graph, it's crucial to choose the right fusion function. We tested lots of fusion funtions and only found the SUM fusion and GATE fusion consistently working well. The fusion function like element-wise production won't work for TDE analysis in most of the cases, because the causal influence from multiple branches can not be linearly separated anymore, which means, it's no longer an identifiable 'influence'.
- For those final predictions having multiple input branches in the causal graph, it may also need to add auxiliary losses for each branch to stablize the causal influence of each independent branch. Because when these branches have different convergent speeds, those hard branches would easily be learned as unimportant tiny floatings that depend on the fastest/stablest converged branch. Auxiliary losses allow different branches to have independent and equal influences. -->

## Frequently Asked Questions:

1. **Q:** Fail to load the given checkpoints.
**A:** The model to be loaded is based on the last_checkpoint file in the OUTPUT_DIR path. If you fail to load the given pretained checkpoints, it probably because the last_checkpoint file still provides the path in my workstation rather than your own path.

2. **Q:** AssertionError on "assert len(fns) == 108073"
**A:** If you are working on VG dataset, it is probably caused by the wrong DATASETS (data path) in sgg_benchmark/config/paths_catlog.py. If you are working on your custom datasets, just comment out the assertions.

3. **Q:** AssertionError on "l_batch == 1" in model_motifs.py
**A:** The original MOTIFS code only supports evaluation on 1 GPU. Since my reimplemented motifs is based on their code, I keep this assertion to make sure it won't cause any unexpected errors.

## Citations

If you find this project helps your research, please kindly consider citing our project or papers in your publications.

```
@misc{neau2024reactrealtimeefficiencyaccuracy,
      title={REACT: Real-time Efficiency and Accuracy Compromise for Tradeoffs in Scene Graph Generation}, 
      author={Maëlic Neau and Paulo E. Santos and Anne-Gwenn Bosser and Cédric Buche},
      year={2024},
      eprint={2405.16116},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.16116}, 
}
```
