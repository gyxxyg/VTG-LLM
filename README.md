<h2 align="center"> <a href="">VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding</a></h2>


## Overview

We introduce 
- VTG-IT-120K, a high-quality and comprehensive instruction tuning dataset that covers VTG tasks such as moment retrieval (63.2K), dense video captioning (37.2K), video summarization (15.2K), and video highlight detection (3.9K).
- VTG-LLM, which (1) effectively integrates timestamp knowledge into visual tokens; (2) incorporates absolute-time tokens that specifically handle timestamp knowledge, thereby avoiding concept shifts; and (3) introduces a lightweight, high-performance slot-based token compression method to facilitate the sampling of more video frames.


## Enviroments

For training, run the following script is sufficient for most cases.
```
bash install_requirements.sh
```
To evaluate our checkpoint, we need pytorch >= 2.0.1, and then run the above script.
To fintune our checkpoint, it's better to use NPU environment. We will try to conduct experiments on GPU in the future.

## Model Checkpoints

The model checkpoint (without finetuning) is avaliable at [huggingface](https://huggingface.co/Yongxin-Guo/VTG-LLM).

## Data

See [DATA.md](./docs/DATA.md) for downloading videos. The annotation can be found in `annotation` forder. You need to update the video paths with your own video paths.

## Requirments

- EVA-ViT-g: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
- InstructBLIP: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth
- Video-LLaMA: https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main

### Script

#### Tuning
Config the checkpoint and dataset paths in [pretrain-slot-sample-fmt-96.yaml](./train_configs/videollama/pretrain-slot-sample-fmt-96.yaml).
```
torchrun --nproc_per_node=16 train.py --cfg-path  train_configs/videollama/pretrain-slot-sample-fmt-96.yaml
```

#### Evaluation
Config the checkpoint and dataset paths in [videollama-slot-96.yaml](./eval_configs/videollama-slot-96.yaml).

Config the downstream task in [eval.sh](eval.sh).
```
bash eval.sh
```

#### Demo
You need to firstly change the path of videos and model checkpoints to your path.
```
python gradio_demo.py
```
## Recommended GPUs
* Instruction-tuning: 16xATN 910B or 16xA100
* Inference: 1xV100

## Acknowledgement
We are grateful for the following awesome projects:
* [TimeChat](https://github.com/RenShuhuai-Andy/TimeChat)
* [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
* [FastChat](https://github.com/lm-sys/FastChat)
* [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
* [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)
* [LLaMA](https://github.com/facebookresearch/llama)
* [VideoChat](https://github.com/OpenGVLab/Ask-Anything)
* [TESTA](https://github.com/RenShuhuai-Andy/TESTA)
* [VTimeLLM](https://github.com/huangb23/VTimeLLM)
* [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
* [entropy_estimators](https://github.com/paulbrodersen/entropy_estimators)

