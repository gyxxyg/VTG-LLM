<h2 align="center"> <a href="">VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding</a></h2>


## Abstract

Video Temporal Grounding (VTG) focuses on accurately identifying event timestamps within a particular video based on a linguistic query, playing a vital role in downstream tasks such as video browsing and editing. While Video Large Language Models (video LLMs) have made significant progress in understanding video content, they often face challenges in accurately pinpointing timestamps within videos, which limits their performance on VTG tasks. Therefore, to improve video LLMs' ability to effectively locate timestamps, we argue that two critical aspects need to be enhanced. 
First, it is essential to have high-quality instructional tuning datasets that encompass mainstream VTG tasks. Second, directly incorporating timestamp knowledge into video LLMs is crucial, as it enables models to efficiently comprehend timestamp information. To address these needs, we first introduce VTG-IT-120K, a high-quality and comprehensive instruction tuning dataset that covers VTG tasks such as moment retrieval, dense video captioning, video summarization, and video highlight detection. Furthermore, we propose a specially designed video LLM model for VTG tasks, VTG-LLM, which (1) effectively integrates timestamp knowledge into visual tokens; (2) incorporates absolute-time tokens that specifically handle timestamp knowledge, thereby avoiding concept shifts; and (3) introduces a lightweight, high-performance slot-based token compression method to facilitate the sampling of more video frames. Comprehensive experiments showcase the superior performance of VTG-LLM in comparison to other video LLM methods across various VTG tasks. We will make our code, model checkpoint, and datasets publicly accessible


## Enviroments

For training, run the following script is sufficient for most cases.
```
bash install_requirements.sh
```
To evaluate our checkpoint, we need pytorch >= 2.0.1, and then run the above script.
To fintune our checkpoint, it's better to use NPU environment. We will try to conduct experiments on GPU in the future.

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

