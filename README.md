<h2 align="center"> <a href="https://arxiv.org/abs/2405.13382">VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding</a></h2>

<h5 align="center"> If our project helps you, please give us a star ⭐ and cite our <a href="#bibliography">paper</a>!</h2>
<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2405.13382)
[![hf_checkpoint](https://img.shields.io/badge/🤗-Checkpoints-9C276A.svg)](https://huggingface.co/Yongxin-Guo/VTG-LLM)
[![hf_data](https://img.shields.io/badge/🤗-Datasets-9C276A.svg)](https://huggingface.co/Yongxin-Guo/VTG-LLM)
[![arxiv](https://img.shields.io/badge/Arxiv-2405.13382-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2405.13382)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgyxxyg%2FVTG-LLM&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)                


## News
- 10/12/2024, VTG-LLM has been accepted to AAAI 2025.
- 10/10/2024, We released a more powerful temporal grounding video LLM [TRACE](https://github.com/gyxxyg/TRACE). 
- 7/22/2024, Update evaluation results using various temperature.
- 5/28/2024, [NPU checkpoints](https://huggingface.co/Yongxin-Guo/VTG-LLM) can be fine-tuned on V100 GPU.

## Overview

We introduce 
- VTG-IT-120K, a high-quality and comprehensive instruction tuning dataset that covers VTG tasks such as moment retrieval (63.2K), dense video captioning (37.2K), video summarization (15.2K), and video highlight detection (3.9K).
- VTG-LLM, which (1) effectively integrates timestamp knowledge into visual tokens; (2) incorporates absolute-time tokens that specifically handle timestamp knowledge, thereby avoiding concept shifts; and (3) introduces a lightweight, high-performance slot-based token compression method to facilitate the sampling of more video frames.

<div align="center">
    <img src="figures/vtg-lm-overview.png" alt="Overview of VTG-LLM" width="700"/>
    <br/>
    <figcaption>Overview of VTG-LLM.</figcaption>
</div>

## Enviroments

We recommend utilizing NPU environments for training, evaluation, and fine-tuning. The environment we use can be found in [environment-npu.yaml](./environment-npu.yaml). Additionally, we have discovered that executing the script below is sufficient for most scenarios.
```
bash install_requirements.sh
```

If an NPU is not available, a V100 can also be employed for training and evaluation, but it cannot be used for fine-tuning checkpoints trained by an NPU. The necessary environments can be found in [requirements-v100.txt](./requirements-v100.txt).

## Model Checkpoints

The model checkpoint (without finetuning) is avaliable at [huggingface](https://huggingface.co/Yongxin-Guo/VTG-LLM):
```
git lfs install

git clone https://huggingface.co/Yongxin-Guo/VTG-LLM
```

## Data

See [DATA.md](./docs/DATA.md) for details. The data annotations are avaliable at [huggingface](https://huggingface.co/datasets/Yongxin-Guo/VTG-IT):
```
git lfs install

git clone https://huggingface.co/datasets/Yongxin-Guo/VTG-IT
```

## Requirments

Please download the following model checkpoints:
- EVA-ViT-g: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
- InstructBLIP: https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth
- Video-LLaMA: https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main
- Bert: https://huggingface.co/google-bert/bert-base-uncased

### Script

#### Tuning
Config the checkpoint and dataset paths in [pretrain-slot-sample-fmt-96.yaml](./train_configs/videollama/pretrain-slot-sample-fmt-96.yaml).
Config the bert checkpoint paths in [blip2.py](./vtgllm/models/blip2.py) and [vtgllm.py](./vtgllm/models/vtgllm.py)
```
torchrun --nproc_per_node=16 train.py --cfg-path  train_configs/videollama/pretrain-slot-sample-fmt-96.yaml
```

#### Evaluation
Config the checkpoint and dataset paths in [videollama-slot-96.yaml](./eval_configs/videollama-slot-96.yaml).

Config the downstream task in [eval.sh](eval.sh).
```
bash eval.sh
```

#### Results

| Youcook2 | CIDER | METEOR | SODA_c | F1 |
| --- | --- | --- | --- | --- |
| t=1.0 (paper) | 5.0 | 1.9 | 1.5 | 17.5 |
| t=0.1 | 5.4 | 1.8 | 1.6 | 18.4 |

| Charades-STA | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- |
| t=1.0 (paper) | 52.0 | 33.8 | 15.7  |
| t=0.1 | 53.9 | 36.3 | 16.6 |

| QVHighlights | mAP | Hit@1 |
| --- | --- | --- |
| t=1.0 (paper) | 16.5 | 33.5 |
| t=0.1 | 16.2 | 30.7 |

| ActivityNet | CIDER | METEOR | SODA_c | F1 |
| --- | --- | --- | --- | --- |
| t=1.0 (paper) | 18.2 | 5.7 | 4.7 | 34.0 |
| t=0.1 | 20.7 | 5.9 | 5.1 | 34.8 |

#### Demo

```
# cat_and_chicken.mp4

# Describe this video

A cute little kitten is sleeping on a couch. A little chicken is sitting on the cats chest and looking at the camera. The cat is purring and the chicken is moving its head.

# Please locate a series of events in the video, output the start and end timestamps of each event, and describe each event in sentences.

0000.0 - 0010.0 seconds, A cute kitten is sleeping on a couch. 0010.0 - 0020.0 seconds, A yellow bird lands on the couch and gently touches the kitten's head. 0020.0 - 0030.0 seconds, The bird picks up the kitten and starts to play with it. 0030.0 - 0040.0 seconds, The kitten tries to push the bird away, but the bird continues to play with it. 0040.0 - 0050.0 seconds, The kitten falls asleep on the couch.
```

#### Gradio Demo
You need to firstly change the path of videos and model checkpoints to your path.
```
python gradio_demo.py
```

## Recommended GPUs
* Instruction-tuning: 16xATN 910B
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

## Bibliography
If you find this repository helpful for your project, please consider citing:
```
@article{guo2024vtg,
  title={VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding},
  author={Guo, Yongxin and Liu, Jingyu and Li, Mingda and Tang, Xiaoying and Chen, Xi and Zhao, Bo},
  journal={arXiv preprint arXiv:2405.13382},
  year={2024}
}
```
