

## Video & Annotations
- YouCook2: http://youcook2.eecs.umich.edu/download
- Charades-STA: https://github.com/jiyanggao/TALL#charades-sta-anno-download
- QVHighlight: https://github.com/jayleicn/moment_detr/blob/main/data/README.md
- ActivityNet Captions: http://activity-net.org/download.html
- ViTT: https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT
- DiDeMo: https://github.com/LisaAnne/LocalizingMoments?tab=readme-ov-file#dataset
- QuerYD: https://www.robots.ox.ac.uk/~vgg/data/queryd/
- HiREST: https://github.com/j-min/HiREST
- TVSum: https://github.com/yalesong/tvsum
- SumMe: http://classif.ai/dataset/ethz-cvl-video-summe/
- COIN: https://github.com/coin-dataset/annotations
- YT-Temporal: https://rowanzellers.com/merlot/#data
- Valley: https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md

### Compressing Videos
We preprocess videos to lower FPS and dimension to reduce storage and to improve data loading. For videos, you may use
```bash
ls -U /path/to/raw/video >> /path/to/video_names.txt

# for YouCook2
python utils/compress_video_data.py \
--input_root=/path/to/raw/videos/ \
--output_root=data/YouCook2-BB/YouCook2_asr_denseCap/youcook2_6fps_224/ \
--input_file_list_path=/path/to/video_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 24

# for ActivityNet Captions
python utils/compress_video_data.py \
--input_root=/path/to/raw/videos/ \
--output_root=data/Activitynet_Captions/anet_6fps_224 \
--input_file_list_path=/path/to/video_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 24
```

Note that the audio is also removed from the video files.

### Process HiREST
For `HiREST_step` dataset, you can use the following commands to crop the videos into clips:
1. download the videos directory from HiREST (see the following file structure)
2. download the annotations directory from https://huggingface.co/datasets/ShuhuaiRen/TimeIT/tree/main/data/step_localization/hirest_step
3. run `python utils/process_hirest.py`


### Process Instruction-Tuning Data

Change the "video" for each item in the annotation path using your own path.