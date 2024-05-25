

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
- VTG-IT: https://huggingface.co/datasets/Yongxin-Guo/VTG-IT

The file structure of videos should be like

```
yourpath/data/
|–– VTG-IT/
|–– youcook2/
    |-- YouCook2_asr_denseCap/
        |-- youcook2_6fps_224/
            |-- GLd3aX16zBg.mp4
            |-- ...
|–– charades/
    |-- videos/
        |-- AO8RW.mp4
        |-- ...
|–– qvhighlights/
    |-- videos/
        |-- train
            |-- j7rJstUseKg_360.0_510.0.mp4
            |-- ...
        |-- val
            |-- Ok-M_V_h-eY_360.0_510.0.mp4
            |-- ...
|-- activitynet/
    |-- anet_6fps_224/
        |-- v_sJFgo9H6zNo.mp4
        |-- ...
|-- vitt/
    |-- raw_videos/
        |-- video---L2yxB3CUg
        |-- ...
|-- didemo/
    |-- videos/
        |-- 54322086@N00_2408598493_274c77d26a.avi
        |-- ...
|–– queryd/
    |-- QuerYD_downloader/
        |-- videos/
            |-- video-fTcSVQJ2h8g
            |-- ...
|-- HiREST/
    |-- clips/
        |-- d1iOnCn2ZAQ_93_320.mp4
        |-- ...
    |-- videos/
        |-- d1iOnCn2ZAQ.mp4
        |-- ...
|-- tvsum/
    |-- videos/
        |-- AwmHb44_ouw.mp4
        |-- ...
|-- summe/
    |-- videos/
        |-- Air_Force_One.webm
        |-- ...
|-- coin/
    |-- videos_ali/
        |-- 116/
            |-- video-NLy71UrHElw
            |-- ...
        |-- ...
|-- yttemporal/
    |-- videos/
        |-- video--7kE8JN7t4c
        |-- ...
|-- valley/
    |-- 026601_026650/
        |-- 1014264308.mp4
        |-- ...
```
Please make sure the suffix of videos are correct.


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
2. download the annotations directory from https://huggingface.co/datasets/Yongxin-Guo/VTG-IT/tree/main/VTG-IT/dense_video_caption/HiREST_step
3. run `python utils/process_hirest.py`