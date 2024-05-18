#### Pre-Requirements

> environment: conda activate /group/20023/messijyliu/envs/torch113-cu117;
> data_path: /cfs/cfs-lugcocyb/messijyliu/data
> model ckpt: /cfs/cfs-lugcocyb/llm_weights/
> pip requirements:
- transformers >= 4.34.1


#### Changed Files Compare to Video-LLama

- timechat\datasets\datasets\video_instruct_dataset.py: add timestampes and stride
- timechat\datasets\builders\instruct_builder.py: add new instruction datasets, and align changes in `video_instruct_dataset.py`
- timechat\models\eva_vit.py: 431-439, change url check
- timechat\models\modeling_llama.py: add candidate rotray
- timechat\models\timechat.py: remove audio branch, add text-aligned QFormer
- timechat\processors\video_processor.py: add `interpolate_frame_pos_embed`, add a new `random` sampling method

#### Re-Organize Videos

- YouCook2-BB --- Done
- Charades --- Done
- QVhighlights --- Done
- Activitynet_Captions --- Done
- vitt --- Downloading --- checking
- didemo --- Done
- QuerYD --- Done
- HiREST --- Downloading
- TVSum --- Done
- SumMe --- Done
- COIN --- Downloading
- yttemporal180m --- Downloading
- vatex --- Done
- jukin --- Downloading


#### Scripts

##### Original TimeChat

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades.yaml

##### Adding time embedding as position embedding to the frames

<!-- torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp.yaml -->

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp_v2.yaml

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp_v3.yaml

##### Adding special time token

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token.yaml

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token_v2.yaml

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token_v3.yaml


export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token_v4-full.yaml


##### Adding special time loss


torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token-time-loss.yaml

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token-time-loss-no-qformer-text.yaml

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token-time-loss-original-qformer-text.yaml

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token-time-loss_v2.yaml

torchrun --nproc_per_node=4 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token-time-loss_v2-full.yaml

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token-time-loss_v3.yaml

torchrun --nproc_per_node=4 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token-time-loss_v4-full.yaml

torchrun --nproc_per_node=4 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token-time-loss_v5-full.yaml


export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_charades-real-timestamp-time-token-time-loss_v3-full.yaml

##### Process YouCook2

ls -U /cfs/cfs-lugcocyb/messijyliu/data/YouCook2-BB/raw_videos/training >> /cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/dense_video_captioning/youcook2/video_names.txt
ls -U /cfs/cfs-lugcocyb/messijyliu/data/YouCook2-BB/raw_videos/testing >> /cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/dense_video_captioning/youcook2/video_names.txt
ls -U /cfs/cfs-lugcocyb/messijyliu/data/YouCook2-BB/raw_videos/validation >> /cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/dense_video_captioning/youcook2/video_names.txt

python /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/utils/compress_video_data.py \
--input_root=/cfs/cfs-lugcocyb/messijyliu/data/YouCook2-BB/raw_videos/training \
--output_root=/cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/dense_video_captioning/youcook2/YouCook2_asr_denseCap/youcook2_6fps_224/ \
--input_file_list_path=/cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/dense_video_captioning/youcook2/video_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 2

python /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/utils/compress_video_data.py \
--input_root=/cfs/cfs-lugcocyb/messijyliu/data/YouCook2-BB/raw_videos/testing \
--output_root=/cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/dense_video_captioning/youcook2/YouCook2_asr_denseCap/youcook2_6fps_224/ \
--input_file_list_path=/cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/dense_video_captioning/youcook2/video_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 2

python /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/utils/compress_video_data.py \
--input_root=/cfs/cfs-lugcocyb/messijyliu/data/YouCook2-BB/raw_videos/validation \
--output_root=/cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/dense_video_captioning/youcook2/YouCook2_asr_denseCap/youcook2_6fps_224/ \
--input_file_list_path=/cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/dense_video_captioning/youcook2/video_names.txt \
--fps=6 --size=224 --file_type=video --num_workers 2


##### Process ActivityNet Captions
ls -U /group/20009/data/ActivityNet/raw_videos >> /workspace/user_code/ft_local/video_names_activity.txt
ls -U /workspace/user_code/ft_local/Activitynet_Captions/anet_6fps_224 >> /workspace/user_code/ft_local/video_names_activity_exist.txt

python /workspace/user_code/project_temporal_video_understanding/utils/compress_video_data.py \
--input_root=/group/20009/data/ActivityNet/raw_videos \
--output_root=/workspace/user_code/ft_local/Activitynet_Captions/anet_6fps_224 \
--exist_file_list_path=/workspace/user_code/ft_local/video_names_activity_exist.txt \
--input_file_list_path=/workspace/user_code/ft_local/video_names_activity.txt \
--fps=6 --size=224 --file_type=video --num_workers 2


### Full Training

##### Original

torchrun --nproc_per_node=8 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2_finetune_time104k_valley72k.yaml

##### Ours

torchrun --nproc_per_node=8 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage1-time-token-and-align.yaml

torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2-lora-tuning.yaml


torchrun --nproc_per_node=2 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/videochat2/stage1-time-token-and-align-bs16.yaml


export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 train.py --cfg-path  /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2-lora-tuning-it.yaml

conda activate /group/20023/messijyliu/envs/torch113-cu117
torchrun --nproc_per_node=2 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/stage2-lora-tuning-it-shorten.yaml


torchrun --nproc_per_node=8 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/videochat2/pretrain.yaml

torchrun --nproc_per_node=8 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/videollama/pretrain.yaml

torchrun --nproc_per_node=16 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/videollama/pretrain_with_qformer.yaml

torchrun --nproc_per_node=16 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/videollama/pretrain-diverse-sample.yaml

torchrun --nproc_per_node=8 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/videochat2/pretrain-full.yaml

torchrun --nproc_per_node=8 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/videochat2/pretrain-diverse-sampler.yaml


torchrun --nproc_per_node=16 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/timechat/pretrain.yaml

torchrun --nproc_per_node=16 train.py --cfg-path /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/train_configs/videochat2/pretrain.yaml

##### Process HIREST

ls -U /cfs/cfs-lugcocyb/yongxinguo/data/HiREST/videos >> /cfs/cfs-lugcocyb/yongxinguo/data/HiREST/videos_names_hirest.txt

python /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/utils/process_hirest.py

% Generate annotations
python /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/utils/construct_it_data/process_action_data.py --dataset hirest --anno_path /cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/step_localization/hirest_step/annotations --outpath /cfs/cfs-lugcocyb/yongxinguo/data/HiREST/

# test

bash eval.sh

# construct it data

python /cfs/cfs-lugcocyb/yongxinguo/projects/project_temporal_video_understanding/utils/construct_it_data/process_action_data.py --dataset coin --anno_path /cfs/cfs-lugcocyb/messijyliu/data/COIN/COIN.json --video_path /cfs/cfs-lugcocyb/messijyliu/data/COIN/videos --outpath /cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/step_localization/coin