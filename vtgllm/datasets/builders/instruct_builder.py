import os
import logging
import warnings

from vtgllm.common.registry import registry
from vtgllm.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from vtgllm.datasets.datasets.laion_dataset import LaionDataset
from vtgllm.datasets.datasets.llava_instruct_dataset import Instruct_Dataset
from vtgllm.datasets.datasets.video_instruct_dataset import Video_Instruct_Dataset


@registry.register_builder("image_instruct")
class Image_Instruct_Builder(BaseDatasetBuilder):
    train_dataset_cls = Instruct_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/instruct/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'

        model_type = self.config.model_type if self.config.model_type else 'vicuna'

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token=num_video_query_token,
            tokenizer_name=tokenizer_name,
            data_type=self.config.data_type,
            model_type=model_type,
        )

        return datasets


@registry.register_builder("video_instruct")
class Video_Instruct_Builder(BaseDatasetBuilder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/instruct/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        if self.config.num_video_query_token:
            num_video_query_token = self.config.num_video_query_token
        else:
            num_video_query_token = 32

        if self.config.tokenizer_name:
            tokenizer_name = self.config.tokenizer_name
        else:
            tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/'

        model_type = self.config.model_type if self.config.model_type else 'vicuna'
        num_frm = self.config.num_frm if self.config.num_frm else 8
        sample_type = self.config.sample_type if self.config.sample_type else 'uniform'
        max_txt_len = self.config.max_txt_len if self.config.max_txt_len else 512
        stride = self.config.stride if self.config.stride else 0
        special_time_token = self.config.special_time_token if self.config.special_time_token else False
        strict_video_token_num = self.config.strict_video_token_num if self.config.strict_video_token_num else 0
        visual_model_type = self.config.visual_model_type if self.config.visual_model_type else "eva_clip"
        use_msg = self.config.use_msg if self.config.use_msg else True

        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir,
            num_video_query_token=num_video_query_token,
            tokenizer_name=tokenizer_name,
            data_type=self.config.data_type,
            model_type=model_type,
            num_frm=num_frm,
            sample_type=sample_type,
            max_txt_len=max_txt_len,
            stride=stride,
            special_time_token=special_time_token,
            strict_video_token_num=strict_video_token_num,
            visual_model_type=visual_model_type,
            use_msg=use_msg
        )

        return datasets


@registry.register_builder("webvid_instruct")
class WebvidInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/webvid_instruct.yaml",
    }


@registry.register_builder("webvid_instruct_zh")
class WebvidInstruct_zh_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/webvid_instruct.yaml",
    }


@registry.register_builder("llava_instruct")
class LlavaInstruct_Builder(Image_Instruct_Builder):
    train_dataset_cls = Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/llava_instruct.yaml",
    }


@registry.register_builder("youcook2_instruct")
class Youcook2Instruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/youcook2_instruct.yaml",
    }


@registry.register_builder("time_instruct")
class TimeInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/time_instruct.yaml",
    }


@registry.register_builder("valley72k_instruct")
class Valley72kInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/valley72k_instruct.yaml",
    }


@registry.register_builder("qvhighlights_instruct")
class QVhighlightsInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/qvhighlights_instruct.yaml",
    }


@registry.register_builder("charades_instruct")
class CharadesInstruct_Builder(Video_Instruct_Builder):
    train_dataset_cls = Video_Instruct_Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/instruct/charades_instruct.yaml",
    }
