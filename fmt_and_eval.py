import copy
import os
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('Using NPU')
except:
    # print(e)
    print('Using GPU')
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList
from math import ceil
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
from vtgllm.common.logger import setup_logger
from vtgllm.common.config import Config
from vtgllm.common.dist_utils import get_rank
from vtgllm.common.registry import registry
from vtgllm.conversation.conversation_video_batch import Chat, Conversation, default_conversation, SeparatorStyle, \
    conv_llava_llama_2
import decord

decord.bridge.set_bridge('torch')
import logging
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms
import pdb
import json
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import random

random.seed(1234)
from utils.format_dvc import format_dvc_output
from utils.format_tvg import format_tvg_output
from utils.format_vhd import format_vhd_output


def read_txt(path):
    with open(path, "r") as fin:
        data = fin.readline().strip()
    return data


def load_data(args, anno_path, split=None):
    '''
    anno data example:
    {"annotations":
        [
            {
                "image_id": "xHr8X2Wpmno.mp4"
                ...
            },
            ...
        ]
    }
    '''
    file_path = os.path.join(anno_path, f'{split}.caption_coco_format.json')
    with open(file_path, 'r') as f:
        data = json.load(f)["annotations"]

    if args.debug:
        data = data[:10]
    return data


def merge_seg_caps(results):
    """merge mulple generated captions from a same video into paragraph."""
    merge_results = {}
    for jterm in results:
        vname = jterm["vname"]
        cap = jterm["generated_cap"]
        postfix = vname.split(".mp4")[-1]
        start_time, end_time = float(postfix.split("_")[-2]), float(postfix.split("_")[-1])
        vid = vname.split(".mp4")[0] + ".mp4"
        if vid not in merge_results:
            merge_results[vid] = []
        merge_results[vid].append({"timestamp": [start_time, end_time], "caption": cap})
    return merge_results


def save_result(args, output_dir, results, split_name='test', format=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result.json'
    if args.timestamp:
        if args.timestamp_file != '':
            file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result_with_pred_timestamp.json'
        else:
            file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result_with_gt_timestamp.json'
    if args.debug:
        file_name = 'debug_' + file_name
    if format:
        file_name = 'fmt_' + file_name
    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f)
    return


def get_timestamp_from_file(timestamp_file):
    timestamp = {}
    with open(timestamp_file, 'r') as f:
        data = json.load(f)
        for vid, vlist in data.items():
            timestamp[vid] = []
            for vterm in vlist:
                timestamp[vid].append(vterm["timestamp"])
    return timestamp


def format_dvc(datas):
    fmt_datas = {}
    timestamp_count = []
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        caption = jterm["generated_cap"]
        timestamps, sents = format_dvc_output(caption)
        if len(timestamps) == 0:
            cnt += 1
            print(vid, caption)
        fmt_datas[vid] = []
        for j in range(len(timestamps)):
            fmt_datas[vid].append({"timestamp": timestamps[j], "caption": sents[j]})
        timestamp_count.append(len(timestamps))
    print(f"predict avg {sum(timestamp_count) / len(timestamp_count)} events per video")
    print(f'parse failed number: {cnt}')
    return fmt_datas


def format_tvg(datas):
    fmt_datas = {}
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        qid = int(jterm["id"])
        timestamps = format_tvg_output(gcap)
        if len(timestamps) == 0:
            cnt += 1
            print(vid, query + "\n", gcap, "\n")
        fmt_datas[qid] = {"timestamp": timestamps, "query": query, "vid": vid}
    print(f'parse failed number: {cnt}')
    return fmt_datas


def format_vhd(datas, gts):
    vid2gts = {}
    for jterm in gts:
        vid2gts[jterm["image_id"]] = jterm
    fmt_datas = []
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        qid = jterm["id"]
        highlights, clipscores = format_vhd_output(gcap, vid2gts[vid])
        if len(highlights) == 0:
            cnt += 1
            print(vid, query + "\n", gcap + "\n")
            # pdb.set_trace()
        else:
            # print(gcap)
            # print(timestamps)
            pass
        result = {}
        result["qid"] = qid
        result["query"] = query
        result["vid"] = vid
        result["pred_saliency_scores"] = clipscores
        fmt_datas.append(result)
    print(f'parse failed number: {cnt}')
    return fmt_datas


def generate(chat, gr_videos, user_messages, num_beams, temperature, top_p, n_frms, chat_states=None, img_lists=None):
    N = len(user_messages)
    if chat_states is None:
        chat_states = []
        for i in range(N):
            if args.model_type == 'vicuna':
                chat_state = default_conversation.copy()
            else:
                chat_state = conv_llava_llama_2.copy()
            chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            chat_states.append(chat_state)
    if img_lists is None:
        img_lists = [[] for i in range(N)]
        llm_message = chat.upload_video_without_audio(gr_videos, chat_states, img_lists, n_frms=n_frms)

    for user_message, chat_state in zip(user_messages, chat_states):
        chat.ask(user_message, chat_state)

    # print(len(img_lists[0][0][0]))

    responses = chat.answer(convs=chat_states,
                            img_lists=img_lists,
                            num_beams=num_beams,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=512,
                            max_length=3000)[0]
    return responses, chat_states, img_lists


def main(args):

    anno_path = args.anno_path
    anno_data = load_data(args, anno_path, split=args.split)

    with open(f'{args.output_dir}/{args.dataset}_{args.split}_f{args.num_frames}_result.json', 'r') as f:
        results = json.load(f)

    # format results to calculate metrics
    if args.task == "dvc":
        fmt_results = format_dvc(results)
    elif args.task == "tvg":
        fmt_results = format_tvg(results)
    elif args.task == "vhd":
        fmt_results = format_vhd(results, anno_data)
    else:
        print(f"Not support formatting samples for task {args.task}")
    # save format results
    save_result(args, args.output_dir, fmt_results, args.split, format=True)

    # total_time = time.time() - eval_start_time
    # # convert seconds to date
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Evaluate time {}'.format(total_time_str))

    # with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #     f.write(json.dumps(cfg.to_dict(), indent=4) + "\n")
    #     f.write(message + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='eval_configs/vtgllm.yaml')
    parser.add_argument('--anno_path', type=str, default='data/YouCook2-BB/YouCook2_asr_denseCap/')
    parser.add_argument('--video_path', type=str, default='data/YouCook2-BB/YouCook2_asr_denseCap/youcook2_6fps_224/')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--task',
                        default='dvc')  # dvc for dense video captioning; tvg for temporal video grounding; vhd for video highlight detection
    parser.add_argument('--dataset', default='youcook')
    parser.add_argument('--output_dir', default='debug')
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--timestamp', action='store_true', help='input the gt/predicted timestamps to the model')
    parser.add_argument('--timestamp_file', type=str, default='', help='the predcited timestamps file')
    parser.add_argument('--debug', action='store_true', help='the debug mode will only use 10 data samples')
    parser.add_argument('--prompt_file', default='prompts/dvc_description.txt')
    parser.add_argument('--vtgllm_model_path',
                        default='ckpt/vtgllm/train_stage2_llama2_7b_time64k_valley72k_bz32_f96_epoch3_open_i_instruct_qformer_lora_bind_time_ws32_mfp96_mtl2048/20231026060/checkpoint_2.pth')
    parser.add_argument('--sample_num', type=int, default=-1, help='fast inference by sampling N instances to evaluate')
    parser.add_argument('--example_output', action='store_true', help='output the example results')
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--post_check', action='store_true', help='post check the format of generated captions')
    parser.add_argument('--post_check_prompt_file', type=str, default='prompts/dvc_post_check.txt')
    parser.add_argument('--asr', action='store_true')
    parser.add_argument('--asr_path', type=str,
                        default='data/YouCook2-BB/YouCook2_asr_denseCap/whisper_outputs_with_time/small.en.cleaned/')
    args = parser.parse_args()
    main(args)
