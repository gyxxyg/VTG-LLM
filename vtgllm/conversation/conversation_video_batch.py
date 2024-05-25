"""
Conversation prompt template of vtgllm.
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/conversation/conversation.py 
"""
import argparse
import time
from PIL import Image
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import re

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
import os
from vtgllm.common.registry import registry
from vtgllm.processors.video_processor import ToTHWC, ToUint8, load_video
from vtgllm.processors import Blip2ImageEvalProcessor

TIME_STR = {
    0: 'ZERO',
    1: 'ONE',
    2: 'TWO',
    3: 'THREE',
    4: 'FOUR',
    5: 'FIVE',
    6: 'SIX',
    7: 'SEVEN',
    8: 'EIGHT',
    9: 'NINE'
}

SPECIAL_TIME_TOKENS = ['<TIME_{}>'.format(TIME_STR[i]) for i in range(10)]
SPECIAL_TIME_TOKENS.append('<TIME_DOT>')

def digit_to_token(digit):
    if digit == '.':
        return '<TIME_DOT>'
    elif digit in [str(i) for i in range(10)]:
        return '<TIME_{}>'.format(digit)
    else:
        return digit

def number_to_token(number):
    if not isinstance(number, str):
        number = str(number)
    
    # res = ''.join([digit_to_token(i) for i in number])

    # v3
    res = []
    # print(number)
    for i, digit in enumerate(number):
        if digit == '.' and i  < 4:
            res = ['<TIME_ZERO>'] * (4 - i) + res
        if  digit == '.':
            res += [SPECIAL_TIME_TOKENS[-1]]
        else:
            res += [SPECIAL_TIME_TOKENS[int(digit)]]
    if  '<TIME_DOT>' not in res:
        res += ['<TIME_DOT>', '<TIME_ZERO>']
    if len(res) < 6:
        res = ['<TIME_ZERO>'] * (6 - len(res)) + res
    return ''.join(res[:6])

def msg_to_token(msg):
    msg = msg.split('at')

    matches = re.findall(r'[0-9]+\.?[0-9]*', msg[1])

    replacements = {match: number_to_token(match) for match in matches} 

    for match in matches:
        msg[1] = msg[1].replace(match, replacements[match], 1)

    return 'at'.join(msg)



class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for i in range(input_ids.shape[0]):
            flag = False
            for stop in self.stops:
                for _pos in range(len(input_ids[i]) - len(stop) + 1):
                    pos = len(input_ids[i]) - len(stop) - _pos
                    if torch.all((stop == input_ids[i][pos:pos + len(stop)])).item():
                        flag = True
                        break
                if flag:
                    break
            if not flag:
                return False

        return True


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

default_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)


class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        # stop_words_ids = [torch.tensor([835]).to(self.device),
        #                   torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        # self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and (
                '</Video>' in conv.messages[-1][1] or '</Image>' in conv.messages[-1][1]):  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, convs, img_lists, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):

        for conv in convs:
            conv.append_message(conv.roles[1], None)

        embs, attn_mask = self.get_context_emb(convs, img_lists)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        # print(embs.shape, begin_idx, len(img_lists))

        embs = embs[:, begin_idx:]
        attn_mask = attn_mask[:, begin_idx:]
        if conv.sep == "###":
            stop_words_ids = [torch.tensor([835]).to(self.device),
                              torch.tensor([2277, 29937]).to(
                                  self.device)]  # '###' can be encoded in two different ways.
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # stopping_criteria
        # print(embs.shape, max_length, max_new_tokens, attn_mask.shape)
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        output_texts = []
        output_tokens = []

        for i, conv in enumerate(convs):
            output_token = outputs[i]
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            if conv.sep == "###":
                output_text = output_text.split('###')[0]  # remove the stop sign '###'
                output_text = output_text.split('Assistant:')[-1].strip()
            else:
                output_text = output_text.split(conv.sep2)[0]  # remove the stop sign '###'
                output_text = output_text.split(conv.roles[1] + ':')[-1].strip()
            conv.messages[-1][1] = output_text
            output_texts.append(output_text)
            output_tokens.append(output_token.cpu().numpy())

        return output_texts, output_tokens

    def upload_video_without_audio(self, video_paths, convs, img_lists, n_frms=8):
        msgs = []
        videos = []
        all_timestamps = [] if self.model.qformer_text_input else None
        all_absolute_timestamps = []
        for video_path in video_paths:
            if isinstance(video_path, str):  # is a video path
                video, msg = load_video(
                    video_path=video_path,
                    n_frms=n_frms,
                    height=224,
                    width=224,
                    sampling="uniform", return_msg=True
                )
                video = self.vis_processor.transform(video)
                videos.append(video)
                if not self.model.special_time_token:
                    msgs.append(msg)
                else:
                    msgs.append(msg_to_token(msg))
                # print(image)
                raw_timestamps = msg.split('at')[1].replace('seconds.', '').strip().split(',') # extract timestamps from msg
                if self.model.qformer_text_input:
                    # timestamp
                    if not self.model.special_time_token:
                        timestamps = [f'This frame is sampled at {t.strip()} second.' for t in raw_timestamps]
                    else:
                        timestamps = [f'This frame is sampled at {number_to_token(t.strip())} second.' for t in raw_timestamps]
                    all_timestamps.append(timestamps)
                # v_1
                all_absolute_timestamps.append(torch.tensor([round(float(t.strip())) for t in raw_timestamps]))
                # v_2
                # all_absolute_timestamps.append(torch.tensor([float(t.strip()) for t in raw_timestamps]))
            else:
                raise NotImplementedError

        n_frms_in_batch = [v.shape[1] for v in videos]
        if min(n_frms_in_batch) == max(n_frms_in_batch):  # nb of frames of all videos is ${num_frm}
            videos = torch.stack(videos, dim=0).to(self.device)
            # conv.system = "You can understand the video that the user provides.  Follow the instructions carefully and explain your answers in detail."
            if self.model.qformer_text_input:
                # timestamp
                all_timestamps = sum(all_timestamps, [])
                all_timestamps = self.model.tokenizer(
                    all_timestamps,
                    return_tensors="pt",
                    padding="longest",
                    max_length=32,
                    truncation=True,
                )
            # print('absolute', all_absolute_timestamps, torch.stack(all_absolute_timestamps))
            video_embs, _, _ = self.model.encode_videoQformer_visual(videos, timestamp=all_timestamps, absolute_timestamp=torch.stack(all_absolute_timestamps).to(self.device))
        else:  # nb of frames of some videos is less than ${num_frm}
            video_embs = []
            for video, timestamps, absolute_timestamps in zip(videos, all_timestamps, all_absolute_timestamps):
                video = video.unsqueeze(0).to(self.device)
                timestamps = self.model.tokenizer(
                    timestamps,
                    return_tensors="pt",
                    padding="longest",
                    max_length=32,
                    truncation=True,
                )
                video_emb, _, _ = self.model.encode_videoQformer_visual(video, timestamp=timestamps, absolute_timestamp=absolute_timestamps.to(self.device))
                video_embs.append(video_emb.squeeze(0))

        for conv, img_list, msg, video_emb in zip(convs, img_lists, msgs, video_embs):
            img_list.append(video_emb.unsqueeze(0))
            conv.append_message(conv.roles[0], "<Video><ImageHere></Video> " + msg)
        return "Received."

    def upload_img(self, image, conv, img_list):

        msg = ""
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')  # 增加一个时间维度
            image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        else:
            raise NotImplementedError

        image_emb, _, _ = self.model.encode_videoQformer_visual(image)
        img_list.append(image_emb)
        # Todo msg=""
        conv.append_message(conv.roles[0], "<Image><ImageHere></Image> " + msg)

        return "Received."

    def get_context_emb(self, convs, img_lists, lora=False):
        if not self.model.lora:
            embed_model = self.model.llama_model.model 
        elif not self.model.second_lora:
            embed_model = self.model.llama_model.get_base_model().model  
        else:
            embed_model = self.model.llama_model.get_base_model().get_base_model().model

        pad_token_id = self.model.llama_tokenizer.pad_token_id
        pad_emb = embed_model.embed_tokens(torch.tensor([pad_token_id], device=self.device))  # 1 * 4096

        max_len = 0
        batch_mixed_embs = []
        for ci, conv in enumerate(convs):
            prompt = conv.get_prompt()
            prompt_segs = prompt.split('<ImageHere>')
            img_list = img_lists[ci]
            # print(prompt_segs)
            assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."

            seg_tokens = [
                self.model.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]

            # print(seg_tokens, seg_tokens[0].shape, seg_tokens[1].shape)

            seg_embs = [embed_model.embed_tokens(seg_t) for seg_t in seg_tokens]
            # print(seg_embs[0].shape, seg_embs[1].shape, img_list[0].shape)
            mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
            mixed_embs = torch.cat(mixed_embs, dim=1)

            # print(mixed_embs.shape)

            batch_mixed_embs.append(mixed_embs)
            max_len = max(max_len, mixed_embs.shape[1])

        mixed_embs = torch.zeros(len(convs), max_len, mixed_embs.shape[-1], device=self.device)
        attn_mask = torch.zeros(len(convs), max_len, device=self.device)


        for bi, mixed_emb in enumerate(batch_mixed_embs):
            # mixed_emb: 1 * L * H
            assert mixed_emb.shape[0] == 1, "Batch size should be 1."
            l = mixed_emb.shape[1]
            pad_len = max_len - l
            mixed_embs[0] = torch.cat([pad_emb] * pad_len + [mixed_emb[0]], dim=0)
            attn_mask[bi, -l:] = 1


        return mixed_embs, attn_mask
