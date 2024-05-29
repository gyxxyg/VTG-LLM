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
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


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

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]
        if conv.sep == "###":
            stop_words_ids = [torch.tensor([835]).to(self.device),
                              torch.tensor([2277, 29937]).to(
                                  self.device)]  # '###' can be encoded in two different ways.
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # stopping_criteria
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
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
        output_token = outputs[0]
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
        return output_text, output_token.cpu().numpy()

    def upload_video_without_audio(self, video_path, conv, img_list, n_frms=8):
        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            # print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            video, msg = load_video(
                video_path=video_path,
                n_frms=n_frms,
                height=224,
                width=224,
                sampling="uniform", return_msg=True
            )
            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device)
            # print(image)
            raw_timestamps = msg.split('at')[1].replace('seconds.', '').strip().split(',')
            if self.model.qformer_text_input:
                # timestamp  # extract timestamps from msg
                if not self.model.special_time_token:
                    timestamps = [f'This frame is sampled at {t.strip()} second.' for t in raw_timestamps]
                else:
                    timestamps = [f'This frame is sampled at {number_to_token(t.strip())} second.' for t in raw_timestamps]
                timestamps = self.model.tokenizer(
                    timestamps,
                    return_tensors="pt",
                    padding="longest",
                    max_length=32,
                    truncation=True,
                )
                # v_1
            absolute_timestamps = torch.tensor([round(float(t.strip())) for t in raw_timestamps])

                # v_2
                # absolute_timestamps = torch.tensor([float(t.strip()) for t in raw_timestamps])
        else:
            raise NotImplementedError
        # conv.system = "You can understand the video that the user provides.  Follow the instructions carefully and explain your answers in detail."
        if self.model.qformer_text_input:
            image_emb, _, _ = self.model.encode_videoQformer_visual(video, timestamp=timestamps, absolute_timestamp=absolute_timestamps.to(self.device))
        else:
            image_emb, _, _ = self.model.encode_videoQformer_visual(video, timestamp=None, absolute_timestamp=absolute_timestamps.to(self.device))
        img_list.append(image_emb)
        if not self.model.special_time_token:
            conv.append_message(conv.roles[0], "<Video><ImageHere></Video> " + msg)
        else:
            conv.append_message(conv.roles[0], "<Video><ImageHere></Video> " + msg_to_token(msg))
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

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        if self.model.lora:
            seg_embs = [self.model.llama_model.get_base_model().model.embed_tokens(seg_t) for seg_t in seg_tokens]
        else:
            seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
