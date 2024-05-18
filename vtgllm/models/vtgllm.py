import logging
import math
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from vtgllm.common.registry import registry
from vtgllm.models.blip2 import Blip2Base, disabled_train
from vtgllm.models.modeling_llama import LlamaForCausalLM
# from vtgllm.models.Qformer import BertEncoder
from transformers import LlamaTokenizer, BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from vtgllm.models.Qformer import BertConfig, BertLMHeadModel
from vtgllm.common.losses import TimeLoss
from vtgllm.common.utils import get_entropy

# from entropy_estimators import continuous


# from flamingo_pytorch import PerceiverResampler

class NoneSampler(nn.Module):
    def __init__(self):
        super(NoneSampler, self).__init__()
        
    def forward(self, x):
        return x,  0.0

class CrossAttentionSampler(nn.Module):

    def __init__(self, num_query, feature_size, nhead=8):
        super(CrossAttentionSampler, self).__init__()
        self.attention = nn.MultiheadAttention(feature_size, nhead, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, num_query, feature_size))

    def forward(self, x):

        attn_output, _ = self.attention(self.query, x, x)

        return attn_output, 0.0

class SlotSampler(nn.Module):

    def __init__(self, num_slots, feature_size):
        super(SlotSampler, self).__init__()
        self.slots = nn.Parameter(torch.randn(feature_size, num_slots))
        self.num_slots = num_slots

    def forward(self, x):

        logits = torch.matmul(x, self.slots)
        logits = torch.softmax(logits, dim=1)

        res = torch.matmul(x.permute(0,2,1), logits).permute(0, 2, 1)

        # print(res, res.shape)

        return res, 0.0


class CodebookSampler(nn.Module):
    def __init__(self, num_codes, feature_size):
        super(CodebookSampler, self).__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, feature_size))

    def forward(self, x):
        batch_size, token_num, dim = x.shape
        codebook_outputs = []
        codebook_losses = []

        for i in range(batch_size):
            # Compute the distances between each code vector and each input token
            distances = (self.codebook.unsqueeze(1) - x[i].unsqueeze(0)).pow(2).sum(-1)

            # Find the indices of the closest input tokens
            _, indices = distances.min(1)

            # Replace each code vector with its closest input token
            codebook_output = x[i, indices]

            # Use the straight-through estimator for the backward pass
            codebook_output_ste = self.codebook + (codebook_output - self.codebook).detach()

            codebook_loss = F.mse_loss(codebook_output, self.codebook.detach()) + F.mse_loss(self.codebook, codebook_output.detach())

            codebook_losses.append(codebook_loss)
            codebook_outputs.append(codebook_output_ste.unsqueeze(0))

        codebook_output = torch.cat(codebook_outputs, dim=0)
        codebook_loss = sum(codebook_losses) / len(codebook_losses)

        return codebook_output, codebook_loss


class DiverseSampler(nn.Module):
    def __init__(self, k):
        super(DiverseSampler, self).__init__()
        self.k = k

    def forward(self, x):
        batch_size, token_num, dim = x.shape
        diverse_tokens = []

        for i in range(batch_size):
            diverse_indices = self.k_means_plus_plus(x[i], self.k)
            # print('diverse_indicies', diverse_indices)
            diverse_tokens.append(x[i, diverse_indices].unsqueeze(0))

        return torch.cat(diverse_tokens, dim=0), 0.0

    @staticmethod
    def k_means_plus_plus(x, k):
        indices = torch.arange(len(x))
        first_index = torch.randint(len(x), (1,)).item()
        selected_indices = [first_index]
        distances = torch.norm(x - x[first_index], dim=-1)

        for _ in range(k - 1):
            next_index = torch.multinomial(distances, 1).item()
            selected_indices.append(next_index)
            new_distances = torch.norm(x - x[next_index], dim=-1)
            distances = torch.min(distances, new_distances)

        return indices[selected_indices]

class EntropySampler(nn.Module):

    def __init__(self, k):
        super(EntropySampler, self).__init__()
        self.k = k

    '''
    sample tokens that have the maximum entropy
    '''
    def forward(self, x):

        batch_size, token_num, dim = x.shape
        sampled_tokens = []

        for i in range(batch_size):
            entropy = torch.tensor(get_entropy(x[i].cpu().detach().numpy(), k=5))
            sampled_indicies = torch.multinomial(entropy, self.k)
            sampled_tokens.append(x[i, sampled_indicies].unsqueeze(0))

        sampled_tokens = torch.cat(sampled_tokens, dim=0)

        # print(sampled_tokens)


        return sampled_tokens, 0.0



@registry.register_model("vtgllm")
class vtgllm(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_llama_v2": "configs/models/vtgllm.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
            self,
            vit_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

            frozen_llama_proj=True,
            use_video_qformer=True,
            frozen_video_Qformer=True,

            llama_proj_model='',
            fusion_header_type="seqTransf",
            max_frame_pos=32,
            max_time_pos=1024,
            fusion_head_layers=2,
            num_video_query_token=32,
            lora=False,
            qformer_text_input=False,
            lora_inference_mode=True,
            window_size=0,
            stride=0,
            real_time_stamp=False,
            real_time_stamp_random_init=False,
            special_time_token=False,
            time_loss=None,
            sampler_type='none',
            sample_num=256
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.real_time_stamp = real_time_stamp
        self.special_time_token = special_time_token
        self.time_loss=time_loss
        self.use_video_qformer=use_video_qformer
        self.sampler_type = sampler_type
        assert not (self.use_video_qformer and self.sampler_type != 'none'), "can not use sampler and video qformer at the same time!"
        print('use_video_qformer', use_video_qformer)
        print('num query token', num_query_token)

        print('Real time stamp: ', real_time_stamp, special_time_token)

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, max_frame_pos
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.Qformer.cls = None
        else:
            print("use text input for Qformer")
            self.Qformer.resize_token_embeddings(len(self.tokenizer))

        # self.Qformer.cls = None
        self.qformer_text_input = qformer_text_input
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

        # if self.special_time_token:
        #    self.llama_tokenizer.add_tokens(SPECIAL_TIME_TOKENS)

        # need to update llama embed dimension

        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            if max_txt_len > 2048:
                logging.info(f"interpolate llama model's rope from 2048 to {max_txt_len}")
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                    max_position_embeddings=max_txt_len,
                    rope_scaling={
                        "type": "linear",
                        "factor": 2.0
                    }
                )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                )

        if use_grad_checkpoint:
            logging.info("use gradient checkpointing for LLAMA")
            self.llama_model.gradient_checkpointing_enable()

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')

        self.lora = lora
        if self.lora:
            logging.info('Using LORA')
            from peft import LoraConfig, get_peft_model, TaskType
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=lora_inference_mode,
                r=32,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []


        self.max_time_pos = max_time_pos
        print('self max time pos', max_time_pos)
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)
        self.video_frame_position_embedding.weight.data = torch.zeros_like(self.video_frame_position_embedding.weight.data).float()

        if self.real_time_stamp:
            # v_1
            # self.video_frame_position_embedding = nn.Embedding(max_time_pos, self.Qformer.config.hidden_size)

            # v_2
            # self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)
            # self.video_time_position_embedding = nn.Embedding(max_time_pos, self.Qformer.config.hidden_size)
            if not real_time_stamp_random_init:
                self.video_time_position_embedding = nn.Parameter(torch.zeros((max_time_pos, self.Qformer.config.hidden_size), requires_grad=True).float())
            else:
                print('Random init real time!')
                self.video_time_position_embedding = nn.Parameter(torch.randn((max_time_pos, self.Qformer.config.hidden_size), requires_grad=True).float())

        self.num_video_query_token = num_video_query_token

        self.window_size = window_size
        self.stride = stride
        
        if self.use_video_qformer:
            self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(num_query_token=num_video_query_token, \
                                                                                vision_width=self.Qformer.config.hidden_size,
                                                                                num_hidden_layers=2)

            self.video_Qformer.cls = None
            self.video_Qformer.bert.embeddings.word_embeddings = None
            self.video_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.video_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

            if frozen_video_Qformer:
                #  todo frozen  llama_proj
                for name, param in self.video_Qformer.named_parameters():
                    param.requires_grad = False
                for name, param in self.video_frame_position_embedding.named_parameters():
                    param.requires_grad = False
                self.video_query_tokens.requires_grad = False

                logging.info('video_Qformer is frozen')
            else:
                for name, param in self.video_Qformer.named_parameters():
                    param.requires_grad = True
                for name, param in self.video_frame_position_embedding.named_parameters():
                    param.requires_grad = True
                self.video_query_tokens.requires_grad = True
                logging.info('video_Qformer is not frozen')
        else:
            self.video_Qformer, self.video_query_tokens = None, None

        if self.sampler_type == 'none':
            self.sampler = NoneSampler()
        elif self.sampler_type == 'diverse':
            self.sampler = DiverseSampler(sample_num)
        elif self.sampler_type == 'codebook':
            self.sampler = CodebookSampler(sample_num, self.Qformer.config.hidden_size)
            for name, param in self.sampler.named_parameters():
                param.requires_grad = True
        elif self.sampler_type == 'entropy':
            self.sampler = EntropySampler(sample_num)
        elif self.sampler_type == 'slot':
            self.sampler = SlotSampler(sample_num, self.Qformer.config.hidden_size)
            if not frozen_video_Qformer:
                for name, param in self.sampler.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in self.sampler.named_parameters():
                    param.requires_grad = False
        elif self.sampler_type == 'cross':
            self.sampler = CrossAttentionSampler(sample_num, self.Qformer.config.hidden_size)
            if not frozen_video_Qformer:
                for name, param in self.sampler.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in self.sampler.named_parameters():
                    param.requires_grad = False
        else:
            raise ValueError('Sampler Type not Supported')

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_videoQformer_visual(self, image, timestamp=None, absolute_timestamp=None):
        device = image.device
        # print(device)

        # self.visual_encoder.to(device)
        # self.ln_vision.to(device)

        # input shape b,c,t,h,w
        # print('timestamp begin', absolute_timestamp, timestamp["input_ids"])
        batch_size, _, time_length, _, _ = image.size()
        # x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            # print(image_embeds.shape)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            if self.qformer_text_input:
                # timestamps_input_ids = einops.rearrange(timestamp["input_ids"], 'b t d -> (b t) d')
                # timestamps_attention_mask = einops.rearrange(timestamp["attention_mask"], 'b t d -> (b t) d')
                timestamps_input_ids = timestamp["input_ids"].to(device)
                timestamps_attention_mask = timestamp["attention_mask"].to(device)
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                Qformer_atts = torch.cat([query_atts, timestamps_attention_mask], dim=1)
                query_output = self.Qformer.bert(
                    timestamps_input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
        
            # print(query_output.last_hidden_state.shape)


            # add frame_pos embedding
            if not self.real_time_stamp:
                position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                frame_position_embeddings = self.video_frame_position_embedding(position_ids)
                frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            else:
                # v_1 -- use absolute time as position embedding
                # position_ids = absolute_timestamp
                # v_2 -- add additional time embedding, and initialize as zero
                position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

                frame_position_embeddings = self.video_frame_position_embedding(position_ids)
                time_position_embeddings = torch.matmul(F.one_hot(absolute_timestamp, self.max_time_pos).float(), self.video_time_position_embedding)

                frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
                time_position_embeddings = time_position_embeddings.unsqueeze(-2)

                frame_position_embeddings = frame_position_embeddings + time_position_embeddings
            q_hidden_state = query_output.last_hidden_state

            # print(frame_position_embeddings.shape, q_hidden_state.shape)

            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # print(frame_hidden_state.shape)


            # frame attention
            if self.use_video_qformer:
                if self.window_size <= 0:
                    # use frames
                    frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size,
                                                        t=time_length)
                    frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
                    video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1,
                                                                        -1)  # expand on batch dim

                    video_query_output = self.video_Qformer.bert(
                        query_embeds=video_query_tokens,
                        encoder_hidden_states=frame_hidden_state,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                    video_hidden = video_query_output.last_hidden_state

                    inputs_llama = self.llama_proj(video_hidden)
                    atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
                else:
                    # use clips
                    inputs_llama_list, atts_llama_list = [], []
                    for i in range(0, time_length, self.stride):
                        clip_hidden_state = frame_hidden_state[:, i:i + self.window_size, ...]
                        clip_hidden_state = einops.rearrange(clip_hidden_state, 'b t q h -> b (t q) h', b=batch_size)
                        clip_atts = torch.ones(clip_hidden_state.size()[:-1], dtype=torch.long).to(device)
                        video_query_tokens = self.video_query_tokens.expand(clip_hidden_state.shape[0], -1,
                                                                            -1)  # expand on batch dim
                        
                        video_query_output = self.video_Qformer.bert(
                            query_embeds=video_query_tokens,
                            encoder_hidden_states=clip_hidden_state,
                            encoder_attention_mask=clip_atts,
                            return_dict=True,
                        )
                        video_hidden = video_query_output.last_hidden_state  # [bsz, t, dim]

                        inputs_llama = self.llama_proj(video_hidden)
                        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
                        inputs_llama_list.append(inputs_llama)
                        atts_llama_list.append(atts_llama)

                    inputs_llama = torch.cat(inputs_llama_list, dim=1)  # [bsz, t, dim]
                    atts_llama = torch.cat(atts_llama_list, dim=1)  # [bsz, t]
                    # print(inputs_llama.shape, atts_llama.shape)
                sample_loss = 0.0
            else:
                video_hidden = frame_hidden_state.view(batch_size, -1, frame_hidden_state.shape[-1])
                # print('before sampler', video_hidden.shape)
                video_hidden, sample_loss = self.sampler(video_hidden)
                # print('after sampler', video_hidden.shape)
                inputs_llama = self.llama_proj(video_hidden)
                atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)



        return inputs_llama, atts_llama, sample_loss

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            # print(prompt)
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            if self.lora:  # peft
                p_before_embeds = self.llama_model.get_base_model().model.embed_tokens(
                    p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.get_base_model().model.embed_tokens(p_after_tokens.input_ids).expand(
                    batch_size, -1, -1)
            else:
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1,
                                                                                                        -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1,
                                                                                                      -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])

            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        if 'conv_type' in samples.keys() and samples['conv_type'] == 'multi':

            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            if isinstance(image, list):  # nb of frames of some videos is less than ${num_frm}
                assert isinstance(samples["timestamps"], list) and isinstance(samples["absolute_timestamps"], list)
                img_embeds_list, atts_img_list, num_patch_tokens_list, sample_loss_list = [], [], [], []
                for img, timestamp, absolute_timestamp in zip(image, samples["timestamps"], samples["absolute_timestamps"]):
                    img = img.unsqueeze(0)
                    if len(img.size()) == 4:
                        time = 1
                        img = einops.repeat(img, 'b c h w -> b c t h w', t=time)

                    # num_patch_tokens = self.num_video_query_token * math.ceil(
                        # img.shape[2] / self.stride) if self.stride > 0 else self.num_video_query_token
                    img_embeds, atts_img, sample_loss = self.encode_videoQformer_visual(img, timestamp=timestamp, absolute_timestamp=absolute_timestamp)
                    num_patch_tokens = img_embeds.shape[1]
                    img_embeds_list.append(img_embeds)
                    atts_img_list.append(atts_img)
                    num_patch_tokens_list.append(num_patch_tokens)
                    sample_loss_list.append(sample_loss)
                img_embeds = img_embeds_list
                atts_img = atts_img_list
                sample_loss = sum(sample_loss_list) / len(sample_loss_list)
            else:  # nb of frames of all videos is ${num_frm}
                if len(image.size()) == 4:
                    time = 1
                    image = einops.repeat(image, 'b c h w -> b c t h w', t=time)

                # num_patch_tokens = self.num_video_query_token * math.ceil(
                #     image.shape[2] / self.stride) if self.stride > 0 else self.num_video_query_token
                # print(image, samples)
                img_embeds, atts_img, sample_loss = self.encode_videoQformer_visual(image, timestamp=samples["timestamps"], absolute_timestamp=samples["absolute_timestamps"])
                num_patch_tokens = img_embeds.shape[1]

            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            if self.lora:
                temp_input_embedding = self.llama_model.get_base_model().model.embed_tokens(temp_input_ids)
            else:
                temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)


            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]  # [num_video_query_token, dim]
                if isinstance(image, list):
                    cur_image_features = cur_image_features.squeeze(0)
                    num_patch_tokens = num_patch_tokens_list[cur_image_idx]
                # print(num_patch_tokens, (cur_input_ids == im_patch_token_id).sum())
                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                    raise ValueError(
                        "The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patch_tokens,
                                                   device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")

                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features,
                                                  cur_input_embeds[mask_index_start + num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)

                cur_image_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            # print(inputs_embeds.shape, targets.shape, attention_mask)
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            # print(loss, sample_loss)
            return {"loss": loss + sample_loss}
        else:
            image = samples["image"]

            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w', t=time)

            img_embeds, atts_img, sample_loss = self.encode_videoQformer_visual(image)

            if self.prompt_list:
                prompt = random.choice(self.prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

            self.llama_tokenizer.padding_side = "right"

            text = [t + self.end_sym for t in samples["text_input"]]

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1] + 1],
                           dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                             dtype=to_regress_tokens.input_ids.dtype,
                             device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            if self.lora:
                bos_embeds = self.llama_model.get_base_model().model.embed_tokens(bos)
                to_regress_embeds = self.llama_model.get_base_model().model.embed_tokens(to_regress_tokens.input_ids)
            else:
                bos_embeds = self.llama_model.model.embed_tokens(bos)
                to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            atts_bos = atts_img[:, :1]

            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss

        return {"loss": loss + sample_loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model",
                            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth")
        q_former_model = cfg.get("q_former_model",
                                 "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        lora = cfg.get("lora", False)
        lora_inference_mode = cfg.get("lora_inference_mode", False)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", '')

        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        max_time_pos = cfg.get("max_time_pos", 1024)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token = cfg.get("num_video_query_token", 32)

        qformer_text_input = cfg.get("qformer_text_input", False)
        window_size = cfg.get("window_size", 0)
        stride = cfg.get("stride", 0)
        real_time_stamp = cfg.get('real_time_stamp', False)
        real_time_stamp_random_init = cfg.get('real_time_stamp_random_init', False)
        special_time_token = cfg.get('special_time_token', False)
        added_time_token =  cfg.get('added_time_token', False)
        time_loss = cfg.get('time_loss', False)
        use_video_qformer = cfg.get('use_video_qformer', True)
        sampler_type = cfg.get('sampler_type', 'none')
        sample_num = cfg.get('sample_num', 256)
        time_embedding_interpolation=cfg.get('time_embedding_interpolation', False)
        time_token_initialization=cfg.get('time_token_initialization', True)
        print('max time pos', max_time_pos)
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            max_time_pos=max_time_pos,
            fusion_head_layers=fusion_head_layers,
            use_video_qformer=use_video_qformer,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            num_video_query_token=num_video_query_token,
            llama_proj_model=llama_proj_model,
            lora=lora,
            qformer_text_input=qformer_text_input,
            lora_inference_mode=lora_inference_mode,
            window_size=window_size,
            stride=stride,
            real_time_stamp=real_time_stamp,
            real_time_stamp_random_init=real_time_stamp_random_init,
            special_time_token=special_time_token,
            time_loss=None,
            sampler_type=sampler_type,
            sample_num=sample_num
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4

        if special_time_token and added_time_token:
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
            
            model.tokenizer.add_tokens(SPECIAL_TIME_TOKENS)
            model.llama_tokenizer.add_tokens(SPECIAL_TIME_TOKENS)
            if qformer_text_input:
                model.Qformer.resize_token_embeddings(len(model.tokenizer))

            model.llama_model.resize_token_embeddings(len(model.llama_tokenizer))

            if lora_inference_mode:
                if lora:
                    model.llama_model.base_model.model.model.embed_tokens.weight.requires_grad = False
                    model.llama_model.base_model.model.lm_head.weight.requires_grad = False
                else:
                    model.llama_model.model.embed_tokens.weight.requires_grad = False
                    model.llama_model.lm_head.weight.requires_grad = False
            else:
                if lora:
                    model.llama_model.base_model.model.model.embed_tokens = model.llama_model.base_model.model.model.embed_tokens.float()
                    model.llama_model.base_model.model.lm_head = model.llama_model.base_model.model.lm_head.float()
                    model.llama_model.base_model.model.model.embed_tokens.weight.requires_grad=True
                    model.llama_model.base_model.model.lm_head.weight.requires_grad=True
                else:
                    model.llama_model.model.embed_tokens = model.llama_model.model.embed_tokens.float()
                    model.llama_model.lm_head = model.llama_model.lm_head.float()
                    model.llama_model.model.embed_tokens.weight.requires_grad=True
                    model.llama_model.lm_head.weight.requires_grad=True

        # ckpt_path_2 = cfg.get("ckpt_2", "")
        # if ckpt_path_2:
        #     print("Load pre Checkpoint: {}".format(ckpt_path_2))
        #     ckpt = torch.load(ckpt_path_2, map_location="cpu")
        #     if 'llama_model.model.embed_tokens.weight' in ckpt['model']:
        #         ckpt['model']['llama_model.base_model.model.model.embed_tokens.weight'] = ckpt['model']['llama_model.model.embed_tokens.weight']
        #     if 'llama_model.lm_head.weight' in ckpt['model']:
        #         ckpt['model']['llama_model.base_model.model.lm_head.weight'] = ckpt['model']['llama_model.lm_head.weight']
        #     msg = model.load_state_dict(ckpt['model'], strict=False)
        #     for key in ckpt['model'].keys():
        #         if key not in msg.missing_keys and key not in msg.unexpected_keys:
        #             print('ckpt pre', key)

        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ori_ckpt = torch.load(ckpt_path, map_location="cpu")
            # print(ori_ckpt.keys())
            ckpt = {'model': {}}
            if 'model' not in ori_ckpt:
                for key in ori_ckpt.keys():
                    # if key.startswith('llama_model.model.layers') or key.startswith('llama_model.model.layers')
                    ckpt['model'][key.replace('vision_encoder', 'visual_encoder').replace('qformer', 'Qformer').replace('vision_layernorm', 'ln_vision')] = ori_ckpt[key]
            else:
                for key in ori_ckpt['model'].keys():
                    ckpt['model'][key.replace('vision_encoder', 'visual_encoder').replace('qformer', 'Qformer').replace('vision_layernorm', 'ln_vision')] = ori_ckpt['model'][key]

            if special_time_token and added_time_token and lora:
                if 'llama_model.model.embed_tokens.weight' in ckpt['model']:
                    ckpt['model']['llama_model.base_model.model.model.embed_tokens.weight'] = ckpt['model']['llama_model.model.embed_tokens.weight']
                if 'llama_model.lm_head.weight' in ckpt['model']:
                    ckpt['model']['llama_model.base_model.model.lm_head.weight'] = ckpt['model']['llama_model.lm_head.weight']

            if 'video_frame_position_embedding.weight' in ckpt['model']:
                old_frame_pos_embed_size = ckpt['model']['video_frame_position_embedding.weight'].size()
                new_frame_pos_embed_size = model.video_frame_position_embedding.weight.size()
                if old_frame_pos_embed_size != new_frame_pos_embed_size:
                    from vtgllm.processors.video_processor import interpolate_frame_pos_embed
                    print(
                        f'video_frame_position_embedding size is not the same, interpolate from {old_frame_pos_embed_size} to {new_frame_pos_embed_size}')
                    ckpt['model']['video_frame_position_embedding.weight'] = interpolate_frame_pos_embed(
                        ckpt['model']['video_frame_position_embedding.weight'], new_n_frm=new_frame_pos_embed_size[0])
            
            if 'video_time_position_embedding' in ckpt['model'] and time_embedding_interpolation:
                print('Start test time interpolation')
                new_weights = torch.zeros_like(ckpt['model']['video_time_position_embedding'])
                all_zero_times = (ckpt['model']['video_time_position_embedding'] == 0).all(dim=1).int()
                print(f'not trained timestamps: {all_zero_times.nonzero().flatten()}')
                bounds = []
                left_margin = 0
                stack = []
                for i in range(len(all_zero_times)):
                    if all_zero_times[i] == 1:
                        stack.append([i, left_margin])
                    else:
                        while len(stack):
                            pre = stack.pop()
                            pre.append(i)
                            bounds.append(pre)
                        left_margin = i
                while len(stack):
                    pre = stack.pop()
                    pre.append(-1)
                    bounds.append(pre)
                
                for bound in bounds:

                    idx, left, right = bound[0], bound[1], bound[2]
                    left_weights = ckpt['model']['video_time_position_embedding'][left]
                    right_weights = ckpt['model']['video_time_position_embedding'][right]
                    idx_weights = left_weights * (idx - left) / (right - left) + right_weights * (right - idx) / (right - left)
                    ckpt['model']['video_time_position_embedding'][idx] = idx_weights

                print(ckpt['model']['video_time_position_embedding'])


                



            

            
            msg = model.load_state_dict(ckpt['model'], strict=False)
            for key in ckpt['model'].keys():
                if key in msg.missing_keys:
                    print('missing', key)

            for key in ckpt['model'].keys():
                if key in msg.unexpected_keys:
                    print('extra', key)

            # for key in ckpt['model'].keys():
            #     if key not in msg.unexpected_keys and key not in msg.missing_keys:
            #         print('loaded', key)

            


        if special_time_token and not added_time_token:
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

            model.tokenizer.add_tokens(SPECIAL_TIME_TOKENS)
            model.llama_tokenizer.add_tokens(SPECIAL_TIME_TOKENS)
            if qformer_text_input:
                model.Qformer.resize_token_embeddings(len(model.tokenizer))
            model.llama_model.resize_token_embeddings(len(model.llama_tokenizer))

            # v2 --- change copy the embedding of 0-9 and dot to new embedding.
            # need to change: 
            # module.Qformer.bert.embeddings.word_embeddings.weight; 
            # module.llama_model.base_model.model.model.embed_tokens.weight; 
            # module.llama_model.base_model.model.lm_head.weight
            # print(model.llama_tokenizer.get_vocab())
            original_tokens = ['{}'.format(i) for i in range(10)]
            original_tokens.append('.')
            print(original_tokens)
            original_bert_ids = [model.tokenizer.encode(i)[1] for i in original_tokens]
            new_bert_ids = [model.tokenizer.encode(i)[1] for i in SPECIAL_TIME_TOKENS]
            original_llama_ids = [ model.llama_tokenizer.encode(i)[-1] for i in original_tokens]
            new_llama_ids = [model.llama_tokenizer.encode(i)[-1] for i in SPECIAL_TIME_TOKENS]



            print('initializing tokenizer weights')
            if qformer_text_input:
                model.Qformer.bert.embeddings.word_embeddings.weight.data[new_bert_ids] = model.Qformer.bert.embeddings.word_embeddings.weight.data[original_bert_ids]
            if lora:
                if time_token_initialization:
                    model.llama_model.base_model.model.model.embed_tokens.weight.data[new_llama_ids] = model.llama_model.base_model.model.model.embed_tokens.weight.data[original_llama_ids]
                    model.llama_model.base_model.model.lm_head.weight.data[new_llama_ids] = model.llama_model.base_model.model.lm_head.weight.data[original_llama_ids]

                model.llama_model.base_model.model.model.embed_tokens = model.llama_model.base_model.model.model.embed_tokens.float()
                model.llama_model.base_model.model.lm_head = model.llama_model.base_model.model.lm_head.float()
                model.llama_model.base_model.model.model.embed_tokens.weight.requires_grad=True
                model.llama_model.base_model.model.lm_head.weight.requires_grad=True
                print(model.llama_model.base_model.model.lm_head.weight)
            else:
                if time_token_initialization:
                    model.llama_model.model.embed_tokens.weight.data[new_llama_ids] = model.llama_model.model.embed_tokens.weight.data[original_llama_ids]
                    model.llama_model.lm_head.weight.data[new_llama_ids] =  model.llama_model.lm_head.weight.data[original_llama_ids]

                model.llama_model.model.embed_tokens = model.llama_model.model.embed_tokens.float()
                model.llama_model.lm_head =model.llama_model.lm_head.float()
                model.llama_model.model.embed_tokens.weight.requires_grad=True
                model.llama_model.lm_head.weight.requires_grad=True
                print(model.llama_model.lm_head.weight)

        print('time loss', time_loss)
        if time_loss:
            if lora:
                model.llama_model.base_model.model.time_loss = TimeLoss(model.llama_tokenizer, special_time_token)
            else:
                model.llama_model.time_loss = TimeLoss(model.llama_tokenizer, special_time_token)

        model.Qformer.cls = None
        # model.llama_model.print_trainable_parameters()
        return model
