import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from transformers import LlamaTokenizer, BertConfig
from torch.distributions import Normal


class TimeLoss:

    def __init__(self, tokenizer, special_time_token=True):

        assert special_time_token == True, "Currently only support speical time token when calculating time loss"

        assert isinstance(tokenizer,  LlamaTokenizer), "Current only support LlamaTokenizer"

        self.special_time_token = special_time_token
        self.tokenizer = tokenizer

        self.time_token_ids = self.get_time_token_ids()

        self.digit_ids = [id for id in self.time_token_ids.keys() if self.time_token_ids[id] != '<TIME_DOT>']

        self.normal_dist = Normal(0.0, 0.5)

    def get_time_token_ids(self):
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

        if isinstance(self.tokenizer,  LlamaTokenizer):
            time_token_ids = {self.tokenizer.encode(i)[-1]:i for i in SPECIAL_TIME_TOKENS}
        else:
            raise ValueError("Unknown Tokenizer Type!")

        print(time_token_ids)

        return time_token_ids

    def loss(self, logits, targets):

        # weights = torch.zeros_like(targets).float()

        # target_indicies_stack = []

        # for i, target in enumerate(targets):
        #     # print(target, target in self.time_token_ids)
        #     if target.item() not in self.time_token_ids:
        #         gap = 0
        #         while len(target_indicies_stack):
        #             index = target_indicies_stack.pop()
        #             weights[index] = 10 ** gap
        #             gap += 1
        #     elif self.time_token_ids[target.item()] != '<TIME_DOT>':
        #         target_indicies_stack.append(i)

        # V2
        # loss = F.cross_entropy(logits, targets, reduction='none')
        # time_loss_count = 0
        # pre_indicies = []
        # for i, target in enumerate(targets):
        #     if target.item() not in self.time_token_ids:
        #         gap = 0
        #         for j in range(len(target_indicies_stack) - 1, -1, -1):
        #             index = target_indicies_stack[j]
        #             weights[index] = 10 ** gap
        #             gap += 1
        #         target_indicies_stack = pre_indicies + target_indicies_stack
        #         if len(target_indicies_stack) > 0:
        #             target_indicies_weights = weights[target_indicies_stack]
        #             # print(target_indicies_weights)
        #             weights[target_indicies_stack] = target_indicies_weights / sum(target_indicies_weights)
        #             time_loss_count += 1
        #             target_indicies_stack = []
        #             pre_indicies = []
        #     elif self.time_token_ids[target.item()] == '<TIME_0>' and len(target_indicies_stack) == 0:
        #         weights[i] = 1.0
        #         pre_indicies.append(i)
        #     elif self.time_token_ids[target.item()] != '<TIME_DOT>':
        #         target_indicies_stack.append(i)

        # # print(weights)


        # time_loss = torch.sum(loss * weights)
        # time_loss = time_loss / time_loss_count if  time_loss_count > 0 else time_loss

        # return time_loss + torch.mean(loss)

        # v3 -- not ture penalty

        # logits = torch.softmax(logits, dim=-1)
        # loss = F.nll_loss(torch.log(logits), targets, reduction='none')

        # position_weights = torch.zeros_like(targets).float() # N
        # class_position_weights = torch.zeros_like(logits).float() # N * C

        # time_loss_count = 0
        # pre_indicies = []

        # target_indicies_stack = []
        # for i, target in enumerate(targets):
        #     if target.item() not in self.time_token_ids:
        #         gap = 0
        #         for j in range(len(target_indicies_stack) - 1, -1, -1):
        #             index = target_indicies_stack[j]
        #             position_weights[index] = 10 ** gap
        #             class_position_weights[index] = torch.tensor([0.0 if (k not in self.time_token_ids or self.time_token_ids[k] == '<TIME_DOT>') else (10 ** gap) * float(abs(targets[index].item() - k)) for k in range(logits.shape[1])])
        #             gap += 1
        #         target_indicies_stack = pre_indicies + target_indicies_stack
        #         if len(target_indicies_stack) > 0:
        #             target_indicies_weights = position_weights[target_indicies_stack]
        #             # print(target_indicies_weights)
        #             position_weights[target_indicies_stack] = target_indicies_weights / sum(target_indicies_weights)

        #             class_position_weights[target_indicies_stack] = class_position_weights[target_indicies_stack] / torch.sum(class_position_weights[target_indicies_stack])
        #             # print(class_position_weights[target_indicies_stack][-20:], targets[target_indicies_stack])

        #             time_loss_count += 1
        #             target_indicies_stack = []
        #             pre_indicies = []
        #     elif self.time_token_ids[target.item()] == '<TIME_0>' and len(target_indicies_stack) == 0:
        #         position_weights[i] = 1.0
        #         class_position_weights[i] = torch.tensor([0.0 if (k not in self.time_token_ids or self.time_token_ids[k] == '<TIME_DOT>') else float(abs(target.item() - k)) for k in range(logits.shape[1])])
        #         pre_indicies.append(i)
        #     elif self.time_token_ids[target.item()] != '<TIME_DOT>':
        #         target_indicies_stack.append(i)

        # time_loss = torch.sum(loss * position_weights)
        # time_loss = time_loss / time_loss_count if  time_loss_count > 0 else time_loss

        # not_true_loss = torch.sum(logits * class_position_weights)
        # not_true_loss =  not_true_loss / time_loss_count if  time_loss_count > 0 else  not_true_loss

        # # print(not_true_loss)

        # return time_loss + torch.mean(loss) + not_true_loss

        # v4
        # logits = torch.softmax(logits, dim=-1)

        # loss = F.nll_loss(torch.log(logits), targets)

        # class_position_weights = torch.zeros_like(logits).float()


        # count = 0
        # for i, target in enumerate(targets):
        #     if target.item() not in self.time_token_ids or self.time_token_ids[target.item()] == '<TIME_DOT>':
        #         continue
        #     else:
        #         class_position_weights[i] = torch.tensor([0.0 if (k not in self.time_token_ids or self.time_token_ids[k] == '<TIME_DOT>' or k == target.item()) else 1 / float(abs(target.item() - k)) for k in range(logits.shape[1])])
        #         class_position_weights[i] = class_position_weights[i] / torch.sum(class_position_weights[i])
        #         count += 1

        # # print(class_position_weights[-7][:-10], - torch.sum(torch.log(logits) * class_position_weights), count)

        # return loss - 0.01 * torch.sum(torch.log(logits) * class_position_weights) / count if count > 0 else loss

        # v5

        loss = F.cross_entropy(logits, targets)

        digit_logits = logits[:, self.digit_ids]
        digit_logits = torch.softmax(digit_logits, dim=-1)

        class_position_weights = torch.zeros_like(logits).float()

        count = 0
        for i, target in enumerate(targets):
            if target.item() not in self.digit_ids:
                continue
            else:
                class_position_weights[i] = torch.tensor([0.0 if (k not in self.digit_ids or k == target.item()) else torch.exp(self.normal_dist.log_prob(torch.tensor(abs(target.item() - k)).float())) for k in range(logits.shape[1])])
                class_position_weights[i] = class_position_weights[i] / torch.sum(class_position_weights[i])
                # print(class_position_weights[i][self.digit_ids])
                count += 1

        class_position_weights = class_position_weights[: ,self.digit_ids]

        # print(torch.sum(torch.log(digit_logits) * class_position_weights), count)
        

        return loss - 0.1 * torch.sum(torch.log(digit_logits) * class_position_weights) / count if count > 0 else loss














# class LegalTimeLossBackward(torch.autograd.Function):
#     # jump the text matching operation as the text matching operation does not have gradients

#     @staticmethod
#     def forward(ctx: Any, logits: Tensor):
#         signed_scores = torch.sign(scores)
#         return signed_scores

#     @staticmethod
#     def backward(ctx:Any, grad_output: Tensor):
#         return grad_output


# Class LegalTimeLoss:

#     def __init__(self, tokenizer, special_time_token=False):
#         self.token2digits = {'<TIME_{}>'.format(i): '{}'.format(i) for i in range(10)}
#         self.token2digits['<TIME_DOT>'] = '.'

#         self.special_time_token = special_time_token

#         # define match pattern
#         self.find_digits_pattern = r'(\d+|(\.+))'
#         self.check_pattern = r'^\d+(\.\d+)?$'

#         self.tokenizer = tokenizer

#     def check_legal(self, number):
#         return 0.0 if re.match(self.check_pattern, number)) else 1.0

#     def translate_outputs(self, outputs):
#         for k in self.token2digits.keys():
#             outputs = outputs.replace(k, v)

#         return outputs


#     def loss(self, logits):

#         print(logits.shape)

#         outputs_id = torch.argmax(logits, dim=-1)

#         outputs = self.tokenizer.decode(outputs_id, add_special_tokens=False)

#         if self.special_time_token:
#             outputs = self.translate_outputs(outputs)

#         matches = re.finditer(self.find_digits_pattern, outputs)
#         matches = [self.check_legal(match.group()) for match in matches]
#         if len(matches) == 0:
#             return 0.0
#         return sum(matches) / len(matches)