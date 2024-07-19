import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import copy
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer, M2M100Model
import random

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.weights_backup = copy.deepcopy(self.model.state_dict())

        self.tokenizers = {'cs': M2M100Tokenizer.from_pretrained("facebook/m2m100_418M",
                                                            src_lang='cs',
                                                            tgt_lang="en",
                                                            padding_side='right',
                                                            truncation_side='right'),
                           'de':M2M100Tokenizer.from_pretrained("facebook/m2m100_418M",
                                                            src_lang='de',
                                                            tgt_lang="en",
                                                            padding_side='right',
                                                            truncation_side='right')}

    def forward_eval(self, batch, lang_code):
        return self.model.generate(**batch['x'], forced_bos_token_id=self.tokenizers[lang_code].get_lang_id("en"))

    def forward_train(self, batch):
        return self.model(**batch['x'], labels=batch['y'])

    def apply_mask(self, mask, sizing):
        start = 0
        copy_state = copy.deepcopy(self.model.state_dict())
        segments = {}
        for i in copy_state:
            if i in sizing:
                end = start + sizing[i]
                segment = np.round(mask[start:end])
                index = np.where(segment == 0)

                final_indices = []
                divisor = int(copy_state[i].shape[0]/sizing[i])
                for j in index[0]:
                    final_indices += [*range(j*divisor, (j*divisor)+divisor)]
                # print(final_indices)
                copy_state[i].data[np.array(final_indices)] = 0
                segments.update({i:index})
                start = end
        self.model.load_state_dict(copy_state)
        for name, param in self.model.named_parameters():
            if name in segments:
                param.data[segments[name]].requires_grad = False
                start = end

    def return_model(self):
        return self.model

    def return_model_state(self):
        return self.model.state_dict()

    def revert_weights(self):
        self.model.load_state_dict(self.weights_backup)
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def update_backup(self):
        self.weights_backup = copy.deepcopy(self.model.state_dict())


class Custom_Dataloader:
    def __init__(self, data, batch_size):

        self.data = data
        self.data_amount = data['x']['input_ids'].shape[0]

        self.batch_size = batch_size
        self.length = int(math.ceil(self.data_amount/batch_size))
        self.available = set([*range(self.length)])


    def select_subset(self, idxs, data, cuda):
        if cuda:
            return {'x':{'input_ids': data['x']['input_ids'][idxs].cuda(),
                         'attention_mask': data['x']['attention_mask'][idxs].cuda()},
                    'y':data['y'][idxs].cuda()}

    def sample_batch(self, cuda=True):

        if len(self.available)==0:
            self.available = set([*range(self.length)])
        idx = random.choice(tuple(self.available))
        self.available.remove(idx)
        start = idx*self.batch_size
        if idx == self.length-1:
            diff = self.data_amount - (idx*self.batch_size)
            batch = self.select_subset([i for i in range(start,start+diff)], self.data, cuda)
            return batch
        else:
            batch = self.select_subset([i for i in range(start,start+self.batch_size)], self.data, cuda)
            return batch

    def reset(self):
        self.history = set()
        self.available = set([*range(self.data_amount)])
