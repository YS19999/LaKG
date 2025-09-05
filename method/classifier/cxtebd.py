import datetime

import torch
import torch.nn as nn
from transformers import BertModel
# from pytorch_transformers import BertModel
import dataset.stats as stats

class CXTEBD(nn.Module):

    def __init__(self, args, return_seq=False):

        super(CXTEBD, self).__init__()

        self.args = args
        self.return_seq = return_seq

        print("{}, Loading pretrainedModel bert".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)

        self.model = BertModel.from_pretrained(args.pretrained)
        self.unfreeze_layers = ['layer.9', 'layer.10', 'layer.11']
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in self.unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size

    def get_bert(self, bert_id, mask, data):

        # need to use smaller batches
        out = self.model(input_ids=bert_id, attention_mask=mask, output_attentions=True)
        # return out
        if self.return_seq:
            return out[0]
        else:
            return out[0][:, 0, :]

    def forward(self, data, weight=None):

        text = data['text']
        attn_mask = data['attn_mask']
        # with torch.no_grad():
        return self.get_bert(text, attn_mask, data)

