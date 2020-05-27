import torch
from torch import nn
import torch.nn.functional as F

from transformers import RobertaModel, RobertaConfig

class HiddenLayerPooling(nn.Module):
    def __init__(self, num_hidden_layer, pooling='average', learnable_weight=False):
        super(HiddenLayerPooling, self).__init__()

        if learnable_weight:
            self.weight = nn.Parameter(torch.ones(num_hidden_layer,1,1,1))
            self.softmax = nn.Softmax(dim=0)
        else:
            self.weight = None
        self.pooling = pooling

    def forward(self, hidden_layers):
        hs = torch.stack(hidden_layers)
        if self.weight is not None:
            hs = self.softmax(self.weight) * hs

        if self.pooling == 'average':
            h = torch.mean(hs, 0)
        elif self.pooling == 'max':
            h = torch.max(hs, 0)[0]
        else:
            h = None

        return h

class TweetModel(nn.Module):
    def __init__(self, config, pretrained_model, 
                 num_use_hid_layers=3, pooling='average', learnable_weight=False):
        super(TweetModel, self).__init__()
        
        config = RobertaConfig.from_pretrained(config, output_hidden_states=True)    
        self.roberta = RobertaModel.from_pretrained(pretrained_model, config=config)

        self.num_use_hid_layers = num_use_hid_layers
        self.hidden_layer_pooing = HiddenLayerPooling(num_use_hid_layers, pooling, learnable_weight)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask) # 12 layers
         
        #x = torch.stack([hs[-1], hs[-2], hs[-3]])
        #x = torch.mean(x, 0)
        x = self.hidden_layer_pooing(hs[-self.num_use_hid_layers:])

        x = self.dropout(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
                
        return start_logits, end_logits

    def get_params(self):
        """
        Returns:
            bert params:
            other params:
        """
        model_params = list(self.named_parameters())

        bert_params = [p for n, p in model_params if "roberta" in n]
        other_params = [p for n, p in model_params if not "roberta" in n]

        return bert_params, other_params
