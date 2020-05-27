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
                 num_use_hid_layers=3, pooling='average', learnable_weight=False, 
                 dropout=0.5):
        super(TweetModel, self).__init__()
        
        config = RobertaConfig.from_pretrained(config, output_hidden_states=True)    
        self.roberta = RobertaModel.from_pretrained(pretrained_model, config=config)

        self.num_use_hid_layers = num_use_hid_layers
        self.hidden_layer_pooing = HiddenLayerPooling(num_use_hid_layers, pooling, learnable_weight)

        self.dropout = nn.Dropout(dropout)
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

class LinearHead(nn.Module):
    def __init__(self, n_input, n_output, ns_hidden=None, dropout=0.1):
        """
        Args:
            ns_hidden : hidden neuron list (ex. [512, 256]
        """
        super(LinearHead, self).__init__()
        
        if ns_hidden is None:
            ns = [n_input] + [n_output]
        else:
            ns = [n_input] + ns_hidden + [n_output]

        self.layers = []
        for i in range(len(ns)-1):
            self.layers.append(nn.Linear(ns[i], ns[i+1]))
            if i < len(ns)-2:
                self.layers.append(nn.ReLU(inplace=True))
                self.layers.append(nn.Dropout(dropout))

        nn.init.normal_(self.layers[-1].weight, std=0.02)
        nn.init.normal_(self.layers[-1].bias, 0)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        h = self.layers(x)
        return h

class Conv1dHead(nn.Module):
    def __init__(self, n_channel, k_size, n_conv, n_output, dropout=0.1):
        super(Conv1dHead, self).__init__()

        self.conv_layers = []
        for i in range(n_conv):
            self.conv_layers.append(nn.Conv1d(n_channel, n_channel, k_size, stride=1, padding=k_size//2))
            self.conv_layers.append(nn.ReLU(inplace=True))
            self.conv_layers.append(nn.Dropout(dropout))
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.linear = nn.Linear(n_channel, n_output)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

        return

    def forward(self, x):
        """
        Args:
            x : shape (Batch, Length, Feature)
        """
        h = x.permute(0, 2, 1) # (Batch, Feature, Length)
        h = self.conv_layers(h)
        
        h = h.permute(0, 2, 1) # (Batch, Length, Feature)
        h = self.linear(h)
        return h

class TweetModel2(nn.Module):
    def __init__(self, config, pretrained_model, 
                 num_use_hid_layers=3, pooling='average', learnable_weight=False, 
                 dropout=0.1,
                 ans_idx_head=None):
        super(TweetModel2, self).__init__()
        
        config = RobertaConfig.from_pretrained(config, output_hidden_states=True)    
        self.roberta = RobertaModel.from_pretrained(pretrained_model, config=config)

        self.num_use_hid_layers = num_use_hid_layers
        self.hidden_layer_pooing = HiddenLayerPooling(num_use_hid_layers, pooling, learnable_weight)

        self.dropout = nn.Dropout(dropout)
        self.ans_idx_head = ans_idx_head

    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask) # 12 layers
        
        x = self.hidden_layer_pooing(hs[-self.num_use_hid_layers:])
        x = self.dropout(x)

        # answer index
        x_ans = self.ans_idx_head(x)
        start_logits, end_logits = x_ans.split(1, dim=-1)
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

