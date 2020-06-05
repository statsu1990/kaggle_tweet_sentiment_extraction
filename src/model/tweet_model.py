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
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout))

        nn.init.normal_(self.layers[-1].weight, std=0.02)
        nn.init.normal_(self.layers[-1].bias, 0)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, *args):
        h = self.layers(x)
        return h

class Conv1dHead(nn.Module):
    def __init__(self, n_channel, k_size, n_conv, n_output, dropout=0.1):
        super(Conv1dHead, self).__init__()

        self.conv_layers = []
        for i in range(n_conv):
            self.conv_layers.append(nn.Conv1d(n_channel, n_channel, k_size, stride=1, padding=k_size//2))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(dropout))
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.linear = nn.Linear(n_channel, n_output)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

        return

    def forward(self, x, *args):
        """
        Args:
            x : shape (Batch, Length, Feature)
        """
        h = x.permute(0, 2, 1) # (Batch, Feature, Length)
        h = self.conv_layers(h)
        
        h = h.permute(0, 2, 1) # (Batch, Length, Feature)
        h = self.linear(h)
        return h

class SentimentAttentionHead(nn.Module):
    def __init__(self, n_input, n_output, 
                 n_element, reduction=4, dropout=0.1,
                 sentiment_index=1,
                 additional_head=None):
        super(SentimentAttentionHead, self).__init__()
        self.sentiment_index = sentiment_index
        self.additional_head = additional_head

        # param
        self.weight = nn.Parameter(torch.Tensor(1, n_output, n_input, n_element))
        self.bias = nn.Parameter(torch.Tensor(1, n_output, n_element))
        if additional_head is None:
            nn.init.normal_(self.weight, std=0.02)
            nn.init.normal_(self.bias, 0)

        # attention
        self.sentiment_attn = nn.Sequential(nn.Linear(n_input, n_input//reduction), 
                                            nn.ReLU(), 
                                            nn.Dropout(dropout),
                                            nn.Linear(n_input//reduction, n_element),
                                            nn.Softmax(dim=-1))

        if additional_head is not None:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args):
        """
        Args:
            x : shape (Batch, Length, N_input)
        """
        sentiment_emb = x[:,self.sentiment_index]
        attn = self.sentiment_attn(sentiment_emb) # (Batch, Element)
        attn = attn.unsqueeze(1) # (Batch, 1, Element)
        attn = attn.unsqueeze(1) # (Batch, 1, 1, Element)

        b = torch.sum(attn * self.bias, dim=-1) # (Batch, 1, N_output)
        w = torch.sum(attn * self.weight, dim=-1).permute(0,2,1) # (Batch, N_input, N_output)
        
        h = torch.bmm(x, w) + b # (Batch, Length, N_output)

        if self.additional_head is not None:
            h = self.relu(h)
            h = self.dropout(h)
            h = self.additional_head(h)

        return h

class XLNetQAHead(nn.Module):
    def __init__(self, n_input, start_n_top):
        super(XLNetQAHead, self).__init__()
        self.start_n_top = start_n_top

        self.start_head = nn.Linear(n_input, 1)
        self.end_head = nn.Sequential(nn.Linear(n_input*2, n_input), 
                                      nn.Tanh(),
                                      nn.LayerNorm(n_input),
                                      nn.Linear(n_input, 1))

    def forward(self, x, start_positions=None):
        start_logits = self.start_head(x)

        if self.training:
            end_logits = self.calc_end_logits(x, start_positions)
        else:
            bsz, slen, hsz = x.size()
            start_log_probs = F.softmax(start_logits.squeeze(-1), dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(x, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = x.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            end_logits = self.calc_end_logits(hidden_states_expanded, start_states=start_states)
            end_logits = torch.sum(end_logits * start_top_log_probs.unsqueeze(1).unsqueeze(-1), dim=2)

        return torch.cat([start_logits, end_logits], dim=-1)

    def calc_end_logits(self, x, start_positions=None, start_states=None):
        """
        x : shape (Batch, Lenght, Feature)
        start_positions : shape (Batch,)
        """
        if start_states is None:
            slen, hsz = x.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = x.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        cat_state = torch.cat([x, start_states], dim=-1)
        end_logit = self.end_head(cat_state)
        return end_logit


class TweetModel2(nn.Module):
    def __init__(self, config, pretrained_model, 
                 num_use_hid_layers=3, pooling='average', learnable_weight=False, 
                 dropout=0.1,
                 ans_idx_head=None, match_sent_head=None):
        super(TweetModel2, self).__init__()
        
        config = RobertaConfig.from_pretrained(config, output_hidden_states=True)    
        self.roberta = RobertaModel.from_pretrained(pretrained_model, config=config)

        self.num_use_hid_layers = num_use_hid_layers
        self.hidden_layer_pooing = HiddenLayerPooling(num_use_hid_layers, pooling, learnable_weight)

        self.dropout = nn.Dropout(dropout)
        self.ans_idx_head = ans_idx_head
        self.match_sent_head = match_sent_head

    def forward(self, input_ids, attention_mask, start_positions=None):
        _, _, hs = self.roberta(input_ids, attention_mask) # 12 layers
        
        x = self.hidden_layer_pooing(hs[-self.num_use_hid_layers:])
        x = self.dropout(x)

        # answer index
        x_ans = self.ans_idx_head(x, start_positions)
        start_logits, end_logits = x_ans.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # match sentiment
        if self.match_sent_head is not None:
            match_sent_logit = self.match_sent_head(x[:,0])
        else:
            match_sent_logit = None

        return start_logits, end_logits, match_sent_logit

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

