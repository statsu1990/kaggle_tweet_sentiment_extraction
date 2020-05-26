import torch
from torch import nn
import torch.nn.functional as F

from transformers import RobertaModel, RobertaConfig

class TweetModel(nn.Module):
    def __init__(self, config, pretrained_model):
        super(TweetModel, self).__init__()
        
        config = RobertaConfig.from_pretrained(config, output_hidden_states=True)    
        self.roberta = RobertaModel.from_pretrained(pretrained_model, config=config)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask)
         
        x = torch.stack([hs[-1], hs[-2], hs[-3]])
        x = torch.mean(x, 0)
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
