import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LabelSmoothingLoss(nn.Module):
    """
    reference : https://github.com/pytorch/pytorch/issues/7455#issuecomment-513735962
    """
    def __init__(self, classes=None, smoothing=0.0, dim=-1, reduce=True):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.reduce = reduce

    def forward(self, pred, target, text_areas=None):
        if self.cls is None:
            if text_areas is None:
                cls = max([pred.size()[self.dim], 2])
            else:
                cls = torch.sum(text_areas, dim=1, keepdim=True)
                cls[cls < 2] = 2
        else:
            cls = self.cls

        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.ones_like(pred)
            true_dist = true_dist * self.smoothing / (cls - 1)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        if text_areas is None:
            loss = torch.sum(-true_dist * pred, dim=self.dim)
        else:
            loss = torch.sum(- true_dist * pred * text_areas, dim=self.dim)

        if self.reduce:
            loss = torch.mean(loss)

        return loss

class IndexLoss(nn.Module):
    """
    Loss for start and end indexes
    """
    def __init__(self, classes=None, smoothing=0.0, dim=-1, reduce=True):
        super(IndexLoss, self).__init__()
        self.loss_func = LabelSmoothingLoss(classes, smoothing, dim, reduce)

    def forward(self, start_logits, end_logits, start_positions, end_positions, text_areas=None, *args, **kargs):
        start_loss = self.loss_func(start_logits, start_positions, text_areas)
        end_loss = self.loss_func(end_logits, end_positions, text_areas)
        total_loss = start_loss + end_loss
        return total_loss

class MatchSentimentLoss(nn.Module):
    def __init__(self, reduce=True):
        super(MatchSentimentLoss, self).__init__()
        reduce = True
        reduction = 'mean' if reduce else 'none'
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logit, match_sentiment):
        loss = self.bce(logit.squeeze(1), match_sentiment)
        return loss

class JaccardExpectationLoss(nn.Module):
    def __init__(self, reduce=True):
        super(JaccardExpectationLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.reduce = True

    def forward(self, start_logits, end_logits, start_positions, end_positions, *args, **kargs):
        indexes = torch.arange(start_logits.size()[1]).unsqueeze(0).cuda()

        start_pred = torch.sum(self.softmax(start_logits) * indexes, dim=1)
        end_pred = torch.sum(self.softmax(end_logits) * indexes, dim=1)

        len_true = end_positions - start_positions + 1
        intersection = len_true - self.relu(start_pred - start_positions) - self.relu(end_positions - end_pred)
        union = len_true + self.relu(start_positions - start_pred) + self.relu(end_pred - end_positions)

        jel = 1 - intersection / union
        if self.reduce:
            jel = torch.mean(jel)
        return jel

class LossCompose(nn.Module):
    def __init__(self, loss_modules, weights):
        super(LossCompose, self).__init__()
        self.loss_modules = nn.ModuleList(loss_modules)
        self.weights = weights

    def forward(self, start_logits, end_logits, start_positions, end_positions, text_areas=None, *args, **kargs):
        for i in range(len(self.loss_modules)):
            if i == 0:
                loss = self.weights[i] * self.loss_modules[i](start_logits, end_logits, start_positions, end_positions, text_areas)
            else:
                loss += self.weights[i] * self.loss_modules[i](start_logits, end_logits, start_positions, end_positions, text_areas)
        return loss

class CombineMatchSentimentLoss(nn.Module):
    def __init__(self, combed_loss, weights):
        super(CombineMatchSentimentLoss, self).__init__()
        self.combed_loss = combed_loss
        self.weights = weights
        self.match_sentiment_loss = MatchSentimentLoss()

    def forward(self, start_logits, end_logits, start_positions, end_positions, text_areas, 
                match_sentiment_logits, match_sentiment):
        if match_sentiment_logits is not None:
            ms_loss = self.match_sentiment_loss(match_sentiment_logits, match_sentiment)
        else:
            ms_loss = 0
            match_sentiment = 1
        comb_loss = self.combed_loss(start_logits, end_logits, start_positions, end_positions, text_areas)
        assert len(comb_loss.size()) > 0, 'combed_loss must be seted reduce=False'

        loss = self.weights[0] * ms_loss + self.weights[1] * torch.mean(match_sentiment * comb_loss)

        return loss

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss