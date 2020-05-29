import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LabelSmoothingLoss(nn.Module):
    """
    reference : https://github.com/pytorch/pytorch/issues/7455#issuecomment-513735962
    """
    def __init__(self, classes=None, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

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
            loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        else:
            loss = torch.mean(torch.sum(- true_dist * pred * text_areas, dim=self.dim))
        return loss

class IndexLoss(nn.Module):
    """
    Loss for start and end indexes
    """
    def __init__(self, classes=None, smoothing=0.0, dim=-1):
        super(IndexLoss, self).__init__()
        self.loss_func = LabelSmoothingLoss(classes, smoothing, dim)

    def forward(self, start_logits, end_logits, start_positions, end_positions, text_areas=None):
        start_loss = self.loss_func(start_logits, start_positions, text_areas)
        end_loss = self.loss_func(end_logits, end_positions, text_areas)
        total_loss = start_loss + end_loss
        return total_loss

class JaccardExpectationLoss(nn.Module):
    def __init__(self):
        super(JaccardExpectationLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, start_logits, end_logits, start_positions, end_positions, text_areas=None):
        indexes = torch.arange(start_logits.size()[1]).unsqueeze(0).cuda()

        start_pred = torch.sum(self.softmax(start_logits) * indexes, dim=1)
        end_pred = torch.sum(self.softmax(end_logits) * indexes, dim=1)

        len_true = end_positions - start_positions + 1
        intersection = len_true - self.relu(start_pred - start_positions) - self.relu(end_positions - end_pred)
        union = len_true + self.relu(start_positions - start_pred) + self.relu(end_pred - end_positions)

        jel = 1 - intersection / union
        jel = torch.mean(jel)
        return jel

class LossCompose(nn.Module):
    def __init__(self, loss_modules, weights):
        super(LossCompose, self).__init__()
        self.loss_modules = nn.ModuleList(loss_modules)
        self.weights = weights

    def forward(self, start_logits, end_logits, start_positions, end_positions, text_areas=None):
        loss = 0
        for i in range(len(self.loss_modules)):
            loss += self.weights[i] * self.loss_modules[i](start_logits, end_logits, start_positions, end_positions, text_areas)
        return loss

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss