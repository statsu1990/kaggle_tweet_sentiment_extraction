import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    """
    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513735962
    """
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        return loss

class IndexLoss(nn.Module):
    """
    Loss for start and end indexes
    """
    def __init__(self, classes=96, smoothing=0.0, dim=-1):
        super(IndexLoss, self).__init__()
        self.loss_func = LabelSmoothingLoss(classes, smoothing, dim)

    def forward(self, start_logits, end_logits, start_positions, end_positions):
        start_loss = self.loss_func(start_logits, start_positions)
        end_loss = self.loss_func(end_logits, end_positions)    
        total_loss = start_loss + end_loss
        return total_loss

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss