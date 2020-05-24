import numpy as np
import torch
from tqdm import tqdm

from model import eval_utils as evut

def predicter(models, dataloader):
    """
    Args:
        models : list of models
    """
    # to cuda, eval mode
    for model in models:
        model = model.cuda()
        model.eval()

    predictions = []

    for batch_idx, data in enumerate(tqdm(dataloader)):
        # data
        ids = data['ids'].cuda()
        masks = data['masks'].cuda()
        tweet = data['tweet']
        offsets = data['offsets'].numpy()

        # pred
        start_probs, end_probs = [], []
        for model in models:
            with torch.no_grad():
                output = model(ids, masks)
                start_probs.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                end_probs.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())
        start_probs = np.mean(start_probs, axis=0)
        end_probs = np.mean(end_probs, axis=0)

        for i in range(len(ids)):
            start_pred = np.argmax(start_probs[i])
            end_pred = np.argmax(end_probs[i])
            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = evut.get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
            predictions.append(pred)

    return predictions
