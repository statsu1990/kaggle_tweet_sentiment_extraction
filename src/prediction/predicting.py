import numpy as np
import torch
from tqdm import tqdm

from model import eval_utils as evut

def remove_excessive_padding(data, pad_id=1):
    """
    ids = data['ids']
    masks = data['masks']
    tweet = data['tweet']
    offsets = data['offsets']
    """
    min_n_pad = torch.min(torch.sum(torch.eq(data['ids'], pad_id), dim=-1))
    max_len = data['ids'].size()[-1] - min_n_pad

    data['ids'] = (data['ids'])[:,:max_len]
    data['masks'] = (data['masks'])[:,:max_len]
    data['offsets'] = (data['offsets'])[:,:max_len]

    return data

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
        # remove excessive padding
        data = remove_excessive_padding(data)

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
            start_pred, end_pred = evut.calc_start_end_index_v1(start_probs[i], end_probs[i])
            pred = evut.get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
            predictions.append(pred)

    return predictions
