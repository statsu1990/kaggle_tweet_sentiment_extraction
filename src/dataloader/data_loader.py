import random

import numpy as np
import torch
import tokenizers


def change_sentiment(row, p, sentiments):
    """
    sentiment : 'positive', 'negative', 'neutral'
    """
    #sentiments = np.array(['positive', 'negative', 'neutral'])

    changed_row = row.copy()
    match_sent = 1

    if random.random() < p:
        cand_sent = sentiments[sentiments != row.sentiment]
        selected_sent = np.random.choice(cand_sent, 1)
        changed_row.sentiment = selected_sent[0]
        match_sent = 0

    match_sent = torch.tensor(match_sent).float()
    return changed_row, match_sent

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=96, 
                 vocab_file='../input/roberta-base/vocab.json',
                 merges_file='../input/roberta-base/merges.txt',
                 change_sentiment_p=0.0,
                 ):
        self.df = df
        self.max_len = max_len
        self.labeled = 'selected_text' in df
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=vocab_file, 
            merges_file=merges_file, 
            lowercase=True,
            add_prefix_space=True)
        self.change_sentiment_p = change_sentiment_p

        self.uniq_sentiment = np.unique(self.df['sentiment'].values)

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        
        # augmentation
        if self.labeled:
            row, match_sent = change_sentiment(row, self.change_sentiment_p, self.uniq_sentiment)
            data['match_sent'] = match_sent

        ids, masks, tweet, offsets, text_areas = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets
        data['text_areas'] = text_areas
        
        if self.labeled:
            # match sentiment and text
            if match_sent > 0:
                start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
                data['start_idx'] = start_idx
                data['end_idx'] = end_idx
            else:
                pad_id = 1
                num_pad = torch.sum(torch.eq(ids, pad_id))
                data['start_idx'] = len(ids) - 1 - int(num_pad)
                data['end_idx'] = len(ids) - 1 - int(num_pad)
        
        return data

    def __len__(self):
        return len(self.df)
    
    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        text_areas = [False] + [False] + [False, False] + [True] * len(encoding.ids) + [False]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]
                
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            text_areas += [False] * pad_len
            offsets += [(0, 0)] * pad_len
        
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        text_areas = torch.tensor(text_areas)
        offsets = torch.tensor(offsets)
        
        return ids, masks, tweet, offsets, text_areas
        
    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " +  " ".join(row.selected_text.lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind+len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        
        return start_idx, end_idx
        
def get_train_val_loaders(df, train_idx, val_idx, batch_size=8, 
                          max_len=96, 
                          vocab_file='../input/roberta-base/vocab.json',
                          merges_file='../input/roberta-base/merges.txt',
                          change_sentiment_p=0.0):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, max_len, vocab_file, merges_file, change_sentiment_p), 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, max_len, vocab_file, merges_file), 
        batch_size=batch_size, 
        shuffle=False, )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict

def get_test_loader(df, batch_size=32, 
                    max_len=96, 
                    vocab_file='../input/roberta-base/vocab.json',
                    merges_file='../input/roberta-base/merges.txt'):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df, max_len, vocab_file, merges_file), 
        batch_size=batch_size, 
        shuffle=False, 
        )    
    return loader
