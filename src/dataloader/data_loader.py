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
                 premake_dataset=False,
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

        self.premake_dataset = premake_dataset
        if self.premake_dataset:
            self.dataset = [self.make_data(i) for i in range(len(self.df))]

    def __getitem__(self, index):
        if self.premake_dataset:
            return self.dataset[index]
        else:
            return self.make_data(index)

    def make_data(self, index):
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

class TweetDatasetWithAug(TweetDataset):
    def __init__(self, df, max_len=96, 
                 vocab_file='../input/roberta-base/vocab.json',
                 merges_file='../input/roberta-base/merges.txt',
                 change_sentiment_p=0.0,
                 add_neutral_p=0.0, 
                 neutral_text=None, 
                 add_nonsentiment_p=0.0, 
                 add_different_sentiment_p=0.0,
                 exchange_selected_text_p=0.0,
                 insert_selected_text_p=0.0,
                 ):
        super(TweetDatasetWithAug, self).__init__(df, max_len, 
                                            vocab_file, merges_file,
                                            change_sentiment_p=change_sentiment_p)

        self.add_neutral_p = add_neutral_p
        if neutral_text is None:
            self.neutral_text = (df[df['sentiment']=='neutral'])['text'].values
        else:
            self.neutral_text = neutral_text
        self.add_nonsentiment_p = add_nonsentiment_p
        self.add_different_sentiment_p = add_different_sentiment_p
        self.exchange_selected_text_p = exchange_selected_text_p
        self.insert_selected_text_p = insert_selected_text_p

    def add_neutral(self, row):
        add_idx = np.random.randint(0, len(self.neutral_text))
        add_text = self.neutral_text[add_idx]
        add_word = add_text.split()

        length = max([1, np.random.randint(int(len(add_word))*0.5, len(add_word)+1)])
        start = np.random.randint(0, len(add_word)-length+1)
        add_text = ' '.join(add_word[start:start+length])

        if np.random.rand() < 0.5:
            row.text = row.text + ' ' + add_text
        else:
            row.text = add_text + ' ' + row.text

        return row

    def add_nonsentiment(self, row):
        while True:
            add_idx = np.random.randint(0, len(self.df))
            add_row = self.df.iloc[add_idx]
            slct_text = add_row['selected_text'].split()
            text = add_row['text'].split()

            if set(slct_text) == (set(text)&set(slct_text)):
                if text != slct_text:
                    add_word = list(filter(lambda x: x not in slct_text, text))
                    if len(add_word) != 0:
                        break

        length = max([1, np.random.randint(int(len(add_word))*0.5, len(add_word)+1)])
        start = np.random.randint(0, len(add_word)-length+1)
        add_text = ' '.join(add_word[start:start+length])

        if np.random.rand() < 0.5:
            row.text = row.text + ' ' + add_text
        else:
            row.text = add_text + ' ' + row.text

        return row

    def add_different_sentiment(self, row):
        while True:
            add_idx = np.random.randint(0, len(self.df))
            add_row = self.df.iloc[add_idx]
            if add_row['sentiment'] != row['sentiment']:
                break
        add_word = add_row['text'].split()

        #length = max([1, np.random.randint(int(len(add_word))*0.5, len(add_word)+1)])
        #start = np.random.randint(0, len(add_word)-length+1)
        #add_text = ' '.join(add_word[start:start+length])
        add_text = ' '.join(add_word)

        if np.random.rand() < 0.5:
            row.text = row.text + ' ' + add_text
        else:
            row.text = add_text + ' ' + row.text

        return row

    def exchange_selected_text(self, row):
        while True:
            exch_idx = np.random.randint(0, len(self.df))
            exch_row = self.df.iloc[exch_idx]
            if exch_row['sentiment'] == row['sentiment']:
                break
        exch_text = exch_row['selected_text']
        row['text'] = row['text'].replace(row['selected_text'], exch_text, 1)
        row['selected_text'] = exch_text

        return row

    def insert_selected_text(self, row):
        neu_idx = np.random.randint(0, len(self.neutral_text))
        neu_word = self.neutral_text[neu_idx].split()

        insert_idx = np.random.randint(0, len(neu_word))
        row['text'] = ' '.join(neu_word[:insert_idx]) + ' ' + row['selected_text'] + ' ' + ' '.join(neu_word[insert_idx:])

        return row

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        # augmentation
        if self.labeled:
            row, match_sent = change_sentiment(row, self.change_sentiment_p, self.uniq_sentiment)
            data['match_sent'] = match_sent

        if np.random.rand() < self.insert_selected_text_p:
            row = self.insert_selected_text(row)

        if np.random.rand() < self.add_neutral_p:
            if row.sentiment != 'neutral':
                row = self.add_neutral(row)

        if np.random.rand() < self.add_nonsentiment_p:
            row = self.add_nonsentiment(row)

        if np.random.rand() < self.add_different_sentiment_p:
            row = self.add_different_sentiment(row)

        if np.random.rand() < self.exchange_selected_text_p:
            row = self.exchange_selected_text(row)



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

def get_train_val_loaders(df, train_idx, val_idx, batch_size=8, 
                          max_len=96, 
                          vocab_file='../input/roberta-base/vocab.json',
                          merges_file='../input/roberta-base/merges.txt',
                          change_sentiment_p=0.0, 
                          premake_dataset=False,
                          add_neutral_p=0.0, 
                          neutral_text=None,
                          add_nonsentiment_p=0.0,
                          add_different_sentiment_p=0.0,
                          exchange_selected_text_p=0.0,
                          insert_selected_text_p=0.0,
                          ):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    if add_neutral_p==0.0 and add_nonsentiment_p==0.0 and add_different_sentiment_p==0.0 and exchange_selected_text_p==0.0 and insert_selected_text_p==0:
        train_loader = torch.utils.data.DataLoader(
            TweetDataset(train_df, max_len, vocab_file, merges_file, change_sentiment_p, 
                         premake_dataset), 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            TweetDatasetWithAug(train_df, max_len, vocab_file, merges_file, change_sentiment_p, 
                                add_neutral_p, neutral_text, add_nonsentiment_p, 
                                add_different_sentiment_p, exchange_selected_text_p,
                                insert_selected_text_p), 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, max_len, vocab_file, merges_file, 
                     premake_dataset=premake_dataset), 
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
