
# # Libraries
import numpy as np
import pandas as pd
import os
import warnings
import random
import torch 
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import tokenizers
from transformers import RobertaModel, RobertaConfig
from tqdm import tqdm

warnings.filterwarnings('ignore')

# # Seed
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# # Data Loader
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=96):
        self.df = df
        self.max_len = max_len
        self.labeled = 'selected_text' in df
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file='../input/roberta-base/vocab.json', 
            merges_file='../input/roberta-base/merges.txt', 
            lowercase=True,
            add_prefix_space=True)

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        
        ids, masks, tweet, offsets = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets
        
        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx
        
        return data

    def __len__(self):
        return len(self.df)
    
    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]
                
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len
        
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        
        return ids, masks, tweet, offsets
        
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
        
def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df), 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df), 
        batch_size=batch_size, 
        shuffle=False, 
        )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict

def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df), 
        batch_size=batch_size, 
        shuffle=False, 
        )    
    return loader

# # Model
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        
        config = RobertaConfig.from_pretrained(
            '../input/roberta-base/config.json', output_hidden_states=True)    
        self.roberta = RobertaModel.from_pretrained(
            '../input/roberta-base/pytorch_model.bin', config=config)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask)
         
        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
                
        return start_logits, end_logits

    def get_params(self):
        model_params = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        bert_params = [p for n, p in model_params if "roberta" in n and not any(nd in n for nd in no_decay)]
        bert_params_nodecay = [p for n, p in model_params if "roberta" in n and any(nd in n for nd in no_decay)]
        other_params = [p for n, p in model_params if not "roberta" in n]

        return bert_params, bert_params_nodecay, other_params

# # Loss Function
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)    
    total_loss = start_loss + end_loss
    return total_loss

# # Evaluation Function
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)
        
    true = get_selected_text(text, start_idx, end_idx, offsets)
    
    return jaccard(true, pred)

# # Training
def remove_excessive_padding(data, pad_id=1):
    """
    Set length to the max length except pad in batch.
    """
    """
    ids = data['ids']
    masks = data['masks']
    offsets = data['offsets']
    """
    min_n_pad = torch.min(torch.sum(torch.eq(data['ids'], pad_id), dim=-1))
    max_len = data['ids'].size()[-1] - min_n_pad

    data['ids'] = (data['ids'])[:,:max_len]
    data['masks'] = (data['masks'])[:,:max_len]
    data['offsets'] = (data['offsets'])[:,:max_len]

    return data

def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename, remove_pad=False):
    model.cuda()

    logger = []
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        if optimizer is not None:
            for gr, param_group in enumerate(optimizer.param_groups):
                print('lr :', param_group['lr'])

        logger.append([epoch+1])
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0
            
            for data in tqdm(dataloaders_dict[phase]):
                if remove_pad:
                    data = remove_excessive_padding(data)

                ids = data['ids'].cuda()
                masks = data['masks'].cuda()
                tweet = data['tweet']
                offsets = data['offsets'].numpy()
                start_idx = data['start_idx'].cuda()
                end_idx = data['end_idx'].cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    start_logits, end_logits = model(ids, masks)

                    loss = criterion(start_logits, end_logits, start_idx, end_idx)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(ids)
                    
                    start_idx = start_idx.cpu().detach().numpy()
                    end_idx = end_idx.cpu().detach().numpy()
                    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
                    
                    for i in range(len(ids)):                        
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            start_idx[i],
                            end_idx[i],
                            start_logits[i], 
                            end_logits[i], 
                            offsets[i])
                        epoch_jaccard += jaccard_score
                    
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)
            
            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
                epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard))
            logger[-1] = logger[-1] + [epoch_loss, epoch_jaccard]
    
    torch.save(model.state_dict(), filename)
    return logger

RESULTS_DIR = '../results2'

class Model2_v1_0_1:
    """
    cv 0.554233, lb 0.708
    train only posi nega
    """
    def __init__(self):
        self.seed = 42
        seed_everything(self.seed)

        self.save_dir = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_only_posi_nega = True
        self.num_fold=3

    def train(self):
        # # Training

        # %% [code]
        num_epochs = 3
        batch_size = 32
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

        # %% [code]

        train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
        if self.train_only_posi_nega:
            train_df = train_df[(train_df['sentiment']=='positive') | (train_df['sentiment']=='negative')].reset_index(drop=True)
        train_df['text'] = train_df['text'].astype(str)
        train_df['selected_text'] = train_df['selected_text'].astype(str)

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=0): 
            if fold < self.num_fold:
                print(f'Fold: {fold}')

                model = TweetModel()
                optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
                criterion = loss_fn    
                dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

                logger = train_model(
                    model, 
                    dataloaders_dict,
                    criterion, 
                    optimizer, 
                    num_epochs,
                    os.path.join(self.save_dir,f'roberta_fold{fold}.pth'))

                # save log
                df = pd.DataFrame(logger)
                df.columns = ['epoch', 'train_loss', 'train_score', 'val_loss', 'val_score']
                df.to_csv(os.path.join(self.save_dir,f'train_log_fold{fold}.csv'))

    def test(self):
        test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
        test_df['text'] = test_df['text'].astype(str)
        test_loader = get_test_loader(test_df)
        predictions = []
        models = []
        for fold in range(self.num_fold):
            model = TweetModel()
            model.cuda()
            model.load_state_dict(torch.load(os.path.join(self.save_dir,f'roberta_fold{fold}.pth')))
            model.eval()
            models.append(model)

        for data in tqdm(test_loader):
            ids = data['ids'].cuda()
            masks = data['masks'].cuda()
            tweet = data['tweet']
            offsets = data['offsets'].numpy()

            start_logits = []
            end_logits = []
            for model in models:
                with torch.no_grad():
                    output = model(ids, masks)
                    start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                    end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

            start_logits = np.mean(start_logits, axis=0)
            end_logits = np.mean(end_logits, axis=0)
            for i in range(len(ids)):    
                start_pred = np.argmax(start_logits[i])
                end_pred = np.argmax(end_logits[i])
                if start_pred > end_pred:
                    pred = tweet[i]
                else:
                    pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                predictions.append(pred)

        if self.train_only_posi_nega:
            neutral_idxs = (test_df['sentiment'].values=='neutral')
            predictions = np.array(predictions)
            predictions[neutral_idxs] = test_df['text'].values[neutral_idxs]

        # # Submission
        sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
        sub_df['selected_text'] = predictions
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
        sub_df.to_csv(os.path.join('submission.csv'), index=False)
        sub_df.head()

class Model2_v1_0_2:
    """
    cv 0.550, lb 0.712
    remove pad

    train only posi nega
    """
    def __init__(self):
        self.seed = 42
        seed_everything(self.seed)

        self.save_dir = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_only_posi_nega = True
        self.num_fold=3
        self.remove_pad = True

    def train(self):
        # # Training

        # %% [code]
        num_epochs = 3
        batch_size = 32
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

        # %% [code]

        train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
        if self.train_only_posi_nega:
            train_df = train_df[(train_df['sentiment']=='positive') | (train_df['sentiment']=='negative')].reset_index(drop=True)
        train_df['text'] = train_df['text'].astype(str)
        train_df['selected_text'] = train_df['selected_text'].astype(str)

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=0): 
            if fold < self.num_fold:
                print(f'Fold: {fold}')

                model = TweetModel()
                optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
                criterion = loss_fn    
                dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

                logger = train_model(
                    model, 
                    dataloaders_dict,
                    criterion, 
                    optimizer, 
                    num_epochs,
                    os.path.join(self.save_dir,f'roberta_fold{fold}.pth'),
                    self.remove_pad)

                # save log
                df = pd.DataFrame(logger)
                df.columns = ['epoch', 'train_loss', 'train_score', 'val_loss', 'val_score']
                df.to_csv(os.path.join(self.save_dir,f'train_log_fold{fold}.csv'))

    def test(self):
        test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
        test_df['text'] = test_df['text'].astype(str)
        test_loader = get_test_loader(test_df)
        predictions = []
        models = []
        for fold in range(self.num_fold):
            model = TweetModel()
            model.cuda()
            model.load_state_dict(torch.load(os.path.join(self.save_dir,f'roberta_fold{fold}.pth')))
            model.eval()
            models.append(model)

        for data in tqdm(test_loader):
            if self.remove_pad:
                data = remove_excessive_padding(data)

            ids = data['ids'].cuda()
            masks = data['masks'].cuda()
            tweet = data['tweet']
            offsets = data['offsets'].numpy()

            start_logits = []
            end_logits = []
            for model in models:
                with torch.no_grad():
                    output = model(ids, masks)
                    start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                    end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

            start_logits = np.mean(start_logits, axis=0)
            end_logits = np.mean(end_logits, axis=0)
            for i in range(len(ids)):    
                start_pred = np.argmax(start_logits[i])
                end_pred = np.argmax(end_logits[i])
                if start_pred > end_pred:
                    pred = tweet[i]
                else:
                    pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                predictions.append(pred)

        if self.train_only_posi_nega:
            neutral_idxs = (test_df['sentiment'].values=='neutral')
            predictions = np.array(predictions)
            predictions[neutral_idxs] = test_df['text'].values[neutral_idxs]

        # # Submission
        sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
        sub_df['selected_text'] = predictions
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
        sub_df.to_csv(os.path.join('submission.csv'), index=False)
        sub_df.head()

class Model2_v1_0_3:
    """
    cv 0., lb 0.
    different learning rate (x30) 

    remove pad
    train only posi nega
    """
    def __init__(self):
        self.seed = 42
        seed_everything(self.seed)

        self.save_dir = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_only_posi_nega = True
        self.num_fold=3
        self.remove_pad = True

    def train(self):
        # # Training

        # %% [code]
        num_epochs = 3
        batch_size = 32
        lr = 1e-5
        dif_lr_rate = 30
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

        # %% [code]

        train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
        if self.train_only_posi_nega:
            train_df = train_df[(train_df['sentiment']=='positive') | (train_df['sentiment']=='negative')].reset_index(drop=True)
        train_df['text'] = train_df['text'].astype(str)
        train_df['selected_text'] = train_df['selected_text'].astype(str)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=0): 
            if fold < self.num_fold:
                print(f'Fold: {fold}')

                model = TweetModel()

                bert_params, bert_params_nodecay, other_params = model.get_params()
                params = [
                    {'params': bert_params, 'lr': lr, 'weight_decay':0.01},
                    {'params': bert_params_nodecay, 'lr': lr, 'weight_decay':0.01},
                    {'params': other_params, 'lr': lr * dif_lr_rate, 'weight_decay':0.01}
                    ]
                optimizer = optim.AdamW(params, lr=3e-5, betas=(0.9, 0.999))
                
                criterion = loss_fn    
                dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

                logger = train_model(
                    model, 
                    dataloaders_dict,
                    criterion, 
                    optimizer, 
                    num_epochs,
                    os.path.join(self.save_dir,f'roberta_fold{fold}.pth'),
                    self.remove_pad)

                # save log
                df = pd.DataFrame(logger)
                df.columns = ['epoch', 'train_loss', 'train_score', 'val_loss', 'val_score']
                df.to_csv(os.path.join(self.save_dir,f'train_log_fold{fold}.csv'))

                scores.append(logger[-1])

        print('scores', scores)
        print('ave score', np.average(scores))
        return

    def test(self):
        test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
        test_df['text'] = test_df['text'].astype(str)
        test_loader = get_test_loader(test_df)
        predictions = []
        models = []
        for fold in range(self.num_fold):
            model = TweetModel()
            model.cuda()
            model.load_state_dict(torch.load(os.path.join(self.save_dir,f'roberta_fold{fold}.pth')))
            model.eval()
            models.append(model)

        for data in tqdm(test_loader):
            if self.remove_pad:
                data = remove_excessive_padding(data)

            ids = data['ids'].cuda()
            masks = data['masks'].cuda()
            tweet = data['tweet']
            offsets = data['offsets'].numpy()

            start_logits = []
            end_logits = []
            for model in models:
                with torch.no_grad():
                    output = model(ids, masks)
                    start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                    end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

            start_logits = np.mean(start_logits, axis=0)
            end_logits = np.mean(end_logits, axis=0)
            for i in range(len(ids)):    
                start_pred = np.argmax(start_logits[i])
                end_pred = np.argmax(end_logits[i])
                if start_pred > end_pred:
                    pred = tweet[i]
                else:
                    pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                predictions.append(pred)

        if self.train_only_posi_nega:
            neutral_idxs = (test_df['sentiment'].values=='neutral')
            predictions = np.array(predictions)
            predictions[neutral_idxs] = test_df['text'].values[neutral_idxs]

        # # Submission
        sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
        sub_df['selected_text'] = predictions
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
        sub_df.to_csv(os.path.join('submission.csv'), index=False)
        sub_df.head()

class Model2_v1_0_3_1:
    """
    cv 0., lb 0.
    different learning rate (x30) 

    remove pad
    train only posi nega
    """
    def __init__(self):
        self.seed = 42
        seed_everything(self.seed)

        self.save_dir = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_only_posi_nega = True
        self.num_fold=3
        self.remove_pad = True

    def train(self):
        # # Training

        # %% [code]
        num_epochs = 3
        batch_size = 32
        lr = 5e-5
        dif_lr_rate = 10
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

        # %% [code]

        train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
        if self.train_only_posi_nega:
            train_df = train_df[(train_df['sentiment']=='positive') | (train_df['sentiment']=='negative')].reset_index(drop=True)
        train_df['text'] = train_df['text'].astype(str)
        train_df['selected_text'] = train_df['selected_text'].astype(str)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=0): 
            if fold < self.num_fold:
                print(f'Fold: {fold}')

                model = TweetModel()

                bert_params, bert_params_nodecay, other_params = model.get_params()
                params = [
                    {'params': bert_params, 'lr': lr, 'weight_decay':0.01},
                    {'params': bert_params_nodecay, 'lr': lr, 'weight_decay':0.01},
                    {'params': other_params, 'lr': lr * dif_lr_rate, 'weight_decay':0.01}
                    ]
                optimizer = optim.AdamW(params, lr=3e-5, betas=(0.9, 0.999))
                
                criterion = loss_fn    
                dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

                logger = train_model(
                    model, 
                    dataloaders_dict,
                    criterion, 
                    optimizer, 
                    num_epochs,
                    os.path.join(self.save_dir,f'roberta_fold{fold}.pth'),
                    self.remove_pad)

                # save log
                df = pd.DataFrame(logger)
                df.columns = ['epoch', 'train_loss', 'train_score', 'val_loss', 'val_score']
                df.to_csv(os.path.join(self.save_dir,f'train_log_fold{fold}.csv'))

                scores.append((logger[-1])[-1])

        print('scores', scores)
        print('ave score', np.average(scores))
        return

    def test(self):
        test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
        test_df['text'] = test_df['text'].astype(str)
        test_loader = get_test_loader(test_df)
        predictions = []
        models = []
        for fold in range(self.num_fold):
            model = TweetModel()
            model.cuda()
            model.load_state_dict(torch.load(os.path.join(self.save_dir,f'roberta_fold{fold}.pth')))
            model.eval()
            models.append(model)

        for data in tqdm(test_loader):
            if self.remove_pad:
                data = remove_excessive_padding(data)

            ids = data['ids'].cuda()
            masks = data['masks'].cuda()
            tweet = data['tweet']
            offsets = data['offsets'].numpy()

            start_logits = []
            end_logits = []
            for model in models:
                with torch.no_grad():
                    output = model(ids, masks)
                    start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                    end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

            start_logits = np.mean(start_logits, axis=0)
            end_logits = np.mean(end_logits, axis=0)
            for i in range(len(ids)):    
                start_pred = np.argmax(start_logits[i])
                end_pred = np.argmax(end_logits[i])
                if start_pred > end_pred:
                    pred = tweet[i]
                else:
                    pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                predictions.append(pred)

        if self.train_only_posi_nega:
            neutral_idxs = (test_df['sentiment'].values=='neutral')
            predictions = np.array(predictions)
            predictions[neutral_idxs] = test_df['text'].values[neutral_idxs]

        # # Submission
        sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
        sub_df['selected_text'] = predictions
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
        sub_df.to_csv(os.path.join('submission.csv'), index=False)
        sub_df.head()

class Model2_v1_1_0:
    """
    cv 0.543579, lb 0.711
    different weight decay

    remove pad
    train only posi nega
    """
    def __init__(self):
        self.seed = 42
        seed_everything(self.seed)

        self.save_dir = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_only_posi_nega = True
        self.num_fold=3
        self.remove_pad = True

    def train(self):
        # # Training

        # %% [code]
        num_epochs = 3
        batch_size = 32
        lr = 5e-5
        dif_lr_rate = 1
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

        # %% [code]

        train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
        if self.train_only_posi_nega:
            train_df = train_df[(train_df['sentiment']=='positive') | (train_df['sentiment']=='negative')].reset_index(drop=True)
        train_df['text'] = train_df['text'].astype(str)
        train_df['selected_text'] = train_df['selected_text'].astype(str)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=0): 
            if fold < self.num_fold:
                print(f'Fold: {fold}')

                model = TweetModel()

                bert_params, bert_params_nodecay, other_params = model.get_params()
                params = [
                    {'params': bert_params, 'lr': lr, 'weight_decay':0.01},
                    {'params': bert_params_nodecay, 'lr': lr, 'weight_decay':0.0},
                    {'params': other_params, 'lr': lr * dif_lr_rate, 'weight_decay':0.0001}
                    ]
                optimizer = optim.AdamW(params, lr=3e-5, betas=(0.9, 0.999))
                
                criterion = loss_fn    
                dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

                logger = train_model(
                    model, 
                    dataloaders_dict,
                    criterion, 
                    optimizer, 
                    num_epochs,
                    os.path.join(self.save_dir,f'roberta_fold{fold}.pth'),
                    self.remove_pad)

                # save log
                df = pd.DataFrame(logger)
                df.columns = ['epoch', 'train_loss', 'train_score', 'val_loss', 'val_score']
                df.to_csv(os.path.join(self.save_dir,f'train_log_fold{fold}.csv'))

                scores.append((logger[-1])[-1])

        print('scores', scores)
        print('ave score', np.average(scores))
        return

    def test(self):
        test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
        test_df['text'] = test_df['text'].astype(str)
        test_loader = get_test_loader(test_df)
        predictions = []
        models = []
        for fold in range(self.num_fold):
            model = TweetModel()
            model.cuda()
            model.load_state_dict(torch.load(os.path.join(self.save_dir,f'roberta_fold{fold}.pth')))
            model.eval()
            models.append(model)

        for data in tqdm(test_loader):
            if self.remove_pad:
                data = remove_excessive_padding(data)

            ids = data['ids'].cuda()
            masks = data['masks'].cuda()
            tweet = data['tweet']
            offsets = data['offsets'].numpy()

            start_logits = []
            end_logits = []
            for model in models:
                with torch.no_grad():
                    output = model(ids, masks)
                    start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                    end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

            start_logits = np.mean(start_logits, axis=0)
            end_logits = np.mean(end_logits, axis=0)
            for i in range(len(ids)):    
                start_pred = np.argmax(start_logits[i])
                end_pred = np.argmax(end_logits[i])
                if start_pred > end_pred:
                    pred = tweet[i]
                else:
                    pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                predictions.append(pred)

        if self.train_only_posi_nega:
            neutral_idxs = (test_df['sentiment'].values=='neutral')
            predictions = np.array(predictions)
            predictions[neutral_idxs] = test_df['text'].values[neutral_idxs]

        # # Submission
        sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
        sub_df['selected_text'] = predictions
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
        sub_df.to_csv(os.path.join('submission.csv'), index=False)
        sub_df.head()

class Model2_v1_1_1:
    """
    cv 0.543579, lb 0.
    weight decay

    remove pad
    train only posi nega
    """
    def __init__(self):
        self.seed = 42
        seed_everything(self.seed)

        self.save_dir = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_only_posi_nega = True
        self.num_fold=3
        self.remove_pad = True

    def train(self):
        # # Training

        # %% [code]
        num_epochs = 3
        batch_size = 32
        lr = 5e-5
        dif_lr_rate = 1
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

        # %% [code]

        train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
        if self.train_only_posi_nega:
            train_df = train_df[(train_df['sentiment']=='positive') | (train_df['sentiment']=='negative')].reset_index(drop=True)
        train_df['text'] = train_df['text'].astype(str)
        train_df['selected_text'] = train_df['selected_text'].astype(str)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=0): 
            if fold < self.num_fold:
                print(f'Fold: {fold}')

                model = TweetModel()

                bert_params, bert_params_nodecay, other_params = model.get_params()
                params = [
                    {'params': bert_params, 'lr': lr, 'weight_decay':0.01},
                    {'params': bert_params_nodecay, 'lr': lr, 'weight_decay':0.0001},
                    {'params': other_params, 'lr': lr * dif_lr_rate, 'weight_decay':0.0001}
                    ]
                optimizer = optim.AdamW(params, lr=3e-5, betas=(0.9, 0.999))
                
                criterion = loss_fn    
                dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

                logger = train_model(
                    model, 
                    dataloaders_dict,
                    criterion, 
                    optimizer, 
                    num_epochs,
                    os.path.join(self.save_dir,f'roberta_fold{fold}.pth'),
                    self.remove_pad)

                # save log
                df = pd.DataFrame(logger)
                df.columns = ['epoch', 'train_loss', 'train_score', 'val_loss', 'val_score']
                df.to_csv(os.path.join(self.save_dir,f'train_log_fold{fold}.csv'))

                scores.append((logger[-1])[-1])

        print('scores', scores)
        print('ave score', np.average(scores))
        return

    def test(self):
        test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
        test_df['text'] = test_df['text'].astype(str)
        test_loader = get_test_loader(test_df)
        predictions = []
        models = []
        for fold in range(self.num_fold):
            model = TweetModel()
            model.cuda()
            model.load_state_dict(torch.load(os.path.join(self.save_dir,f'roberta_fold{fold}.pth')))
            model.eval()
            models.append(model)

        for data in tqdm(test_loader):
            if self.remove_pad:
                data = remove_excessive_padding(data)

            ids = data['ids'].cuda()
            masks = data['masks'].cuda()
            tweet = data['tweet']
            offsets = data['offsets'].numpy()

            start_logits = []
            end_logits = []
            for model in models:
                with torch.no_grad():
                    output = model(ids, masks)
                    start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                    end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

            start_logits = np.mean(start_logits, axis=0)
            end_logits = np.mean(end_logits, axis=0)
            for i in range(len(ids)):    
                start_pred = np.argmax(start_logits[i])
                end_pred = np.argmax(end_logits[i])
                if start_pred > end_pred:
                    pred = tweet[i]
                else:
                    pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                predictions.append(pred)

        if self.train_only_posi_nega:
            neutral_idxs = (test_df['sentiment'].values=='neutral')
            predictions = np.array(predictions)
            predictions[neutral_idxs] = test_df['text'].values[neutral_idxs]

        # # Submission
        sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
        sub_df['selected_text'] = predictions
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
        sub_df.to_csv(os.path.join('submission.csv'), index=False)
        sub_df.head()

class Model2_v1_1_2:
    """
    cv 0.549316, lb 0.711
    weight decay

    remove pad
    train only posi nega
    """
    def __init__(self):
        self.seed = 42
        seed_everything(self.seed)

        self.save_dir = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_only_posi_nega = True
        self.num_fold=3
        self.remove_pad = True

    def train(self):
        # # Training

        # %% [code]
        num_epochs = 3
        batch_size = 32
        lr = 5e-5
        dif_lr_rate = 1
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

        # %% [code]

        train_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
        if self.train_only_posi_nega:
            train_df = train_df[(train_df['sentiment']=='positive') | (train_df['sentiment']=='negative')].reset_index(drop=True)
        train_df['text'] = train_df['text'].astype(str)
        train_df['selected_text'] = train_df['selected_text'].astype(str)

        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=0): 
            if fold < self.num_fold:
                print(f'Fold: {fold}')

                model = TweetModel()

                bert_params, bert_params_nodecay, other_params = model.get_params()
                params = [
                    {'params': bert_params, 'lr': lr, 'weight_decay':0.01},
                    {'params': bert_params_nodecay, 'lr': lr, 'weight_decay':0.0},
                    {'params': other_params, 'lr': lr * dif_lr_rate, 'weight_decay':0.01}
                    ]
                optimizer = optim.AdamW(params, lr=3e-5, betas=(0.9, 0.999))
                
                criterion = loss_fn    
                dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

                logger = train_model(
                    model, 
                    dataloaders_dict,
                    criterion, 
                    optimizer, 
                    num_epochs,
                    os.path.join(self.save_dir,f'roberta_fold{fold}.pth'),
                    self.remove_pad)

                # save log
                df = pd.DataFrame(logger)
                df.columns = ['epoch', 'train_loss', 'train_score', 'val_loss', 'val_score']
                df.to_csv(os.path.join(self.save_dir,f'train_log_fold{fold}.csv'))

                scores.append((logger[-1])[-1])

        print('scores', scores)
        print('ave score', np.average(scores))
        return

    def test(self):
        test_df = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
        test_df['text'] = test_df['text'].astype(str)
        test_loader = get_test_loader(test_df)
        predictions = []
        models = []
        for fold in range(self.num_fold):
            model = TweetModel()
            model.cuda()
            model.load_state_dict(torch.load(os.path.join(self.save_dir,f'roberta_fold{fold}.pth')))
            model.eval()
            models.append(model)

        for data in tqdm(test_loader):
            if self.remove_pad:
                data = remove_excessive_padding(data)

            ids = data['ids'].cuda()
            masks = data['masks'].cuda()
            tweet = data['tweet']
            offsets = data['offsets'].numpy()

            start_logits = []
            end_logits = []
            for model in models:
                with torch.no_grad():
                    output = model(ids, masks)
                    start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                    end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

            start_logits = np.mean(start_logits, axis=0)
            end_logits = np.mean(end_logits, axis=0)
            for i in range(len(ids)):    
                start_pred = np.argmax(start_logits[i])
                end_pred = np.argmax(end_logits[i])
                if start_pred > end_pred:
                    pred = tweet[i]
                else:
                    pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                predictions.append(pred)

        if self.train_only_posi_nega:
            neutral_idxs = (test_df['sentiment'].values=='neutral')
            predictions = np.array(predictions)
            predictions[neutral_idxs] = test_df['text'].values[neutral_idxs]

        # # Submission
        sub_df = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
        sub_df['selected_text'] = predictions
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
        sub_df.to_csv(os.path.join('submission.csv'), index=False)
        sub_df.head()