import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
import torch.optim as optim

from data import data_utils
from dataloader import data_loader
from model import tweet_model, loss
from training import training
from prediction import predicting, pred_utils

RESULTS_DIR = '../results'

def get_checkpoint(path):
    cp = torch.load(path, map_location=lambda storage, loc: storage)
    return cp

class Proto_v0_0_1():
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        # constants
        self.MAX_LEN = 96

        # pretrained
        self.PRETRAINED_DIR = '../input/roberta-base'
        self.VOCAB_FILE = os.path.join(self.PRETRAINED_DIR, 'vocab.json')
        self.MERGES_FILE = os.path.join(self.PRETRAINED_DIR, 'merges.txt')

        return

    def get_model(self):
        MODEL_CONFIG = os.path.join(self.PRETRAINED_DIR, 'config.json')
        PRETRAINED_MODEL = os.path.join(self.PRETRAINED_DIR, 'pytorch_model.bin')

        model = tweet_model.TweetModel(MODEL_CONFIG, PRETRAINED_MODEL)
        return model

    def run(self):
        EPOCHS = 3
        BATCH_SIZE = 32
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)

        train_df = data_utils.get_original_data(is_train=True)

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1): 
            print(f'Fold: {fold}')
            model = self.get_model()
            
            optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
            criterion = loss.loss_fn
            dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                 BATCH_SIZE, self.MAX_LEN, 
                                                                 self.VOCAB_FILE, self.MERGES_FILE)

            training.train_model(
                model, 
                dataloaders_dict,
                criterion, 
                optimizer, 
                EPOCHS,
                f'roberta_fold{fold}.pth')

class Proto_v0_0_2():
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.FILENAME_HEAD = self.__class__.__name__ + '_'
        self.CHECK_POINT = 'Proto_v0_0_2_checkpoint.pth'

        # constants
        self.MAX_LEN = 96

        # pretrained
        self.PRETRAINED_DIR = '../input/roberta-base'
        self.VOCAB_FILE = os.path.join(self.PRETRAINED_DIR, 'vocab.json')
        self.MERGES_FILE = os.path.join(self.PRETRAINED_DIR, 'merges.txt')
        return

    def get_model(self):
        MODEL_CONFIG = os.path.join(self.PRETRAINED_DIR, 'config.json')
        PRETRAINED_MODEL = os.path.join(self.PRETRAINED_DIR, 'pytorch_model.bin')

        model = tweet_model.TweetModel(MODEL_CONFIG, PRETRAINED_MODEL)
        return model

    def train(self):
        # constants
        EPOCHS = 6
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 2
        GRAD_ACCUM_STEP = 1
        LR = 3e-5

        # data
        train_df = data_utils.get_original_data(is_train=True)

        # train
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment)):
            if fold < 1:
                print(f'Fold: {fold}')

                model = self.get_model()
            
                optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
                criterion = loss.loss_fn
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE)

                model = training.train_model(
                            model, dataloaders_dict, criterion, optimizer, 
                            EPOCHS, GRAD_ACCUM_STEP, WARMUP_EPOCHS, step_scheduler,
                            self.FILENAME_HEAD)

    def pred_test(self, model=None):
        if model is None:
            model = self.get_model()
            cp = get_checkpoint(self.CHECK_POINT)
            model.load_state_dict(cp['state_dict'])

        # data
        test_df = data_utils.get_original_data(is_train=False)
        test_loader = data_loader.get_test_loader(test_df, batch_size=32)

        # pred
        preds = predicting.predicter(model, test_loader)
        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Proto_v0_1_0():
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.FILENAME_HEAD = self.__class__.__name__ + '_'
        self.CHECK_POINT = 'Proto_v0_1_0_checkpoint.pth' #None

        # constants
        self.MAX_LEN = 96

        # pretrained
        self.PRETRAINED_DIR = '../input/roberta-base'
        self.VOCAB_FILE = os.path.join(self.PRETRAINED_DIR, 'vocab.json')
        self.MERGES_FILE = os.path.join(self.PRETRAINED_DIR, 'merges.txt')
        return

    def get_model(self):
        MODEL_CONFIG = os.path.join(self.PRETRAINED_DIR, 'config.json')
        PRETRAINED_MODEL = os.path.join(self.PRETRAINED_DIR, 'pytorch_model.bin')

        model = tweet_model.TweetModel(MODEL_CONFIG, PRETRAINED_MODEL)
        return model

    def train(self):
        ONLY_VAL = True

        # constants
        EPOCHS = 3
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 0
        GRAD_ACCUM_STEP = 1
        LR = 3e-5

        # data
        train_df = data_utils.get_original_data(is_train=True)

        # train
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment)):
            if fold < 1:
                print(f'Fold: {fold}')

                model = self.get_model()
                if ONLY_VAL:
                    cp = get_checkpoint(self.CHECK_POINT)
                    model.load_state_dict(cp['state_dict'])
            
                optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
                criterion = loss.loss_fn
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE)

                model = training.train_model(
                            model, dataloaders_dict, criterion, optimizer, 
                            EPOCHS, GRAD_ACCUM_STEP, WARMUP_EPOCHS, step_scheduler,
                            self.FILENAME_HEAD, ONLY_VAL)

    def pred_test(self, model=None):
        if model is None:
            model = self.get_model()
            cp = get_checkpoint(self.CHECK_POINT)
            model.load_state_dict(cp['state_dict'])

        # data
        test_df = data_utils.get_original_data(is_train=False)
        test_loader = data_loader.get_test_loader(test_df, 32, self.MAX_LEN, 
                                                  self.VOCAB_FILE, self.MERGES_FILE)

        # pred
        preds = predicting.predicter(model, test_loader)
        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Proto_v0_2_0():
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.CHECK_POINT = ['Proto_v0_2_0_checkpoint_fold0.pth',
                            'Proto_v0_2_0_checkpoint_fold1.pth',
                            'Proto_v0_2_0_checkpoint_fold2.pth',]

        # constants
        self.MAX_LEN = 96

        # pretrained
        self.PRETRAINED_DIR = '../input/roberta-base'
        self.VOCAB_FILE = os.path.join(self.PRETRAINED_DIR, 'vocab.json')
        self.MERGES_FILE = os.path.join(self.PRETRAINED_DIR, 'merges.txt')
        return

    def get_model(self):
        MODEL_CONFIG = os.path.join(self.PRETRAINED_DIR, 'config.json')
        PRETRAINED_MODEL = os.path.join(self.PRETRAINED_DIR, 'pytorch_model.bin')

        model = tweet_model.TweetModel(MODEL_CONFIG, PRETRAINED_MODEL)
        return model

    def train(self):
        ONLY_VAL = False

        # constants
        NUM_FOLD = 3

        EPOCHS = 3
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 0
        GRAD_ACCUM_STEP = 1
        LR = 3e-5

        # data
        train_df = data_utils.get_original_data(is_train=True)

        # train
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment)):
            if fold < NUM_FOLD:
                print(f'Fold: {fold}')

                model = self.get_model()
                if ONLY_VAL:
                    cp = get_checkpoint(self.CHECK_POINT)
                    model.load_state_dict(cp['state_dict'])
            
                optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
                criterion = loss.loss_fn
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE)

                cp_filename = self.FILENAME_HEAD + 'checkpoint_fold' + str(fold) + '.pth'
                model, val_score = training.train_model(
                            model, dataloaders_dict, criterion, optimizer, 
                            EPOCHS, GRAD_ACCUM_STEP, WARMUP_EPOCHS, step_scheduler,
                            self.FILENAME_HEAD, fold, ONLY_VAL)
                val_scores.append(val_score)

        # summary
        print()
        for fold, score in enumerate(val_scores):
            print('fold {0} : score {1}'.format(fold, score))
        print('average score {0}'.format(np.mean(val_scores)))

    def pred_test(self, models=None):
        if models is None:
            models = []
            for cpfile in self.CHECK_POINT:
                model = self.get_model()
                cp = get_checkpoint(cpfile)
                model.load_state_dict(cp['state_dict'])
                models.append(model)

        # data
        test_df = data_utils.get_original_data(is_train=False)
        test_loader = data_loader.get_test_loader(test_df, 32, self.MAX_LEN, 
                                                  self.VOCAB_FILE, self.MERGES_FILE)

        # pred
        preds = predicting.predicter(models, test_loader)
        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return
