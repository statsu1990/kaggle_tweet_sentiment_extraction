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

class Model_v1_0_0():
    """
    score 0.7204
    baseline
    """
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold0.pth',
                            self.FILENAME_HEAD+'checkpoint_fold1.pth',
                            self.FILENAME_HEAD+'checkpoint_fold2.pth',]

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
        NUM_FOLD = 3

        EPOCHS = 3 if not ONLY_VAL else 1
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
                    cp = get_checkpoint(self.CHECK_POINT[fold])
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

class Model_v1_1_0():
    """
    train only positive and negative
    score 0.5444 (only posi and nega)
    """
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold0.pth',
                            self.FILENAME_HEAD+'checkpoint_fold1.pth',
                            self.FILENAME_HEAD+'checkpoint_fold2.pth',]

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
        NUM_FOLD = 3

        EPOCHS = 3 if not ONLY_VAL else 1
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 0
        GRAD_ACCUM_STEP = 1
        LR = 3e-5

        # data
        train_df = data_utils.get_original_data(is_train=True)
        train_df = data_utils.remove_neutral(train_df) # to train only positive and negative

        # train
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment)):
            if fold < NUM_FOLD:
                print(f'Fold: {fold}')

                model = self.get_model()
                if ONLY_VAL:
                    cp = get_checkpoint(self.CHECK_POINT[fold])
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
        preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_2_1():
    """
    score 0.547774713656 (only posi and nega)

    train only positive and negative
    label smoothing 0.05

    ----------------------
    label smoothing results

    Model_v1_1_0 (w/o LS) : score 0.5444 (only posi and nega)
    Model_v1_2_0 (LS 0.1) : score 0.541514 (only posi and nega)
    Model_v1_2_1 (LS 0.05) : score 0.54777 (only posi and nega)
    ----------------------
    """
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold0.pth',
                            self.FILENAME_HEAD+'checkpoint_fold1.pth',
                            self.FILENAME_HEAD+'checkpoint_fold2.pth',]

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

        EPOCHS = 3 if not ONLY_VAL else 1
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 0
        GRAD_ACCUM_STEP = 1
        LR = 3e-5

        LABEL_SMOOTHING = 0.05

        # data
        train_df = data_utils.get_original_data(is_train=True)
        train_df = data_utils.remove_neutral(train_df) # to train only positive and negative

        # train
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment)):
            if fold < NUM_FOLD:
                print(f'Fold: {fold}')

                model = self.get_model()
                if ONLY_VAL:
                    cp = get_checkpoint(self.CHECK_POINT[fold])
                    model.load_state_dict(cp['state_dict'])
            
                optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
                criterion = loss.IndexLoss(classes=self.MAX_LEN, smoothing=LABEL_SMOOTHING, dim=-1)
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
        preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_3_4():
    """
    score 0.553748246 (only posi and nega), LB 0.705

    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 

    ----
    different learning rate results

    (epoch, warmup epoch, learning rate, different learning rate)
    Model_v1_2_1, (3,0,3e-5,x1) : score 0.547774 (only posi and nega)
    Model_v1_3_0, (3,0,3e-5,x100) : score 0.546235 (only posi and nega)
    Model_v1_3_1, (3,1,3e-5,x100) : score 0.540208 (only posi and nega)
    Model_v1_3_2, (3,0,3e-5,x30) : score 0.550897 (only posi and nega)
    Model_v1_3_3, (3,0,1e-5,x100) : score 0.547941 (only posi and nega)
    Model_v1_3_4, (3,0,1e-5,x30) : score 0.553748, 0.551891 (only posi and nega)
    Model_v1_3_5, (3,1,1e-5,x100) : score 0.544455 (only posi and nega)
    Model_v1_3_6, (4,1,1e-5,x30) : score 0.547059 (only posi and nega)
    ----

    """
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold0.pth',
                            self.FILENAME_HEAD+'checkpoint_fold1.pth',
                            self.FILENAME_HEAD+'checkpoint_fold2.pth',]

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

        EPOCHS = 3 if not ONLY_VAL else 1
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 0
        GRAD_ACCUM_STEP = 1
        LR = 1e-5
        DIF_LR_RATE = 30

        LABEL_SMOOTHING = 0.05

        # data
        train_df = data_utils.get_original_data(is_train=True)
        train_df = data_utils.remove_neutral(train_df) # to train only positive and negative

        # train
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment)):
            if fold < NUM_FOLD:
                print(f'Fold: {fold}')

                model = self.get_model()
                if ONLY_VAL:
                    cp = get_checkpoint(self.CHECK_POINT[fold])
                    model.load_state_dict(cp['state_dict'])
                
                bert_params, other_params = model.get_params()
                params = [{'params': bert_params, 'lr': LR},
                          {'params': other_params, 'lr': LR * DIF_LR_RATE}]
                optimizer = optim.AdamW(params, betas=(0.9, 0.999))
                
                criterion = loss.IndexLoss(classes=self.MAX_LEN, smoothing=LABEL_SMOOTHING, dim=-1)
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
        preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_3_4_fold10():
    """
    score 0.54125 (only posi and nega), LB 0.709

    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 

    ----
    different learning rate results

    (epoch, warmup epoch, learning rate, different learning rate)
    Model_v1_2_1, (3,0,3e-5,x1) : score 0.547774 (only posi and nega)
    Model_v1_3_0, (3,0,3e-5,x100) : score 0.546235 (only posi and nega)
    Model_v1_3_1, (3,1,3e-5,x100) : score 0.540208 (only posi and nega)
    Model_v1_3_2, (3,0,3e-5,x30) : score 0.550897 (only posi and nega)
    Model_v1_3_3, (3,0,1e-5,x100) : score 0.547941 (only posi and nega)
    Model_v1_3_4, (3,0,1e-5,x30) : score 0.553748, 0.551891 (only posi and nega)
    Model_v1_3_5, (3,1,1e-5,x100) : score 0.544455 (only posi and nega)
    Model_v1_3_6, (4,1,1e-5,x30) : score 0.547059 (only posi and nega)

    Model_v1_3_4_fold10, (3,0,1e-5,x30) : score 0.541253 (only posi and nega)
    ----

    """
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 10
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

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
        NUM_FOLD = self.NUM_FOLD

        EPOCHS = 3 if not ONLY_VAL else 1
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 0
        GRAD_ACCUM_STEP = 1
        LR = 1e-5
        DIF_LR_RATE = 30

        LABEL_SMOOTHING = 0.05

        # data
        train_df = data_utils.get_original_data(is_train=True)
        train_df = data_utils.remove_neutral(train_df) # to train only positive and negative

        # train
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)
        val_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment)):
            if fold < NUM_FOLD:
                print(f'Fold: {fold}')

                model = self.get_model()
                if ONLY_VAL:
                    cp = get_checkpoint(self.CHECK_POINT[fold])
                    model.load_state_dict(cp['state_dict'])
                
                bert_params, other_params = model.get_params()
                params = [{'params': bert_params, 'lr': LR},
                          {'params': other_params, 'lr': LR * DIF_LR_RATE}]
                optimizer = optim.AdamW(params, betas=(0.9, 0.999))
                
                criterion = loss.IndexLoss(classes=self.MAX_LEN, smoothing=LABEL_SMOOTHING, dim=-1)
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
        preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return
