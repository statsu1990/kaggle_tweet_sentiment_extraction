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

class Model_v1_4_0():
    """
    score 0.54476 (only posi and nega), 
    implement remove_excessive_padding

    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
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

class Model_v1_5_0():
    """
    score 0.54651 (only posi and nega), 
    implement consideration of text_areas

    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
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

class Model_v1_6_6():
    """
    score 0.553753 (only posi and nega), lb 0.711
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False

    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    """
    score (cv is only posi and nega)
    param (n_hid, pooling, learnable)

    Model_v1_6_0 (3, ave, True) : cv 0.5476964
    Model_v1_6_1 (6, ave, True) : cv 0.5541571
    Model_v1_6_2 (9, ave, True) : cv 0.5484257
    Model_v1_6_3 (12, ave, True): cv 0.5489246
    Model_v1_5_0 (3, ave, False): cv 0.54651
    Model_v1_6_4 (6, ave, False): cv 0.5482903
    Model_v1_6_5 (9, ave, False): cv 0.553167
    Model_v1_6_6 (12, ave, False): cv 0.553753, lb 0.711
    Model_v1_6_7 (3, max, False): cv 0.543612
    Model_v1_6_8 (6, max, False): cv 0.548280
    Model_v1_6_9 (9, max, False): cv 0.549519
    Model_v1_6_10 (12, max, False): cv 0.551035

    train all sentiments
    Model_v1_6_11(12, ave, False) : cv 0.72232, lb 0.707
    """
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
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

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False

        model = tweet_model.TweetModel(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT)
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
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

class Model_v1_6_11():
    """
    score 0.72232, 
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False

    implement consideration of text_areas
    implement remove_excessive_padding
    [NO USE] train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    """
    score (cv is only posi and nega)
    param (n_hid, pooling, learnable)

    Model_v1_6_0 (3, ave, True) : cv 0.5476964
    Model_v1_6_1 (6, ave, True) : cv 0.5541571
    Model_v1_6_2 (9, ave, True) : cv 0.5484257
    Model_v1_6_3 (12, ave, True): cv 0.5489246
    Model_v1_5_0 (3, ave, False): cv 0.54651
    Model_v1_6_4 (6, ave, False): cv 0.5482903
    Model_v1_6_5 (9, ave, False): cv 0.553167
    Model_v1_6_6 (12, ave, False): cv 0.553753, lb 0.711
    Model_v1_6_7 (3, max, False): cv 0.543612
    Model_v1_6_8 (6, max, False): cv 0.548280
    Model_v1_6_9 (9, max, False): cv 0.549519
    Model_v1_6_10 (12, max, False): cv 0.551035

    train all sentiments
    Model_v1_6_11(12, ave, False) : cv 0.72232, lb 0.707
    """
    def __init__(self):
        self.set_config()

        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.TRAIN_ONLY_POSI_NEGA = False

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

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False

        model = tweet_model.TweetModel(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT)
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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_7_0():
    """
    score 0.551518 (only posi and nega), 
    dropout 0.1
    
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.TRAIN_ONLY_POSI_NEGA = True

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

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT)
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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_8_2():
    """
    score 0.5567775 (only posi and nega), lb 0.709
    multi linear head (hidden=[128], dropout=0.1)

    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    """
    Model_v1_5_0 (linear head) : cv 0.54651 (only posi and nega)

    3 hidden layer, average, not learnable
    Model_v1_8_0 (multi linear head, 768-128-2) : cv 0.54918 (only posi and nega)
    Model_v1_8_1 (conv head, k=3, n_conv=1, 768-768) : cv 0.55192 (only posi and nega), lb 0.709

    12 hidden layer, average, not learnable
    Model_v1_8_2 (multi linear head, 768-128-2) : cv 0.55677 (only posi and nega)
    Model_v1_8_3 (conv head, k=3, n_conv=1, 768-768) : cv 0.55003 (only posi and nega)

    other condition
    dropout=0.1
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.TRAIN_ONLY_POSI_NEGA = True

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

        NS_HIDDEN = [128]
        DROPOUT = 0.1
        ans_idx_head = tweet_model.LinearHead(768, 2, NS_HIDDEN, DROPOUT)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_9_7():
    """
    score 0.552004 (only posi and nega), 
    SentimentAttentionHead(n_element=8, reduction=4, dropout=0.0, add=linear_head(hidden=None))

    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    """
    Sentiment attention head

    mode (n_element, reduction, dropout, add) : score

    Model_v1_9_0 (5, 4, 0.1, None) : cv 0.546743 (only posi and nega)

    Model_v1_9_2 (3, 4, 0.0, None) : cv 0.547488 (only posi and nega)
    Model_v1_9_1 (5, 4, 0.0, None) : cv 0.551155 (only posi and nega)
    Model_v1_9_3 (8, 4, 0.0, None) : cv 0.551247 (only posi and nega)
    Model_v1_9_4 (16, 4, 0.0, None) : cv 0.545075 (only posi and nega)
    Model_v1_9_5 (32, 4, 0.0, None) : cv 0.550671 (only posi and nega)
    Model_v1_9_6 (64, 4, 0.0, None) : cv 0.551484 (only posi and nega)

    Model_v1_9_7 (8, 4, 0.0, linear_head) : cv 0.552004 (only posi and nega)
    Model_v1_9_8 (8, 4, 0.0, linear_head) : cv 723900

    other condition
    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    [No use]train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.TRAIN_ONLY_POSI_NEGA = True

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

        N_ELEMENT = 8
        REDUCTION = 4
        DROPOUT = 0.0
        ADDITIONAL_HEAD = tweet_model.LinearHead(768, 2, None, 0.1)
        ans_idx_head = tweet_model.SentimentAttentionHead(768, 768, N_ELEMENT, REDUCTION, DROPOUT, 
                                                          additional_head=ADDITIONAL_HEAD)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_10_2():
    """
    score 0.551860 (only posi and nega), 
    JaccardExpectationLoss * 1.0 + IndexLoss * 0.5

    multi linear head (hidden=[128], dropout=0.1)
    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    """
    https://www.kaggle.com/koza4ukdmitrij/jaccard-expectation-loss
    (JaccardExpectationLoss rate, IndexLoss rate)

    Model_v1_8_2 (0, 1) : cv 0.556777(only posi and nega)
    Model_v1_10_0 (1, 0) : cv 0.520439 (only posi and nega)
    Model_v1_10_1 (1, 0.2) : cv 0.547114(only posi and nega)
    Model_v1_10_2 (1, 0.5) : cv 0.551860(only posi and nega)
    Model_v1_10_3 (1, 1) : cv 0.546657(only posi and nega)

    Other conditions
    multi linear head (hidden=[128], dropout=0.1)
    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.TRAIN_ONLY_POSI_NEGA = True

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

        NS_HIDDEN = [128]
        DROPOUT = 0.1
        ans_idx_head = tweet_model.LinearHead(768, 2, NS_HIDDEN, DROPOUT)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                criterion = loss.LossCompose([loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1), 
                                              loss.JaccardExpectationLoss(),], 
                                             [0.5, 1.0])
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1) #learning rate decay

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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_11_0():
    """
    score 0.5478125 (only posi and nega), lb 
    roberta-base_from_kaggle_dataset

    multi linear head (hidden=[128], dropout=0.1)
    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.TRAIN_ONLY_POSI_NEGA = True

        # constants
        self.MAX_LEN = 96

        # pretrained
        self.PRETRAINED_DIR = '../input/roberta-base_from_kaggle_dataset'
        self.VOCAB_FILE = os.path.join(self.PRETRAINED_DIR, 'vocab.json')
        self.MERGES_FILE = os.path.join(self.PRETRAINED_DIR, 'merges.txt')
        return

    def get_model(self):
        MODEL_CONFIG = os.path.join(self.PRETRAINED_DIR, 'config.json')
        PRETRAINED_MODEL = os.path.join(self.PRETRAINED_DIR, 'pytorch_model.bin')

        NS_HIDDEN = [128]
        DROPOUT = 0.1
        ans_idx_head = tweet_model.LinearHead(768, 2, NS_HIDDEN, DROPOUT)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_12_0():
    """
    score 0.549195 (only posi and nega), lb 
    sub task answering mutch sentiment and text (pretraining)

    multi linear head (hidden=[128], dropout=0.1)
    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    """
    pretraining sub task answering mutch sentiment and text
    (change sentiment p, batch size)

    - Model_v1_12_0 (0.5, 64) : cv 0.549195 (train with posi nega)
    - Model_v1_12_1 (0.5, 64) : cv 0.726108, lb 0.708
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.TRAIN_ONLY_POSI_NEGA = True

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

        NS_HIDDEN = [128]
        DROPOUT = 0.1
        ans_idx_head = tweet_model.LinearHead(768, 2, NS_HIDDEN, DROPOUT)

        NS_HIDDEN = None
        DROPOUT = 0.1
        match_sent_head = tweet_model.LinearHead(768, 1, NS_HIDDEN, DROPOUT)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head, match_sent_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

        # constants
        NUM_FOLD = self.NUM_FOLD

        CHANGE_SENTIMENT_P = 0.5

        EPOCHS = 4 if not ONLY_VAL else 1
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 1
        GRAD_ACCUM_STEP = 2
        LR = 1e-5
        DIF_LR_RATE = 30

        LABEL_SMOOTHING = 0.05

        # data
        train_df = data_utils.get_original_data(is_train=True)
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                criterion = loss.CombineMatchSentimentLoss(loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1, reduce=False), 
                                                           [1.0, 0.0])


                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE, 
                                                                     CHANGE_SENTIMENT_P)

                cp_filename = self.FILENAME_HEAD + 'checkpoint_fold' + str(fold) + '.pth'
                model, val_score = training.train_model(
                            model, dataloaders_dict, criterion, optimizer, 
                            EPOCHS, GRAD_ACCUM_STEP, WARMUP_EPOCHS, step_scheduler,
                            self.FILENAME_HEAD+'pretrain_', fold, ONLY_VAL)

                # ----------------
                print('fineturning')
                bert_params, other_params = model.get_params()
                params = [{'params': bert_params, 'lr': LR},
                          {'params': other_params, 'lr': LR * DIF_LR_RATE}]
                optimizer = optim.AdamW(params, betas=(0.9, 0.999))
                
                criterion = loss.CombineMatchSentimentLoss(loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1, reduce=False), 
                                                           [0.0, 1.0])


                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     32, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE, 
                                                                     0.0)

                cp_filename = self.FILENAME_HEAD + 'checkpoint_fold' + str(fold) + '.pth'
                model, val_score = training.train_model(
                            model, dataloaders_dict, criterion, optimizer, 
                            3, GRAD_ACCUM_STEP, 1, step_scheduler,
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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_13_14():
    """
    score 0.538617, lb 
    (ep5 cv, ep3 cv)
    insert_selected_text_p = 0.3

    multi linear head (hidden=[128], dropout=0.1)
    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.TRAIN_ONLY_POSI_NEGA = True
        self.ADD_NEUTRAL_P = 0.0
        self.ADD_NONSENTIMENT_P = 0.0
        self.ADD_DIFFERENT_SENTIMENT_P = 0.0
        self.EXCHANGE_SELECTED_TEXT_P = 0.0
        self.INSERT_SELECTED_TEXT_P = 0.3

        # constants
        self.MAX_LEN = 96 * 3

        # pretrained
        self.PRETRAINED_DIR = '../input/roberta-base'
        self.VOCAB_FILE = os.path.join(self.PRETRAINED_DIR, 'vocab.json')
        self.MERGES_FILE = os.path.join(self.PRETRAINED_DIR, 'merges.txt')
        return

    def get_model(self):
        MODEL_CONFIG = os.path.join(self.PRETRAINED_DIR, 'config.json')
        PRETRAINED_MODEL = os.path.join(self.PRETRAINED_DIR, 'pytorch_model.bin')

        NS_HIDDEN = [128]
        DROPOUT = 0.1
        ans_idx_head = tweet_model.LinearHead(768, 2, NS_HIDDEN, DROPOUT)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

        # constants
        NUM_FOLD = self.NUM_FOLD

        EPOCHS = 7 if not ONLY_VAL else 1
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 0
        GRAD_ACCUM_STEP = 1
        LR = 1e-5
        DIF_LR_RATE = 30

        LABEL_SMOOTHING = 0.05

        # data
        train_df = data_utils.get_original_data(is_train=True)
        if self.TRAIN_ONLY_POSI_NEGA:
            neutral_df = train_df[train_df['sentiment']=='neutral'].reset_index(drop=True)
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE, 
                                                                     add_neutral_p=self.ADD_NEUTRAL_P, 
                                                                     neutral_text=neutral_df['text'].values, 
                                                                     add_nonsentiment_p=self.ADD_NONSENTIMENT_P,
                                                                     add_different_sentiment_p=self.ADD_DIFFERENT_SENTIMENT_P, 
                                                                     exchange_selected_text_p=self.EXCHANGE_SELECTED_TEXT_P, 
                                                                     insert_selected_text_p=self.INSERT_SELECTED_TEXT_P)

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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_14_0():
    """
    premake dataset 2:05/epoch (defalt 2:30/epoch)

    multi linear head (hidden=[128], dropout=0.1)
    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.PREMAKE_DATASET = True
        self.TRAIN_ONLY_POSI_NEGA = True

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

        NS_HIDDEN = [128]
        DROPOUT = 0.1
        ans_idx_head = tweet_model.LinearHead(768, 2, NS_HIDDEN, DROPOUT)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE, 
                                                                     premake_dataset=self.PREMAKE_DATASET)

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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_15_2():
    """
    score 0.551031 (only posi and nega), lb 
    OHEM 0.8

    (fork Model_v1_8_2)
    multi linear head (hidden=[128], dropout=0.1)
    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.PREMAKE_DATASET = True
        self.TRAIN_ONLY_POSI_NEGA = True

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

        NS_HIDDEN = [128]
        DROPOUT = 0.1
        ans_idx_head = tweet_model.LinearHead(768, 2, NS_HIDDEN, DROPOUT)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

        # constants
        NUM_FOLD = self.NUM_FOLD

        EPOCHS = 4 if not ONLY_VAL else 1
        BATCH_SIZE = 32
        WARMUP_EPOCHS = 0
        GRAD_ACCUM_STEP = 1
        LR = 1e-5
        DIF_LR_RATE = 30

        LABEL_SMOOTHING = 0.05
        OHEM_RATE = 0.8

        # data
        train_df = data_utils.get_original_data(is_train=True)
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1, ohem_rate=OHEM_RATE)
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE, 
                                                                     premake_dataset=self.PREMAKE_DATASET)

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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_16_1():
    """
    cv 0.553225 (only posi and nega), lb 
    XLNetQAHead(n_top=5)

    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    """
    Model_v1_16_0, XLNetQAHead(n_top=3) : cv 0.546599 (only posi and nega)
    Model_v1_16_1, XLNetQAHead(n_top=5) : cv 0.553225 (only posi and nega)
    Model_v1_16_6, XLNetQAHead(n_top=10) : cv 0.545139 (only posi and nega)
    Model_v1_16_2, XLNetQAHead2(n_top=3) : cv 0.550632 (only posi and nega)
    Model_v1_16_3, XLNetQAHead2(n_top=5) : cv 0.545782 (only posi and nega)
    Model_v1_16_7, XLNetQAHead2(n_top=10) : cv 0.548332 (only posi and nega)
    Model_v1_16_4, XLNetQAHead3(n_top=3) : cv 0.544662 (only posi and nega)
    Model_v1_16_5, XLNetQAHead3(n_top=5) : cv 0.550960 (only posi and nega)
    Model_v1_16_8, XLNetQAHead3(n_top=10) : cv 0.550018 (only posi and nega)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.PREMAKE_DATASET = True
        self.TRAIN_ONLY_POSI_NEGA = True

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

        START_TOP_N = 5
        ans_idx_head = tweet_model.XLNetQAHead(768, START_TOP_N)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                optimizer = optim.AdamW(params, betas=(0.9, 0.999), weight_decay=0.01)
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE, 
                                                                     premake_dataset=self.PREMAKE_DATASET)

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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_17_3():
    """
    cv 0.554637 (only posi and nega), lb 

    XLNetQAHead(n_top=5)
    Learnable weight of averaging hidden layer, n_hid=3, average, learn=True

    dropout=0.1
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    """
    Model_v1_16_1, (n_hid=12, average, learn=False), cv 553225 (only posi and nega)
    Model_v1_17_0, (n_hid=12, average, learn=True), cv 550804 (only posi and nega)
    Model_v1_17_1, (n_hid=12, max, learn=False), cv 548205 (only posi and nega)
    Model_v1_17_2, (n_hid=3, average, learn=False), cv 0.551376 (only posi and nega)
    Model_v1_17_3, (n_hid=3, average, learn=True), cv 0.554637 (only posi and nega)
    Model_v1_17_4, (n_hid=3, max, learn=False), cv 0.548902 (only posi and nega)
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.PREMAKE_DATASET = True
        self.TRAIN_ONLY_POSI_NEGA = True

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

        START_TOP_N = 5
        ans_idx_head = tweet_model.XLNetQAHead(768, START_TOP_N)

        NUM_USE_HID_LAYERS = 3
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = True
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                optimizer = optim.AdamW(params, betas=(0.9, 0.999), weight_decay=0.01)
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE, 
                                                                     premake_dataset=self.PREMAKE_DATASET)

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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return

class Model_v1_18_1():
    """
    cv 0.555713 (only posi and nega), lb 
    weight decay (hidden(LN, bias)=0.0, hidden(other)=0.01, head=0.0001)

    multi linear head (hidden=[128], dropout=0.1)
    dropout=0.1
    Learnable weight of averaging hidden layer, n_hid=12, average, learn=False
    implement consideration of text_areas
    implement remove_excessive_padding
    train only positive and negative
    label smoothing 0.05
    lr 1e-5
    different learning rate (x30) 
    """
    """

    weight decay (hidden(LN, bias)=xxx, hidden(other)=xxx, head=xxx)

    Model_v1_18_0 (0.0, 0.01,  0.01): cv 0.54893
    Model_v1_18_4 (0.0, 0.01,  0.001): cv 0.548809
    Model_v1_18_1 (0.0, 0.01,  0.0001): cv 0.555713
    Model_v1_18_2 (0.0, 0.01,  0.00001): cv 0.553674
    Model_v1_18_3 (0.0, 0.001, 0.0001): cv 0.549853
    Model_v1_18_4 (0.0, 0.1,   0.0001): cv 0.552835
    """
    def __init__(self):
        self.set_config()
        return

    def set_config(self):
        self.SAVE_DIR = os.path.join(RESULTS_DIR, self.__class__.__name__)
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FILENAME_HEAD = os.path.join(self.SAVE_DIR, '')

        self.NUM_FOLD = 3
        self.CHECK_POINT = [self.FILENAME_HEAD+'checkpoint_fold'+str(i)+'.pth' for i in range(self.NUM_FOLD)]

        # data
        self.PREMAKE_DATASET = True
        self.TRAIN_ONLY_POSI_NEGA = True

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

        NS_HIDDEN = [128]
        DROPOUT = 0.1
        ans_idx_head = tweet_model.LinearHead(768, 2, NS_HIDDEN, DROPOUT)

        NUM_USE_HID_LAYERS = 12
        POOLING = 'average' # 'average', 'max'
        LEARNABLE_WEIGHT = False
        DROPOUT = 0.1

        model = tweet_model.TweetModel2(MODEL_CONFIG, PRETRAINED_MODEL, 
                                       NUM_USE_HID_LAYERS, POOLING, LEARNABLE_WEIGHT,
                                       DROPOUT, 
                                       ans_idx_head)
        return model

    def train(self, only_val=False):
        ONLY_VAL = only_val

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
        if self.TRAIN_ONLY_POSI_NEGA:
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
                
                no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
                bert_params, bert_params_sep, other_params = model.get_params(separate_names=no_decay)
                params = [
                    #{'params': bert_params, 'lr': LR, 'weight_decay':0.01},
                    {'params': bert_params, 'lr': LR, 'weight_decay':0.01},
                    {'params': bert_params_sep, 'lr': LR, 'weight_decay':0.0},
                    {'params': other_params, 'lr': LR * DIF_LR_RATE, 'weight_decay':0.0001}
                    ]
                optimizer = optim.AdamW(params, betas=(0.9, 0.999))
                
                criterion = loss.IndexLoss(classes=None, smoothing=LABEL_SMOOTHING, dim=-1)
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1) #learning rate decay

                dataloaders_dict = data_loader.get_train_val_loaders(train_df, train_idx, val_idx, 
                                                                     BATCH_SIZE, self.MAX_LEN, 
                                                                     self.VOCAB_FILE, self.MERGES_FILE, 
                                                                     premake_dataset=self.PREMAKE_DATASET)

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
        if self.TRAIN_ONLY_POSI_NEGA:
            preds = pred_utils.neutral_pred_to_text(preds, test_df['text'], test_df['sentiment'])

        pred_utils.make_submission(preds, self.FILENAME_HEAD)
        return
