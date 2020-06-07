import torch
from tqdm import tqdm
import numpy as np

from model import eval_utils as evut
from .scheduler import WarmUpLR
from .train_utils import save_log, save_checkpoint

def remove_excessive_padding(data, pad_id=1):
    """
    Set length to the max length except pad in batch.
    """
    """
    ids = data['ids']
    masks = data['masks']
    offsets = data['offsets']
    text_areas = data['text_areas'].numpy()
    """
    min_n_pad = torch.min(torch.sum(torch.eq(data['ids'], pad_id), dim=-1))
    max_len = data['ids'].size()[-1] - min_n_pad

    data['ids'] = (data['ids'])[:,:max_len]
    data['masks'] = (data['masks'])[:,:max_len]
    data['text_areas'] = (data['text_areas'])[:,:max_len]
    data['offsets'] = (data['offsets'])[:,:max_len]

    return data

def trainer(model, dataloaders_dict, criterion, optimizer, 
            grad_accum_steps, warmup_scheduler, 
            only_val=False, remove_pad=True):
    model = model.cuda()

    if not only_val:
        phases = ['train', 'val']
    else:
        phases = ['val']

    for phase in phases:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        epoch_loss = 0.0
        epoch_jaccard = 0.0
            
        n_high_start = 0
        n_all = 0

        if phase == 'train':
            optimizer.zero_grad()

        for batch_idx, data in enumerate(tqdm(dataloaders_dict[phase])):
            # remove excessive padding
            if remove_pad:
                data = remove_excessive_padding(data)

            # data
            ids = data['ids'].cuda()
            masks = data['masks'].cuda()
            tweet = data['tweet']
            offsets = data['offsets'].numpy()
            text_areas = data['text_areas'].cuda()
            start_idx = data['start_idx'].cuda()
            end_idx = data['end_idx'].cuda()
            match_sent = data['match_sent'].cuda()

            with torch.set_grad_enabled(phase == 'train'):
                # pred, loss
                outputs = model(ids, masks, start_idx, text_areas)
                if len(outputs) == 3:
                    start_logits, end_logits, match_sent_logits = outputs[0], outputs[1], outputs[2]
                    start_logits[~text_areas] = torch.finfo(torch.float32).min
                    end_logits[~text_areas] = torch.finfo(torch.float32).min
                    loss = criterion(start_logits, end_logits, start_idx, end_idx, text_areas, match_sent_logits, match_sent) / grad_accum_steps

                else:
                    start_logits, end_logits, start_logits2, end_logits2, match_sent_logits = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
                    start_logits[~text_areas] = torch.finfo(torch.float32).min
                    end_logits[~text_areas] = torch.finfo(torch.float32).min
                    start_logits2[~text_areas] = torch.finfo(torch.float32).min
                    end_logits2[~text_areas] = torch.finfo(torch.float32).min
                    loss = criterion(start_logits, end_logits, start_idx, end_idx, text_areas, match_sent_logits, match_sent) / grad_accum_steps
                    loss2 = criterion(start_logits2, end_logits2, start_idx, end_idx, text_areas, match_sent_logits, match_sent) / grad_accum_steps
                    loss = (loss + loss2) * 0.5
                
                # update
                if phase == 'train':
                    loss.backward()
                    if (batch_idx + 1) % grad_accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    # warmup
                    if warmup_scheduler is not None:
                        warmup_scheduler.step()

                # store loss
                epoch_loss += loss.item() * grad_accum_steps
                    
                # calc score
                start_idx = start_idx.cpu().detach().numpy()
                end_idx = end_idx.cpu().detach().numpy()
                start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
                
                n_high_start += np.sum(np.argmax(start_logits, axis=1)>np.argmax(end_logits, axis=1))
                n_all += len(start_idx)

                for i in range(len(ids)):                        
                    jaccard_score = evut.compute_jaccard_score(
                        tweet[i],
                        start_idx[i],
                        end_idx[i],
                        start_logits[i], 
                        end_logits[i], 
                        offsets[i])
                    epoch_jaccard += jaccard_score

        # summary
        epoch_loss = epoch_loss / len(dataloaders_dict[phase])
        epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)
            
        print('{:^5} | Loss: {:.4f} | Jaccard: {:.4f}'.format(
            phase, epoch_loss, epoch_jaccard))
        print('n_high_start / n_all : {0} / {1}'.format(n_high_start, n_all))

        if phase == 'train':
            tr_loss = epoch_loss
            tr_score = epoch_jaccard
        else:
            vl_loss = epoch_loss
            vl_score = epoch_jaccard

    if not only_val:
        return tr_loss, tr_score, vl_loss, vl_score
    else:
        return vl_score, vl_score

def train_model(model, dataloaders_dict, criterion, optimizer, 
                 num_epochs, grad_accum_steps, 
                 warmup_epoch, step_scheduler, 
                 filename_head='', fold=0, only_val=False, remove_pad=True, save_best_cp=False):

    # warmup_scheduler
    if warmup_epoch > 0:
        warmup_scheduler = WarmUpLR(optimizer, len(dataloaders_dict['train']) * warmup_epoch)
    
    # train
    loglist = []
    best_score = None
    for epoch in range(num_epochs):
        # scheduler
        if epoch > warmup_epoch - 1:
            warm_sch = None
            if step_scheduler is not None:
                step_scheduler.step()
        else:
            warm_sch = warmup_scheduler

        print('\nepoch ', epoch)
        if optimizer is not None:
            for gr, param_group in enumerate(optimizer.param_groups):
                print('lr :', param_group['lr'])
                if gr == 0:
                    now_lr = param_group['lr']

        log = trainer(model, dataloaders_dict, criterion, optimizer, grad_accum_steps, warm_sch, only_val, remove_pad)

        return_score = None
        if not only_val:
            # save checkpoint
            if save_best_cp:
                if best_score is None or log[-1] > best_score:
                    best_score = log[-1]
                    return_score = log[-1]
                    #save_checkpoint(epoch, model, optimizer, step_scheduler, cp_filename)
                    save_checkpoint(None, model, None, None, filename_head+'checkpoint_fold'+str(fold)+'.pth')
                    print('save checkpoint')
            else:
                return_score = log[-1]
                save_checkpoint(None, model, None, None, filename_head+'checkpoint_fold'+str(fold)+'.pth')
                print('save checkpoint')

            # save log
            loglist.append([epoch] + [now_lr] + list(log))
            colmuns = ['epoch', 'lr', 'tr_loss', 'tr_score', 'vl_loss', 'vl_score']
            save_log(loglist, colmuns, filename_head + 'training_log_fold'+str(fold)+'.csv')
        else:
            return_score = log[-1]
            # save log
            loglist.append([epoch] + [now_lr] + list(log))
            colmuns = ['epoch', 'lr', 'vl_loss', 'vl_score']
            save_log(loglist, colmuns, filename_head + 'val_log_fold'+str(fold)+'.csv')

    return model, return_score

