import numpy as np
from prediction import post_proc

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
    return float(len(c)) / (len(a) + len(b) - len(c) + 1e-5)

def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred, end_pred = calc_start_end_index_v1(start_logits, end_logits)
    pred = get_selected_text(text, start_pred, end_pred, offsets)
    true = get_selected_text(text, start_idx, end_idx, offsets)
    
    #pred = post_proc.postproc_selected_text_v7(text, pred)
    #pred = post_proc.postproc_selected_text_v12(text, pred)
    #pred = post_proc.postproc_selected_text_v13(text, pred)
    #pred = post_proc.postproc_selected_text_v14(text, pred)

    score = jaccard(true, pred)

    #with open('../consideration/pred_text/pred_text.csv', mode='a', encoding='utf_8') as f:
    #    f.write('"'+text+'"'+','+'"'+true+'"'+','+'"'+pred+'"'+','+str(score)+"\n")

    return score

def calc_start_end_index_v1(start_probs, end_probs):
    start_pred = np.argmax(start_probs)
    end_pred = np.argmax(end_probs)

    if start_pred > end_pred:
        start_pred = 0
        end_pred = len(end_probs) - 1

    return start_pred, end_pred

def calc_start_end_index_v2(start_probs, end_probs):
    comb_mat = start_probs[:,None] + end_probs
    #comb_mat = start_probs[:,None] * end_probs
    comb_mat = np.triu(comb_mat, k=1)
    start_pred, end_pred = np.unravel_index(np.argmax(comb_mat), comb_mat.shape)

    if start_pred > end_pred:
        start_pred = 0
        end_pred = len(end_probs) - 1

    return start_pred, end_pred

def calc_start_end_index_v3(start_probs, end_probs):
    indexes = np.arange(len(start_probs))
    
    start_pred = np.round(np.sum(start_probs * indexes)).astype('int')
    end_pred = np.round(np.sum(end_probs * indexes)).astype('int')

    if start_pred > end_pred:
        start_pred = 0
        end_pred = len(end_probs) - 1

    return start_pred, end_pred

def calc_start_end_index_v4(start_probs, end_probs):
    """
    calc_start_end_index_v1 : 0.558128
    calc_start_end_index_v4 : 0.558228
    """
    start_pred = np.argsort(start_probs)[::-1][0]
    end_pred = np.argsort(end_probs)[::-1][0]

    start_pred2 = np.argsort(start_probs)[::-1][1]
    end_pred2 = np.argsort(end_probs)[::-1][1]

    if start_pred > end_pred:
        start_pred = min([start_pred, start_pred2])
        end_pred = max([end_pred, end_pred2])
        #start_pred = start_pred2
        #end_pred = end_pred2

    if start_pred > end_pred:
        print(start_pred, end_pred)
        start_pred = 0
        end_pred = len(end_probs) - 1

    return start_pred, end_pred

def calc_start_end_index_v5(start_probs, end_probs):
    start_pred = np.argsort(start_probs)[::-1][0]
    end_pred = np.argsort(end_probs)[::-1][0]

    start_pred2 = np.argsort(start_probs)[::-1][1]
    end_pred2 = np.argsort(end_probs)[::-1][1]

    if start_probs[start_pred] - start_probs[start_pred2] < 0.01:
        #print(start_pred, start_pred2)
        #start_pred = start_pred2
        start_pred = min([start_pred, start_pred2])
        #start_pred = int((start_pred + start_pred2)*0.5)
    if end_probs[end_pred] - end_probs[end_pred2] < 0.01:
        #end_pred = end_pred2
        end_pred = max([end_pred, end_pred2])
        #end_pred = int((end_pred + end_pred2)*0.5)

    if start_pred > end_pred:
        start_pred = 0
        end_pred = len(end_probs) - 1

    return start_pred, end_pred

def calc_start_end_index_v6(start_probs, end_probs):
    start_pred = np.argsort(start_probs)[::-1][0]
    end_pred = np.argsort(end_probs)[::-1][0]

    start_pred2 = np.argsort(start_probs)[::-1][1]
    end_pred2 = np.argsort(end_probs)[::-1][1]

    if start_probs[start_pred] < 0.2:
        print(start_pred, start_pred2)
        #start_pred = start_pred2
        start_pred = min([start_pred, start_pred2])
        #start_pred = int((start_pred + start_pred2)*0.5)
    if end_probs[end_pred] < 0.2:
        #end_pred = end_pred2
        end_pred = max([end_pred, end_pred2])
        #end_pred = int((end_pred + end_pred2)*0.5)

    if start_pred > end_pred:
        start_pred = 0
        end_pred = len(end_probs) - 1

    return start_pred, end_pred

