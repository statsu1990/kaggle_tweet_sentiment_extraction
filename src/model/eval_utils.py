import numpy as np

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
    start_pred, end_pred = calc_start_end_index_v1(start_logits, end_logits)
    pred = get_selected_text(text, start_pred, end_pred, offsets)
        
    true = get_selected_text(text, start_idx, end_idx, offsets)
    
    return jaccard(true, pred)

def calc_start_end_index_v1(start_probs, end_probs):
    start_pred = np.argmax(start_probs)
    end_pred = np.argmax(end_probs)

    if start_pred > end_pred:
        start_pred = 0
        end_pred = len(end_probs) - 1

    return start_pred, end_pred

def calc_start_end_index_v2(start_probs, end_probs):
    #comb_mat = start_probs[:,None] + end_probs
    comb_mat = start_probs[:,None] * end_probs
    comb_mat = np.triu(comb_mat, k=1)
    start_pred, end_pred = np.unravel_index(np.argmax(comb_mat), comb_mat.shape)

    if start_pred > end_pred:
        start_pred = 0
        end_pred = len(end_probs) - 1

    return start_pred, end_pred