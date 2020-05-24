import pandas as pd
import numpy as np

SAMP_SUB_FILE = '../input/tweet-sentiment-extraction/sample_submission.csv'

def make_submission(predictions, filename_head):
    sub_df = pd.read_csv(SAMP_SUB_FILE)
    sub_df['selected_text'] = predictions
    sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
    sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)
    sub_df.to_csv(filename_head + 'submission.csv', index=False)
    return

def neutral_pred_to_text(pred_selected_text, text, sentiments):
    neutral_idxs = sentiments.values=='neutral'
    conv_preds = np.array(pred_selected_text)
    conv_preds[neutral_idxs] = text.values[neutral_idxs]
    return conv_preds.tolist()
