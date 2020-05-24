import os
import numpy as np
import pandas as pd

ORIGIN_DATA_DIR = '../input/tweet-sentiment-extraction'

def get_original_data(is_train=True):
    """
    Returns:
        columns = [textID	text	selected_text	sentiment]
    """
    if is_train:
        file = 'train.csv'
    else:
        file = 'test.csv'
    file = os.path.join(ORIGIN_DATA_DIR, file)

    df = pd.read_csv(file)
    df['text'] = df['text'].astype(str)
    if is_train:
        df['selected_text'] = df['selected_text'].astype(str)

    print('data length : {0}'.format(len(df)))
    return df



