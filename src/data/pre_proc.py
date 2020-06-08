import pandas as pd

class PreprocSelectedText:
    def __init__(self, proc_func):
        self.proc_func = proc_func

    def __call__(self, tr_df):
        new_slc_txt = tr_df.apply(lambda x: self.proc_func(x['text'], x['selected_text']) , axis=1)
        tr_df.loc[:,'selected_text'] = new_slc_txt.values

        return tr_df
