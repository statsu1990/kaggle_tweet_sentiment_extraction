from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import BertModel, BertConfig, BertTokenizer

def download_robert_base():
    file = '../input/roberta-base'

    config = RobertaConfig.from_pretrained('roberta-base')
    config.save_pretrained(file)
    
    model = RobertaModel.from_pretrained('roberta-base')
    model.save_pretrained(file)

    tkn = RobertaTokenizer.from_pretrained('roberta-base')
    tkn.save_pretrained(file)

def download_bert_base():
    file = '../input/bert-base-uncased'

    config = BertConfig.from_pretrained('bert-base-uncased')
    config.save_pretrained(file)
    
    model = BertModel.from_pretrained('bert-base-uncased')
    model.save_pretrained(file)

    tkn = BertTokenizer.from_pretrained('bert-base-uncased')
    tkn.save_pretrained(file)