from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import BertModel, BertConfig, BertTokenizer
from transformers import AlbertModel, AlbertConfig, AlbertTokenizer
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

def download_robert_base():
    file = '../input/roberta-base'

    config = RobertaConfig.from_pretrained('roberta-base')
    config.save_pretrained(file)
    
    model = RobertaModel.from_pretrained('roberta-base')
    model.save_pretrained(file)

    tkn = RobertaTokenizer.from_pretrained('roberta-base')
    tkn.save_pretrained(file)

def download_robert_large():
    file = '../input/roberta-large'

    config = RobertaConfig.from_pretrained('roberta-large')
    config.save_pretrained(file)
    
    model = RobertaModel.from_pretrained('roberta-large')
    model.save_pretrained(file)

    tkn = RobertaTokenizer.from_pretrained('roberta-large')
    tkn.save_pretrained(file)

def download_bert_base():
    file = '../input/bert-base-uncased'

    config = BertConfig.from_pretrained('bert-base-uncased')
    config.save_pretrained(file)
    
    model = BertModel.from_pretrained('bert-base-uncased')
    model.save_pretrained(file)

    tkn = BertTokenizer.from_pretrained('bert-base-uncased')
    tkn.save_pretrained(file)

def download_bert_large():
    file = '../input/bert-large-uncased'

    config = BertConfig.from_pretrained('bert-large-uncased')
    config.save_pretrained(file)
    
    model = BertModel.from_pretrained('bert-large-uncased')
    model.save_pretrained(file)

    tkn = BertTokenizer.from_pretrained('bert-large-uncased')
    tkn.save_pretrained(file)

def download_albert_base():
    file = '../input/albert-base-v2'

    config = AlbertConfig.from_pretrained('albert-base-v2')
    config.save_pretrained(file)
    
    model = AlbertModel.from_pretrained('albert-base-v2')
    model.save_pretrained(file)

    tkn = AlbertTokenizer.from_pretrained('albert-base-v2')
    tkn.save_pretrained(file)

def download_distilbert_base():
    file = '../input/distilbert-base-uncased'

    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    config.save_pretrained(file)
    
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.save_pretrained(file)

    tkn = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tkn.save_pretrained(file)