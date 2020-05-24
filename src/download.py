from transformers import RobertaModel, RobertaConfig, RobertaTokenizer

def download_robert_base():
    file = '../input/roberta-base'

    config = RobertaConfig.from_pretrained('roberta-base')
    config.save_pretrained(file)
    
    model = RobertaModel.from_pretrained('roberta-base')
    model.save_pretrained(file)

    tkn = RobertaTokenizer.from_pretrained('roberta-base')
    tkn.save_pretrained(file)
