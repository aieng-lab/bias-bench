import transformers


# dirty solution for a problem with the Roberta Tokenizer
def load_tokenizer(name_or_path):
    if 'roberta-large' in name_or_path:
        name_or_path = 'roberta-large'
    return transformers.AutoTokenizer.from_pretrained(name_or_path)