import transformers


# dirty solution for a problem with the Roberta Tokenizer
def load_tokenizer(name_or_path):
    if 'roberta-large' in name_or_path:
        name_or_path = 'roberta-large'
    elif 'gpt2' in name_or_path:
        name_or_path = 'gpt2'

    tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer