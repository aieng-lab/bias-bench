import transformers



class InstructTokenizerWrapper:

    system_prompt = ""


    def __init__(self, tokenizer, user_prompt_header="user", assistant_prompt_header="assistant"):
        self.tokenizer = tokenizer
        self.user_prompt_header = user_prompt_header
        self.assistant_prompt_header = assistant_prompt_header

        # You can change these markers depending on the model
        self.BEGIN = "<|begin_of_text|>"
        self.START = "<|start_header_id|>"
        self.END = "<|end_header_id|>"
        self.EOT = "<|eot_id|>"

    def _wrap_prompt(self, user_text):
        if isinstance(user_text, str):
            user_texts = [user_text]
        elif isinstance(user_text, list):
            user_texts = user_text
        else:
            raise TypeError("user_text must be a string or a list of strings")

        prompts = []

        for text in user_texts:
            parts = [self.BEGIN]

            parts.append(f"{self.START}system{self.END}\n{self.system_prompt}\n{self.EOT}")

            # Add user prompt
            parts.append(f"{self.START}{self.user_prompt_header}{self.END}\n{text}\n{self.EOT}")

            # Indicate the assistant is expected to reply
            parts.append(f"{self.START}{self.assistant_prompt_header}{self.END}")

            prompts.append(''.join(parts))

        return prompts if len(prompts) > 1 else prompts[0]

    def __call__(self, text, **kwargs):
        """
        Fully mimic Hugging Face tokenizer call: return dict with 'input_ids', 'attention_mask', etc.
        """
        if 'add_special_tokens' not in kwargs or kwargs['add_special_tokens'] is True:
            text = self._wrap_prompt(text)
        return self.tokenizer(text, **kwargs)

    def tokenize(self, text, **kwargs):
        wrapped = self._wrap_prompt(text)
        return self.tokenizer.tokenize(wrapped, **kwargs)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def __getattr__(self, name):
        # Fallback to base tokenizer for any other attributes/methods
        return getattr(self.tokenizer, name)

    def __setattr__(self, key, value):
        if key in ['tokenizer', 'system_prompt']:
            super().__setattr__(key, value)
        else:
            setattr(self.tokenizer, key, value)

    @classmethod
    def from_pretrained(cls, name_or_path, **kwargs):
        tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path, **kwargs)
        return cls(tokenizer, **kwargs)


def load_tokenizer(name_or_path):
    tokenizer = None

    if 'roberta-large' in name_or_path:
        name_or_path = 'roberta-large'
    elif 'gpt2' in name_or_path:
        name_or_path = 'gpt2'
    elif 'llama' in name_or_path.lower():
        if 'instruct' in name_or_path.lower():
            # a special tokenizer is used that automatically adds a system prompt
            tokenizer = InstructTokenizerWrapper.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')

        name_or_path = 'meta-llama/Llama-3.2-3B'

    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer