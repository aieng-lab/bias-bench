def _is_generative(model):
    # Checks if we are running an autoregressive model.
    return model in [
        "GPT2LMHeadModel",
        "SentenceDebiasGPT2LMHeadModel",
        "INLPGPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel",
        "SelfDebiasGPT2LMHeadModel",
    ] or 'llama' in model.lower()


def _is_self_debias(model):
    # Checks if we are running a Self-Debias model.
    return 'SelfDebias' in model or 'self-debias' in model.lower() or 'self_debias' in model.lower()
