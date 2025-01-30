from .experiment_id import generate_experiment_id
from .util import _is_generative
from .util import _is_self_debias

from transformers import TrainingArguments


# Subclass TrainingArguments without saving any checkpoints
class CustomTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        kwargs['save_total_limit'] = 0
        kwargs['save_strategy'] = 'no'
        super().__init__(*args, **kwargs)