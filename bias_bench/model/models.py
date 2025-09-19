import os
from functools import partial

import torch
import transformers
from torch import nn
from transformers import PreTrainedModel, AutoModelForMultipleChoice, AutoConfig, GPT2PreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput, ModelOutput

from bias_bench.debias.self_debias.modeling import MaskedLMWrapper, GPT2Wrapper, LlamaWrapper

HF_TOKEN = os.getenv('HF_TOKEN')


BASE_MODELS = [
    'Bert',
    'Roberta',
    'Distilbert',
    'GPT2',
    'Llama',
]

def is_encoder(model):
    return model in {'Bert', 'Roberta', 'Distilbert'}

ENCODER_SUFFIXES = [
    'Model',
    'ForMaskedLM',
    'ForSequenceClassification',
    'ForMultipleChoice',
    'ForRecord',
]

DECODER_SUFFIXES = [
    'Model',
    'ForCausalLM', # forLlama
    'LMHeadModel', # for GPT2
    'ForSequenceClassification',
    'ForMultipleChoice',
    'ForRecord',
]


VARIANTS = [
    '',
    'SentenceDebias',
    'INLP',
    'CDA',
    'Dropout',
    'SelfDebias',
]

DUPLICATORS = {
    'Bert': 'BertLarge',
    'Llama': 'LlamaInstruct',
}

SPECIALS = {
    "Llama": {"token": "HF_TOKEN"}  # Llama may need extra kwargs
}





class _SentenceDebiasModel:
    def __init__(self, model_name_or_path, bias_direction):
        def _hook(module, input_, output, bias_direction):
            # Debias the last hidden state.
            x = output["last_hidden_state"]

            # Ensure that everything is on the same device.
            bias_direction = bias_direction.to(x.device)

            # Debias the representation.
            for t in range(x.size(1)):
                x[:, t] = x[:, t] - torch.ger(
                    torch.matmul(x[:, t], bias_direction), bias_direction
                ) / bias_direction.dot(bias_direction)

            # Update the output.
            output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, bias_direction=bias_direction)


class _SentenceDebiasLlamaModel:
    def __init__(self, model_name_or_path, bias_direction):
        def _hook(module, input_, output, bias_direction):
            # Extract logits (shape: [batch_size, sequence_length, vocab_size])
            x = output.hidden_states[-1]

            # Ensure that everything is on the same device.
            bias_direction = bias_direction.to(x.device)

            # Debias the representation.
            for t in range(x.size(1)):
                x[:, t] = x[:, t] - torch.ger(
                    torch.matmul(x[:, t], bias_direction), bias_direction
                ) / bias_direction.dot(bias_direction)

            # Update the output.
            output["last_hidden_state"] = x
            return output


        self.func = partial(_hook, bias_direction=bias_direction)


class _INLPModel:
    def __init__(self, model_name_or_path, projection_matrix):
        def _hook(module, input_, output, projection_matrix):
            # Debias the last hidden state.
            x = output["last_hidden_state"]

            # Ensure that everything is on the same device.
            projection_matrix = projection_matrix.to(x.device)

            for t in range(x.size(1)):
                x[:, t] = torch.matmul(projection_matrix, x[:, t].T).T

            # Update the output.
            output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, projection_matrix=projection_matrix)



class _GPT2ForMultipleChoice(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        from transformers import GPT2Model # GPT2Model needs to imported locally here to avoid name clash with custom GPT2Model!
        self.gpt2 = GPT2Model(config)
        self.dropout = nn.Dropout(config.summary_first_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    @property
    def transformer(self):
        return self.gpt2

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Custom from_pretrained that loads encoder weights into self.gpt2.
        """
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Create instance
        model = cls(config)

        # Load pretrained encoder weights
        from transformers import GPT2Model
        gpt2_model = GPT2Model.from_pretrained(pretrained_model_name_or_path, config=config)
        model.gpt2 = gpt2_model

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        # input_ids shape: (batch_size, num_choices, seq_len)
        batch_size, num_choices, seq_len = input_ids.shape

        # Flatten input for GPT-2
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len) if attention_mask is not None else None

        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        # Take last hidden state of the last token
        pooled_output = outputs.last_hidden_state[:, -1, :]  # shape: (batch_size*num_choices, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # shape: (batch_size*num_choices, 1)
        logits = logits.view(batch_size, num_choices)  # shape: (batch_size, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM, BertPreTrainedModel


class EntityClassifierModel(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        #if config is None:
        #    config = AutoConfig.from_pretrained(name)
        super().__init__(config)

        self.model_type = config.model_type
        #self.bert = AutoModel.from_pretrained(name, config=config)
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            entity_spans=None,  # list of (start, end) per entity per example
            labels=None,
    ):
        # Only pass token_type_ids if supported
        token_type_ids_to_pass = token_type_ids if self.model_type in ["bert", "xlnet"] else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids_to_pass,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)

        batch_size = sequence_output.size(0)
        logits = []

        # Iterate over batch → gather entity representations
        for i in range(batch_size):
            spans = entity_spans[i]  # list of (start, end) for this example
            if len(spans) == 0:
                logits.append(torch.empty(0, device=sequence_output.device))
                continue

            ent_reps = []
            for (s, e) in spans:
                ent_reps.append(sequence_output[i, s:e].mean(dim=0))
            ent_reps = torch.stack(ent_reps, dim=0)
            ent_logits = self.classifier(self.dropout(ent_reps)).squeeze(-1)
            logits.append(ent_logits)

        # Pad logits to rectangular tensor
        max_ents = max(l.size(0) for l in logits)
        padded_logits = torch.full(
            (batch_size, max_ents), -1e9, device=sequence_output.device
        )
        for i, l in enumerate(logits):
            padded_logits[i, : l.size(0)] = l
        logits = padded_logits

        # Compute loss
        loss = None
        if labels is not None:
            # Pad labels to match logits
            max_ents = logits.size(1)
            padded_labels = torch.full(
                (batch_size, max_ents), -100, device=labels.device
            )
            for i, l in enumerate(labels):
                padded_labels[i, : l.size(0)] = l[:max_ents]  # truncate if longer

            loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=logits.device))
            active_mask = padded_labels != -100
            if active_mask.any():
                loss = loss_fct(
                    logits[active_mask].view(-1),
                    padded_labels[active_mask].float().view(-1),
                )

        return {"loss": loss, "logits": logits}


import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel

class ReCoRDModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.1)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
        self.model_name = config._name_or_path if hasattr(config, '_name_or_path') else 'unknown'

    @property
    def transformer(self):
        return self.encoder

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Custom from_pretrained that loads encoder weights into self.encoder.
        """
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Create instance
        model = cls(config)

        # Load pretrained encoder weights
        encoder_model = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config)
        model.encoder = encoder_model

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_spans=None,  # list of (start, end) per entity per example
        labels=None,
    ):
        if entity_spans is None:
            raise ValueError("entity_spans must be provided for ReCoRDModel")

        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Only BERT actually needs token_type_ids
        if "bert" in self.model_name and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        # GPT-2 doesn't support attention_mask in the same way (causal LM)
        if "gpt2" in self.model_name:
            kwargs.pop("attention_mask", None)

        outputs = self.encoder(**kwargs, return_dict=True)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)

        batch_size = sequence_output.size(0)
        logits = []
        loss = None

        # Iterate over batch → gather entity representations
        for i in range(batch_size):
            spans = entity_spans[i]  # list of (start, end) for this example
            if len(spans) == 0:
                logits.append(torch.empty(0, device=sequence_output.device))
                continue

            ent_reps = []
            for (s, e) in spans:
                ent_reps.append(sequence_output[i, s:e].mean(dim=0))
            ent_reps = torch.stack(ent_reps, dim=0)  # (num_entities, hidden)

            ent_logits = self.classifier(self.dropout(ent_reps)).squeeze(-1)
            logits.append(ent_logits)

        # Pad logits to rectangular tensor
        max_ents = max(l.size(0) for l in logits)
        padded_logits = torch.full(
            (batch_size, max_ents), -1e9, device=sequence_output.device
        )
        for i, l in enumerate(logits):
            padded_logits[i, : l.size(0)] = l
        logits = padded_logits  # (batch, max_entities)

        # Compute loss
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=sequence_output.device))
            active_mask = labels != -100
            if active_mask.any():
                loss = loss_fct(
                    logits[active_mask].view(-1),
                    labels[active_mask].float().view(-1)
                )

        return {"loss": loss, "logits": logits}


def make_model_class(base, suffix, variant):
    cls_name = f"{variant}{base}{suffix}"

    super_cls = tuple()
    if variant == 'SentenceDebias':
        if base == 'Llama':
            super_cls = (_SentenceDebiasLlamaModel, )
        else:
            super_cls = (_SentenceDebiasModel, )
    elif variant == 'INLP':
        super_cls = (_INLPModel, )

    if base == 'Distilbert':
        model_id = f'DistilBert{suffix}'
    else:
        model_id = f"{base}{suffix}"

    if base == 'GPT2' and suffix == 'ForMultipleChoice':
        tf_cls = _GPT2ForMultipleChoice
    elif suffix == 'ForRecord':
        tf_cls = ReCoRDModel
    else:
        tf_cls = getattr(transformers, model_id)

    def register_forward_hook(model, hook):
        if suffix:
            if hasattr(model, 'base_model'):
                model = model.base_model
            elif hasattr(model, 'transformer'):
                model = model.transformer
            elif hasattr(model, 'model'):
                model = model.model
            else:
                raise NotImplementedError(f"Model {model_id} does not have a base model or transformer attribute")

        if base in ['Bert', 'Roberta']:
            model.encoder.register_forward_hook(hook)
        elif base in ['Distilbert']:
            model.transformer.register_forward_hook(hook)
        elif base in ['GPT2', 'Llama']:
            if hasattr(model, 'transformer'):
                model.transformer.register_forward_hook(hook)
            else:
                model.register_forward_hook(hook)
        else:
            raise NotImplementedError(f"Forward hook not implemented for {base}")


    if variant == 'SentenceDebias':
        def __new__(self, model_name_or_path, bias_direction, **kwargs):
            super_cls[0].__init__(self, model_name_or_path, bias_direction)
            if 'llama' in model_name_or_path.lower():
                kwargs['output_hidden_states'] = True
            model = tf_cls.from_pretrained(model_name_or_path, **kwargs)
            register_forward_hook(model, self.func)
            return model
    elif variant == 'INLP':
        def __new__(self, model_name_or_path, projection_matrix, **kwargs):
            super_cls[0].__init__(self, model_name_or_path, projection_matrix)
            model = tf_cls.from_pretrained(model_name_or_path, **kwargs)
            register_forward_hook(model, self.func)
            return model

    elif variant == 'SelfDebias':
        if base == 'GPT2':
            wrapper_cls = GPT2Wrapper
        elif base == 'Llama':
            wrapper_cls = LlamaWrapper
        else:
            wrapper_cls = MaskedLMWrapper

        def __new__(self, model_name_or_path, **kwargs):
            model = wrapper_cls(model_name_or_path, **kwargs)
            return model
    else:
        def __new__(cls, model_name_or_path, **kwargs):
            # Inject any special kwargs
            if base in SPECIALS:
                kwargs.update(SPECIALS[base])
            return tf_cls.from_pretrained(model_name_or_path, **kwargs)

    return type(cls_name, super_cls, {"__new__": __new__})


# Dynamically create all classes
for base in BASE_MODELS:

    if is_encoder(base):
        SUFFIXES = ENCODER_SUFFIXES
    else:
        SUFFIXES = DECODER_SUFFIXES

    for suffix in SUFFIXES:

        for variant in VARIANTS:
            if variant == 'SelfDebias' and suffix not in ['ForMaskedLM', 'ForCausalLM', 'LMHeadModel']:
                continue

            try:
                cls = make_model_class(base, suffix, variant)

                names = [cls.__name__]
                if base in DUPLICATORS:
                    duplicated_name = names[0].replace(base, DUPLICATORS[base])
                    names.append(duplicated_name)

                for name in names:
                    if name not in globals():
                        # Register the class in the global namespace
                        globals()[name] = cls
                        #print(f"Registered {name} in the global namespace.")
                    else:
                        print(f"Skipping {name} as it already exists in the global namespace.")
            except AttributeError as e:
                # Skip if the transformers class doesn't exist (some suffixes don't exist for some models)
                continue


