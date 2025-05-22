import os
import pickle
from functools import partial

import torch.nn as nn
import torch
import transformers

from bias_bench.debias.self_debias.modeling import MaskedLMWrapper, GPT2Wrapper, LlamaWrapper

HF_TOKEN = os.getenv('HF_TOKEN')

class BertModel:
    def __new__(self, model_name_or_path):
        return transformers.BertModel.from_pretrained(model_name_or_path)


class AlbertModel:
    def __new__(self, model_name_or_path):
        return transformers.AlbertModel.from_pretrained(model_name_or_path)

class RobertaModel:
    def __new__(self, model_name_or_path):
        return transformers.RobertaModel.from_pretrained(model_name_or_path)

class DistilbertModel:
    def __new__(self, model_name_or_path):
        return transformers.DistilBertModel.from_pretrained(model_name_or_path)

class DebertaModel:
    def __new__(self, model_name_or_path):
        return transformers.DebertaV2Model.from_pretrained(model_name_or_path)

class AutoModel:
    def __new__(self, model_name_or_path):
        return transformers.AutoModel.from_pretrained(model_name_or_path)


class GPT2Model:
    def __new__(self, model_name_or_path):
        return transformers.GPT2Model.from_pretrained(model_name_or_path)

class LlamaModel:
    def __new__(self, model_name_or_path, **kwargs):
        return transformers.AutoModel.from_pretrained(model_name_or_path, token=HF_TOKEN)

LlamaInstructModel = LlamaModel

class BertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.BertForMaskedLM.from_pretrained(model_name_or_path)

BertLargeForMaskedLM = BertForMaskedLM


class AlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)


class RobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)

class DebertaForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.DebertaV2ForMaskedLM.from_pretrained(model_name_or_path)

class DistilbertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.DistilBertForMaskedLM.from_pretrained(model_name_or_path)

class GPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        return transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)

class LlamaForCausalLM:
    def __new__(self, model_name_or_path, **kwargs):
        return transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, token=HF_TOKEN)

LlamaInstructForCausalLM = LlamaForCausalLM

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


class SentenceDebiasBertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model
SentenceDebiasBertLargeModel = SentenceDebiasBertModel

class SentenceDebiasAlbertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model

class SentenceDebiasDebertaModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.DebertaV2Model.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model

class SentenceDebiasDistilbertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.DistilBertModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2Model(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        return model

class SentenceDebiasLlamaModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AutoModel.from_pretrained(model_name_or_path, token=HF_TOKEN)




        model.register_forward_hook(self.func)
        return model

SentenceDebiasLlamaInstructModel = SentenceDebiasLlamaModel

class SentenceDebiasBertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model
SentenceDebiasBertLargeForMaskedLM = SentenceDebiasBertForMaskedLM


class SentenceDebiasAlbertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model

class SentenceDebiasDebertaForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.DebertaV2ForMaskedLM.from_pretrained(model_name_or_path)
        model.base_model.register_forward_hook(self.func)
        return model

class SentenceDebiasDistilbertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.DistilBertForMaskedLM.from_pretrained(model_name_or_path)
        model.base_model.transformer.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2LMHeadModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model

class SentenceDebiasLlamaForCausalLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, token=HF_TOKEN)
        model.model.register_forward_hook(self.func)

        return model

SentenceDebiasLlamaInstructForCausalLM = SentenceDebiasLlamaForCausalLM

class INLPBertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model
INLPBertLargeModel = INLPBertModel

class INLPAlbertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model

class INLPDebertaModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.DebertaV2Model.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model

class INLPDistilbertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.DistilBertModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model

class INLPGPT2Model(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        return model

class INLPLlamaModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AutoModel.from_pretrained(model_name_or_path, token=HF_TOKEN)
        model.register_forward_hook(self.func)
        return model

INLPLlamaInstructModel = INLPLlamaModel



class INLPBertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.base_model.register_forward_hook(self.func)
        return model
INLPBertLargeForMaskedLM = INLPBertForMaskedLM

class INLPAlbertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class INLPRobertaForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model

class INLPDebertaForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.DebertaV2ForMaskedLM.from_pretrained(model_name_or_path)
        model.base_model.register_forward_hook(self.func)
        return model

class INLPDistilbertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.DistilBertForMaskedLM.from_pretrained(model_name_or_path)
        model.base_model.transformer.register_forward_hook(self.func)
        return model


class INLPGPT2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model

class INLPLlamaForCausalLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, token=HF_TOKEN)
        model.model.register_forward_hook(self.func)
        return model

INLPLlamaInstructForCausalLM = INLPLlamaForCausalLM

class CDABertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model
CDABertLargeModel = CDABertModel

class CDAAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class CDARobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model

class CDDebertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.DebertaV2Model.from_pretrained(model_name_or_path)
        return model

class CDADistilbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.DistilBertModel.from_pretrained(model_name_or_path)
        return model


class CDAGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model

class CDALlamaModel:
    def __new__(self, model_name_or_path):
        model = transformers.AutoModel.from_pretrained(model_name_or_path, token=HF_TOKEN)
        return model

CDALlamaInstructModel = CDALlamaModel

class CDABertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model
CDABertLargeForMaskedLM = CDABertForMaskedLM

class CDAAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDARobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model

class CDADebertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.DebertaV2ForMaskedLM.from_pretrained(model_name_or_path)
        return model

class CDADistilbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.DistilBertForMaskedLM.from_pretrained(model_name_or_path)
        return model

class CDAGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        return model

class CDALlamaForCausalLM:
    def __new__(self, model_name_or_path):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, token=HF_TOKEN)
        return model

CDALlamaInstructForCausalLM = CDALlamaForCausalLM

class DropoutBertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model
DropoutBertLargeModel = DropoutBertModel

class DropoutAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model

class DropoutDebertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.DebertaV2Model.from_pretrained(model_name_or_path)
        return model

class DropoutDistilbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.DistilBertModel.from_pretrained(model_name_or_path)
        return model

class DropoutGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model

class DropoutLlamaModel:
    def __new__(self, model_name_or_path):
        model = transformers.AutoModel.from_pretrained(model_name_or_path, token=HF_TOKEN)
        return model

DropoutLlamaInstructModel = DropoutLlamaModel

class DropoutBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model
DropoutBertLargeForMaskedLM = DropoutBertForMaskedLM

class DropoutAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model

class DropoutDebertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.DebertaV2ForMaskedLM.from_pretrained(model_name_or_path)
        return model

class DropoutDistilbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.DistilBertForMaskedLM.from_pretrained(model_name_or_path)
        return model

class DropoutGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        return model

class DropoutLlamaForCausalLM:
    def __new__(self, model_name_or_path):
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, token=HF_TOKEN)
        return model

DropoutLlamaInstructForCausalLM = DropoutLlamaForCausalLM

class BertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model
BertLargeForSequenceClassification = BertForSequenceClassification

class AlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class RobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model

class DebertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.DebertaV2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model

class DistilbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.DistilBertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class GPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


LlamaForSequenceClassification = transformers.LlamaForSequenceClassification

class SentenceDebiasBertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model
SentenceDebiasBertLargeForSequenceClassification = SentenceDebiasBertForSequenceClassification

class SentenceDebiasAlbertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model

class SentenceDebiasDebertaForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.DebertaV2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.base_model.encoder.register_forward_hook(self.func)
        return model

class SentenceDebiasDistilbertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.DistilBertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.base_model.transformer.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2ForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model

class SentenceDebiasLlamaForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = LlamaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config, token=HF_TOKEN
        )
        model.llama.register_forward_hook(self.func)
        return model

SentenceDebiasLlamaInstructForSequenceClassification = SentenceDebiasLlamaForSequenceClassification

class SentenceDebiasLlamaForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = LlamaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config, token=HF_TOKEN
        )
        model.llama.register_forward_hook(self.func)
        return model

SentenceDebiasLlamaInstructForSequenceClassification = SentenceDebiasLlamaForSequenceClassification

class INLPBertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model
INLPBertLargeForSequenceClassification = INLPBertForSequenceClassification

class INLPAlbertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model

class INLPDebertaForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.DebertaV2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.base_model.encoder.register_forward_hook(self.func)
        return model

class INLPDistilbertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.DistilBertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.base_model.transformer.register_forward_hook(self.func)
        return model


class INLPGPT2ForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model

class INLPLlamaForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = LlamaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config, token=HF_TOKEN
        )
        model.llama.register_forward_hook(self.func)
        return model

INLPLlamaInstructForSequenceClassification = INLPLlamaForSequenceClassification

class CDABertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model
CDABertLargeForSequenceClassification = CDABertForSequenceClassification

class CDAAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDARobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model

class CDADebertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.DebertaV2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model

class CDADistilbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.DistilBertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDAGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model

class CDALlamaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=config, token=HF_TOKEN
        )
        return model

CDALlamaInstructForSequenceClassification = CDALlamaForSequenceClassification

class DropoutBertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model
DropoutBertLargeForSequenceClassification = DropoutBertForSequenceClassification

class DropoutAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutRobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model

class DropoutDebertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.DebertaV2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model

class DropoutDistilbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.DistilBertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model

class DropoutLlamaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=config, token=HF_TOKEN
        )
        return model

DropoutLlamaInstructForSequenceClassification = DropoutLlamaForSequenceClassification

class SelfDebiasBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model
SelfDebiasBertLargeForMaskedLM = SelfDebiasBertForMaskedLM

class SelfDebiasAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasDebertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model

class SelfDebiasDistilbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model

class SelfDebiasGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = GPT2Wrapper(model_name_or_path, use_cuda=False)
        return model

class SelfDebiasLlamaForCausalLM:
    def __new__(self, model_name_or_path):
        model = LlamaWrapper(model_name_or_path, token=HF_TOKEN)
        return model

SelfDebiasLlamaInstructForCausalLM = SelfDebiasLlamaForCausalLM

class MovementPruningBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model
    
class MovementPruningRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model

class MovementPruningDistilbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model