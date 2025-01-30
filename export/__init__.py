

def _parse_experiment_id(experiment_id):
    model = None
    model_name_or_path = None
    bias_type = None
    seed = None

    items = experiment_id.replace('-gradient-results-changed_models', '').replace('_id', '-id').replace('_^2', '-^2').replace('_^3', '-^3').replace('_^10', '-^10').split("_")[1:]
    for item in items:
        id_, val = item[:1], item[2:]
        if id_ == "m":
            model = val
        elif id_ == "c":
            model_name_or_path = val
        elif id_ == "t":
            bias_type = val
        elif id_ == "s":
            seed = int(val)
        else:
            raise ValueError(f"Unrecognized ID {id_} for {experiment_id}.")

    return model, model_name_or_path, bias_type, seed


def _label_model_type(row):
    if "BertLarge" in row["model"] or 'bert-large' in row['model_name_or_path']:
        # if "BertLarge" is not in row['model']
        if 'BertLarge' not in row['model']:
            row['model'] = row['model'].replace('Bert', 'BertLarge')

        return "bert-large"
    elif "Bert" in row["model"]:
        return "bert"
    elif "Albert" in row["model"]:
        return "albert"
    elif "Roberta" in row["model"]:
        return "roberta"
    elif 'gpt2' in row['model'].lower():
        return "gpt2"
    elif 'distilbert' in row['model'].lower():
        return "distilbert"
    elif 'roberta' in row['model'].lower():
        return "roberta"
    else:
        return row['model']


def _pretty_model_name(row):
    pretty_name_mapping = {
        "BertForMaskedLM": "BERT",
        "SentenceDebiasBertForMaskedLM": r"\, + \textsc{SentenceDebias}",
        "INLPBertForMaskedLM": r"\, + \textsc{INLP}",
        "CDABertForMaskedLM": r"\, + \textsc{CDA}",
        "DropoutBertForMaskedLM": r"\, + \textsc{Dropout}",
        "SelfDebiasBertForMaskedLM": r"\, + \textsc{Self-Debias}",
        "AlbertForMaskedLM": "ALBERT",
        "SentenceDebiasAlbertForMaskedLM": r"\, + \textsc{SentenceDebias}",
        "INLPAlbertForMaskedLM": r"\, + \textsc{INLP}",
        "CDAAlbertForMaskedLM": r"\, + \textsc{CDA}",
        "DropoutAlbertForMaskedLM": r"\, + \textsc{Dropout}",
        "SelfDebiasAlbertForMaskedLM": r"\, + \textsc{Self-Debias}",
        "RobertaForMaskedLM": "RoBERTa",
        "SentenceDebiasRobertaForMaskedLM": r"\, + \textsc{SentenceDebias}",
        "INLPRobertaForMaskedLM": r"\, + \textsc{INLP}",
        "CDARobertaForMaskedLM": r"\, + \textsc{CDA}",
        "DropoutRobertaForMaskedLM": r"\, + \textsc{Dropout}",
        "SelfDebiasRobertaForMaskedLM": r"\, + \textsc{Self-Debias}",
        "GPT2LMHeadModel": "GPT-2",
        "SentenceDebiasGPT2LMHeadModel": r"\, + \textsc{SentenceDebias}",
        "INLPGPT2LMHeadModel": r"\, + \textsc{INLP}",
        "CDAGPT2LMHeadModel": r"\, + \textsc{CDA}",
        "DropoutGPT2LMHeadModel": r"\, + \textsc{Dropout}",
        "SelfDebiasGPT2LMHeadModel": r"\, + \textsc{Self-Debias}",
    }

    name = row['model']
    model_name_or_path = row['model_name_or_path']
    return pretty_name_mapping.get(name, name) + ' - ' + model_name_or_path


def _get_baseline_metric(df, model_type, metric, masked_lm=True):
    model_type_to_baseline = {
        "bert": "BertModel",
        "bert-large": "BertLargeModel",
        "albert": "AlbertModel",
        "roberta": "RobertaModel",
        "distilbert": "DistilbertModel",
        "gpt2": "GPT2LMHeadModel",
    }
    if masked_lm:
        model_type_to_baseline = {k: v.removesuffix('Model') + 'ForMaskedLM' for k, v in model_type_to_baseline.items()}
    #baseline = model_type_to_baseline[model_type]
    #return df[df["model"] == baseline][metric].values[0]
    df_model_type = df['model_type'].values[0]

    return df[(df["model_name_or_path"] == model_type) & (df['model'] == model_type_to_baseline[df_model_type])][metric].values[0]

