#!/bin/bash
# Variables for running batch jobs.


######################################################
# Adjust the directories to your local setup.
persistent_dir="/srv/data/drechsel/Git/bias-bench" # base dir of bias-bench repo
gradiend_dir="${persistent_dir}/../gradiend" # base dir of gradiend repo
#####################################################

changed_models_dir="${gradiend_dir}/results/changed_models"
eval "$(conda shell.bash hook)"
conda activate bias-bench
export PYTHONPATH="${PYTHONPATH}:${persistent_dir}"

suffix=""

male_model="${changed_models_dir}/bert-base-cased-M${suffix}"
female_model="${gradiend_dir}/results/changed_models/bert-base-cased-F${suffix}"
unbiased_model="${gradiend_dir}/results/changed_models/bert-base-cased-N${suffix}"

male_model_bert_large_cased="${gradiend_dir}/results/changed_models/bert-large-cased-M${suffix}"
female_model_bert_large_cased="${gradiend_dir}/results/changed_models/bert-large-cased-F${suffix}"
unbiased_model_bert_large_cased="${gradiend_dir}/results/changed_models/bert-large-cased-N${suffix}"

male_roberta_model="${gradiend_dir}/results/changed_models/roberta-large-M${suffix}"
female_roberta_model="${gradiend_dir}/results/changed_models/roberta-large-F${suffix}"
unbiased_roberta_model="${gradiend_dir}/results/changed_models/roberta-large-N${suffix}"

male_distilbert_model="${gradiend_dir}/results/changed_models/distilbert-base-cased-M${suffix}"
female_distilbert_model="${gradiend_dir}/results/changed_models/distilbert-base-cased-F${suffix}"
unbiased_distilbert_model="${gradiend_dir}/results/changed_models/distilbert-base-cased-N${suffix}"


debiased_bert_models="${male_model} ${female_model} ${unbiased_model} ${male_model_bert_large_cased} ${female_model_bert_large_cased} ${unbiased_model_bert_large_cased}"
debiased_roberta_models="${male_roberta_model} ${female_roberta_model} ${unbiased_roberta_model}"
debiased_distilbert_models="${male_distilbert_model} ${female_distilbert_model} ${unbiased_distilbert_model}"


model_name_or_paths=(
  "bert-base-cased"
  "bert-large-cased"
  "roberta-large"
  "distilbert-base-cased"
)

declare -A model_name_or_path_to_model=(
    ["bert-base-cased"]="BertModel"
    ["bert-large-cased"]="BertModel"
    ["roberta-large"]="RobertaModel"
    ["distilbert-base-cased"]="DistilbertModel"
)

declare -A model_to_debiased_models=(
    ["BertModel"]="${debiased_bert_models}"
    ["RobertaModel"]="${debiased_roberta_models}"
    ["DistilbertModel"]="${debiased_distilbert_models}"
)

seeds=(0 1 2 3 4 5 6 7 8 9)
seeds=(0 1 2)


bias_types=(
    "gender"
    #"race"
    #"religion"
)

# Baseline models.
models=(
    "BertModel"
    "RobertaModel"
    "DistilbertModel"
)

# Baseline masked language models.
masked_lm_models=(
    "BertForMaskedLM"
    "BertLargeForMaskedLM"
    "DistilbertForMaskedLM"
    "RobertaForMaskedLM"
)

# Baseline causal language models.
causal_lm_models=(
    #"GPT2LMHeadModel"
)

# Debiased masked language models.
sentence_debias_masked_lm_models=(
    "SentenceDebiasBertForMaskedLM"
    "SentenceDebiasBertLargeForMaskedLM"
    "SentenceDebiasDistilbertForMaskedLM"
    "SentenceDebiasRobertaForMaskedLM"
)

inlp_masked_lm_models=(
    "INLPBertForMaskedLM"
    "INLPBertLargeForMaskedLM"
    "INLPRobertaForMaskedLM"
    "INLPDistilbertForMaskedLM"
)

cda_masked_lm_models=(
    "CDABertForMaskedLM"
    "CDABertLargeForMaskedLM"
    "CDARobertaForMaskedLM"
    "CDADistilbertForMaskedLM"
)

dropout_masked_lm_models=(
    "DropoutBertForMaskedLM"
    "DropoutBertLargeForMaskedLM"
    "DropoutRobertaForMaskedLM"
    "DropoutDistilbertForMaskedLM"
)

self_debias_masked_lm_models=(
    "SelfDebiasBertForMaskedLM"
    "SelfDebiasBertLargeForMaskedLM"
    "SelfDebiasRobertaForMaskedLM"
    "SelfDebiasDistilbertForMaskedLM"
)

# Debiased causal language models.
sentence_debias_causal_lm_models=(
    "SentenceDebiasGPT2LMHeadModel"
)

inlp_causal_lm_models=(
    "INLPGPT2LMHeadModel"
)

cda_causal_lm_models=(
    "CDAGPT2LMHeadModel"
)

dropout_causal_lm_models=(
    "DropoutGPT2LMHeadModel"
)

self_debias_causal_lm_models=(
    "SelfDebiasGPT2LMHeadModel"
)

# Debiased base models.
sentence_debias_models=(
    "SentenceDebiasBertModel"
    "SentenceDebiasBertLargeModel"
    "SentenceDebiasRobertaModel"
    "SentenceDebiasDistilbertModel"
)

inlp_models=(
    "INLPBertModel"
    "INLPBertLargeModel"
    "INLPRobertaModel"
    "INLPDistilbertModel"
)

cda_models=(
    "CDABertModel"
    "CDABertLargeModel"
    "CDARobertaModel"
    "CDADistilbertModel"
)

dropout_models=(
    "DropoutBertModel"
    "DropoutBertLargeModel"
    "DropoutRobertaModel"
    "DropoutDistilbertModel"
)


declare -A model_to_model_name_or_path=(
    ["BertModel"]="bert-base-cased"
    ["BertLargeModel"]="bert-large-cased"
    ["AlbertModel"]="albert-base-v2"
    ["RobertaModel"]="roberta-large"
    ["DebertaModel"]="microsoft/deberta-v3-large"
    ["DistilbertModel"]="distilbert-base-cased"
    ["GPT2Model"]="gpt2"
    ["BertForMaskedLM"]="bert-base-cased"
    ["BertLargeForMaskedLM"]="bert-large-cased"
    ["AlbertForMaskedLM"]="albert-base-v2"
    ["RobertaForMaskedLM"]="roberta-large"
    ["DebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["DistilbertForMaskedLM"]="distilbert-base-cased"
    ["GPT2LMHeadModel"]="gpt2"
    ["BertForSequenceClassification"]="bert-base-cased"
    ["BertLargeForSequenceClassification"]="bert-large-cased"
    ["AlbertForSequenceClassification"]="albert-base-v2"
    ["RobertaForSequenceClassification"]="roberta-large"
    ["DebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["DistilbertForSequenceClassification"]="distilbert-base-cased"
    ["GPT2ForSequenceClassification"]="gpt2"
    ["SentenceDebiasBertModel"]="bert-base-cased"
    ["SentenceDebiasBertLargeModel"]="bert-large-cased"
    ["SentenceDebiasAlbertModel"]="albert-base-v2"
    ["SentenceDebiasRobertaModel"]="roberta-large"
    ["SentenceDebiasDebertaModel"]="microsoft/deberta-v3-large"
    ["SentenceDebiasDistilbertModel"]="distilbert-base-cased"
    ["SentenceDebiasGPT2Model"]="gpt2"
    ["SentenceDebiasBertForMaskedLM"]="bert-base-cased"
    ["SentenceDebiasBertLargeForMaskedLM"]="bert-large-cased"
    ["SentenceDebiasAlbertForMaskedLM"]="albert-base-v2"
    ["SentenceDebiasRobertaForMaskedLM"]="roberta-large"
    ["SentenceDebiasDebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["SentenceDebiasDistilbertForMaskedLM"]="distilbert-base-cased"
    ["SentenceDebiasGPT2LMHeadModel"]="gpt2"
    ["SentenceDebiasBertForSequenceClassification"]="bert-base-cased"
    ["SentenceDebiasBertLargeForSequenceClassification"]="bert-large-cased"
    ["SentenceDebiasAlbertForSequenceClassification"]="albert-base-v2"
    ["SentenceDebiasRobertaForSequenceClassification"]="roberta-large"
    ["SentenceDebiasDebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["SentenceDebiasDistilbertForSequenceClassification"]="distilbert-base-cased"
    ["SentenceDebiasGPT2ForSequenceClassification"]="gpt2"
    ["INLPBertModel"]="bert-base-cased"
    ["INLPBertLargeModel"]="bert-large-cased"
    ["INLPAlbertModel"]="albert-base-v2"
    ["INLPRobertaModel"]="roberta-large"
    ["INLPDebertaModel"]="microsoft/deberta-v3-large"
    ["INLPDistilbertModel"]="distilbert-base-cased"
    ["INLPGPT2Model"]="gpt2"
    ["INLPBertForMaskedLM"]="bert-base-cased"
    ["INLPBertLargeForMaskedLM"]="bert-large-cased"
    ["INLPAlbertForMaskedLM"]="albert-base-v2"
    ["INLPRobertaForMaskedLM"]="roberta-large"
    ["INLPDebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["INLPDistilbertForMaskedLM"]="distilbert-base-cased"
    ["INLPGPT2LMHeadModel"]="gpt2"
    ["INLPBertForSequenceClassification"]="bert-base-cased"
    ["INLPBertLargeForSequenceClassification"]="bert-large-cased"
    ["INLPAlbertForSequenceClassification"]="albert-base-v2"
    ["INLPRobertaForSequenceClassification"]="roberta-large"
    ["INLPDebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["INLPDistilbertForSequenceClassification"]="distilbert-base-cased"
    ["INLPGPT2ForSequenceClassification"]="gpt2"
    ["CDABertModel"]="bert-base-cased"
    ["CDABertLargeModel"]="bert-large-cased"
    ["CDAAlbertModel"]="albert-base-v2"
    ["CDARobertaModel"]="roberta-large"
    ["CDADebertaModel"]="microsoft/deberta-v3-large"
    ["CDADistilbertModel"]="distilbert-base-cased"
    ["CDAGPT2Model"]="gpt2"
    ["CDABertForMaskedLM"]="bert-base-cased"
    ["CDABertLargeForMaskedLM"]="bert-large-cased"
    ["CDAAlbertForMaskedLM"]="albert-base-v2"
    ["CDARobertaForMaskedLM"]="roberta-large"
    ["CDADebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["CDADistilbertForMaskedLM"]="distilbert-base-cased"
    ["CDAGPT2LMHeadModel"]="gpt2"
    ["CDABertForSequenceClassification"]="bert-base-cased"
    ["CDABertLargeForSequenceClassification"]="bert-large-cased"
    ["CDAAlbertForSequenceClassification"]="albert-base-v2"
    ["CDARobertaForSequenceClassification"]="roberta-large"
    ["CDADebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["CDADistilbertForSequenceClassification"]="distilbert-base-cased"
    ["CDAGPT2ForSequenceClassification"]="gpt2"
    ["DropoutBertModel"]="bert-base-cased"
    ["DropoutBertLargeModel"]="bert-large-cased"
    ["DropoutAlbertModel"]="albert-base-v2"
    ["DropoutRobertaModel"]="roberta-large"
    ["DropoutDebertaModel"]="microsoft/deberta-v3-large"
    ["DropoutDistilbertModel"]="distilbert-base-cased"
    ["DropoutGPT2Model"]="gpt2"
    ["DropoutBertForMaskedLM"]="bert-base-cased"
    ["DropoutBertLargeForMaskedLM"]="bert-large-cased"
    ["DropoutAlbertForMaskedLM"]="albert-base-v2"
    ["DropoutRobertaForMaskedLM"]="roberta-large"
    ["DropoutDebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["DropoutDistilbertForMaskedLM"]="distilbert-base-cased"
    ["DropoutGPT2LMHeadModel"]="gpt2"
    ["DropoutBertForSequenceClassification"]="bert-base-cased"
    ["DropoutBertLargeForSequenceClassification"]="bert-large-cased"
    ["DropoutAlbertForSequenceClassification"]="albert-base-v2"
    ["DropoutRobertaForSequenceClassification"]="roberta-large"
    ["DropoutDebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["DropoutDistilbertForSequenceClassification"]="distilbert-base-cased"
    ["DropoutGPT2ForSequenceClassification"]="gpt2"
    ["SelfDebiasBertForMaskedLM"]="bert-base-cased"
    ["SelfDebiasBertLargeForMaskedLM"]="bert-large-cased"
    ["SelfDebiasAlbertForMaskedLM"]="albert-base-v2"
    ["SelfDebiasRobertaForMaskedLM"]="roberta-large"
    ["SelfDebiasDebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["SelfDebiasDistilbertForMaskedLM"]="distilbert-base-cased"
    ["SelfDebiasGPT2LMHeadModel"]="gpt2"
)

# For SentenceDebias and INLP, it is useful to have the base model
# that was used to compute the subspace or projection matrix.
# We map als the ...BertLargeModels to BertModel instead of BertLargeModel (which is an alias), since BertModel is used in file paths
declare -A debiased_model_to_base_model=(
    ["SentenceDebiasBertModel"]="BertModel"
    ["SentenceDebiasBertLargeModel"]="BertModel"
    ["SentenceDebiasAlbertModel"]="AlbertModel"
    ["SentenceDebiasRobertaModel"]="RobertaModel"
    ["SentenceDebiasDebertaModel"]="DebertaModel"
    ["SentenceDebiasDistilbertModel"]="DistilbertModel"
    ["SentenceDebiasGPT2Model"]="GPT2Model"
    ["SentenceDebiasBertForMaskedLM"]="BertModel"
    ["SentenceDebiasBertLargeForMaskedLM"]="BertModel"
    ["SentenceDebiasAlbertForMaskedLM"]="AlbertModel"
    ["SentenceDebiasRobertaForMaskedLM"]="RobertaModel"
    ["SentenceDebiasDebertaForMaskedLM"]="DebertaModel"
    ["SentenceDebiasDistilbertForMaskedLM"]="DistilbertModel"
    ["SentenceDebiasGPT2LMHeadModel"]="GPT2Model"
    ["SentenceDebiasBertForSequenceClassification"]="BertModel"
    ["SentenceDebiasBertLargeForSequenceClassification"]="BertModel"
    ["SentenceDebiasAlbertForSequenceClassification"]="AlbertModel"
    ["SentenceDebiasRobertaForSequenceClassification"]="RobertaModel"
    ["SentenceDebiasDebertaForSequenceClassification"]="DebertaModel"
    ["SentenceDebiasDistilbertForSequenceClassification"]="DistilbertModel"
    ["SentenceDebiasGPT2ForSequenceClassification"]="GPT2Model"
    ["INLPBertModel"]="BertModel"
    ["INLPBertLargeModel"]="BertModel"
    ["INLPAlbertModel"]="AlbertModel"
    ["INLPRobertaModel"]="RobertaModel"
    ["INLPDebertaModel"]="DebertaModel"
    ["INLPDistilbertModel"]="DistilbertModel"
    ["INLPGPT2Model"]="GPT2Model"
    ["INLPBertForMaskedLM"]="BertModel"
    ["INLPBertLargeForMaskedLM"]="BertModel"
    ["INLPAlbertForMaskedLM"]="AlbertModel"
    ["INLPRobertaForMaskedLM"]="RobertaModel"
    ["INLPDebertaForMaskedLM"]="DebertaModel"
    ["INLPDistilbertForMaskedLM"]="DistilbertModel"
    ["INLPGPT2LMHeadModel"]="GPT2Model"
    ["INLPBertForSequenceClassification"]="BertModel"
    ["INLPBertLargeForSequenceClassification"]="BertModel"
    ["INLPAlbertForSequenceClassification"]="AlbertModel"
    ["INLPRobertaForSequenceClassification"]="RobertaModel"
    ["INLPDebertaForSequenceClassification"]="DebertaModel"
    ["INLPDistilbertForSequenceClassification"]="DistilbertModel"
    ["INLPGPT2ForSequenceClassification"]="GPT2Model"
)

declare -A debiased_model_to_masked_lm_model=(
    ["CDABertModel"]="BertForMaskedLM"
    ["CDABertLargeModel"]="BertLargeForMaskedLM"
    ["CDAAlbertModel"]="AlbertForMaskedLM"
    ["CDARobertaModel"]="RobertaForMaskedLM"
    ["CDADebertaModel"]="DebertaForMaskedLM"
    ["CDADistilbertModel"]="DistilbertForMaskedLM"
    ["CDABertForMaskedLM"]="BertForMaskedLM"
    ["CDABertLargeForMaskedLM"]="BertLargeForMaskedLM"
    ["CDAAlbertForMaskedLM"]="AlbertForMaskedLM"
    ["CDARobertaForMaskedLM"]="RobertaForMaskedLM"
    ["CDADebertaForMaskedLM"]="DebertaForMaskedLM"
    ["CDADistilbertForMaskedLM"]="DistilbertForMaskedLM"
    ["CDAGPT2Model"]="GPT2LMHeadModel"
    ["CDAGPT2LMHeadModel"]="GPT2LMHeadModel"
    ["DropoutBertModel"]="BertForMaskedLM"
    ["DropoutBertLargeModel"]="BertLargeForMaskedLM"
    ["DropoutAlbertModel"]="AlbertForMaskedLM"
    ["DropoutRobertaModel"]="RobertaForMaskedLM"
    ["DropoutDebertaModel"]="DebertaForMaskedLM"
    ["DropoutDistilbertModel"]="DistilbertForMaskedLM"
    ["DropoutGPT2Model"]="GPT2LMHeadModel"
    ["DropoutBertForMaskedLM"]="BertForMaskedLM"
    ["DropoutBertLargeForMaskedLM"]="BertLargeForMaskedLM"
    ["DropoutAlbertForMaskedLM"]="AlbertForMaskedLM"
    ["DropoutRobertaForMaskedLM"]="RobertaForMaskedLM"
    ["DropoutDebertaForMaskedLM"]="DebertaForMaskedLM"
    ["DropoutDistilbertForMaskedLM"]="DistilbertForMaskedLM"
    ["DropoutGPT2LMHeadModel"]="GPT2LMHeadModel"
)

# StereoSet specific variables.
stereoset_score_types=(
    "likelihood"
)

stereoset_splits=(
    # "dev"
    "test"
)

# Types of representations to use for computing SentenceDebias subspace
# and INLP projection matrix.
representation_types=(
    "cls"
    "mean"
)

# GLUE variables.
glue_tasks=(
    "cola"
    "mnli"
    "mrpc"
    "qnli"
    "qqp"
    "rte"
    "sst2"
    "stsb"
    "wnli"
)

# SEAT variables.
# Space separated list of SEAT tests to run.
seat_tests="sent-religion1 "\
"sent-religion1b "\
"sent-religion2 "\
"sent-religion2b "\
"sent-angry_black_woman_stereotype "\
"sent-angry_black_woman_stereotype_b "\
"sent-weat3 "\
"sent-weat3b "\
"sent-weat4 "\
"sent-weat5 "\
"sent-weat5b "\
"sent-weat6 "\
"sent-weat6b "\
"sent-weat7 "\
"sent-weat7b "\
"sent-weat8 "\
"sent-weat8b"

# only gender tests
seat_tests="sent-weat6 "\
"sent-weat6b "\
"sent-weat7 "\
"sent-weat7b "\
"sent-weat8 "\
"sent-weat8b"


declare -A model_to_n_classifiers=(["BertModel"]="80" ["BertLargeModel"]="80" ["AlbertModel"]="80" ["RobertaModel"]="80" ["DebertaModel"]="80" ["DistilbertModel"]="80" ["GPT2Model"]="10")
