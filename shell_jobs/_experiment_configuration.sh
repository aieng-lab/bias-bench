#!/bin/bash
# Variables for running batch jobs.

######################################################
# Adjust the directories to your local setup.
persistent_dir="/srv/data/drechsel/Git/bias-bench" # base dir of bias-bench repo
gradiend_dir="${persistent_dir}/../gradient" # base dir of gradiend repo
if [ -d "/root/bias-bench" ]; then
    persistent_dir="/root/bias-bench"
    gradiend_dir="${persistent_dir}/../gradiend" # base dir of gradiend repo
fi
#####################################################

changed_models_dir="${gradiend_dir}/results/changed_models"
checkpoint_dir="${persistent_dir}/results/checkpoints"
eval "$(conda shell.bash hook)"
conda activate bias-bench
export PYTHONPATH="${PYTHONPATH}:${persistent_dir}"


llama_model="meta-llama/Llama-3.2-3B"
llama_instruct_model="meta-llama/Llama-3.2-3B-Instruct"

suffix=""
race_suffix="-race_white_black"
race_suffix="-race_white_black_asian_combined"
religion_suffix="-religion_christian_muslim_jewish_combined"

male_model="${changed_models_dir}/bert-base-cased-M${suffix}"
female_model="${changed_models_dir}/bert-base-cased-F${suffix}"
unbiased_model_gender="${changed_models_dir}/bert-base-cased-N${suffix}"
unbiased_model_race="${changed_models_dir}/bert-base-cased-v7-race_white_black ${changed_models_dir}/bert-base-cased-v7-race_white_asian ${changed_models_dir}/bert-base-cased-v7-race_black_asian"
unbiased_model_religion="${changed_models_dir}/bert-base-cased-v7-religion_christian_jewish ${changed_models_dir}/bert-base-cased-v7-religion_christian_muslim ${changed_models_dir}/bert-base-cased-v7-religion_muslim_jewish"
cda_model_gender="${checkpoint_dir}/cda_c-bert-base-cased_t-gender_s-0"
cda_model_race="${checkpoint_dir}/cda_c-bert-base-cased_t-race_s-0"
cda_model_religion="${checkpoint_dir}/cda_c-bert-base-cased_t-religion_s-0"
dropout_model="${checkpoint_dir}/dropout_c-bert-base-cased_s-0"

male_model_bert_large_cased="${changed_models_dir}/bert-large-cased-M${suffix}"
female_model_bert_large_cased="${changed_models_dir}/bert-large-cased-F${suffix}"
unbiased_model_bert_large_cased_gender="${changed_models_dir}/bert-large-cased-N${suffix}"
unbiased_model_bert_large_cased_race="${changed_models_dir}/bert-large-cased-v7-race_white_black ${changed_models_dir}/bert-large-cased-v7-race_white_asian ${changed_models_dir}/bert-large-cased-v7-race_black_asian"
unbiased_model_bert_large_cased_religion="${changed_models_dir}/bert-large-cased-v7-religion_christian_jewish ${changed_models_dir}/bert-large-cased-v7-religion_christian_muslim ${changed_models_dir}/bert-large-cased-v7-religion_muslim_jewish"
cda_model_bert_large_cased_gender="${checkpoint_dir}/cda_c-bert-large-cased_t-gender_s-0"
cda_model_bert_large_cased_race="${checkpoint_dir}/cda_c-bert-large-cased_t-race_s-0"
cda_model_bert_large_cased_religion="${checkpoint_dir}/cda_c-bert-large-cased_t-religion_s-0"
dropout_model_bert_large_cased="${checkpoint_dir}/dropout_c-bert-large-cased_s-0"

male_roberta_model="${changed_models_dir}/roberta-large-M${suffix}"
female_roberta_model="${changed_models_dir}/roberta-large-F${suffix}"
unbiased_roberta_model_gender="${changed_models_dir}/roberta-large-N${suffix}"
unbiased_roberta_model_race="${changed_models_dir}/roberta-large-v7-race_white_asian ${changed_models_dir}/roberta-large-v7-race_black_asian ${changed_models_dir}/roberta-large-v7-race_white_black "
unbiased_roberta_model_religion="${changed_models_dir}/roberta-large-v7-religion_muslim_jewish ${changed_models_dir}/roberta-large-v7-religion_christian_jewish ${changed_models_dir}/roberta-large-v7-religion_christian_muslim"
cda_roberta_model_gender="${checkpoint_dir}/cda_c-roberta-large_t-gender_s-0"
cda_roberta_model_race="${checkpoint_dir}/cda_c-roberta-large_t-race_s-0"
cda_roberta_model_religion="${checkpoint_dir}/cda_c-roberta-large_t-religion_s-0"
dropout_roberta_model="${checkpoint_dir}/dropout_c-roberta-large_s-0"

male_distilbert_model="${changed_models_dir}/distilbert-base-cased-M${suffix}"
female_distilbert_model="${changed_models_dir}/distilbert-base-cased-F${suffix}"
unbiased_distilbert_model_gender="${changed_models_dir}/distilbert-base-cased-N${suffix}"
unbiased_distilbert_model_race="${changed_models_dir}/distilbert-base-cased-v7-race_white_black ${changed_models_dir}/distilbert-base-cased-v7-race_white_asian ${changed_models_dir}/distilbert-base-cased-v7-race_black_asian"
unbiased_distilbert_model_religion="${changed_models_dir}/distilbert-base-cased-v7-religion_christian_jewish ${changed_models_dir}/distilbert-base-cased-v7-religion_muslim_jewish ${changed_models_dir}/distilbert-base-cased-v7-religion_christian_muslim"
cda_distilbert_model_gender="${checkpoint_dir}/cda_c-distilbert-base-cased_t-gender_s-0"
cda_distilbert_model_race="${checkpoint_dir}/cda_c-distilbert-base-cased_t-race_s-0"
cda_distilbert_model_religion="${checkpoint_dir}/cda_c-distilbert-base-cased_t-religion_s-0"
dropout_distilbert_model="${checkpoint_dir}/dropout_c-distilbert-base-cased_s-0"


male_gpt2_model="${changed_models_dir}/gpt2-M${suffix}"
female_gpt2_model="${changed_models_dir}/gpt2-F${suffix}"
unbiased_gpt2_model_gender="${changed_models_dir}/gpt2-N${suffix}"
unbiased_gpt2_model_race="${changed_models_dir}/gpt2-v7-race_white_black ${changed_models_dir}/gpt2-v7-race_black_asian ${changed_models_dir}/gpt2-v7-race_white_black ${changed_models_dir}/gpt2-v7-race_white_asian"
unbiased_gpt2_model_religion="${changed_models_dir}/gpt2-v7-religion_christian_muslim ${changed_models_dir}/gpt2-v7-religion_christian_jewish ${changed_models_dir}/gpt2-v7-religion_muslim_jewish"
cda_gpt2_model_gender="${checkpoint_dir}/cda_c-gpt2_t-gender_s-0"
cda_gpt2_model_race="${checkpoint_dir}/cda_c-gpt2_t-race_s-0"
cda_gpt2_model_religion="${checkpoint_dir}/cda_c-gpt2_t-religion_s-0"
dropout_gpt2_model="${checkpoint_dir}/dropout_c-gpt2_s-0"

male_llama_model="${changed_models_dir}/Llama-3.2-3B-M"
female_llama_model="${changed_models_dir}/Llama-3.2-3B-F"
unbiased_llama_model_gender="${changed_models_dir}/Llama-3.2-3B-N"
unbiased_llama_model_race="${changed_models_dir}/Llama-3.2-3B-v5-race_white_black ${changed_models_dir}/Llama-3.2-3B-v5-race_white_asian ${changed_models_dir}/Llama-3.2-3B-v5-race_black_asian"
unbiased_llama_model_religion="${changed_models_dir}/Llama-3.2-3B-v5-religion_christian_jewish ${changed_models_dir}/Llama-3.2-3B-v5-religion_christian_muslim ${changed_models_dir}/Llama-3.2-3B-v5-religion_muslim_jewish"

male_llama_instruct_model="${changed_models_dir}/Llama-3.2-3B-Instruct-M"
female_llama_instruct_model="${changed_models_dir}/Llama-3.2-3B-Instruct-F"
unbiased_llama_instruct_model_gender="${changed_models_dir}/Llama-3.2-3B-Instruct-N"
unbiased_llama_instruct_model_race="${changed_models_dir}/Llama-3.2-3B-Instruct-v5-race_white_black ${changed_models_dir}/Llama-3.2-3B-Instruct-v5-race_white_asian ${changed_models_dir}/Llama-3.2-3B-Instruct-v5-race_black_asian"
unbiased_llama_instruct_model_religion="${changed_models_dir}/Llama-3.2-3B-Instruct-v5-religion_christian_muslim ${changed_models_dir}/Llama-3.2-3B-Instruct-v5-religion_christian_jewish ${changed_models_dir}/Llama-3.2-3B-Instruct-v5-religion_muslim_jewish"



gradiend_debiased_bert_models_gender="${male_model} ${female_model} ${unbiased_model_gender} ${male_model_bert_large_cased} ${female_model_bert_large_cased} ${unbiased_model_bert_large_cased_gender}"
debiased_bert_models_gender="${cda_model_bert_large_cased_gender} ${dropout_model_bert_large_cased} ${gradiend_debiased_bert_models_gender} ${cda_model_gender} ${dropout_model}"
gradiend_debiased_roberta_models_gender="${male_roberta_model} ${female_roberta_model} ${unbiased_roberta_model_gender}"
debiased_roberta_models_gender="${gradiend_debiased_roberta_models_gender} ${cda_roberta_model_gender} ${dropout_roberta_model}"
gradiend_debiased_distilbert_models_gender="${male_distilbert_model} ${female_distilbert_model} ${unbiased_distilbert_model_gender}"
debiased_distilbert_models_gender="${gradiend_debiased_distilbert_models_gender} ${cda_distilbert_model_gender} ${dropout_distilbert_model}"
gradiend_debiased_gpt2_models_gender="${unbiased_gpt2_model_gender} ${female_gpt2_model} ${male_gpt2_model}"
debiased_gpt2_models_gender="${gradiend_debiased_gpt2_models_gender} ${cda_gpt2_model_gender} ${dropout_gpt2_model}"
debiased_llama_models_gender="${unbiased_llama_model_gender} ${female_llama_model} ${male_llama_model}"
gradiend_debiased_llama_models_gender="${debiased_llama_models_gender}"
debiased_llama_instruct_models_gender="${unbiased_llama_instruct_model_gender} ${female_llama_instruct_model} ${male_llama_instruct_model}"
gradiend_debiased_llama_instruct_models_gender="${debiased_llama_instruct_models_gender}"

gradiend_debiased_bert_models_race="${unbiased_model_bert_large_cased_race} ${unbiased_model_race}"
debiased_bert_models_race=" ${cda_model_bert_large_cased_race} ${dropout_model_bert_large_cased} ${gradiend_debiased_bert_models_race} ${cda_model_race} ${dropout_model}"
gradiend_debiased_roberta_models_race="${unbiased_roberta_model_race}"
debiased_roberta_models_race="${gradiend_debiased_roberta_models_race} ${cda_roberta_model_race} ${dropout_roberta_model}"
gradiend_debiased_distilbert_models_race="${unbiased_distilbert_model_race}"
debiased_distilbert_models_race="${gradiend_debiased_distilbert_models_race} ${cda_distilbert_model_race} ${dropout_distilbert_model}"
gradiend_debiased_gpt2_models_race="${unbiased_gpt2_model_race}"
debiased_gpt2_models_race="${gradiend_debiased_gpt2_models_race} ${cda_gpt2_model_race} ${dropout_gpt2_model}"
debiased_llama_models_race="${unbiased_llama_model_race}"
gradiend_debiased_llama_models_race="${debiased_llama_models_race}"
debiased_llama_instruct_models_race="${unbiased_llama_instruct_model_race}"
gradiend_debiased_llama_instruct_models_race="${debiased_llama_instruct_models_race}"

gradiend_debiased_bert_models_religion="${unbiased_model_religion} ${male_model_bert_large_cased} ${female_model_bert_large_cased} ${unbiased_model_bert_large_cased_religion}"
debiased_bert_models_religion="${cda_model_bert_large_cased_religion} ${dropout_model_bert_large_cased} ${gradiend_debiased_bert_models_religion} ${cda_model_religion} ${dropout_model}"
gradiend_debiased_roberta_models_religion="${unbiased_roberta_model_religion}"
debiased_roberta_models_religion="${gradiend_debiased_roberta_models_religion} ${cda_roberta_model_religion} ${dropout_roberta_model}"
gradiend_debiased_distilbert_models_religion="${unbiased_distilbert_model_religion}"
debiased_distilbert_models_religion="${gradiend_debiased_distilbert_models_religion} ${cda_distilbert_model_religion} ${dropout_distilbert_model}"
gradiend_debiased_gpt2_models_religion="${unbiased_gpt2_model_religion}"
debiased_gpt2_models_religion="${gradiend_debiased_gpt2_models_religion} ${cda_gpt2_model_religion} ${dropout_gpt2_model}"
debiased_llama_models_religion="${unbiased_llama_model_religion}"
gradiend_debiased_llama_models_religion="${debiased_llama_models_religion}"
debiased_llama_instruct_models_religion="${unbiased_llama_instruct_model_religion}"
gradiend_debiased_llama_instruct_models_religion="${debiased_llama_instruct_models_religion}"


model_name_or_paths=(
  "bert-base-cased"
  "bert-large-cased"
  "roberta-large"
  "distilbert-base-cased"
  "distilbert-base-cased"
  "gpt2"
  $llama_model
  $llama_instruct_model
)

declare -A model_to_debiased_gradiend_models=(
    ["BertModel_gender"]="${gradiend_debiased_bert_models_gender}"
    ["BertModel_race"]="${gradiend_debiased_bert_models_race}"
    ["BertModel_religion"]="${gradiend_debiased_bert_models_religion}"
    ["RobertaModel_gender"]="${gradiend_debiased_roberta_models_gender}"
    ["RobertaModel_race"]="${gradiend_debiased_roberta_models_race}"
    ["RobertaModel_religion"]="${gradiend_debiased_roberta_models_religion}"
    ["DistilbertModel_gender"]="${gradiend_debiased_distilbert_models_gender}"
    ["DistilbertModel_race"]="${gradiend_debiased_distilbert_models_race}"
    ["DistilbertModel_religion"]="${gradiend_debiased_distilbert_models_religion}"
    ["GPT2Model_gender"]="${gradiend_debiased_gpt2_models_gender}"
    ["GPT2Model_race"]="${gradiend_debiased_gpt2_models_race}"
    ["GPT2Model_religion"]="${gradiend_debiased_gpt2_models_religion}"
    ["LlamaModel_gender"]="${gradiend_debiased_llama_models_gender}"
    ["LlamaModel_race"]="${gradiend_debiased_llama_models_race}"
    ["LlamaModel_religion"]="${gradiend_debiased_llama_models_religion}"
    ["LlamaInstructModel_gender"]="${gradiend_debiased_llama_instruct_models_gender}"
    ["LlamaInstructModel_race"]="${gradiend_debiased_llama_instruct_models_race}"
    ["LlamaInstructModel_religion"]="${gradiend_debiased_llama_instruct_models_religion}"
)

declare -A model_to_debiased_models=(
    ["BertModel_gender"]="${debiased_bert_models_gender}"
    ["BertModel_race"]="${debiased_bert_models_race}"
    ["BertModel_religion"]="${debiased_bert_models_religion}"
    ["RobertaModel_gender"]="${debiased_roberta_models_gender}"
    ["RobertaModel_race"]="${debiased_roberta_models_race}"
    ["RobertaModel_religion"]="${debiased_roberta_models_religion}"
    ["DistilbertModel_gender"]="${debiased_distilbert_models_gender}"
    ["DistilbertModel_race"]="${debiased_distilbert_models_race}"
    ["DistilbertModel_religion"]="${debiased_distilbert_models_religion}"
    ["GPT2Model_gender"]="${debiased_gpt2_models_gender}"
    ["GPT2Model_race"]="${debiased_gpt2_models_race}"
    ["GPT2Model_religion"]="${debiased_gpt2_models_religion}"
    ["LlamaModel_gender"]="${debiased_llama_models_gender}"
    ["LlamaModel_race"]="${debiased_llama_models_race}"
    ["LlamaModel_religion"]="${debiased_llama_models_religion}"
    ["LlamaInstructModel_gender"]="${debiased_llama_instruct_models_gender}"
    ["LlamaInstructModel_race"]="${debiased_llama_instruct_models_race}"
    ["LlamaInstructModel_religion"]="${debiased_llama_instruct_models_religion}"
)

declare -A model_name_or_path_to_model=(
    ["bert-base-cased"]="BertModel"
    ["bert-large-cased"]="BertModel"
    ["roberta-large"]="RobertaModel"
    ["distilbert-base-cased"]="DistilbertModel"
    ["gpt2"]="GPT2Model"
    [$llama_model]="LlamaModel"
    [$llama_instruct_model]="LlamaInstructModel"
)

bias_types=(
    "gender"
    "race"
    "religion"
)

seeds=(0 1 2)

projection_matrix_prefixes=(
  ""
  "rlace_"
  "leace_"
)



# Baseline models.
models=(
    "LlamaModel"
    "LlamaInstructModel"
    "BertModel"
    "DistilbertModel"
    "GPT2Model"
    "RobertaModel"
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
    "LlamaForCausalLM"
    "LlamaInstructForCausalLM"
    "GPT2LMHeadModel"
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
    "SentenceDebiasLlamaForCausalLM"
    "SentenceDebiasLlamaInstructForCausalLM"
)

inlp_causal_lm_models=(
    "INLPGPT2LMHeadModel"
    "INLPLlamaForCausalLM"
    "INLPLlamaInstructForCausalLM"
)


cda_causal_lm_models=(
    "CDAGPT2LMHeadModel"
    "CDALlamaForCausalLM"
    "CDALlamaInstructForCausalLM"
)

dropout_causal_lm_models=(
    "DropoutGPT2LMHeadModel"
    "DropoutLlamaForCausalLM"
    "DropoutLlamaInstructForCausalLM"
)

self_debias_causal_lm_models=(
    "SelfDebiasGPT2LMHeadModel"
    "SelfDebiasLlamaForCausalLM"
    "SelfDebiasLlamaInstructForCausalLM"
)

# Debiased base models.
sentence_debias_models=(
    "SentenceDebiasLlamaInstructModel"
    "SentenceDebiasLlamaModel"
    "SentenceDebiasGPT2Model"
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
    "INLPGPT2Model"
    "INLPLlamaModel"
    "INLPLlamaInstructModel"
)

cda_models=(
    "CDABertModel"
    "CDABertLargeModel"
    "CDARobertaModel"
    "CDADistilbertModel"
    "CDAGPT2Model"
    #"CDALlamaModel"
    #"CDALlamaInstructModel"
)

dropout_models=(
    "DropoutBertModel"
    "DropoutBertLargeModel"
    "DropoutRobertaModel"
    "DropoutDistilbertModel"
    "DropoutGPT2Model"
    #"DropoutLlamaModel"
    #"DropoutLlamaInstructModel"
)



declare -A model_to_model_name_or_path=(
    ["BertModel"]="bert-base-cased"
    ["BertLargeModel"]="bert-large-cased"
    ["AlbertModel"]="albert-base-v2"
    ["RobertaModel"]="roberta-large"
    ["DebertaModel"]="microsoft/deberta-v3-large"
    ["DistilbertModel"]="distilbert-base-cased"
    ["GPT2Model"]="gpt2"
    ["LlamaModel"]=$llama_model
    ["LlamaInstructModel"]=$llama_instruct_model
    ["BertForMaskedLM"]="bert-base-cased"
    ["BertLargeForMaskedLM"]="bert-large-cased"
    ["AlbertForMaskedLM"]="albert-base-v2"
    ["RobertaForMaskedLM"]="roberta-large"
    ["DebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["DistilbertForMaskedLM"]="distilbert-base-cased"
    ["GPT2LMHeadModel"]="gpt2"
    ["LlamaForCausalLM"]=$llama_model
    ["LlamaInstructForCausalLM"]=$llama_instruct_model
    ["BertForSequenceClassification"]="bert-base-cased"
    ["BertLargeForSequenceClassification"]="bert-large-cased"
    ["AlbertForSequenceClassification"]="albert-base-v2"
    ["RobertaForSequenceClassification"]="roberta-large"
    ["DebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["DistilbertForSequenceClassification"]="distilbert-base-cased"
    ["GPT2ForSequenceClassification"]="gpt2"
    ["LlamaForSequenceClassification"]=$llama_model
    ["LlamaInstructForSequenceClassification"]=$llama_instruct_model
    ["SentenceDebiasBertModel"]="bert-base-cased"
    ["SentenceDebiasBertLargeModel"]="bert-large-cased"
    ["SentenceDebiasAlbertModel"]="albert-base-v2"
    ["SentenceDebiasRobertaModel"]="roberta-large"
    ["SentenceDebiasDebertaModel"]="microsoft/deberta-v3-large"
    ["SentenceDebiasDistilbertModel"]="distilbert-base-cased"
    ["SentenceDebiasGPT2Model"]="gpt2"
    ["SentenceDebiasLlamaModel"]=$llama_model
    ["SentenceDebiasLlamaInstructModel"]=$llama_instruct_model
    ["SentenceDebiasBertForMaskedLM"]="bert-base-cased"
    ["SentenceDebiasBertLargeForMaskedLM"]="bert-large-cased"
    ["SentenceDebiasAlbertForMaskedLM"]="albert-base-v2"
    ["SentenceDebiasRobertaForMaskedLM"]="roberta-large"
    ["SentenceDebiasDebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["SentenceDebiasDistilbertForMaskedLM"]="distilbert-base-cased"
    ["SentenceDebiasGPT2LMHeadModel"]="gpt2"
    ["SentenceDebiasLlamaForCausalLM"]=$llama_model
    ["SentenceDebiasLlamaInstructForCausalLM"]=$llama_instruct_model
    ["SentenceDebiasBertForSequenceClassification"]="bert-base-cased"
    ["SentenceDebiasBertLargeForSequenceClassification"]="bert-large-cased"
    ["SentenceDebiasAlbertForSequenceClassification"]="albert-base-v2"
    ["SentenceDebiasRobertaForSequenceClassification"]="roberta-large"
    ["SentenceDebiasDebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["SentenceDebiasDistilbertForSequenceClassification"]="distilbert-base-cased"
    ["SentenceDebiasGPT2ForSequenceClassification"]="gpt2"
    ["SentenceDebiasLlamaForSequenceClassification"]=$llama_model
    ["SentenceDebiasLlamaInstructForSequenceClassification"]=$llama_instruct_model
    ["INLPBertModel"]="bert-base-cased"
    ["INLPBertLargeModel"]="bert-large-cased"
    ["INLPAlbertModel"]="albert-base-v2"
    ["INLPRobertaModel"]="roberta-large"
    ["INLPDebertaModel"]="microsoft/deberta-v3-large"
    ["INLPDistilbertModel"]="distilbert-base-cased"
    ["INLPGPT2Model"]="gpt2"
    ["INLPLlamaModel"]=$llama_model
    ["INLPLlamaInstructModel"]=$llama_instruct_model
    ["INLPBertForMaskedLM"]="bert-base-cased"
    ["INLPBertLargeForMaskedLM"]="bert-large-cased"
    ["INLPAlbertForMaskedLM"]="albert-base-v2"
    ["INLPRobertaForMaskedLM"]="roberta-large"
    ["INLPDebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["INLPDistilbertForMaskedLM"]="distilbert-base-cased"
    ["INLPGPT2LMHeadModel"]="gpt2"
    ["INLPLlamaForCausalLM"]=$llama_model
    ["INLPLlamaInstructForCausalLM"]=$llama_instruct_model
    ["INLPBertForSequenceClassification"]="bert-base-cased"
    ["INLPBertLargeForSequenceClassification"]="bert-large-cased"
    ["INLPAlbertForSequenceClassification"]="albert-base-v2"
    ["INLPRobertaForSequenceClassification"]="roberta-large"
    ["INLPDebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["INLPDistilbertForSequenceClassification"]="distilbert-base-cased"
    ["INLPGPT2ForSequenceClassification"]="gpt2"
    ["INLPLlamaForSequenceClassification"]=$llama_model
    ["INLPLlamaInstructForSequenceClassification"]=$llama_instruct_model
    ["CDABertModel"]="bert-base-cased"
    ["CDABertLargeModel"]="bert-large-cased"
    ["CDAAlbertModel"]="albert-base-v2"
    ["CDARobertaModel"]="roberta-large"
    ["CDADebertaModel"]="microsoft/deberta-v3-large"
    ["CDADistilbertModel"]="distilbert-base-cased"
    ["CDAGPT2Model"]="gpt2"
    ["CDALlamaModel"]=$llama_model
    ["CDALlamaInstructModel"]=$llama_instruct_model
    ["CDABertForMaskedLM"]="bert-base-cased"
    ["CDABertLargeForMaskedLM"]="bert-large-cased"
    ["CDAAlbertForMaskedLM"]="albert-base-v2"
    ["CDARobertaForMaskedLM"]="roberta-large"
    ["CDADebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["CDADistilbertForMaskedLM"]="distilbert-base-cased"
    ["CDAGPT2LMHeadModel"]="gpt2"
    ["CDALlamaForCausalLM"]=$llama_model
    ["CDALlamaInstructForCausalLM"]=$llama_instruct_model
    ["CDABertForSequenceClassification"]="bert-base-cased"
    ["CDABertLargeForSequenceClassification"]="bert-large-cased"
    ["CDAAlbertForSequenceClassification"]="albert-base-v2"
    ["CDARobertaForSequenceClassification"]="roberta-large"
    ["CDADebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["CDADistilbertForSequenceClassification"]="distilbert-base-cased"
    ["CDAGPT2ForSequenceClassification"]="gpt2"
    ["CDALlamaForSequenceClassification"]=$llama_model
    ["CDALlamaInstructForSequenceClassification"]=$llama_instruct_model
    ["DropoutBertModel"]="bert-base-cased"
    ["DropoutBertLargeModel"]="bert-large-cased"
    ["DropoutAlbertModel"]="albert-base-v2"
    ["DropoutRobertaModel"]="roberta-large"
    ["DropoutDebertaModel"]="microsoft/deberta-v3-large"
    ["DropoutDistilbertModel"]="distilbert-base-cased"
    ["DropoutGPT2Model"]="gpt2"
    ["DropoutLlamaModel"]=$llama_model
    ["DropoutLlamaInstructModel"]=$llama_instruct_model
    ["DropoutBertForMaskedLM"]="bert-base-cased"
    ["DropoutBertLargeForMaskedLM"]="bert-large-cased"
    ["DropoutAlbertForMaskedLM"]="albert-base-v2"
    ["DropoutRobertaForMaskedLM"]="roberta-large"
    ["DropoutDebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["DropoutDistilbertForMaskedLM"]="distilbert-base-cased"
    ["DropoutGPT2LMHeadModel"]="gpt2"
    ["DropoutLlamaForCausalLM"]=$llama_model
    ["DropoutLlamaInstructForCausalLM"]=$llama_instruct_model
    ["DropoutBertForSequenceClassification"]="bert-base-cased"
    ["DropoutBertLargeForSequenceClassification"]="bert-large-cased"
    ["DropoutAlbertForSequenceClassification"]="albert-base-v2"
    ["DropoutRobertaForSequenceClassification"]="roberta-large"
    ["DropoutDebertaForSequenceClassification"]="microsoft/deberta-v3-large"
    ["DropoutDistilbertForSequenceClassification"]="distilbert-base-cased"
    ["DropoutGPT2ForSequenceClassification"]="gpt2"
    ["DropoutLlamaForSequenceClassification"]=$llama_model
    ["DropoutLlamaInstructForSequenceClassification"]=$llama_instruct_model
    ["SelfDebiasBertForMaskedLM"]="bert-base-cased"
    ["SelfDebiasBertLargeForMaskedLM"]="bert-large-cased"
    ["SelfDebiasAlbertForMaskedLM"]="albert-base-v2"
    ["SelfDebiasRobertaForMaskedLM"]="roberta-large"
    ["SelfDebiasDebertaForMaskedLM"]="microsoft/deberta-v3-large"
    ["SelfDebiasDistilbertForMaskedLM"]="distilbert-base-cased"
    ["SelfDebiasGPT2LMHeadModel"]="gpt2"
    ["SelfDebiasLlamaForCausalLM"]=$llama_model
    ["SelfDebiasLlamaInstructForCausalLM"]=$llama_instruct_model
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
    ["SentenceDebiasLlamaModel"]="LlamaModel"
    ["SentenceDebiasLlamaInstructModel"]="LlamaInstructModel"
    ["SentenceDebiasBertForMaskedLM"]="BertModel"
    ["SentenceDebiasBertLargeForMaskedLM"]="BertModel"
    ["SentenceDebiasAlbertForMaskedLM"]="AlbertModel"
    ["SentenceDebiasRobertaForMaskedLM"]="RobertaModel"
    ["SentenceDebiasDebertaForMaskedLM"]="DebertaModel"
    ["SentenceDebiasDistilbertForMaskedLM"]="DistilbertModel"
    ["SentenceDebiasGPT2LMHeadModel"]="GPT2Model"
    ["SentenceDebiasLlamaForCausalLM"]="LlamaModel"
    ["SentenceDebiasLlamaInstructForCausalLM"]="LlamaInstructModel"
    ["SentenceDebiasBertForSequenceClassification"]="BertModel"
    ["SentenceDebiasBertLargeForSequenceClassification"]="BertModel"
    ["SentenceDebiasAlbertForSequenceClassification"]="AlbertModel"
    ["SentenceDebiasRobertaForSequenceClassification"]="RobertaModel"
    ["SentenceDebiasDebertaForSequenceClassification"]="DebertaModel"
    ["SentenceDebiasDistilbertForSequenceClassification"]="DistilbertModel"
    ["SentenceDebiasGPT2ForSequenceClassification"]="GPT2Model"
    ["SentenceDebiasLlamaForSequenceClassification"]="LlamaModel"
    ["SentenceDebiasLlamaInstructForSequenceClassification"]="LlamaInstructModel"
    ["INLPBertModel"]="BertModel"
    ["INLPBertLargeModel"]="BertModel"
    ["INLPAlbertModel"]="AlbertModel"
    ["INLPRobertaModel"]="RobertaModel"
    ["INLPDebertaModel"]="DebertaModel"
    ["INLPDistilbertModel"]="DistilbertModel"
    ["INLPGPT2Model"]="GPT2Model"
    ["INLPLlamaModel"]="LlamaModel"
    ["INLPLlamaInstructModel"]="LlamaInstructModel"
    ["INLPBertForMaskedLM"]="BertModel"
    ["INLPBertLargeForMaskedLM"]="BertModel"
    ["INLPAlbertForMaskedLM"]="AlbertModel"
    ["INLPRobertaForMaskedLM"]="RobertaModel"
    ["INLPDebertaForMaskedLM"]="DebertaModel"
    ["INLPDistilbertForMaskedLM"]="DistilbertModel"
    ["INLPGPT2LMHeadModel"]="GPT2Model"
    ["INLPLlamaForCausalLM"]="LlamaModel"
    ["INLPLlamaInstructForCausalLM"]="LlamaInstructModel"
    ["INLPBertForSequenceClassification"]="BertModel"
    ["INLPBertLargeForSequenceClassification"]="BertModel"
    ["INLPAlbertForSequenceClassification"]="AlbertModel"
    ["INLPRobertaForSequenceClassification"]="RobertaModel"
    ["INLPDebertaForSequenceClassification"]="DebertaModel"
    ["INLPDistilbertForSequenceClassification"]="DistilbertModel"
    ["INLPGPT2ForSequenceClassification"]="GPT2Model"
    ["INLPLlamaForSequenceClassification"]="LlamaModel"
    ["INLPLlamaInstructForSequenceClassification"]="LlamaInstructModel"
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
    ["CDALlamaModel"]="LlamaForCausalLM"
    ["CDALlamaInstructModel"]="LlamaForCausalLM"
    ["CDALlamaForCausalLM"]="LlamaForCausalLM"
    ["CDALlamaInstructForCausalLM"]="LlamaForCausalLM"
    ["DropoutBertModel"]="BertForMaskedLM"
    ["DropoutBertLargeModel"]="BertLargeForMaskedLM"
    ["DropoutAlbertModel"]="AlbertForMaskedLM"
    ["DropoutRobertaModel"]="RobertaForMaskedLM"
    ["DropoutDebertaModel"]="DebertaForMaskedLM"
    ["DropoutDistilbertModel"]="DistilbertForMaskedLM"
    ["DropoutGPT2Model"]="GPT2LMHeadModel"
    ["DropoutLlamaModel"]="LlamaForCausalLM"
    ["DropoutLlamaInstructModel"]="LlamaForCausalLM"
    ["DropoutBertForMaskedLM"]="BertForMaskedLM"
    ["DropoutBertLargeForMaskedLM"]="BertLargeForMaskedLM"
    ["DropoutAlbertForMaskedLM"]="AlbertForMaskedLM"
    ["DropoutRobertaForMaskedLM"]="RobertaForMaskedLM"
    ["DropoutDebertaForMaskedLM"]="DebertaForMaskedLM"
    ["DropoutDistilbertForMaskedLM"]="DistilbertForMaskedLM"
    ["DropoutGPT2LMHeadModel"]="GPT2LMHeadModel"
    ["DropoutLlamaForCausalLM"]="LlamaForCausalLM"
    ["DropoutLlamaInstructForCausalLM"]="LlamaForCausalLM"
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
    "wnli"
    "stsb"
    "sst2"
    "cola"
    "qqp"
    "rte"
    "mrpc"
    "mnli"
    "qnli"
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



seat_gender_tests=${seat_tests}

seat_race_tests="sent-angry_black_woman_stereotype "\
"sent-angry_black_woman_stereotype_b "\
"sent-weat3 "\
"sent-weat3b "\
"sent-weat4 "\
"sent-weat5 "\
"sent-weat5b "


seat_religion_tests="sent-religion1 "\
"sent-religion1b "\
"sent-religion2 "\
"sent-religion2b "

declare -A model_to_n_classifiers=(["BertModel"]="80" ["BertLargeModel"]="80" ["AlbertModel"]="80" ["RobertaModel"]="80" ["DebertaModel"]="80" ["DistilbertModel"]="80" ["GPT2Model"]="10" ["LlamaModel"]="80" ["LlamaInstructModel"]="80")
