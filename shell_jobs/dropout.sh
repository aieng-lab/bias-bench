#!/bin/bash

source "shell_jobs/_experiment_configuration.sh"

seeds=(0) # only seed 0 is used for evaluation

models=("gpt2")
for seed in ${seeds[@]}; do
    for model in ${models[@]}; do
        model_id=$(basename "$model")
        experiment_id="dropout_c-${model_id}_s-${seed}"
        if [ ! -d "${persistent_dir}/results/checkpoints/${experiment_id}" ]; then
            echo ${experiment_id}
            python experiments/run_clm.py \
                --model_name_or_path ${model} \
                --do_train \
                --train_file "${persistent_dir}/data/text/wikipedia-10.txt" \
                --max_steps 2000 \
                --per_device_train_batch_size 8 \
                --gradient_accumulation_steps 32 \
                --save_steps 500 \
                --preprocessing_num_workers 4 \
                --dropout_debias \
                --seed ${seed} \
                --output_dir "${persistent_dir}/results/checkpoints/${experiment_id}" \
                --persistent_dir ${persistent_dir}
            else
                echo "${experiment_id} already computed"
        fi
    done
done

models=("bert-base-cased" "bert-large-cased" "roberta-large" "distilbert-base-cased")
for seed in ${seeds[@]}; do
  for model in ${models[@]}; do
        experiment_id="dropout_c-${model}_s-${seed}"

        if [ ! -d "${persistent_dir}/results/checkpoints/${experiment_id}" ]; then
            echo ${experiment_id}
            python experiments/run_mlm.py \
                --model_name_or_path ${model} \
                --do_train \
                --train_file "${persistent_dir}/data/text/wikipedia-10.txt" \
                --max_steps 2000 \
                --per_device_train_batch_size 32 \
                --gradient_accumulation_steps 16 \
                --max_seq_length 512 \
                --save_steps 500 \
                --preprocessing_num_workers 4 \
                --dropout_debias \
                --seed ${seed} \
                --output_dir "${persistent_dir}/results/checkpoints/${experiment_id}" \
                --persistent_dir ${persistent_dir}
        else
            echo "${experiment_id} already computed"
        fi
    done
done

