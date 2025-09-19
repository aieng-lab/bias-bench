import json
import re

import numpy as np
from datasets import load_dataset

from experiments import glue_bootstrap_evaluation


def get_jsonl_file(model_path, task):
    """
    Search for a .jsonl file in the given folder that matches the task name.
    Allows for dynamic timestamp suffix.

    Args:
        model_path (str): Path to the folder containing the jsonl files.
        task (str): Task name (e.g., 'mnli', 'mnli_mismatch', 'cola', etc.)

    Returns:
        str or None: Full path to the matching file, or None if not found.
    """
    pattern = re.compile(rf'^samples_{re.escape(task)}_\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}-\d{{2}}-\d{{2}}\.\d+\.jsonl$')

    for filename in os.listdir(model_path):
        if pattern.match(filename):
            return os.path.join(model_path, filename)

    return None


def read_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

glue_tasks = [
    'wnli',
    #'stsb',
    'sst2',
    'rte',
    'qqp',
    'qnli',
    'mrpc',
    'mnli_matched',
    'mnli_mismatched',
    'cola',
]

super_glue_tasks = [
    'boolq',
    'cb',
    'copa',
    'multirc',
    'record',
    'rte',
    'wic',
    'wsc.fixed',
    'axb',
    'axg'
]

dataset_cache = {gt: {} for gt in ['glue', 'super_glue']}




def convert_model(input_path, seeds=(0, 1, 2), output_dir='results', model_type='Llama', output_suffix='', glue_type='glue'):

    tasks = glue_tasks if glue_type == 'glue' else super_glue_tasks

    model_id = input_path.split('/')[-1]
    for suffix in {'-inlp', '-rlace_', '-leace_', '-sentence_debias'}:
        model_id = model_id.replace(suffix, '').replace('-gender', '')

    model_output_dir = f'{output_dir}/{glue_type}/{glue_type}_m-{model_type}ForSequenceClassification_c-{model_id}{output_suffix}'

    for seed in seeds:
        seed_path = os.path.join(input_path, str(seed))
        # Check if the seed path exists
        if not os.path.exists(seed_path):
            print(f"Seed path {seed_path} does not exist.")
            continue

        # one folder should be contained in the seed path
        folders = [
            f for f in os.listdir(seed_path) if os.path.isdir(os.path.join(seed_path, f))
        ]

        if len(folders) == 0:
            print(f"No folders found in {seed_path}.")
            continue
        elif len(folders) != 1:
            # check if folder with start "meta-llama" exists
            folder = None
            for f in folders:
                if f.startswith('meta-llama') or f.startswith('__root__'):
                    folder = f
                    break
            if folder is None:
                # use first one as fallback
                folder = folders[0]
                print(f"Multiple folders found in {seed_path}, using the first one: {folder}")
            else:
                print(f"Expected one folder in {seed_path}, found {len(folders)}, using {folder}")
        else:
            folder = folders[0]
        model_path = os.path.join(seed_path, folder)
        model_output_dir_seed = f'{model_output_dir}/{seed}'

        for task in tasks:
            task_id = task
            task_name = task
            task_jsonl_id = task
            if task == 'mnli_mismatched':
                task_id = 'mnli'
                task_name = 'mnli-mm'
                task_jsonl_id = 'mnli_mismatch'
            elif task == 'mnli_matched':
                task_id = 'mnli'
                task_name = task_id
                task_jsonl_id = task_name
            elif task == 'rte' and glue_type == 'super_glue':
                task_jsonl_id = 'sglue_rte'
            elif task == 'wsc.fixed':
                task_jsonl_id = 'wsc'

            output_file = f'{model_output_dir_seed}/{task_id}/eval_results_{task_name}.txt'

            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists. Skipping conversion for task {task}.")
                #continue

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            jsonl_file = get_jsonl_file(model_path, task_jsonl_id)

            if jsonl_file is None:
                print(f"No jsonl file found for task {task} in {model_path}.")
                continue

            data = read_jsonl(jsonl_file)

            if task in dataset_cache[glue_type]:
                ds = dataset_cache[glue_type][task]
            else:
                ds = load_dataset(glue_type, task, split='validation')
                dataset_cache[glue_type][task] = ds

            converted_data = {'index': ['index'], 'prediction': ['prediction']}
            if task == 'record':
                i = 0
                for d in data:
                    index = d['doc']['idx']
                    resps = d['resps'] # [[['-85.5', 'False']], [['-98.0', 'False']], [['-105.5', 'False']], [['-98.0', 'False']], [['-78.0', 'False']], [['-85.0', 'False']], [['-94.5', 'False']], [['-95.5', 'False']], [['-97.0', 'False']], [['-99.0', 'False']], [['-92.5', 'False']], [['-93.5', 'False']], [['-87.5', 'False']], [['-99.5', 'False']]]
                    logits = [float(x[0][0]) for x in resps]
                    argmax = np.argmax(logits)

                    predicted = d['doc']['entities'][argmax]
                    converted_data['index'].append(i)
                    converted_data['prediction'].append(predicted)
                    i += 1
            else:
                labels = ds.features['label'].names
                for d in data:
                    index = d['doc']['idx']
                    prediction = labels[d['pred']]
                    converted_data['index'].append(index)
                    converted_data['prediction'].append(prediction)

            with open(output_file, 'w') as f:
                for index, prediction in zip(converted_data['index'], converted_data['prediction']):
                    f.write(f"{index}\t{prediction}\n")

            print(f"Converted data for task {task} saved to {output_file}")

    #mean, margin_of_error = glue_bootstrap_evaluation.glue_bootstrap_scores(model_output_dir, glue_type=glue_type)
    #print(f"Mean: {mean}, Margin of Error: {margin_of_error}")


if __name__ == '__main__':
    # Convert all models in the directory
    import os
    import shutil



    glue_types = ['glue', 'super_glue']
    #glue_types = ['super_glue']
    #glue_types = ['glue']
    result_dirs = {
        'glue': 'results/glue_zero_shot',
        'super_glue': 'results/super-glue-lm-eval-v1_zero_shot',
    }

    for glue_type in glue_types:
        result_dir = result_dirs[glue_type]

        #convert_model(f'{result_dir}/Llama-3.2-3B', model_type='Llama', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-N-gender', model_type='Llama', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-F-gender', model_type='Llama', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-M-gender', model_type='Llama', output_suffix='_t-gender', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-v5-race_white_asian', model_type='Llama', output_suffix='_t-race', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-v5-race_white_black', model_type='Llama', output_suffix='_t-race', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-v5-race_black_asian', model_type='Llama', output_suffix='_t-race', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-v5-religion_christian_jewish', model_type='Llama', output_suffix='_t-religion', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-v5-religion_christian_muslim', model_type='Llama', output_suffix='_t-religion', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-v5-religion_muslim_jewish', model_type='Llama', output_suffix='_t-religion', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-inlp-gender', model_type='INLPLlama', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-inlp-race', model_type='INLPLlama', output_suffix='_t-race', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-inlp-religion', model_type='INLPLlama', output_suffix='_t-religion', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-rlace_-gender', model_type='rlace_INLPLlama', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-leace_-gender', model_type='leace_INLPLlama', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-N-inlp-gender', model_type='INLPLlama', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-sentence_debias-gender', model_type='SentenceDebiasLlama', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-sentence_debias-race', model_type='SentenceDebiasLlama', output_suffix='_t-race', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-sentence_debias-religion', model_type='SentenceDebiasLlama', output_suffix='_t-religion', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-N-sentence_debias-gender', model_type='SentenceDebiasLlama', output_suffix='_t-gender', glue_type=glue_type)

        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct', model_type='LlamaInstruct', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-N-gender', model_type='LlamaInstruct', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-F-gender', model_type='LlamaInstruct', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-M-gender', model_type='LlamaInstruct', output_suffix='_t-gender', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-v5--race_white_asian', model_type='LlamaInstruct', output_suffix='_t-race', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-v5-race_white_black', model_type='LlamaInstruct', output_suffix='_t-race', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-v5-race_black_asian', model_type='LlamaInstruct', output_suffix='_t-race', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-v5-religion_christian_jewish', model_type='LlamaInstruct', output_suffix='_t-religion', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-v5-religion_christian_muslim', model_type='LlamaInstruct', output_suffix='_t-religion', glue_type=glue_type)
        convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-v5-religion_muslim_jewish', model_type='LlamaInstruct', output_suffix='_t-religion', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-inlp-gender', model_type='INLPLlamaInstruct', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-inlp-race', model_type='INLPLlamaInstruct', output_suffix='_t-race', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-inlp-religion', model_type='INLPLlamaInstruct', output_suffix='_t-religion', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-rlace_-gender', model_type='rlace_INLPLlamaInstruct', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-leace_-gender', model_type='leace_INLPLlamaInstruct', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-N-inlp-gender', model_type='INLPLlamaInstruct', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-sentence_debias-gender', model_type='SentenceDebiasLlamaInstruct', output_suffix='_t-gender', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-sentence_debias-race', model_type='SentenceDebiasLlamaInstruct', output_suffix='_t-race', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-sentence_debias-religion', model_type='SentenceDebiasLlamaInstruct', output_suffix='_t-religion', glue_type=glue_type)
        #convert_model(f'{result_dir}/Llama-3.2-3B-Instruct-N-sentence_debias-gender', model_type='SentenceDebiasLlamaInstruct', output_suffix='_t-gender', glue_type=glue_type)
