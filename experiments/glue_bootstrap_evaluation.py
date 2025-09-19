import json
import os
import random
import time

from frozendict import frozendict
import pandas as pd
import numpy as np
import tqdm
from datasets import load_metric, load_dataset
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import spearmanr, norm
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
import multiprocessing

def bootstrap_glue_scores(df: pd.DataFrame, n_samples: int = 1000, seed: int = 42) -> list:
    """
    Perform bootstrapping on a pandas DataFrame containing GLUE predicted and correct values.
    Different metrics are applied based on the task.

    Metrics:
    - Accuracy: Default for most tasks.
    - F1 Score: MRPC
    - Spearman Correlation: STS-B
    - Matthew's Correlation Coefficient: CoLA

    Parameters:
    - df (pd.DataFrame): DataFrame containing columns ['task', 'seed', 'prediction', 'label', 'idx'].
    - n_samples (int): Number of bootstrap samples to generate.
    - seed (int): Random seed for reproducibility.

    Returns:
    - list: A list containing the bootstrapped GLUE scores.
    """
    np.random.seed(seed)  # Set random seed for reproducibility

    # Precompute unique tasks and seeds
    unique_tasks = df['task'].unique()
    unique_seeds = df['seed'].unique()

    # Pre-load metrics to avoid repeated loading
    task_mapper = {
        'mnli': 'mnli_matched',
        'mnli-mm': 'mnli_mismatched',
    }
    metrics = {task: load_metric("glue", task_mapper.get(task, task)) for task in unique_tasks}
    from experiments.util import metric_mapping
    metric_mapping['mnli-mm'] = metric_mapping['mnli']

    # Function to compute metrics
    def compute_metric(task, correct, predicted):
        metric = metrics[task]
        result = metric.compute(predictions=predicted, references=correct)
        metric_key = metric_mapping[task].removeprefix('eval_')
        return result[metric_key]

    # Prepare the pivot table once for efficiency
    pivot_tables = {
        task: df[df['task'] == task]
        .pivot_table(index='idx', columns='seed', values='prediction')
        .merge(df[df['task'] == task][['idx', 'label']].drop_duplicates(), on='idx')
        for task in unique_tasks
    }

    # Predefine bootstrap scores list
    bootstrap_scores = []

    mean_tasks = unique_tasks
    if 'mnli' in unique_tasks and 'mnli-mm' in unique_tasks:
        mean_tasks = [t for t in unique_tasks if t not in ['mnli', 'mnli-mm']] + ['mnli_mean']

    # Bootstrap loop
    for sample_idx in tqdm.tqdm(range(n_samples), desc="Bootstrapping"):
        task_scores = {}

        for task in unique_tasks:
            task_pivot_df = pivot_tables[task]

            # Bootstrap sample
            sampled_pivot_df = task_pivot_df.sample(
                n=len(task_pivot_df), replace=True, random_state=seed + sample_idx
            )

            # Compute scores for each seed
            scores = [
                compute_metric(task, sampled_pivot_df['label'], sampled_pivot_df[seed])
                for seed in unique_seeds
            ]

            if 0.0 in scores:
                print(f"Sample {sample_idx} has 0.0 in scores for task {task}")
                #raise ValueError(f"Sample {sample_idx} has 0.0 in scores for task {task}")

            task_scores[task] = np.mean(scores)

        if 'mnli' in unique_tasks and 'mnli-mm' in unique_tasks:
            task_scores['mnli_mean'] = np.mean([task_scores['mnli'], task_scores['mnli-mm']])

        bootstrap_scores.append({'mean': np.mean([task_scores[task] for task in mean_tasks]), **task_scores})

    # compute the mean of the original data
    result = {}
    for task in unique_tasks:
        task_pivot_df = pivot_tables[task]
        scores = [
            compute_metric(task, task_pivot_df['label'], task_pivot_df[seed])
            for seed in unique_seeds
        ]

        result[task] = np.mean(scores)

    if 'mnli' in unique_tasks and 'mnli-mm' in unique_tasks:
        result['mnli_mean'] = np.mean([result['mnli'], result['mnli-mm']])

    result['mean'] = np.mean([result[task] for task in mean_tasks])
    result['bootstrap'] = bootstrap_scores

    return result

import numpy as np
import pandas as pd
import tqdm
from datasets import load_metric

metrics_cache = {}

dataset_cache = {}

def bootstrap_super_glue_scores(df: pd.DataFrame, n_samples: int = 1000, seed: int = 42, n_jobs=-1) -> dict:
    """
    Perform bootstrapping on a pandas DataFrame containing SuperGLUE predictions.
    For CB, MultiRC, and ReCoRD, individual metrics are reported separately,
    and their mean is also reported as the task score.

    Parameters:
    - df (pd.DataFrame): Must contain ['task', 'seed', 'prediction', 'label', 'idx'].
    - n_samples (int): Number of bootstrap samples.
    - seed (int): Random seed.

    Returns:
    - dict: Dictionary with per-task scores, sub-metrics, mean, and bootstrap distribution.
    """
    # Precompute unique tasks and seeds
    unique_tasks = df["task"].unique()
    unique_seeds = df["seed"].unique()


    if 'multirc' in unique_tasks and not 'multirc' in dataset_cache:
        eval_dataset = load_dataset("super_glue", "multirc", split="validation")
        dataset_cache['multirc'] = eval_dataset["idx"]

    # Load metrics for all tasks once
    for task in unique_tasks:
        if task not in metrics_cache:
            metrics_cache[task] = load_metric("super_glue", task)
    metrics = {t: metrics_cache[t] for t in unique_tasks}

    # Task-specific metric logic
    def compute_metric(task, correct, predicted):
        metric = metrics[task]
        result = metric.compute(predictions=predicted, references=correct)

        if task == "cb":
            return {
                "cb_acc": result["accuracy"],
                "cb_f1": result["f1"],
                "cb": (result["accuracy"] + result["f1"]) / 2,
            }
        elif task == "multirc":
            return {
                "multirc_f1a": result["f1_a"],
                "multirc_em": result["exact_match"],
                "multirc": (result["f1_a"] + result["exact_match"]) / 2,
            }
        elif task == "record":
            return {
                "record_f1": result["f1"],
                "record_em": result["exact_match"],
                "record": (result["f1"] + result["exact_match"]) / 2,
            }
        # Default: take the first key
        k = next(iter(result))
        return {task: result[k]}

    # Prepare pivot tables once
    pivot_tables = {
        task: df[df["task"] == task]
        .pivot_table(index="idx", columns="seed", values="prediction", aggfunc='first')
        .merge(df[df["task"] == task][["idx", "label"]].drop_duplicates(subset=["idx"]), on="idx")
        for task in unique_tasks
    }

    n_tasks = {t: len(pivot_tables[t]) for t in unique_tasks}

    # Extend dataset_cache with grouping info
    if "multirc" in unique_tasks and "multirc_groups" not in dataset_cache:
        eval_dataset = load_dataset("super_glue", "multirc", split="validation")
        dataset_cache["multirc"] = eval_dataset["idx"]
        # group by (paragraph, question)
        question_groups = {}
        for i, idx in enumerate(eval_dataset["idx"]):
            qid = (idx["paragraph"], idx["question"])
            question_groups.setdefault(qid, []).append(i)
        dataset_cache["multirc_groups"] = list(question_groups.values())

    if "record" in unique_tasks and "record_groups" not in dataset_cache:
        eval_dataset = load_dataset("super_glue", "record", split="validation")
        dataset_cache["record"] = eval_dataset["idx"]
        # group by (passage, query)
        query_groups = {}
        for i, idx in enumerate(eval_dataset["idx"]):
            qid = idx["passage"]
            query_groups.setdefault(qid, []).append(i)
        dataset_cache["record_groups"] = list(query_groups.values())

    multirc_df = pivot_tables.get("multirc", None)
    # extend the seed columns with the prediction format (i.e. dict with idx and prediction)
    if multirc_df is not None:
        for seed in unique_seeds:
            multirc_df[seed] = [
                {"idx": idx, "prediction": int(pred)}
                for idx, pred in zip(dataset_cache["multirc"], multirc_df[seed])
            ]
        pivot_tables["multirc"] = multirc_df

    def one_bootstrap(rep_seed: int):
        rng_local = np.random.default_rng(rep_seed)
        task_scores = {}
        for task in unique_tasks:
            task_pivot = pivot_tables[task]
            if task == "multirc":
                groups = dataset_cache["multirc_groups"]
                sampled_groups = rng_local.integers(0, len(groups), size=len(groups))
                sampled_rows = [i for g in sampled_groups for i in groups[g]]
                sampled = task_pivot.iloc[sampled_rows]
            elif task == "record":
                groups = dataset_cache["record_groups"]
                sampled_groups = rng_local.integers(0, len(groups), size=len(groups))
                sampled_rows = [i for g in sampled_groups for i in groups[g]]
                sampled = task_pivot.iloc[sampled_rows]
            else:
                sampled_idx = rng_local.integers(0, n_tasks[task], size=n_tasks[task])
                sampled = task_pivot.iloc[sampled_idx]

            metric_dicts = [
                compute_metric(task, sampled["label"], sampled[s])
                for s in unique_seeds
            ]
            merged = {k: np.mean([d[k] for d in metric_dicts]) for k in metric_dicts[0]}
            task_scores.update(merged)

        mean_score = np.mean([
            v for k, v in task_scores.items()
            if not any(x in k for x in ["_acc", "_f1", "_f1a", "_em"])
        ])
        return {"mean": mean_score, **task_scores}

    # Run bootstrap in
    if n_jobs > 0:
        print("Running bootstrapping in parallel...")
        start = time.time()
        bootstrap_scores = Parallel(n_jobs=n_jobs)(
            delayed(one_bootstrap)(seed + i) for i in range(n_samples)
        )
        print(f"Bootstrapping completed in {time.time() - start:.2f} seconds.")
    else:
        bootstrap_scores = []
        for i in tqdm.tqdm(range(n_samples), desc="Bootstrapping"):
            bootstrap_scores.append(one_bootstrap(seed + i))

    # Compute scores for original (non-bootstrap) data
    result = {}
    for task in unique_tasks:
        task_pivot_df = pivot_tables[task]

        metric_dicts = [
            compute_metric(task, task_pivot_df["label"], task_pivot_df[seed])
            for seed in unique_seeds
        ]

        merged = {}
        for key in metric_dicts[0].keys():
            merged[key] = np.mean([d[key] for d in metric_dicts])

        result.update(merged)

    result["mean"] = np.mean(
        [v for k, v in result.items() if not any(x in k for x in ["_acc", "_f1", "_f1a", "_em"])]
    )
    result["bootstrap"] = bootstrap_scores

    return result


def preprocess_record(examples):
    """
    Prepare ReCoRD examples for sequence classification.
    Each row corresponds to one candidate entity with a True/False label.
    """
    texts_a = []
    texts_b = []
    labels = []
    original_idxs = []
    entity_idxs = []
    entities = []
    answers = []

    has_labels = "answers" in examples
    answers_list = examples["answers"] if has_labels else [[] for _ in range(len(examples["passage"]))]

    for ex_idx in range(len(examples["passage"])):
        passage = examples["passage"][ex_idx]
        query = examples["query"][ex_idx]
        curr_entities = examples["entities"][ex_idx]
        curr_answers = answers_list[ex_idx]

        for ent_idx, entity in enumerate(curr_entities):
            # Input is passage + query with entity filled in
            texts_a.append(passage)
            texts_b.append(query.replace("@placeholder", entity))

            original_idxs.append(frozendict(examples["idx"][ex_idx]))
            entity_idxs.append(ent_idx)
            entities.append(entity)
            answers.append(curr_answers)

            if has_labels:
                labels.append(1.0 if entity in curr_answers else 0.0)

    # Add extra info
    result = {}
    result["original_idx"] = original_idxs
    result["entity_idx"] = entity_idxs
    result["entity"] = entities
    result["answers"] = answers
    return result


def process_record_logits(df):

    eval_dataset = load_dataset("super_glue", "record", split="validation")
    eval_dataset = eval_dataset.map(preprocess_record, batched=True, remove_columns=eval_dataset.column_names)

    gold_answers = {frozendict(example["original_idx"]): example["answers"] for example in
                    eval_dataset}
    original_idxs = eval_dataset["original_idx"]
    entities = eval_dataset["entity"]
    original_idxs = [frozendict(oi) for oi in original_idxs]
    unique_idxs = sorted(set(original_idxs), key=lambda x: (x['passage'], x['query']))

    logits = df['prediction'].tolist()
    if len(logits) != len(original_idxs):
        raise ValueError(f"Number of logits {len(logits)} does not match number of examples {len(original_idxs)}")
    probs = F.sigmoid(torch.tensor(logits))
    # Create a dictionary to store the most likely prediction for each idx
    pred_dict = {}
    prob_dict = {}  # Track highest probability for each idx
    for i, o_idx in enumerate(original_idxs):
        if probs[i] > 0.5:
            if o_idx not in pred_dict or probs[i] > prob_dict[o_idx]:
                pred_dict[o_idx] = entities[i]  # Store entity with highest probability
                prob_dict[o_idx] = probs[i]  # Update highest probability

    # Format predictions: one prediction_text per idx
    predictions = [
        {
            "idx": frozendict(dict(idx)),
            "prediction_text": pred_dict.get(idx, "")  # Use empty string if no prediction
        }
        for idx in unique_idxs
    ]

    # Format references: get gold answers from dataset
    references = [
        {
            "idx": frozendict(dict(idx)),
            "answers": gold_answers.get(idx, [])
        }
        for idx in unique_idxs
    ]

    record_df = pd.DataFrame.from_dict({
        'label': references,
        'prediction': predictions,
    })

    return record_df




def process_record_predictions(df):

    dataset_key = 'super_glue__record'
    if dataset_key not in dataset_cache:
        eval_dataset = load_dataset("super_glue", "record", split="validation")
        eval_dataset = eval_dataset.map(preprocess_record, batched=True, remove_columns=eval_dataset.column_names)
        dataset_cache[dataset_key] = eval_dataset
    else:
        eval_dataset = dataset_cache[dataset_key]

    gold_answers = {frozendict(example["original_idx"]): example["answers"] for example in
                    eval_dataset}
    original_idxs = eval_dataset["original_idx"]
    original_idxs = [frozendict(oi) for oi in original_idxs]
    unique_idxs = sorted(set(original_idxs), key=lambda x: (x['passage'], x['query']))

    predictions = df['prediction'].tolist()
    if len(predictions) != len(unique_idxs):
        raise ValueError(f"Number of predictions {len(predictions)} does not match number of examples {len(unique_idxs)}")

    # Format predictions: one prediction_text per idx
    predictions = [
        frozendict({
            "idx": dict(idx),
            "prediction_text": pred,
        })
        for pred, idx in zip(predictions, unique_idxs)
    ]

    # Format references: get gold answers from dataset
    references = [
        frozendict({
            "idx": dict(idx),
            "answers": gold_answers.get(idx, [])
        })
        for idx in unique_idxs
    ]

    record_df = pd.DataFrame.from_dict({
        'label': references,
        'prediction': predictions,
    })

    return record_df

all_glue_tasks = ('wnli', 'stsb', 'sst2', 'rte', 'qqp', 'qnli', 'mrpc', 'cola', 'mnli', 'mnli-mm')
all_super_glue_tasks = ('boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc.fixed', 'axb', 'axg')

def read_glue_results(base_dir, seeds=(0, 1, 2), tasks=all_glue_tasks, glue_type='glue'):
    print('Read GLUE results for ', base_dir)

    df_data = []
    for task in tasks:
        print('Task:', task)
        raw_task = task
        if task == 'mnli':
            raw_task = 'mnli_matched'
        elif task == 'mnli-mm':
            raw_task = 'mnli_mismatched'

        dataset_key = f'{glue_type}_{raw_task}'
        if dataset_key in dataset_cache:
            raw_datasets = dataset_cache[dataset_key]
        else:
            raw_datasets = load_dataset(glue_type, raw_task)
            dataset_cache[dataset_key] = raw_datasets

        split = 'validation'
        if task in {'axb', 'axg'}:
            split = 'test'

        if split not in raw_datasets:
            raise ValueError(f'Split {split} not found in dataset {glue_type} {raw_task}')

        validation_data = raw_datasets[split]
        if task == 'record':
            label_type = None
        else:
            label_type = validation_data.features['label']

        for seed in seeds:
            if task == 'mnli-mm':
                file = f'{base_dir}/{seed}/mnli/eval_results_{task}.txt'
            elif task in {'axb', 'axg'}:
                file = f'{base_dir}/{seed}/cb/eval_results_{task}.txt'
            else:
                file = f'{base_dir}/{seed}/{task}/eval_results_{task}.txt'

            df = pd.read_csv(file, sep='\t')

            if len(df) == 0:
                print('WARNING: Empty results for', file)
                # rename folder to mark for recomputation
                #folder = os.path.dirname(file)
                #new_folder = folder + '_error'
                #os.rename(folder, new_folder)
                raise ValueError(f'Empty results for {file}')


            if task == 'record':
                if df['prediction'].dtype == float:
                    df = process_record_logits(df)
                else:
                    df = process_record_predictions(df)
            #elif task == 'multirc':
            #    labels = []
            #    for lab in validation_data["label"]:
            #        if isinstance(lab, str):
            #            labels.append(label_to_id.get(lab, 0) if label_to_id else int(lab == "True"))
            #        else:
            #            labels.append(int(lab))
            #    df['label'] = labels
            else:
                labels = validation_data['label']
                df['label'] = labels

            if task in ['axb', 'axg']:
                label_mapping = lambda x: 0 if x == 'entailment' else 1
                df['prediction'] = df['prediction'].apply(label_mapping)
            elif hasattr(label_type, 'str2int') and 'True' not in label_type.names:
                label_mapping = label_type.str2int
                df['prediction'] = df['prediction'].apply(label_mapping)

            df['idx'] = df.index
            df['task'] = task
            df['seed'] = seed

            df_data.append(df)

    df = pd.concat(df_data)

    return df


def calculate_symmetric_ci(bootstrap_results, confidence_level=0.95):
    """
    Calculate the symmetric confidence intervals for the bootstrap results.
    """
    mean_score = np.mean(bootstrap_results)
    std_error = np.std(bootstrap_results, ddof=1)
    margin_of_error = norm.ppf(1 - (1 - confidence_level) / 2) * std_error

    lower_bound = mean_score - margin_of_error
    upper_bound = mean_score + margin_of_error

    return mean_score, lower_bound, upper_bound, margin_of_error


def test_needed_samples(input):
    """
    Test how many bootstrap samples are needed to stabilize the confidence intervals.
    Compares confidence intervals for smaller subsets (e.g., n=10, 25, 50, etc.) of bootstrapped results.

    Parameters:
    - input: path to the GLUE results input data.
    - full_n_samples: number of bootstrap samples to use for the full bootstrap.
    - sample_sizes: list of sample sizes to test (e.g., [10, 25, 50]).
    """

    bootstrap_results = glue_bootstrap(input, n_samples=1000)
    full_n_samples = len(bootstrap_results)
    sample_sizes = [[10*10**i, 25*10**i, 50*10**i] for i in range(0, np.log10(full_n_samples).astype(int))]
    sample_sizes = [item for sublist in sample_sizes for item in sublist if item <= full_n_samples]
    if not full_n_samples in sample_sizes:
        sample_sizes.append(full_n_samples)

    # Calculate the confidence intervals for the full number of bootstrap samples
    full_mean, full_lower, full_upper, full_margin_of_error = calculate_symmetric_ci(bootstrap_results, 0.95)
    print(f"Full Bootstrap ({full_n_samples} samples):")
    print(f"  Mean: {full_mean:.4f}, Symmetric 95% CI: {full_lower:.2f} - {full_upper:.2f}, ± {full_margin_of_error:.4f}")

    # Now calculate for the smaller sample sizes
    for size in sample_sizes:
        if size <= len(bootstrap_results):
            sampled_results = bootstrap_results[:size]  # Take the first 'size' results
            mean, lower, upper, margin_of_error = calculate_symmetric_ci(sampled_results, 0.95)
            print(f"\nSample Size = {size}:")
            print(f"  Mean: {mean:.4f}, Symmetric 95% CI: {lower:.2f} - {upper:.2f}, ± {margin_of_error:.4f}")
        else:
            print(f"Skipping sample size {size} as it exceeds the total number of samples available ({len(bootstrap_results)}).")

def glue_bootstrap(input, n_samples=1000, confidence_level=0.95, suffix='', glue_type='glue'):
    assert glue_type in ['glue', 'super_glue'], "glue_type must be either 'glue' or 'super_glue'"


    output = f"{input}/{glue_type}_results_{n_samples}_{confidence_level}{suffix}.json"

    tasks = all_glue_tasks if glue_type == 'glue' else all_super_glue_tasks
    if 'llama' in input.lower():
        tasks = [t for t in tasks if t not in {'stsb', 'axb', 'axg'}]

    bootstrap_results = None
    if os.path.exists(output):
        with open(output, "r") as f:
            bootstrap_results = json.load(f)

    else:
        df_output = f'{input}/{glue_type}_results{suffix}.csv'
        if False and os.path.exists(df_output):
            df = pd.read_csv(df_output)
        else:
            df = read_glue_results(input, tasks=tasks, glue_type=glue_type)
            df.to_csv(df_output, index=False)

        # Run bootstrapping
        print(f"Running bootstrapping on GLUE scores for {input}...")
        if glue_type == 'glue':
            bootstrap_results = bootstrap_glue_scores(df, n_samples=n_samples, seed=42)
        else:
            bootstrap_results = bootstrap_super_glue_scores(df, n_samples=n_samples, seed=42)

        # Save bootstrapped scores to a JSON file
        with open(output, "w") as f:
            json.dump(bootstrap_results, f)

    mean_bootstrap_results = [result['mean'] for result in bootstrap_results['bootstrap']]
    print(f"Bootstrapped GLUE scores (first 10): {mean_bootstrap_results[:10]}")
    print(f"Mean Bootstrapped GLUE score: {np.mean(mean_bootstrap_results):.2f}")
    # 0.95 and 0.99 confidence intervals
    print(f"95% CI: {np.percentile(mean_bootstrap_results, 2.5):.2f} - {np.percentile(mean_bootstrap_results, 97.5):.2f}")
    print(f"99% CI: {np.percentile(mean_bootstrap_results, 0.5):.2f} - {np.percentile(mean_bootstrap_results, 99.5):.2f}")

    # Step 1: Compute the mean
    mean_score = np.mean(mean_bootstrap_results)

    # Step 2: Compute the standard error (SE)
    std_error = np.std(mean_bootstrap_results, ddof=1)  # Use ddof=1 for sample std dev
    margin_of_error = norm.ppf(1 - (1 - confidence_level) / 2) * std_error

    # Step 3: Calculate symmetric confidence intervals
    lower_bound = mean_score - margin_of_error
    upper_bound = mean_score + margin_of_error
    # Print results
    print(f"Mean Bootstrapped GLUE score: {mean_score:.4f}")
    print(f"Symmetric 95% CI: {lower_bound:.2f} - {upper_bound:.2f}")
    print(f"Mean Bootstrapped GLUE score: {mean_score:.4f} ± {margin_of_error:.4f}")

    return bootstrap_results

def glue_bootstrap_scores(input, n_samples=1000, confidence_level=0.95, glue_type='glue'):
    bootstrap_results = glue_bootstrap(input, n_samples, confidence_level, glue_type=glue_type)

    if isinstance(bootstrap_results, dict):
        bootstrap_results = [x['mean'] for x in bootstrap_results['bootstrap']]

    # Step 1: Compute the mean
    mean_score = np.mean(bootstrap_results)

    # Step 2: Compute the standard error (SE)
    std_error = np.std(bootstrap_results, ddof=1)  # Use ddof=1 for sample std dev
    margin_of_error = norm.ppf(1 - (1 - confidence_level) / 2) * std_error

    return mean_score, margin_of_error

def calculate_bootstrap_for_all_models(base_path, suffix=''):
    # find all folders in base_path that start with 'glue_m'
    models = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and 'glue_m' in f]
    errors = []

    random.shuffle(models)

    glue_type = 'super_glue' if 'super_glue' in base_path else 'glue'
    #models = [m for m in models if m.endswith('super_glue_m-RobertaForSequenceClassification_c-cda_c-roberta-large_t-gender_s-0')]
    for model in models:
        try:
            glue_bootstrap(f'{base_path}/{model}', suffix=suffix, glue_type=glue_type)
        except Exception as e:
            print(f"Error for model {model}: {e}")
            errors.append(model)

    print(f"Errors: {errors}")


if __name__ == "__main__":
    while True:
        calculate_bootstrap_for_all_models('results/glue')
        calculate_bootstrap_for_all_models('results/super_glue')

        print("Sleeping for 10 seconds before checking again...")
        time.sleep(1000)
