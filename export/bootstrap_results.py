import json
import os.path

import numpy as np
from datasets import load_dataset
from scipy.stats import norm
from tabulate import tabulate


class ConfidenceMetric:
    def __init__(self, score,
                 margin=None,
                 baseline_diff_command=None,
                 reference_score=None,
                 print_baseline_diff=True,
                 print_margin=True,
                 ):
        self.score = score
        self.margin = margin
        self.baseline_diff_command = baseline_diff_command
        self.reference_score = reference_score
        self.print_baseline_diff = print_baseline_diff
        self.print_margin = print_margin

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.score == other
        if isinstance(other, ConfidenceMetric):
            return self.score == other.score
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)


    def print(self, fmt='.2f', baseline=None, is_best_metric=False, is_significant=False):
        s = rf'{self.score:{fmt}}'

        if is_best_metric:
            if is_significant:
                s = rf'\boldsymbol{{\mathit{{{s}}}}}'
            else:
                s = rf'\mathbf{{{s}}}'

        if is_significant:
            s = rf'\mathit{{{s + self.print_reference_score_match()}}}'
        else:
            s += self.print_reference_score_match()

        if self.print_margin and self.margin is not None:
            if is_significant:
                s += rf' \pm \mathit{{{self.margin:{fmt}}}}'
            else:
                s += rf' \pm {self.margin:{fmt}}'


        s = f'${s}$'

        if baseline is not None and self.print_baseline_diff and self.baseline_diff_command is not None:
            baseline_diff = self.baseline_diff_command(self.score, baseline, fmt, is_significant=is_significant, is_best_metric=is_best_metric)
            s = f'{baseline_diff} {s}'

        if is_best_metric:
            s = rf'\!\!{s}'

        return s

    def print_reference_score_match(self):
        if self.margin is None:
            return ''

        lower = self.score - self.margin
        upper = self.score + self.margin
        if self.reference_score is None or lower <= self.reference_score <= upper:
            return ''

        print(f'Reference score {self.reference_score:.3f} not in range {lower} - {upper}')
        return f'*!({self.reference_score})'

class BestAtFiftyConfidenceMetric(ConfidenceMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(baseline_diff_command=self.best_at_fifty, *args, **kwargs)

    def best_at_fifty(self, score, baseline, fmt='.2f', is_significant=False, is_best_metric=False):
        score_diff = abs(50 - score)
        baseline_diff = abs(50 - baseline)

        change = abs(score_diff - baseline_diff)
        change_fmt = f'{change:{fmt}}'

        if is_significant:
            change_fmt = rf'\mathit{{{change_fmt}}}'

        if is_best_metric:
            change_fmt = rf'\boldsymbol{{{change_fmt}}}'

        if score_diff == baseline_diff:
            return ""
        elif score_diff < baseline_diff:
            return r'\da{$' + change_fmt + '$}'
        else: # score > baseline:
            return r'\ua{$' + change_fmt + '$}'

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return abs(self.score - 50) > abs(other - 50)
        if isinstance(other, ConfidenceMetric):
            return abs(self.score - 50) > abs(other.score - 50)
        return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

class BestAtZeroConfidenceMetric(ConfidenceMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(baseline_diff_command=self.best_at_zero, *args, **kwargs)

    def best_at_zero(self, score, baseline, fmt='.2f', is_significant=False, is_best_metric=False):
        score_diff = abs(score)
        baseline_diff = abs(baseline)

        change = abs(score_diff - baseline_diff)
        change_fmt = f'{change:{fmt}}'

        if is_significant:
            change_fmt = rf'\mathit{{{change_fmt}}}'

        if is_best_metric:
            change_fmt = rf'\boldsymbol{{{change_fmt}}}'

        if score_diff == baseline_diff:
            return ""
        elif score_diff < baseline_diff:
            return r'\da{$' + change_fmt + '$}'
        else: # score > baseline:
            return r'\ua{$' + change_fmt + '$}'

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return abs(self.score) > abs(other)
        if isinstance(other, ConfidenceMetric):
            return abs(self.score) > abs(other.score)
        return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


class BestAtLargestConfidenceMetric(ConfidenceMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(baseline_diff_command=self.best_at_largest, *args, **kwargs)

    def best_at_largest(self, score, baseline, fmt='.2f', is_significant=False, is_best_metric=False):
        change = abs(score - baseline)
        change_fmt = f'{change:{fmt}}'

        if is_significant:
            change_fmt = rf'\mathit{{{change_fmt}}}'

        if is_best_metric:
            change_fmt = rf'\boldsymbol{{{change_fmt}}}'

        if score == baseline:
            return ""
        elif score < baseline:
            return r'\dab{$' + change_fmt + '$}'
        else: # score > baseline:
            return r'\uag{$' + change_fmt + '$}'

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.score < other
        if isinstance(other, ConfidenceMetric):
            return self.score < other.score
        return NotImplemented
    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

def fmt_eq(float_1, float_2, fmt):
    return f'{float_1:{fmt}}' == f'{float_2:{fmt}}'

class ModelResults:
    def __init__(self, base_model, scores, variant=None, base_model_pretty=None, fmt='.2f'):
        self.model = base_model
        self.variant = variant
        self.base_model_pretty = base_model_pretty or base_model
        self.fmt = fmt
        self.scores = scores
        self.baseline = None

    def set_baseline(self, baseline):
        self.baseline = baseline

    def __setitem__(self, key, value):
        self.scores[key] = value

    def __getitem__(self, key):
        return self.scores[key]

    def __contains__(self, item):
        return item in self.scores

    def print(self, *scores, best_metrics=None):
        if self.variant is BASE:
            result = self.base_model_pretty
        else:
            result = rf'\, + {self.variant}'

        for metric in scores:
            if self.scores.get(metric) is None:
                result += ' & -'
            else:
                baseline_value = self.baseline[metric].score if (self.baseline is not None and metric in self.baseline) else None
                #fmt = ".3f" if metric == SEAT else self.fmt
                fmt = self.fmt
                is_best_metric = best_metrics is not None and metric in best_metrics and fmt_eq(self.scores[metric].score, best_metrics[metric], fmt)
                is_significant = self.is_significant(metric)
                value = self.scores[metric].print(fmt, baseline_value, is_best_metric=is_best_metric, is_significant=is_significant)
                if self.scores[metric].score < 0:
                    value = f'\!\!\! {value}'
                result += rf' & {value}'

        return result

    def is_significant(self, metric):
        if self.baseline is None:
            return False

        if metric not in self.baseline:
            return False

        # check if there is an intersection between the confidence intervals of baseline and score
        metric_score = self.scores[metric]
        baseline_score = self.baseline[metric]
        if metric_score.margin is None or baseline_score.margin is None:
            return False

        lower_score = metric_score.score - metric_score.margin
        upper_score = metric_score.score + metric_score.margin
        lower_baseline = baseline_score.score - baseline_score.margin
        upper_baseline = baseline_score.score + baseline_score.margin

        return lower_score > upper_baseline or upper_score < lower_baseline

class Variant:
    def __init__(self, model, suffix, pretty, model_prefix='', model_suffix=''):
        self.model = model
        self.suffix = suffix
        self.pretty = pretty
        self.model_prefix = model_prefix
        self.model_suffix = model_suffix

    def __str__(self):
        return self.pretty

BASE = Variant('', '', 'Base Model')
GRADIEND_BPI = Variant('', '-N', r'\gradiendbpi')
GRADIEND_FPI = Variant('', '-F', r'\gradiendfpi')
GRADIEND_MPI = Variant('', '-M', r'\gradiendmpi')
CDA = Variant('CDA', '-gender_s-0', r'\cda')
DROPOUT = Variant('Dropout', '-gender_s-0', r'\dropout')
INLP = Variant('INLP', '', r'\inlp')
RLACE = Variant('rlace_INLP', '', r'\rlace')
LEACE = Variant('leace_INLP', '', r'\leace')
SELF_DEBIAS = Variant('SelfDebias', '', r'\selfdebias')
SENTENCE_DEBIAS = Variant('SentenceDebias', '', r'\sentencedebias')

GRADIEND_BPI_INLP = Variant('INLP', '-N', r'\gradiendbpi\ + \inlp')
GRADIEND_BPI_RLACE = Variant('rlace_INLP', '-N', r'\gradiendbpi\ + \rlace')
GRADIEND_BPI_LEACE = Variant('leace_INLP', '-N', r'\gradiendbpi\ + \leace')
GRADIEND_BPI_SENTENCE_DEBIAS = Variant('SentenceDebias', '-N', r'\gradiendbpi\ + \sentencedebias \!\!')
CDA_INLP = Variant('INLP', '', r'\cda\, + \inlp', model_prefix='cda_c-', model_suffix='_t-gender_s-0')
CDA_SENTENCE_DEBIAS = Variant('SentenceDebias', '', r'\cda\, + \sentencedebias', model_prefix='cda_c-', model_suffix='_t-gender_s-0')
DROPOUT_INLP = Variant('INLP', '', r'\dropout \, + \inlp', model_prefix='dropout_c-', model_suffix='_s-0')
DROPOUT_SENTENCE_DEBIAS = Variant('SentenceDebias', '', r'\dropout \, + \sentencedebias', model_prefix='dropout_c-', model_suffix='_s-0')

VARIANTS = [
    GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI, GRADIEND_BPI_INLP, GRADIEND_BPI_RLACE, GRADIEND_BPI_LEACE, GRADIEND_BPI_SENTENCE_DEBIAS, CDA_INLP, DROPOUT_INLP,
    CDA, DROPOUT, INLP, RLACE, LEACE, SELF_DEBIAS, SENTENCE_DEBIAS
]
STRUCTURED_VARIANTS = [
    [BASE],
    [GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI],
    [CDA, DROPOUT, INLP, RLACE, LEACE, SELF_DEBIAS, SENTENCE_DEBIAS],
    [GRADIEND_BPI_INLP, GRADIEND_BPI_RLACE, GRADIEND_BPI_LEACE, GRADIEND_BPI_SENTENCE_DEBIAS],
    [CDA_INLP, DROPOUT_INLP, DROPOUT_SENTENCE_DEBIAS, CDA_SENTENCE_DEBIAS],
]

STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS = [
    [BASE],
    [GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI],
    [CDA, DROPOUT, INLP, RLACE, LEACE, SENTENCE_DEBIAS],
    [GRADIEND_BPI_INLP, GRADIEND_BPI_RLACE, GRADIEND_BPI_LEACE, GRADIEND_BPI_SENTENCE_DEBIAS],
    [CDA_INLP, DROPOUT_INLP, DROPOUT_SENTENCE_DEBIAS, CDA_SENTENCE_DEBIAS],
]


class Metric:
    def __init__(self, id, pretty_name, best=max):
        self.id = id
        self.pretty_name = pretty_name
        self.best = best

    def __str__(self):
        return self.pretty_name

    def __eq__(self, other):
        return self.id == other

def best_at_fifty(values, default=None):
    if not values:
        return default

    best = None
    best_diff = float('inf')
    for value in values:
        diff = abs(50 - value)
        if diff < best_diff:
            best = value
            best_diff = diff

    return best

def best_at_zero(values, default=None):
    if not values:
        return default

    best = None
    best_diff = float('inf')
    for value in values:
        diff = abs(value)
        if diff < best_diff:
            best = value
            best_diff = diff

    return best

SS = Metric('stereoset', '\\textbf{\\acrshort{ss}} (\\%) \\bestatfiftytiny', best_at_fifty)
LMS = Metric('lms', '\\textbf{\\acrshort{lms}} (\\%) $\\uparrow$')
GLUE = Metric('glue', '\\textbf{\\acrshort{glue}} (\\%) $\\uparrow$')
GLUE_AVG = Metric('glue', '\\textbf{Average} $\\uparrow$')
SEAT = Metric('seat', '\\textbf{\\acrshort{seat}} \\bestatzerotiny', best_at_zero)
SEAT_AVG = Metric('seat', '\\textbf{Absolute Average} \\bestatzerotiny', best_at_zero)
CROWS = Metric('crows', '\\textbf{CrowS} (\\%) \\bestatfiftytiny', best_at_fifty)

GLUE_COLA = Metric('glue_cola', '\\textbf{CoLA}')
GLUE_MNLI = Metric('glue_mnli_mean', '\\textbf{MNLI}')
GLUE_MNLI_M = Metric('glue_mnli', '\!\!\\textbf{MNLI-M}')
GLUE_MNLI_MM = Metric('glue_mnli-mm', '\!\!\\textbf{MNLI-MM}')
GLUE_MRPC = Metric('glue_mrpc', '\\textbf{MRPC}')
GLUE_QNLI = Metric('glue_qnli', '\\textbf{QNLI}')
GLUE_QQP = Metric('glue_qqp', '\\textbf{QQP}')
GLUE_RTE = Metric('glue_rte', '\\textbf{RTE}')
GLUE_SST2 = Metric('glue_sst2', '\\textbf{SST-2}')
GLUE_STSB = Metric('glue_stsb', '\\textbf{STS-B}')
GLUE_WNLI = Metric('glue_wnli', '\\textbf{WNLI}')

SEAT_6 = Metric('seat-6', '\\textbf{SEAT-6}', best_at_zero)
SEAT_6B = Metric('seat-6b', '\\textbf{SEAT-6b}', best_at_zero)
SEAT_7 = Metric('seat-7', '\\textbf{SEAT-7}', best_at_zero)
SEAT_7b = Metric('seat-7b', '\\textbf{SEAT-7b}', best_at_zero)
SEAT_8 = Metric('seat-8', '\\textbf{SEAT-8}', best_at_zero)
SEAT_8b = Metric('seat-8b', '\\textbf{SEAT-8b}', best_at_zero)

models = {
    'bert-base-cased': 'Bert',
    'bert-large-cased': 'BertLarge',
    'distilbert-base-cased': 'Distilbert',
    'roberta-large': 'Roberta',
    'gpt2': 'GPT2',
}

model_type_pretty = {
    'Bert': r'\bertbase',
    'BertLarge': r'\bertlarge',
    'Distilbert': r'\distilbert',
    'Roberta': r'\roberta',
    'GPT2': r'\gpttwo',
}

def get_experiments(base_model, variant, metrics=None):
    model_type = models[base_model]
    base_model_pretty = model_type_pretty[model_type]
    variant_suffix = (variant.suffix if variant in [BASE, GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI, GRADIEND_BPI_INLP, GRADIEND_BPI_RLACE, GRADIEND_BPI_LEACE, GRADIEND_BPI_SENTENCE_DEBIAS, ] else "") + ("_t-gender" if variant in [CDA, SELF_DEBIAS, SENTENCE_DEBIAS, INLP, RLACE, LEACE, GRADIEND_BPI_INLP, GRADIEND_BPI_RLACE, GRADIEND_BPI_LEACE, GRADIEND_BPI_SENTENCE_DEBIAS, CDA_INLP, CDA_SENTENCE_DEBIAS, DROPOUT_INLP, DROPOUT_SENTENCE_DEBIAS] else "")

    model_head = 'LMHeadModel' if model_type in {'GPT2'} else 'ForMaskedLM'

    model_type_modified = model_type.removesuffix("Large") if variant in [GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI] else model_type
    crows = f'results/crows/crows_m-{variant.model}{model_type_modified}{model_head}_c-{variant.model_prefix}{base_model}{variant.model_suffix}{variant.suffix.removesuffix("-gender_s-0")}_t-gender.json'
    seat = f'results/seat/seat_m-{variant.model}{model_type.removesuffix("Large") if variant in [BASE] else model_type_modified}Model_c-{variant.model_prefix}{base_model}{variant.model_suffix}{variant_suffix}.json'
    stereoset_data = 'results/stereoset.json'
    stereoset = f'stereoset_m-{variant.model}{model_type_modified}{model_head}_c-{variant.model_prefix}{base_model}{variant.model_suffix}{variant_suffix}{"_s-0" if variant in [DROPOUT, CDA] else ""}'
    glue = f'results/checkpoints/glue_m-{variant.model}{model_type.removesuffix("Large")}ForSequenceClassification_c-{variant.model_prefix}{base_model}{variant.model_suffix}{variant_suffix}/glue_results_1000_0.95.json'

    # load the files
    def load_if_exists(file):
        if os.path.isfile(file):
            try:
                with open(file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"File {file} not found")
        return None

    crows = load_if_exists(crows)
    seat = load_if_exists(seat)
    stereoset_data = load_if_exists(stereoset_data) or []
    glue = load_if_exists(glue)

    scores = {}
    # create crows metrics
    if (metrics is None or CROWS in metrics) and crows is not None:
        score = crows['ci_mean']
        margin = crows['ci_margin']
        ref_score = crows['score']
        scores['crows'] = BestAtFiftyConfidenceMetric(score, margin, reference_score=ref_score)

    # create seat metrics
    seat_metrics = [SEAT, SEAT_AVG, SEAT_6, SEAT_6B, SEAT_7, SEAT_7b, SEAT_8, SEAT_8b]
    if (metrics is None or any(s in metrics for s in seat_metrics)) and seat is not None:
        score = seat['ci_abs_mean']
        margin = seat['ci_margin']
        ref_score = seat['effect_size']
        scores['seat'] = BestAtZeroConfidenceMetric(score, margin, reference_score=ref_score)

        for suffix in ['6', '6b', '7', '7b', '8', '8b']:
            sub_seat = seat[f'sent-weat{suffix}']
            score = sub_seat['ci_mean']
            margin = sub_seat['ci_margin']
            ref_score = sub_seat['effect_size']
            scores[f'seat-{suffix}'] = BestAtZeroConfidenceMetric(score, margin, reference_score=ref_score, print_baseline_diff=False, print_margin=True)

    # create stereoset metrics
    if (metrics is None or SS in metrics or LMS in metrics) and  stereoset_data is not None and stereoset in stereoset_data:
        lms_data = stereoset_data[stereoset]['intrasentence']['overall']
        stereoset_data = stereoset_data[stereoset]['intrasentence']['gender']

        score = stereoset_data['SS Score_ci_mean']
        margin = stereoset_data['SS Score_ci_margin']
        ref_score = stereoset_data['SS Score']
        scores['stereoset'] = BestAtFiftyConfidenceMetric(score, margin, reference_score=ref_score)

        score = lms_data['LM Score_ci_mean']
        margin = lms_data['LM Score_ci_margin']
        ref_score = lms_data['LM Score']
        scores['lms'] = BestAtLargestConfidenceMetric(score, margin, reference_score=ref_score)
    elif stereoset not in stereoset_data:
        print(f"Missing stereoset data for {stereoset}")

    # create glue metrics
    glue_sub_metrics = [GLUE_COLA, GLUE_MNLI, GLUE_MRPC, GLUE_QNLI, GLUE_QQP, GLUE_RTE, GLUE_SST2, GLUE_STSB, GLUE_WNLI]
    glue_metrics = [GLUE, GLUE_AVG, GLUE_COLA, GLUE_MNLI, GLUE_MRPC, GLUE_QNLI, GLUE_QQP, GLUE_RTE, GLUE_SST2, GLUE_STSB, GLUE_WNLI]
    if (metrics is None or any(g in metrics for g in glue_metrics)) and  glue is not None:
        bootstraps = glue['bootstrap']
        confidence_level = 0.95
        keys = bootstraps[0].keys()

        glue_metric_ids = [metric.id for metric in glue_sub_metrics]
        if not all(id in keys for id in keys):
            raise ValueError(f"Expected 10 metrics, got {len(keys)}")

        for metric in keys:
            # Step 1: Compute the mean
            bootstrap_results = [score[metric] * 100 for score in bootstraps]
            mean_score = np.mean(bootstrap_results)

            # Step 2: Compute the standard error (SE)
            std_error = np.std(bootstrap_results, ddof=1)  # Use ddof=1 for sample std dev
            margin_of_error = norm.ppf(1 - (1 - confidence_level) / 2) * std_error

            ref_score = glue[metric] * 100
            if metric == 'mean':
                conf_metric = BestAtLargestConfidenceMetric(mean_score, margin_of_error, reference_score=ref_score)
                scores['glue'] = conf_metric
            else:
                conf_metric = BestAtLargestConfidenceMetric(mean_score, margin_of_error, print_baseline_diff=False, print_margin=False)
                scores[f'glue_{metric}'] = conf_metric

    results = ModelResults(base_model, scores, variant, base_model_pretty)
    return results


def read_results(models, metrics, variants):

    results = {}
    for model in models:
        model_results = {}
        base_results = get_experiments(model, BASE)
        model_results[BASE] = base_results

        for variant in variants:
            variant_results = get_experiments(model, variant, metrics=metrics)
            variant_results.set_baseline(base_results)
            model_results[variant] = variant_results

        results[model] = model_results

    return results


def print_table(models=list(models),
                metrics=[SS, SEAT, CROWS, LMS, GLUE],
                variants=STRUCTURED_VARIANTS,
                fmt='.2f'):
    all_variants = [variant for variant_group in variants for variant in variant_group]
    results = read_results(metrics=metrics, variants=all_variants, models=models)

    n = len(metrics)
    table = '\\begin{tabular}{l' + n * 'r' + '}\n\\toprule'
    table += '\\textbf{Model}'
    for metric in metrics:
        table += ' & ' + str(metric)

    table += '\\\\\n'

    for i, model in enumerate(models):
        print(model)
        model_results = results[model]
        table += '\\midrule\n'

        all_variants = [variant for variant_group in variants for variant in variant_group]
        best_metrics={} # maps metrics to their best value
        for metric in metrics:
            # Extract scores for the current metric from all valid variants
            scores = [
                model_results[variant][metric.id].score
                for variant in all_variants
                if metric.id in model_results[variant]
            ]
            # Update the best score for the metric, or None if no scores are found
            best_metrics[metric.id] = metric.best(scores, default=None)

        for j, variant_group in enumerate(variants):
            for variant in variant_group:
                model_results[variant].fmt = fmt
                variant_str = model_results[variant].print(*[metric.id for metric in metrics], best_metrics=best_metrics)
                table += variant_str + '\\\\\n'

            if j < len(variants) - 1:
                table += '\\lightcmidrule{1-' + str(n+1) + '}\n'

        if i < len(results) - 1:
            table += '\n'

    table += '\\bottomrule\n\\end{tabular}'
    print(table)

    return results

def print_main_table():
    print_table() # fmt='.1f'

def print_full_glue_table():
    data = print_table(
        metrics=[GLUE_COLA, GLUE_MNLI_M, GLUE_MNLI_MM, GLUE_MRPC, GLUE_QNLI, GLUE_QQP, GLUE_RTE, GLUE_SST2, GLUE_STSB, GLUE_WNLI, GLUE_AVG],
        variants=STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
    )

    # calculate the average confidence margin for the GLUE sub-metrics
    margins = [metric for metric in list(data.values())[0][BASE].scores.keys() if metric.startswith('glue')]
    max_margin = {task: max([result.scores[task].margin for model, model_result in data.items() for variant, result in model_result.items() if task in result.scores]) for task in margins}
    min_margin = {task: min([result.scores[task].margin for model, model_result in data.items() for variant, result in model_result.items() if task in result.scores]) for task in margins}

    task_mapping = {
        'glue': 'GLUE',
        'glue_wnli': 'WNLI',
        'glue_stsb': 'STSB',
        'glue_sst2': 'SST-2',
        'glue_rte': 'RTE',
        'glue_qqp': 'QQP',
        'glue_qnli': 'QNLI',
        'glue_mrpc': 'MRPC',
        'glue_cola': 'CoLA',
        'glue_mnli': 'MNLI-M',
        'glue_mnli-mm': 'MNLI-MM',
        'glue_mnli_mean': 'MNLI',
    }

    # Initialize task samples dictionary
    task_samples = {}

    # Iterate through tasks and fetch dataset sizes
    for task in task_mapping.keys():
        if task == 'glue' or task == 'glue_mnli_mean':
            continue

        raw_task = task.removeprefix('glue_')
        if raw_task == 'mnli':
            raw_task = 'mnli_matched'
        elif raw_task == 'mnli-mm':
            raw_task = 'mnli_mismatched'

        # Load dataset and count samples
        dataset = load_dataset("glue", raw_task)
        task_samples[task] = len(dataset['validation']) if 'validation' in dataset else 0

    task_samples['glue'] = sum([task_samples[task] for task in task_samples.keys()])

    table_data = [[task_mapping.get(task, ''), min_margin.get(task, '-'), max_margin.get(task, '-'), task_samples.get(task, -1)] for task in margins]
    # Sort table data by the number of samples (4th column)
    table_data.sort(key=lambda x: x[3], reverse=True)

    headers = ["Task", "Min (\%)", "Max (\%)", "\# Samples"]

    # Generate LaTeX table
    latex_table = tabulate(table_data, headers=headers, tablefmt="latex", floatfmt=".2f")

    print('\n'*3)
    print(latex_table)

def print_full_seat_table():
    print_table(
        variants=STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        metrics=[SEAT_6, SEAT_6B, SEAT_7, SEAT_7b, SEAT_8, SEAT_8b, SEAT_AVG],
        fmt='.2f',
    )

if __name__ == '__main__':
    print_main_table()
    #print_full_glue_table()
    #print_full_seat_table()