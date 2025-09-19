import json
import os.path

import numpy as np
from datasets import load_dataset
from scipy.stats import norm
from tabulate import tabulate


class ConfidenceMetric:
    def __init__(self,
                 score,
                 margin=None,
                 baseline_diff_command=None,
                 reference_score=None,
                 print_baseline_diff=True,
                 print_margin=True,
                 scaling_factor=1.0,
                 ):
        self.score = score
        self.margin = margin
        self.baseline_diff_command = baseline_diff_command
        self.reference_score = reference_score
        self.print_baseline_diff = print_baseline_diff
        self.print_margin = print_margin
        self.scaling_factor = scaling_factor

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.score == other
        if isinstance(other, ConfidenceMetric):
            return self.score == other.score
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)


    def print(self, fmt='.2f', baseline=None, is_best_metric=False, is_significant=False):
        s = rf'{self.score * self.scaling_factor:{fmt}}'

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
                s += rf'\tinymath{{\pm \mathit{{{self.margin * self.scaling_factor:{fmt}}}}}}'
            else:
                s += rf'\tinymath{{\pm {self.margin * self.scaling_factor:{fmt}}}}'


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
        score_diff = abs(50 - score) * self.scaling_factor
        baseline_diff = abs(50 - baseline) * self.scaling_factor

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


    def best_at_zero(self, score, baseline, fmt='.1f', is_significant=False, is_best_metric=False):
        score_diff = abs(score) * self.scaling_factor
        baseline_diff = abs(baseline) * self.scaling_factor

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

        if score < baseline:
            return r'\dab{$' + change_fmt + '$}'
        else: # score >= baseline:
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


class DualConfidenceMetric:
    def __init__(self,
                 score1, score2,
                 margin1=None, margin2=None,
                 baseline_diff_command=None,
                 reference_score1=None, reference_score2=None,
                 print_baseline_diff=True,
                 print_margin=True,
                 labels=("Score1", "Score2")):
        self.score1 = score1
        self.score2 = score2
        self.margin1 = margin1
        self.margin2 = margin2
        self.baseline_diff_command = baseline_diff_command
        self.reference_score1 = reference_score1
        self.reference_score2 = reference_score2
        self.print_baseline_diff = print_baseline_diff
        self.print_margin = print_margin
        self.labels = labels

    @property
    def score(self):
        return np.mean([self.score1, self.score2])

    @property
    def margin(self):
        if self.margin1 is None or self.margin2 is None:
            return None
        return np.mean([self.margin1, self.margin2])

    def __eq__(self, other):
        if isinstance(other, DualConfidenceMetric):
            return (self.score1, self.score2) == (other.score1, other.score2)
        return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def print(self, fmt='.2f', baseline=None, is_best_metric=False, is_significant=False):
        s1 = self._format_score(self.score1, self.margin1,
                                self.reference_score1, fmt,
                                is_best_metric, is_significant)
        s2 = self._format_score(self.score2, self.margin2,
                                self.reference_score2, fmt,
                                is_best_metric, is_significant)

        s = f'{s1}/{s2}'  # join with slash

        if baseline is not None and self.print_baseline_diff and self.baseline_diff_command is not None:
            baseline_diff = self.baseline_diff_command(
                (self.score1, self.score2),
                baseline, fmt,
                is_significant=is_significant,
                is_best_metric=is_best_metric
            )
            s = f'{baseline_diff} {s}'

        if is_best_metric:
            s = rf'\!\!{s}'

        return s

    def _format_score(self, score, margin, ref, fmt, is_best_metric, is_significant):
        s = rf'{score:{fmt}}'

        if is_best_metric:
            if is_significant:
                s = rf'\boldsymbol{{\mathit{{{s}}}}}'
            else:
                s = rf'\mathbf{{{s}}}'

        if is_significant:
            s = rf'\mathit{{{s + self._print_reference_score_match(score, margin, ref)}}}'
        else:
            s += self._print_reference_score_match(score, margin, ref)

        if self.print_margin and margin is not None:
            if is_significant:
                s += rf'\tinymath{{\pm \mathit{{{margin:{fmt}}}}}}'
            else:
                s += rf'\tinymath{{\pm {margin:{fmt}}}}'

        return f'${s}$'

    def _print_reference_score_match(self, score, margin, ref):
        if margin is None:
            return ''
        lower = score - margin
        upper = score + margin
        if ref is None or lower <= ref <= upper:
            return ''
        print(f'Reference score {ref:.3f} not in range {lower} - {upper}')
        return f'*!({ref})'


class BestAtLargestDualConfidenceMetric(DualConfidenceMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(baseline_diff_command=self.best_at_largest, *args, **kwargs)

    def best_at_largest(self, scores, baseline, fmt='.2f', is_significant=False, is_best_metric=False):
        """Compare based on sum of absolute improvements in both scores."""
        s1, s2 = scores
        b1, b2 = baseline

        change1 = abs(s1 - b1)
        change2 = abs(s2 - b2)
        total_change = change1 + change2

        change_fmt = f'{total_change:{fmt}}'

        if is_significant:
            change_fmt = rf'\mathit{{{change_fmt}}}'
        if is_best_metric:
            change_fmt = rf'\boldsymbol{{{change_fmt}}}'

        if (s1 + s2) < (b1 + b2):
            return r'\dab{$' + change_fmt + '$}'
        else:
            return r'\uag{$' + change_fmt + '$}'




def fmt_eq(float_1, float_2, fmt):
    return float_1 == float_2
    #return f'{float_1:{fmt}}' == f'{float_2:{fmt}}'

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
                result += ' & --'
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
    def __init__(self, model, suffix, pretty, model_prefix='', model_suffix='', changes_weights=False, is_post_processing=False):
        self.model = model
        self.suffix = suffix
        self.pretty = pretty
        self.model_prefix = model_prefix
        self.model_suffix = model_suffix
        self.changes_weights = changes_weights
        self.is_post_processing = is_post_processing

    def __str__(self):
        return self.pretty

BASE = Variant('', '', 'Base Model')
GRADIEND_BPI = Variant('', '-N', r'\gradiendbpi', changes_weights=True)
GRADIEND_FPI = Variant('', '-F', r'\gradiendfpi', changes_weights=True)
GRADIEND_MPI = Variant('', '-M', r'\gradiendmpi', changes_weights=True)

GRADIEND_RACE = Variant('', '-race_white_black_asian_combined', r'\gradiendrace', changes_weights=True)
GRADIEND_RACE1 = Variant('', '-v7-race_white_black', r'\gradiendracebw', changes_weights=True)
GRADIEND_RACE2 = Variant('', '-v7-race_white_asian', r'\gradiendraceaw', changes_weights=True)
GRADIEND_RACE3 = Variant('', '-v7-race_black_asian', r'\gradiendraceab', changes_weights=True)
GRADIEND_RELIGION = Variant('', '-religion_christian_muslim_jewish_combined', r'\gradiendreligion', changes_weights=True)
GRADIEND_RELIGION1 = Variant('', '-v7-religion_christian_muslim', r'\gradiendreligioncm', changes_weights=True)
GRADIEND_RELIGION2 = Variant('', '-v7-religion_christian_jewish', r'\gradiendreligioncj', changes_weights=True)
GRADIEND_RELIGION3 = Variant('', '-v7-religion_muslim_jewish', r'\gradiendreligionjm', changes_weights=True)


GRADIEND_MODELS = [GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI, GRADIEND_RACE, GRADIEND_RACE1, GRADIEND_RACE2, GRADIEND_RACE3, GRADIEND_RELIGION, GRADIEND_RELIGION1, GRADIEND_RELIGION2, GRADIEND_RELIGION3]

CDA = Variant('', '_t-{bias_type}_s-0', r'\cda', model_prefix='cda_c-', changes_weights=True)
DROPOUT = Variant('', '_s-0', r'\dropout', model_prefix='dropout_c-', changes_weights=True)
INLP = Variant('INLP', '', r'\inlp', is_post_processing=True)
RLACE = Variant('rlace_INLP', '', r'\rlace', is_post_processing=True)
LEACE = Variant('leace_INLP', '', r'\leace', is_post_processing=True)
SELF_DEBIAS = Variant('SelfDebias', '', r'\selfdebias', is_post_processing=True)
SENTENCE_DEBIAS = Variant('SentenceDebias', '', r'\sentencedebias', is_post_processing=True)

GRADIEND_BPI_INLP = Variant('INLP', '-N', r'\gradiendbpi\ + \inlp', changes_weights=True, is_post_processing=True)
GRADIEND_BPI_RLACE = Variant('rlace_INLP', '-N', r'\gradiendbpi\ + \rlace', changes_weights=True, is_post_processing=True)
GRADIEND_BPI_LEACE = Variant('leace_INLP', '-N', r'\gradiendbpi\ + \leace', changes_weights=True, is_post_processing=True)
GRADIEND_BPI_SENTENCE_DEBIAS = Variant('SentenceDebias', '-N', r'\gradiendbpi\ + \sentencedebias \!\!\!', changes_weights=True, is_post_processing=True)
CDA_INLP = Variant('INLP', '', r'\cda\, + \inlp', model_prefix='cda_c-', model_suffix='_t-{bias_type}_s-0', changes_weights=True, is_post_processing=True)
CDA_SENTENCE_DEBIAS = Variant('SentenceDebias', '', r'\cda\, + \sentencedebias', model_prefix='cda_c-', model_suffix='_t-{bias_type}_s-0', changes_weights=True, is_post_processing=True)
DROPOUT_INLP = Variant('INLP', '', r'\dropout \, + \inlp', model_prefix='dropout_c-', model_suffix='_s-0', changes_weights=True, is_post_processing=True)
DROPOUT_SENTENCE_DEBIAS = Variant('SentenceDebias', '', r'\dropout \, + \sentencedebias', model_prefix='dropout_c-', model_suffix='_s-0', changes_weights=True, is_post_processing=True)

VARIANTS = [
    GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI,
    CDA, DROPOUT, INLP, RLACE, LEACE, SELF_DEBIAS, SENTENCE_DEBIAS,
    GRADIEND_BPI_INLP, #GRADIEND_BPI_RLACE, GRADIEND_BPI_LEACE,
    GRADIEND_BPI_SENTENCE_DEBIAS,
    CDA_INLP, CDA_SENTENCE_DEBIAS, DROPOUT_INLP, DROPOUT_SENTENCE_DEBIAS,
]
STRUCTURED_VARIANTS = [
    [BASE],
    [GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI],
    [CDA, DROPOUT, INLP, RLACE, LEACE, SELF_DEBIAS, SENTENCE_DEBIAS],
    [GRADIEND_BPI_INLP, GRADIEND_BPI_SENTENCE_DEBIAS, CDA_INLP, DROPOUT_SENTENCE_DEBIAS, CDA_SENTENCE_DEBIAS, DROPOUT_INLP],
]

STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS = [
    [BASE],
    [GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI],
    [CDA, DROPOUT, INLP, RLACE, LEACE, SENTENCE_DEBIAS],
    [GRADIEND_BPI_INLP, GRADIEND_BPI_SENTENCE_DEBIAS, CDA_INLP, DROPOUT_SENTENCE_DEBIAS, CDA_SENTENCE_DEBIAS, DROPOUT_INLP],
]

RACE_STRUCTURED_VARIANTS_WITH_SELF_DEBIAS = [
    [BASE],
    [GRADIEND_RACE3, GRADIEND_RACE2, GRADIEND_RACE1],
    [CDA, DROPOUT, INLP, SELF_DEBIAS, SENTENCE_DEBIAS],
]

RACE_STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS = [[xx for xx in x if xx != SELF_DEBIAS] for x in RACE_STRUCTURED_VARIANTS_WITH_SELF_DEBIAS]


RELIGION_STRUCTURED_VARIANTS_WITH_SELF_DEBIAS = [
    [BASE],
    [GRADIEND_RELIGION2, GRADIEND_RELIGION1, GRADIEND_RELIGION3],
    [CDA, DROPOUT, INLP, SELF_DEBIAS, SENTENCE_DEBIAS],
]
RELIGION_STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS = [[xx for xx in x if xx != SELF_DEBIAS] for x in RELIGION_STRUCTURED_VARIANTS_WITH_SELF_DEBIAS]



class Metric:
    def __init__(self, id, pretty_name, best=max, suffix='', metric_type=None):
        self.id = id
        self.pretty_name = pretty_name
        self.best = best
        self.suffix = suffix
        self.metric_type = metric_type

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
SUPER_GLUE = Metric('super_glue', '\\textbf{\\acrshort{sglue}} (\\%) $\\uparrow$')
SUPER_GLUE_AVG = Metric('super_glue', '\\textbf{Average} $\\uparrow$')
SEAT = Metric('seat', '\\textbf{\\acrshort{seat}} $\\downarrow$', best_at_zero)
SEAT_AVG = Metric('seat', '\\textbf{Absolute Average} \\bestatzerotiny', best_at_zero)
CROWS = Metric('crows', '\\textbf{CrowS} (\\%) \\bestatfiftytiny', best_at_fifty)

GLUE_COLA = Metric('glue_cola', '\\textbf{CoLA}', metric_type='Mat. Cor.') # Matthews Correlation
GLUE_MNLI = Metric('glue_mnli_mean', '\\textbf{MNLI}', metric_type='Acc.')
GLUE_MNLI_M = Metric('glue_mnli', '\!\!\\textbf{MNLI-M}', metric_type='Acc.')
GLUE_MNLI_MM = Metric('glue_mnli-mm', '\!\!\\textbf{MNLI-MM}', metric_type='Acc.')
GLUE_MRPC = Metric('glue_mrpc', '\\textbf{MRPC}', metric_type='F1')
GLUE_QNLI = Metric('glue_qnli', '\\textbf{QNLI}', metric_type='Acc.')
GLUE_QQP = Metric('glue_qqp', '\\textbf{QQP}', metric_type='Acc.')
GLUE_RTE = Metric('glue_rte', '\\textbf{RTE}', metric_type='Acc.')
GLUE_SST2 = Metric('glue_sst2', '\\textbf{SST-2}', metric_type='Acc.')
GLUE_STSB = Metric('glue_stsb', '\\textbf{STS-B}', metric_type='Pear. Cor.')
GLUE_WNLI = Metric('glue_wnli', '\\textbf{WNLI}', metric_type='Acc.')

SUPER_GLUE_BOOLQ = Metric('super_glue_boolq', '\\textbf{BoolQ}', metric_type='Acc.')
SUPER_GLUE_CB = Metric('super_glue_cb', '\\textbf{CB}', metric_type='F1/Acc.')
SUPER_GLUE_COPA = Metric('super_glue_copa', '\\textbf{COPA}', metric_type='Acc.')
SUPER_GLUE_MULTIRC = Metric('super_glue_multirc', '\\textbf{MultiRC}', metric_type=r'$\text{F1}_\alpha$/EM')
SUPER_GLUE_RECORD = Metric('super_glue_record', '\\textbf{ReCoRD}', metric_type='F1/EM')
SUPER_GLUE_RTE = Metric('super_glue_rte', '\\textbf{RTE}', metric_type='Acc.')
SUPER_GLUE_WIC = Metric('super_glue_wic', '\\textbf{WiC}', metric_type='Acc.')
SUPER_GLUE_WSC = Metric('super_glue_wsc.fixed', '\\textbf{WSC}', metric_type='Acc.')
SUPER_GLUE_AXB = Metric('super_glue_axb', '\\textbf{AX-b}')
SUPER_GLUE_AXG = Metric('super_glue_axg', '\\textbf{AX-g}')

# race SEAT tests
SEAT_ABW1 = Metric('seat-abw1', '\\textbf{SEAT-ABW1} \\bestatzerotiny', best_at_zero, suffix='abw1')
SEAT_ABW2 = Metric('seat-abw2', '\\textbf{SEAT-ABW2} \\bestatzerotiny', best_at_zero, suffix='abw2')
SEAT_3 = Metric('seat-3', '\\textbf{SEAT-3} \\bestatzerotiny', best_at_zero, suffix='3')
SEAT_3b = Metric('seat-3b', '\\textbf{SEAT-3b} \\bestatzerotiny', best_at_zero, suffix='3b')
SEAT_4 = Metric('seat-4', '\\textbf{SEAT-4} \\bestatzerotiny', best_at_zero, suffix='4')
SEAT_5 = Metric('seat-5', '\\textbf{SEAT-5} \\bestatzerotiny', best_at_zero, suffix='5')
SEAT_5b = Metric('seat-5b', '\\textbf{SEAT-5b} \\bestatzerotiny', best_at_zero, suffix='5b')

# gender SEAT tests
SEAT_6 = Metric('seat-6', '\\textbf{SEAT-6} \\bestatzerotiny', best_at_zero, suffix='6')
SEAT_6b = Metric('seat-6b', '\\textbf{SEAT-6b} \\bestatzerotiny', best_at_zero, suffix='6b')
SEAT_7 = Metric('seat-7', '\\textbf{SEAT-7} \\bestatzerotiny', best_at_zero, suffix='7')
SEAT_7b = Metric('seat-7b', '\\textbf{SEAT-7b} \\bestatzerotiny', best_at_zero, suffix='7b')
SEAT_8 = Metric('seat-8', '\\textbf{SEAT-8} \\bestatzerotiny', best_at_zero, suffix='8')
SEAT_8b = Metric('seat-8b', '\\textbf{SEAT-8b} \\bestatzerotiny', best_at_zero, suffix='8b')

# religion SEAT tests
SEAT_REL1 = Metric('seat-rel1', '\\textbf{SEAT-REL1} \\bestatzerotiny', best_at_zero, suffix='rel1')
SEAT_REL1b = Metric('seat-rel1b', '\\textbf{SEAT-REL1b} \\bestatzerotiny', best_at_zero, suffix='rel1b')
SEAT_REL2 = Metric('seat-rel2', '\\textbf{SEAT-REL2} \\bestatzerotiny', best_at_zero, suffix='rel2')
SEAT_REL2b = Metric('seat-rel2b', '\\textbf{SEAT-REL2b} \\bestatzerotiny', best_at_zero, suffix='rel2b')


BERT = 'bert-base-cased'
BERT_LARGE = 'bert-large-cased'
DISTILBERT = 'distilbert-base-cased'
ROBERTA = 'roberta-large'
GPT2 = 'gpt2'
LLAMA = 'Llama-3.2-3B'
LLAMA_INSTRUCT = 'Llama-3.2-3B-Instruct'

models = {
    BERT: 'Bert',
    BERT_LARGE: 'BertLarge',
    DISTILBERT: 'Distilbert',
    ROBERTA: 'Roberta',
    GPT2: 'GPT2',
    LLAMA: 'Llama',
    LLAMA_INSTRUCT: 'LlamaInstruct',
}

model_type_pretty = {
    'Bert': r'\bertbase',
    'BertLarge': r'\bertlarge',
    'Distilbert': r'\distilbert',
    'Roberta': r'\roberta',
    'GPT2': r'\gpttwo',
    'Llama': r'\llama',
    'LlamaInstruct': r'\llamai',
}


SEAT_METRICS_BY_BIAS_TYPE = {
    'gender': [SEAT_6, SEAT_6b, SEAT_7, SEAT_7b, SEAT_8, SEAT_8b],
    'race': [SEAT_ABW1, SEAT_ABW2, SEAT_3, SEAT_3b, SEAT_4, SEAT_5, SEAT_5b],
    'religion': [SEAT_REL1, SEAT_REL1b, SEAT_REL2, SEAT_REL2b],
}


def get_experiments(base_model, variant, bias_type='gender', metrics=None):
    model_type = models[base_model]
    base_model_pretty = model_type_pretty[model_type]
    variant_suffix = (variant.suffix if variant in [CDA, DROPOUT, BASE, GRADIEND_BPI_INLP, GRADIEND_BPI_RLACE, GRADIEND_BPI_LEACE, GRADIEND_BPI_SENTENCE_DEBIAS] + GRADIEND_MODELS else "") + (f"_t-{bias_type}" if variant in [SELF_DEBIAS, SENTENCE_DEBIAS, INLP, RLACE, LEACE, GRADIEND_BPI_INLP, GRADIEND_BPI_RLACE, GRADIEND_BPI_LEACE, GRADIEND_BPI_SENTENCE_DEBIAS, DROPOUT_INLP, DROPOUT_SENTENCE_DEBIAS, CDA_INLP, CDA_SENTENCE_DEBIAS, GRADIEND_BPI, GRADIEND_MPI, GRADIEND_FPI, GRADIEND_RACE, GRADIEND_RACE1, GRADIEND_RACE2, GRADIEND_RACE3, GRADIEND_RELIGION, GRADIEND_RELIGION1, GRADIEND_RELIGION2, GRADIEND_RELIGION3] else "")

    variant_suffix = variant_suffix.replace('{bias_type}', bias_type)
    raw_variant_suffix = variant.suffix.replace('{bias_type}', bias_type)
    variant_model_suffix = variant.model_suffix.replace('{bias_type}', bias_type)

    model_head = 'LMHeadModel' if model_type in {'GPT2'} else ('ForCausalLM' if 'llama' in model_type.lower() else 'ForMaskedLM')

    base_model_full = base_model.replace('/', '-')

    seat_variant_suffix = variant_suffix.removesuffix(f'_t-{bias_type}') if variant in [GRADIEND_RELIGION, GRADIEND_RELIGION1, GRADIEND_RELIGION2, GRADIEND_RELIGION3, GRADIEND_RACE, GRADIEND_RACE1, GRADIEND_RACE2, GRADIEND_RACE3] else variant_suffix
    model_type_modified = model_type.removesuffix("Large") if variant in [GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI, GRADIEND_RACE1, GRADIEND_RACE2, GRADIEND_RACE3, GRADIEND_RACE, GRADIEND_RELIGION1, GRADIEND_RELIGION2, GRADIEND_RELIGION3, GRADIEND_RELIGION] else model_type
    crows = f'results/crows/crows_m-{variant.model}{model_type_modified}{model_head}_c-{variant.model_prefix}{base_model}{variant_model_suffix}{raw_variant_suffix.removesuffix(f"-{bias_type}_s-0")}_t-{bias_type}.json'
    crows = None
    seat = f'results/seat/{bias_type}/seat_m-{variant.model}{model_type.removesuffix("Large") if variant in [BASE, CDA, DROPOUT] else model_type_modified}Model_c-{variant.model_prefix}{base_model_full}{variant_model_suffix}{seat_variant_suffix}.json'
    seat_file = seat
    stereoset_data = 'results/stereoset.json'
    ss_variant_suffix = variant_suffix.removesuffix(f'_t-{bias_type}') if variant in {GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI, GRADIEND_RACE, GRADIEND_RACE1, GRADIEND_RACE2, GRADIEND_RACE3, GRADIEND_RELIGION3, GRADIEND_RELIGION, GRADIEND_RELIGION1, GRADIEND_RELIGION2} else variant_suffix
    glue_variant_suffix = variant_suffix.removesuffix(f'_t-{bias_type}') if variant in [CDA_INLP, CDA_SENTENCE_DEBIAS] else variant_suffix
    stereoset = f'stereoset_m-{variant.model}{model_type_modified}{model_head}_c-{variant.model_prefix}{base_model}{variant_model_suffix}{ss_variant_suffix}'
    glue = f'results/glue/glue_m-{variant.model}{model_type.removesuffix("Large")}ForSequenceClassification_c-{variant.model_prefix}{base_model}{variant_model_suffix}{glue_variant_suffix}/glue_results_1000_0.95.json'
    super_glue = f'results/super_glue/super_glue_m-{variant.model}{model_type.removesuffix("Large")}ForSequenceClassification_c-{variant.model_prefix}{base_model}{variant_model_suffix}{glue_variant_suffix}/super_glue_results_1000_0.95.json'
    super_glue_file = super_glue

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

    #crows = load_if_exists(crows)
    seat = load_if_exists(seat)
    stereoset_data = load_if_exists(stereoset_data) or []
    glue = load_if_exists(glue)
    super_glue = load_if_exists(super_glue)

    scores = {}
    # create crows metrics
    if (metrics is None or CROWS in metrics) and crows is not None:
        score = crows['ci_mean']
        margin = crows['ci_margin']
        ref_score = crows['score']
        scores['crows'] = BestAtFiftyConfidenceMetric(score, margin, reference_score=ref_score)

    # create seat metrics
    seat_metrics = SEAT_METRICS_BY_BIAS_TYPE[bias_type]
    all_seat_metrics = [SEAT, SEAT_AVG] + seat_metrics
    if (metrics is None or any(s in metrics for s in all_seat_metrics)) and seat is not None:
        score = seat['ci_abs_mean']
        margin = seat['ci_margin']
        ref_score = seat['effect_size']
        scores['seat'] = BestAtZeroConfidenceMetric(score, margin, reference_score=ref_score, scaling_factor=1)

        seat_suffixes = [s.suffix for s in seat_metrics]
        for suffix in seat_suffixes:
            key = f'sent-weat{suffix}'
            if suffix == 'abw1':
                key = 'sent-angry_black_woman_stereotype'
            elif suffix == 'abw2':
                key = 'sent-angry_black_woman_stereotype_b'
            elif suffix == 'rel1':
                key = 'sent-religion1'
            elif suffix == 'rel1b':
                key = 'sent-religion1b'
            elif suffix == 'rel2':
                key = 'sent-religion2'
            elif suffix == 'rel2b':
                key = 'sent-religion2b'

            if key in seat:
                sub_seat = seat[key]
                score = sub_seat['ci_mean']
                margin = sub_seat['ci_margin']
                ref_score = sub_seat['effect_size']

                ci_scores = sub_seat['ci_scores']
                if len(ci_scores) != 1000:
                    # rename the seat_file
                    print(f"WARNING: Expected 1000 bootstrap samples for {key} in {seat_file}, got {len(ci_scores)}")
                    outdated_seat_file = seat_file.replace('.json', '_outdated.json')
                    if os.path.isfile(seat_file):
                        os.rename(seat_file, outdated_seat_file)

                scores[f'seat-{suffix}'] = BestAtZeroConfidenceMetric(score, margin, reference_score=ref_score, print_baseline_diff=False, print_margin=True, scaling_factor=1)
            else:
                print(f"Missing SEAT data for {key} in {seat}")
                # rename the seat file to mark them as outdated
                if os.path.isfile(seat_file):
                    outdated_seat_file = seat_file.replace('.json', '_outdated.json')
                    os.rename(seat_file, outdated_seat_file)

    # create stereoset metrics
    if (metrics is None or SS in metrics or LMS in metrics) and  stereoset_data is not None and stereoset in stereoset_data:
        lms_data = stereoset_data[stereoset]['intrasentence']['overall']
        stereoset_data = stereoset_data[stereoset]['intrasentence'][bias_type]

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
    glue_metrics = [GLUE, GLUE_AVG] + glue_sub_metrics
    if (metrics is None or any(g in metrics for g in glue_metrics)) and  glue is not None:
        bootstraps = glue['bootstrap']
        confidence_level = 0.95
        keys = bootstraps[0].keys()

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

    # create super glue metrics
    super_glue_sub_metrics = [SUPER_GLUE_BOOLQ, SUPER_GLUE_CB, SUPER_GLUE_COPA, SUPER_GLUE_MULTIRC, SUPER_GLUE_RECORD, SUPER_GLUE_RTE, SUPER_GLUE_WIC, SUPER_GLUE_WSC]
    super_glue_metrics = [SUPER_GLUE, SUPER_GLUE_AVG] + super_glue_sub_metrics
    if (metrics is None or any(g in metrics for g in super_glue_metrics)) and  super_glue is not None:
        bootstraps = super_glue['bootstrap']
        confidence_level = 0.95
        keys = bootstraps[0].keys()

        super_glue_metric_ids = [metric.id.removeprefix('super_glue_') for metric in super_glue_sub_metrics]
        if not all(id in keys for id in super_glue_metric_ids):
            missing = [id for id in super_glue_metric_ids if id not in keys]
            raise ValueError(f"Expected 9 metrics, got {len(keys)}. Missing: {missing}", super_glue_file)

        def compute_bootstrap_scores(metric):
            bootstrap_results = [score[metric] * 100 for score in bootstraps]
            mean_score = np.mean(bootstrap_results)

            # Step 2: Compute the standard error (SE)
            std_error = np.std(bootstrap_results, ddof=1)  # Use ddof=1 for sample std dev
            margin_of_error = norm.ppf(1 - (1 - confidence_level) / 2) * std_error

            ref_score = super_glue[metric] * 100
            return mean_score, margin_of_error, ref_score


        for metric in keys:
            mean_score, margin_of_error, ref_score = compute_bootstrap_scores(metric)

            # Step 1: Compute the mean
            if metric == 'mean':
                conf_metric = BestAtLargestConfidenceMetric(mean_score, margin_of_error, reference_score=ref_score)
                scores['super_glue'] = conf_metric
            elif metric == 'cb':
                cb_f1, cb_f1_margin, cb_f1_ref = compute_bootstrap_scores('cb_f1')
                cb_acc, cb_acc_margin, cb_acc_ref = compute_bootstrap_scores('cb_acc')
                scores['super_glue_cb'] = BestAtLargestDualConfidenceMetric(cb_f1, cb_acc, margin1=cb_f1_margin, margin2=cb_acc_margin, reference_score1=cb_f1_ref, reference_score2=cb_acc_ref, print_baseline_diff=False, print_margin=False)
            elif metric == 'multirc':
                multirc_f1a, multirc_f1a_margin, multirc_f1a_ref = compute_bootstrap_scores('multirc_f1a')
                multirc_em, multirc_em_margin, multirc_em_ref = compute_bootstrap_scores('multirc_em')
                scores['super_glue_multirc'] = BestAtLargestDualConfidenceMetric(multirc_f1a, multirc_em, margin1=multirc_f1a_margin, margin2=multirc_em_margin, reference_score1=multirc_f1a_ref, reference_score2=multirc_em_ref, print_baseline_diff=False, print_margin=False)
            elif metric == 'record':
                record_f1, record_f1_margin, record_f1_ref = compute_bootstrap_scores('record_f1')
                record_em, record_em_margin, record_em_ref = compute_bootstrap_scores('record_em')
                scores['super_glue_record'] = BestAtLargestDualConfidenceMetric(record_f1, record_em, margin1=record_f1_margin, margin2=record_em_margin, reference_score1=record_f1_ref, reference_score2=record_em_ref, print_baseline_diff=False, print_margin=False)
            elif metric in {'cb_f1', 'cb_acc', 'multirc_f1a', 'multirc_em', 'record_f1', 'record_em'}:
                continue # these metrics are used above
            else:
                conf_metric = BestAtLargestConfidenceMetric(mean_score, margin_of_error, print_baseline_diff=False, print_margin=False)
                scores[f'super_glue_{metric}'] = conf_metric


    results = ModelResults(base_model, scores, variant, base_model_pretty)
    return results


def read_results(models, metrics, variants, bias_type='gender'):

    results = {}
    for model in models:
        model_results = {}
        base_results = get_experiments(model, BASE, bias_type=bias_type)
        model_results[BASE] = base_results

        for variant in variants:

            if 'llama' in model.lower() and variant in [CDA, DROPOUT, CDA_INLP, DROPOUT_INLP, CDA_SENTENCE_DEBIAS, DROPOUT_SENTENCE_DEBIAS]:
                # skip variants that are not applicable to llama models
                continue

            variant_results = get_experiments(model, variant, metrics=metrics, bias_type=bias_type)
            if variant != BASE:
                variant_results.set_baseline(base_results)
            model_results[variant] = variant_results

        results[model] = model_results

    return results


def print_table(models=list(models),
                metrics=[SS, SEAT, LMS, GLUE, SUPER_GLUE],
                variants=[BASE, GRADIEND_BPI, CDA, DROPOUT, INLP, RLACE, LEACE, SELF_DEBIAS, SENTENCE_DEBIAS],
                bias_type='gender',
                fmt='.2f',
                print_metric_types=False,
                ):

    if isinstance(variants[0], Variant):
        variants = [variants]
    all_variants = [variant for variant_group in variants for variant in variant_group]
    results = read_results(metrics=metrics, variants=all_variants, models=models, bias_type=bias_type)

    n = len(metrics)
    table = '\\begin{tabular}{l' + n * 'r' + '}\n\\toprule'
    table += '\\textbf{Model}'
    for metric in metrics:
        table += ' & ' + str(metric)

    table += '\\\\\n'

    if print_metric_types:
        table += '\\textbf{Metrics}'
        for metric in metrics:
            table += r' & \textbf{' + (metric.metric_type if metric.metric_type is not None else '') + '}'
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
                if variant in model_results and metric.id in model_results[variant]
            ]
            # Update the best score for the metric, or None if no scores are found
            best_metrics[metric.id] = metric.best(scores, default=None)

        for j, variant_group in enumerate(variants):
            for variant in variant_group:
                if variant in model_results:
                    model_results[variant].fmt = fmt
                    variant_str = model_results[variant].print(*[metric.id for metric in metrics], best_metrics=best_metrics)
                    table += variant_str + '\\\\\n'
                else:
                    print('WARNING: Missing results for model', model, 'variant', variant)

            if j < len(variants) - 1:
                table += '\\lightcmidrule{1-' + str(n+1) + '}\n'

        if i < len(results) - 1:
            table += '\n'

    table += '\\bottomrule\n\\end{tabular}'
    print(table)

    return results

def print_reduced_main_table():
    print_table(
        variants=[
            [BASE],
            [GRADIEND_BPI, GRADIEND_FPI, GRADIEND_MPI],
            [INLP, RLACE, LEACE, SELF_DEBIAS, SENTENCE_DEBIAS],
            [GRADIEND_BPI_INLP, GRADIEND_BPI_SENTENCE_DEBIAS]
        ],
        models=[GPT2, LLAMA, LLAMA_INSTRUCT],
    )

def print_full_main_table():
    print_table(
        variants=STRUCTURED_VARIANTS,
    ) # fmt='.1f'


def print_full_gender_glue_table():
    data = print_table(
        metrics=[GLUE_COLA, GLUE_MNLI_M, GLUE_MNLI_MM, GLUE_MRPC, GLUE_QNLI, GLUE_QQP, GLUE_RTE, GLUE_SST2, GLUE_STSB, GLUE_WNLI, GLUE_AVG],
        variants=STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        bias_type='gender',
        print_metric_types=True,
    )

def print_full_gender_super_glue_table():
    data = print_table(
        metrics=[SUPER_GLUE_BOOLQ, SUPER_GLUE_CB, SUPER_GLUE_COPA, SUPER_GLUE_MULTIRC, SUPER_GLUE_RECORD, SUPER_GLUE_RTE, SUPER_GLUE_WIC, SUPER_GLUE_WSC, SUPER_GLUE_AVG],
        variants=STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        bias_type='gender',
        print_metric_types=True,
    )

def print_full_race_glue_table():
    data = print_table(
        metrics=[GLUE_COLA, GLUE_MNLI_M, GLUE_MNLI_MM, GLUE_MRPC, GLUE_QNLI, GLUE_QQP, GLUE_RTE, GLUE_SST2, GLUE_STSB, GLUE_WNLI, GLUE_AVG],
        variants=RACE_STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        bias_type='race',
        print_metric_types=True,
    )


def print_full_race_super_glue_table():
    data = print_table(
        metrics=[SUPER_GLUE_BOOLQ, SUPER_GLUE_CB, SUPER_GLUE_COPA, SUPER_GLUE_MULTIRC, SUPER_GLUE_RECORD, SUPER_GLUE_RTE, SUPER_GLUE_WIC, SUPER_GLUE_WSC, SUPER_GLUE_AVG],
        variants=STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        bias_type='race',
        print_metric_types=True,
    )

def print_full_religion_glue_table():
    data = print_table(
        metrics=[GLUE_COLA, GLUE_MNLI_M, GLUE_MNLI_MM, GLUE_MRPC, GLUE_QNLI, GLUE_QQP, GLUE_RTE, GLUE_SST2, GLUE_STSB, GLUE_WNLI, GLUE_AVG],
        variants=RELIGION_STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        models=[DISTILBERT],
        bias_type='religion',
        print_metric_types=True,
    )

def print_full_religion_super_glue_table():
    data = print_table(
        metrics=[SUPER_GLUE_BOOLQ, SUPER_GLUE_CB, SUPER_GLUE_COPA, SUPER_GLUE_MULTIRC, SUPER_GLUE_RECORD, SUPER_GLUE_RTE, SUPER_GLUE_WIC, SUPER_GLUE_WSC, SUPER_GLUE_AVG],
        variants=STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        bias_type='religion',
        print_metric_types=True,
    )

def print_full_glue_table(glue_type='glue'):
    all_variants = STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS + RACE_STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS + RELIGION_STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS
    all_variants = [variant for variant_group in all_variants for variant in variant_group]
    variants = list(set(all_variants))
    task_mapping_glue = {
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

    task_mapping_super_glue = {
        'super_glue': 'SuperGLUE',
        'super_glue_wsc.fixed': 'WSC',
        'super_glue_wic': 'WiC',
        'super_glue_rte': 'RTE',
        'super_glue_record': 'ReCoRD',
        'super_glue_multirc': 'MultiRC',
        'super_glue_copa': 'COPA',
        'super_glue_cb': 'CB',
        'super_glue_boolq': 'BoolQ',

    }
    if glue_type == 'glue':
        metrics = [GLUE_COLA, GLUE_MNLI_M, GLUE_MNLI_MM, GLUE_MRPC, GLUE_QNLI, GLUE_QQP, GLUE_RTE, GLUE_SST2, GLUE_STSB, GLUE_WNLI, GLUE_AVG]
        task_mapping = task_mapping_glue
    elif glue_type == 'super_glue':
        metrics = [SUPER_GLUE_BOOLQ, SUPER_GLUE_CB, SUPER_GLUE_COPA, SUPER_GLUE_MULTIRC, SUPER_GLUE_RECORD, SUPER_GLUE_RTE, SUPER_GLUE_WIC, SUPER_GLUE_WSC, SUPER_GLUE_AVG]
        task_mapping = task_mapping_super_glue
    else:
        raise ValueError(f"Invalid glue_type: {glue_type}")

    data = print_table(
        metrics=metrics,
        variants=variants,
        print_metric_types=True,
    )

    # calculate the average confidence margin for the GLUE sub-metrics
    margins = [metric for metric in list(data.values())[0][BASE].scores.keys() if metric.startswith(glue_type)]
    max_margin = {task: max([result.scores[task].margin for model, model_result in data.items() for variant, result in model_result.items() if task in result.scores]) for task in margins}
    min_margin = {task: min([result.scores[task].margin for model, model_result in data.items() for variant, result in model_result.items() if task in result.scores]) for task in margins}


    # Initialize task samples dictionary
    task_samples = {}

    # Iterate through tasks and fetch dataset sizes
    for task in task_mapping.keys():
        if task == 'glue' or task == 'glue_mnli_mean' or task == 'super_glue':
            continue

        raw_task = task.removeprefix('glue_').removeprefix('super_glue_')
        if raw_task == 'mnli':
            raw_task = 'mnli_matched'
        elif raw_task == 'mnli-mm':
            raw_task = 'mnli_mismatched'

        # Load dataset and count samples
        dataset = load_dataset(glue_type, raw_task)
        task_samples[task] = len(dataset['validation']) if 'validation' in dataset else 0

    task_samples[glue_type] = sum([task_samples[task] for task in task_samples.keys()])

    table_data = [[task_mapping.get(task, ''), min_margin.get(task, '-'), max_margin.get(task, '-'), task_samples.get(task, -1)] for task in margins]
    # Sort table data by the number of samples (4th column)
    table_data.sort(key=lambda x: x[3], reverse=True)

    headers = ["Task", "Min (\%)", "Max (\%)", "\# Samples"]

    # Generate LaTeX table
    latex_table = tabulate(table_data, headers=headers, tablefmt="latex", floatfmt=".2f")

    print('\n'*3)
    print(latex_table)

def print_full_seat_table(bias_type='gender'):
    print_table(
        variants=STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        metrics=[SEAT_6, SEAT_6b, SEAT_7, SEAT_7b, SEAT_8, SEAT_8b, SEAT_AVG],
        bias_type=bias_type,
        fmt='.2f',
    )


def print_full_religion_seat_table():
    print_table(
        variants=RELIGION_STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        metrics=SEAT_METRICS_BY_BIAS_TYPE['religion'] + [SEAT_AVG],
        bias_type='religion',
        fmt='.2f',
    )


def print_full_race_seat_table():
    print_table(
        variants=RACE_STRUCTURED_VARIANTS_WITHOUT_SELF_DEBIAS,
        metrics=SEAT_METRICS_BY_BIAS_TYPE['race'] + [SEAT_AVG],
        bias_type='race',
        fmt='.2f',
    )

def print_full_main_race_table():
    print_table(
        variants=RACE_STRUCTURED_VARIANTS_WITH_SELF_DEBIAS,
        bias_type='race',
    )

def print_full_main_religion_table():
    print_table(
        variants=RELIGION_STRUCTURED_VARIANTS_WITH_SELF_DEBIAS,
        #variants=[[GRADIEND_RELIGION]],
        bias_type='religion',
    )


if __name__ == '__main__':
    print_full_race_super_glue_table()
    #print_full_religion_super_glue_table()

    #print_full_race_glue_table()
    #print_full_religion_glue_table()

    #print_full_glue_table(glue_type='glue')
    #print_full_glue_table(glue_type='super_glue')

    #print_full_main_race_table()
    #print_full_main_religion_table()

    # DONE
    #print_full_main_table()
    #print_full_gender_glue_table()
    #print_full_gender_super_glue_table()
    #print_full_seat_table()
    #print_full_race_seat_table()
    #print_full_religion_seat_table()
