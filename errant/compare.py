from errant.edit import Edit
from typing import List
import copy
import pprint
from typing import List, Tuple, Union, Dict
import errant
from dataclasses import dataclass, field
from tabulate import tabulate

class Score:
    def __init__(
        self,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
        beta: float = 0.5
    ):
        self.tp: int = tp
        self.fp: int = fp
        self.fn: int = fn
        self.beta: float = beta
        self.precision: float = 0
        self.recall: float = 0
        self.f: float = 0
        self.calc_f()
    
    def add(self, other):
        '''Changes its own values.
        '''
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.calc_f()

    def __add__(self, other):
        '''Returns another object and does not change its own values.
        This overloads "+".
        '''
        return self.__class__(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            beta=self.beta
        )
    
    def add_tp(self, num=1):
        self.tp += num
        self.calc_f()

    def add_fp(self, num=1):
        self.fp += num
        self.calc_f()

    def add_fn(self, num=1):
        self.fn += num
        self.calc_f()
    
    def get_beta(self):
        return self.beta

    def calc_f(self):
        tp = self.tp
        fp = self.fp
        fn = self.fn
        beta = self.beta
        p = float(tp)/(tp+fp) if fp else 1.0
        r = float(tp)/(tp+fn) if fn else 1.0
        self.f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
        self.precision = p
        self.recall = r

    def __repr__(self):
        return f"[TP={self.tp}, FP={self.fp}, FN={self.fn}, Precision={self.precision}, Recall={self.recall}, F_{self.beta}={self.f}]"

@dataclass
class ERRANTCompareOutput:
    overall: Score
    etype: Dict[str, Score]
    best_ref_ids: List[int]

def check_num_sents(
    srcs: List[str],
    hyps: List[str],
    refs: List[List[str]]
) -> None:
    for r in refs:
        assert len(r) == len(srcs) 
        assert len(r) == len(hyps)
    return

def compare_from_raw(
    orig: List[str],
    cor: List[str],
    refs: List[List[str]],
    beta: float=0.5,
    cat: int=2,
    mode='cs',
    single: bool=False,
    multi: bool=False,
    filt: List[str]=[],
    verbose: bool=False,
    annotator=None
) -> ERRANTCompareOutput:
    '''errant_compare given the raw text
    Args:
        orig: Original sentences
        cor: Corrected sentences
        refs: References
            refs can be multiple. The shape is (num_annotations, num_sents)
        beta: the beta for F_{beta}
        cat: 1 or 2 or 3.
            1: Only show operation tier scores; e.g. R.
            2: Only show main tier scores; e.g. NOUN.
            3: Show all category scores; e.g. R:NOUN.
        mode:
            'cs': Span-based correction
            'cse': Span-based correction with error types
            'ds': Span-based detection
            'dt': Token-based detection
        single: Only evaluate single token edits; i.e. 0:1, 1:0 or 1:1
        mulit: Only evaluate multi token edits; i.e. 2+:n or n:2+
        filt: Do not evaluate the specified error types.
        verbose: Print verbose output.
    Returns:
        ERRANTCompareOutput: The evaluation scores.
    '''
    # The references must be a two dimensions list
    if not isinstance(refs[0], list):
        # The shape of refs must be (num_annotations, num_sents)
        refs = [refs]
    check_num_sents(orig, cor, refs)
    if annotator is None:
        annotator = errant.load('en')
    # Parse each sentences
    orig = [annotator.parse(o) for o in orig]
    cor = [annotator.parse(c) for c in cor]
    refs = [[annotator.parse(r) for r in ref] for ref in refs]
    # Generate Edit objects
    hyp_edits = [annotator.annotate(o, c) for o, c in zip(orig, cor)]
    ref_edits = [[annotator.annotate(o, r) for o, r in zip(orig, ref)] for ref in refs]
    output = compare_from_edits(
        hyp_edits,
        ref_edits,
        beta=beta,
        cat=cat,
        mode=mode,
        single=single,
        multi=multi,
        filt=filt,
        verbose=verbose
    )
    return output

def can_update_best(
    best: Score,
    candidate: Score
):
    '''Check whether the new_score outperforms the current best score.
    Compare in order of priority F, TP, FP, FN.
    '''
    return [best.f, best.tp, -best.fp, -best.fn] \
        < [candidate.f, candidate.tp, -candidate.fp, -candidate.fn]

def filter_edits(
    edits: List[List[Edit]],
    cat: int=2,
    mode: str='cs',
    single: bool=False,
    multi: bool=False,
    filt: list=[],
):
    '''To remove the corrections that not to be evaluated.
    '''
    assert not (single and multi)
    new_edits = []
    for edit in edits:
        l = []
        for e in edit:
            # Process error type filtering condition
            if e.type in filt:
                continue
            # Process single or multi condition
            if single and e.is_multi():
                continue
            if multi and e.is_single():
                continue 
            # Process detection or correction condition
            if mode in ['dt', 'ds']:
                # To ignore c_str in the detection scoring,
                #   it is replaced with an empty string.
                e.c_str = ''
            elif e.type in ['UNK']:
                # UNK is ignored in correction scoring
                continue
            # Only dt treats noop edits.
            if mode != 'dt' and e.o_start == -1:
                continue

            # Process the error type
            if mode == 'cse':
                e.c_str = e.c_str + '[SEP]' + e.type
            if cat == 1:
                # e.g. 'M:NOUN:NUM' -> 'M'
                e.type = e.type[0] if e.type != 'UNK' else 'UNK'
            elif cat == 2:
                # e.g. 'M:NOUN:NUM' -> 'NOUN:NUM'
                e.type = e.type[2:] if e.type != 'UNK' else 'UNK'

            if mode == 'dt':
                # Insertion edit but not noop
                if e.o_start == e.o_end and e.o_start >= 0:
                    e.o_end = e.o_start + 1
                    l.append(e)
                elif e.o_start != e.o_end:
                    for tok_id in range(e.o_start, e.o_end):
                        new_edit = copy.copy(e)
                        new_edit.o_start = tok_id
                        new_edit.o_end = tok_id + 1
                        l.append(new_edit)
                else:
                    l.append(e)
            else:
                l.append(e)
        new_edits.append(l)
    return new_edits
    
def calc_overall_score(
    score: Dict[str, Score]
) -> Score:
    '''Convert error type based scores into an entire score.
    '''
    # If there is no scores, it retuns initialized instance.
    # This corresponds to the situation of src == trg == hyp.
    if len(score) == 0:
        return Score()
    beta = list(score.values())[0].get_beta()
    overall = Score(beta=beta)
    for etype in score:
        overall.add(score[etype])
    return overall

def merge_etype_scores(
    d1: Dict[str, Score],
    d2: Dict[str, Score]
) -> Dict[str, Score]:
    '''Add d2 information to d1
    '''
    if d2 == {}:
        return d1
    beta = list(d2.values())[0].get_beta()
    for etype in d2.keys():
        d1[etype] = d1.get(etype, Score(beta=beta))
        d1[etype].add(d2[etype])
    return d1

def print_table(table):
    '''This is for verbose setting.
    This is copied from official imlementation:
        https://github.com/chrisjbryant/errant
    '''
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3)
        for i in range(len(table[0]))
    ]
    row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
    for row in table:
        print(row_format.format(*row))

def compare_from_edits(
    hyp_edits: List[List[Edit]],  # (num_sents, num_edits)
    ref_edits: List[List[List[Edit]]],  # (num_annotations, num_sents, num_edts)
    beta: float=0.5,
    cat: int=2,
    mode='cs',
    single: bool=False,
    multi: bool=False,
    filt: List[str]=[],
    verbose: bool=False
):
    '''errant_compare given edits, which are the results of errant.Annotator.annotate()
    Args:
        hyp_edits: The edits between original and correction.
        ref_edits: The edits between original and references.
            This can be multiple. The shape is (num_annotations, num_sents, num_edits)
    Other args and returns is the same as compare_from_raw().
    '''
    filter_args = {
        'cat': cat,
        'mode': mode,
        'single': single,
        'multi': multi,
        'filt': filt
    }
    # Removed correction not to be evaluated
    hyp_edits = filter_edits(hyp_edits, **filter_args)
    ref_edits = [filter_edits(r, **filter_args) for r in ref_edits]
    for ref_id in range(len(ref_edits)):
        assert len(hyp_edits) == len(ref_edits[ref_id])
    final_etype_score = {}
    final_overall_score = Score(beta=beta)
    num_annotator = len(ref_edits)
    num_sents = len(ref_edits[0])
    best_ref_ids = []
    for sent_id in range(num_sents):
        # best_score: sentence-level best score
        best_score: Dict[str, Score] = dict()
        best_ref_id = 0
        best_r_edits = None
        best_h_edits = None
        for ref_id in range(num_annotator):
            candidate_score = dict()
            h_edits = hyp_edits[sent_id]
            r_edits = ref_edits[ref_id][sent_id]
            # True positive and False negative
            for edit in r_edits:
                candidate_score[edit.type] = candidate_score.get(
                    edit.type,
                    Score(beta=beta)
                )
                if edit in h_edits:
                    candidate_score[edit.type].add_tp()
                else:
                    candidate_score[edit.type].add_fn()
            # False positive
            for edit in h_edits:
                if edit not in r_edits:
                    candidate_score[edit.type] = candidate_score.get(
                        edit.type,
                        Score(beta=beta)
                    )
                    candidate_score[edit.type].add_fp()
            # Update the best sentence-level score
            if ref_id == 0:
                best_score = candidate_score
                best_ref_id = ref_id
                best_h_edits = h_edits
                best_r_edits = r_edits
            else:
                best_overall = calc_overall_score(best_score)
                candidate_overall = calc_overall_score(candidate_score)
                if can_update_best(
                    (final_overall_score + best_overall),
                    (final_overall_score + candidate_overall),
                ):
                    best_score = candidate_score
                    best_ref_id = ref_id
                    best_h_edits = h_edits
                    best_r_edits = r_edits
        final_etype_score = merge_etype_scores(final_etype_score, best_score)
        final_overall_score = calc_overall_score(final_etype_score)
        best_ref_ids.append(best_ref_id)
        if verbose:
            print('{:-^40}'.format(""))
            print(f'^^ HYP 0, REF {best_ref_id} chosen for sentence {sent_id}')
            print('Local results:')
            header = ["Category", "TP", "FP", "FN"]
            body = [[k, v.tp, v.fp, v.fn] for k, v in best_score.items()]
            print_table([header] + body)
            print('Hypothesis:', [(e.o_start, e.o_end, e.c_str) for e in best_h_edits])
            print('Best References:', [(e.o_start, e.o_end, e.c_str) for e in best_r_edits])
            print('Local:', calc_overall_score(best_score))
            print('Global:', final_overall_score)
    
    return ERRANTCompareOutput(
        overall=final_overall_score,
        etype=final_etype_score,
        best_ref_ids=best_ref_ids
    )