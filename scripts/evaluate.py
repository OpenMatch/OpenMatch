import argparse
import csv
import logging

import numpy as np
import pytrec_eval
from datasets import load_dataset
from openmatch.qa_utils import SimpleTokenizer, has_answers
from openmatch.utils import load_from_trec
from tqdm import tqdm

logger = logging.getLogger(__name__)


def eval_mrr(qrel, run, cutoff=None):
    """
    Compute MRR@cutoff manually.
    """
    mrr = 0.0
    num_ranked_q = 0
    results = {}
    for qid in qrel:
        if qid not in run:
            continue
        num_ranked_q += 1
        docid_and_score = [(docid, score) for docid, score in run[qid].items()]
        docid_and_score.sort(key=lambda x: x[1], reverse=True)
        for i, (docid, _) in enumerate(docid_and_score):
            rr = 0.0
            if cutoff is None or i < cutoff:
                if docid in qrel[qid] and qrel[qid][docid] > 0:
                    rr = 1.0 / (i + 1)
                    break
        results[qid] = rr
        mrr += rr
    mrr /= num_ranked_q
    results["all"] = mrr
    return results


def print_line(measure, scope, value):
    print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query_eval_wanted", action="store_true")
    parser.add_argument("-m", "--measure", type=str, default=None)

    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--collection", type=str, required=False)
    parser.add_argument("--answer", type=str, required=False)

    parser.add_argument("qrel")
    parser.add_argument("run")
    args = parser.parse_args()

    if args.qa:
        if args.collection is None or args.answer is None:
            raise ValueError("Must provide collection and answer files for QA eval")
        collection = {}
        with open(args.collection, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                id_ = row.pop("id")
                collection[id_] = row
        answer = {}
        with open(args.answer, "r") as f:
            for line in f:
                id_, question, answers = line.strip().split("\t")
                if answers[0] == '"':
                    answers = answers[1:-1].replace('""', '"')
                answer[id_] = eval(answers)

        run = load_from_trec(args.run, as_list=True)

        tokenizer = SimpleTokenizer()
        accuracy = {1: [], 5: [], 20: [], 100: []}

        for qid, rank_list in run.items():
            answers = answer[qid]
            has_ans_idx = 100
            for doc_rank, (docid, _) in enumerate(rank_list):
                text = collection[docid]["text"]
                if has_answers(text, answers, tokenizer):
                    has_ans_idx = doc_rank
                    break

            for k in accuracy:
                accuracy[k].append(int(has_ans_idx < k))

        for k in accuracy:
            print_line("Accuracy@{}".format(k), "all", np.mean(accuracy[k]))

        exit(0)

    with open(args.qrel, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(args.run, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)


    if args.measure is not None and "mrr" in args.measure:

        if "mrr_cut" in args.measure:
            mrr_result = eval_mrr(qrel, run, cutoff=int(args.measure.split(".")[-1]))
        else:
            mrr_result = eval_mrr(qrel, run)
        if not args.query_eval_wanted:
            print("MRR: ", mrr_result["all"])
        else:
            for qid, mrr in mrr_result.items():
                print_line("MRR", qid, mrr)
            print("MRR: ", mrr_result["all"])

    else:
        if args.measure is None:
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
        else:
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, {args.measure})
        results = evaluator.evaluate(run)

        for query_id, query_measures in sorted(results.items()):
            for measure, value in sorted(query_measures.items()):
                if args.query_eval_wanted:
                    print_line(measure, query_id, value)

        for measure in sorted(query_measures.keys()):
            print_line(
                measure,
                'all',
                pytrec_eval.compute_aggregated_measure(
                    measure,
                    [query_measures[measure] for query_measures in results.values()]))
