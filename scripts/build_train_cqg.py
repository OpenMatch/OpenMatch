# Adapted from Tevatron (https://github.com/texttron/tevatron)

import copy
import csv
import json
import os
import random
from datetime import datetime
from functools import partial

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset
from openmatch.utils import load_beir_positives, load_from_trec, load_positives


def get_positive_and_negative_samples(
        query_dataset, 
        corpus_dataset_pos, 
        corpus_dataset_neg,
        qrel, 
        n_sample, 
        depth, 
        q_and_docs
    ):
    qid = q_and_docs[0]
    rank_list = q_and_docs[1]
    origin_positives = qrel.get(qid, [])
    negatives = []
    for docid, _ in rank_list:
        if docid not in origin_positives:
            negatives.append(docid)
    negatives = negatives[:depth]
    random.shuffle(negatives)
    if (len(origin_positives)) >= 1:
        item = process_one(
            query_dataset, 
            corpus_dataset_pos, 
            corpus_dataset_neg,
            qid, 
            origin_positives, 
            negatives[:n_sample]
        )
        if item["positives"]:
            return item
        else:
            return None
    else:
        return None


def process_one(
        query_dataset, 
        corpus_dataset_pos, 
        corpus_dataset_neg,
        q, 
        poss, 
        negs
    ):
    train_example = {
        'query': query_dataset[q]["input_ids"],
        'positives': [corpus_dataset_pos[p]["input_ids"] for p in poss if corpus_dataset_pos[p]["input_ids"]],
        'negatives': [corpus_dataset_neg[n]["input_ids"] for n in negs if corpus_dataset_neg[n]["input_ids"]],
    }

    return train_example


random.seed(datetime.now())
parser = HfArgumentParser(DataArguments)
parser.add_argument("--pos_doc_template", type=str, default="Positive: <text>")
parser.add_argument("--neg_doc_template", type=str, default="Negative: <text>")
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--hn_file', type=str, default=None)
parser.add_argument("--qrels_file", type=str, required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--n_sample', type=int, default=40)
parser.add_argument('--depth', type=int, default=200)
parser.add_argument('--beir', action='store_true')

data_args, other_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(other_args.tokenizer_name, use_fast=True)
query_dataset = InferenceDataset.load(
    tokenizer=tokenizer,
    data_args=data_args,
    is_query=True,
    full_tokenization=False,
    stream=False
)
corpus_dataset_pos = InferenceDataset.load(
    tokenizer=tokenizer,
    data_args=data_args,
    template=other_args.pos_doc_template,
    is_query=False,
    full_tokenization=False,
    stream=False
)
corpus_dataset_neg = copy.copy(corpus_dataset_pos)
corpus_dataset_neg.template = other_args.neg_doc_template

qrel = load_positives(other_args.qrels_file) if not other_args.beir else load_beir_positives(other_args.qrels_file)
run_list = []
if other_args.hn_file is not None:
    run = load_from_trec(other_args.hn_file, as_list=True)
    for qid, rank_list in run.items():
        run_list.append((qid, rank_list))
else:
    for qid in query_dataset.dataset.keys():
        run_list.append((qid, []))

get_positive_and_negative_samples_partial = partial(
                                                get_positive_and_negative_samples, 
                                                query_dataset, 
                                                corpus_dataset_pos, 
                                                corpus_dataset_neg,
                                                qrel, 
                                                other_args.n_sample, 
                                                other_args.depth
                                            )

save_dir = os.path.split(other_args.save_to)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

contents = list(tqdm(map(get_positive_and_negative_samples_partial, run_list), total=len(run_list)))

with open(other_args.save_to, 'w') as f:
    for result in tqdm(contents):
        if result is not None:
            f.write(json.dumps(result))
            f.write("\n")