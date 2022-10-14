# Adapted from Tevatron (https://github.com/texttron/tevatron)

import csv
import json
import os
import random
from datetime import datetime
from functools import partial

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset
from openmatch.qa_utils import SimpleTokenizer, has_answers
from openmatch.utils import load_from_trec
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser


def get_positive_and_negative_samples(query_dataset, corpus_dataset, qa_tokenizer, qrel, n_sample, depth, q_and_docs):
    qid = q_and_docs[0]
    rank_list = q_and_docs[1]
    origin_positives = qrel.get(qid, [])
    negatives = []
    new_positives = []
    answers = eval(query_dataset.get_raw(qid)["ans"])
    for docid, _ in rank_list:
        content = corpus_dataset.get_raw(docid)['text']
        if not has_answers(content, answers, qa_tokenizer, regex=False):
            negatives.append(docid)
        else:
            if docid not in origin_positives:
                new_positives.append(docid)
    negatives = negatives[:depth]
    random.shuffle(negatives)
    if (len(origin_positives) + len(new_positives)) >= 1 and len(negatives) >= 1:
        return process_one(query_dataset, corpus_dataset, qid, origin_positives, new_positives[:1], negatives[:n_sample])
    else:
        return None


def read_qrel(relevance_file):
    qrel = {}
    with open(relevance_file, encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel


def process_one(query_dataset, corpus_dataset, q, ori_poss, new_poss, negs):
    pp = ori_poss + new_poss
    train_example = {
        'query': query_dataset[q]["input_ids"],
        'positives': [corpus_dataset[p]["input_ids"] for p in pp],
        'negatives': [corpus_dataset[n]["input_ids"] for n in negs],
    }

    return train_example


random.seed(datetime.now())
parser = HfArgumentParser(DataArguments)
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--hn_file', required=True)
parser.add_argument("--qrels_file", type=str, required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--n_sample', type=int, default=40)
parser.add_argument('--depth', type=int, default=200)

data_args, other_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(other_args.tokenizer_name, use_fast=True)
query_dataset = InferenceDataset.load(
    tokenizer=tokenizer,
    data_args=data_args,
    is_query=True,
    final=False,
    stream=False
)
corpus_dataset = InferenceDataset.load(
    tokenizer=tokenizer,
    data_args=data_args,
    is_query=False,
    final=False,
    stream=False
)
qrel = read_qrel(other_args.qrels_file)
run = load_from_trec(other_args.hn_file, as_list=True)
run_list = []
for qid, rank_list in run.items():
    run_list.append((qid, rank_list))

qa_tokenizer = SimpleTokenizer()
get_positive_and_negative_samples_partial = partial(
                                                get_positive_and_negative_samples, 
                                                query_dataset, 
                                                corpus_dataset, 
                                                qa_tokenizer, 
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

