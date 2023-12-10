import json
import os
import random

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset
from openmatch.utils import load_beir_positives, load_from_trec, load_positives

parser = HfArgumentParser(DataArguments)
parser.add_argument("--tokenizer_name", type=str, required=True)
parser.add_argument("--trec_file", type=str, required=True)
parser.add_argument("--qrels_file", type=str, default=None)
parser.add_argument("--n_sample", type=int, default=10)
parser.add_argument("--beir", action="store_true")
parser.add_argument("--save_to", type=str, required=True)
data_args, other_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(other_args.tokenizer_name, use_fast=True)
query_dataset = InferenceDataset.load(
    tokenizer=tokenizer, data_args=data_args, is_query=True, full_tokenization=False, stream=False
)
corpus_dataset = InferenceDataset.load(
    tokenizer=tokenizer, data_args=data_args, is_query=False, full_tokenization=False, stream=False
)
qrels = (
    load_beir_positives(other_args.qrels_file)
    if other_args.beir
    else load_positives(other_args.qrels_file)
)
run = load_from_trec(other_args.trec_file)

save_dir = os.path.split(other_args.save_to)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(other_args.save_to, "w") as fout:
    for qid, doc_dict in tqdm(run.items()):
        if qid not in qrels:
            continue
        pos_doc = qrels[qid][0]
        pos_doc_score = doc_dict[pos_doc]
        negs = list(doc_dict.items())
        negs = [x for x in negs if x[0] != pos_doc]
        negs = random.sample(negs, min(other_args.n_sample, len(negs)))
        for neg_doc, neg_doc_score in negs:
            train_example = {
                "query": query_dataset[qid]["input_ids"],
                "positive": corpus_dataset[pos_doc]["input_ids"],
                "negative": corpus_dataset[neg_doc]["input_ids"],
                "score": pos_doc_score - neg_doc_score,
            }
            fout.write(json.dumps(train_example) + "\n")
