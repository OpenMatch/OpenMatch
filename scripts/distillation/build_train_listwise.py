import json
import os
import random

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset
from openmatch.utils import load_from_trec

parser = HfArgumentParser(DataArguments)
parser.add_argument("--tokenizer_name", type=str, required=True)
parser.add_argument("--trec_file", type=str, required=True)
parser.add_argument("--n_sample", type=int, default=10)
parser.add_argument("--save_to", type=str, required=True)
data_args, other_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(other_args.tokenizer_name, use_fast=True)
query_dataset = InferenceDataset.load(
    tokenizer=tokenizer,
    data_args=data_args,
    is_query=True,
    full_tokenization=False,
    stream=False
)
corpus_dataset = InferenceDataset.load(
    tokenizer=tokenizer,
    data_args=data_args,
    is_query=False,
    full_tokenization=False,
    stream=False
)
run = load_from_trec(other_args.trec_file)

save_dir = os.path.split(other_args.save_to)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(other_args.save_to, "w") as fout:
    for qid, doc_dict in tqdm(run.items()):
        docs_and_scores = list(doc_dict.items())
        docs_and_scores = random.sample(docs_and_scores, min(other_args.n_sample, len(docs_and_scores)))
        train_example = {
            "query": query_dataset[qid]["input_ids"],
            "docs": [corpus_dataset[doc_id]["input_ids"] for doc_id, _ in docs_and_scores],
            "scores": [score for _, score in docs_and_scores],
        }
        fout.write(json.dumps(train_example) + "\n")