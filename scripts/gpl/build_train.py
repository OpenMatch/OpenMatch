import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset

parser = HfArgumentParser(DataArguments)
parser.add_argument('--tokenizer_name', type=str, required=True)
parser.add_argument('--gpl_data_dir', type=str, required=True)
parser.add_argument('--save_to', type=str, required=True)
data_args, other_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(other_args.tokenizer_name, use_fast=True)
data_args.query_path = data_args.query_path or os.path.join(other_args.gpl_data_dir, "qgen-queries.jsonl")
data_args.corpus_path = data_args.corpus_path or os.path.join(other_args.gpl_data_dir, "corpus.jsonl")
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

save_dir = os.path.split(other_args.save_to)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(other_args.save_to, "w") as fout, open(os.path.join(other_args.gpl_data_dir, "gpl-training-data.tsv"), "r") as fin:
    for line in tqdm(fin):
        qid, pos, neg, score = line.strip().split("\t")
        score = float(score)
        train_example = {
            "query": query_dataset[qid]["input_ids"],
            "positive": corpus_dataset[pos]["input_ids"],
            "negative": corpus_dataset[neg]["input_ids"],
            "score": score,
        }
        fout.write(json.dumps(train_example) + "\n")