import json
import os
import random
from functools import partial
from multiprocessing import Pool

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser


def random_crop(x, ratio_min, ratio_max):
    x = x["input_ids"]

    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop_a = x[start:end]

    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop_b = x[start:end]

    return {
        "query": crop_a,
        "positives": [crop_b],
        "negatives": [],
    }


if __name__ == "__main__":
    parser = HfArgumentParser(DataArguments)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--ratio_min", type=float, default=0.1)
    parser.add_argument("--ratio_max", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_to", type=str)

    data_args, other_args = parser.parse_args_into_dataclasses()
    random.seed(other_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(other_args.tokenizer_name, use_fast=True)
    corpus_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=False,
        final=False,
        stream=True
    )

    save_dir = os.path.split(other_args.save_to)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    random_crop_partial = partial(random_crop, ratio_min=other_args.ratio_min, ratio_max=other_args.ratio_max)
    with open(other_args.save_to, 'w') as f:
        with Pool(other_args.num_workers) as p:
            for item in tqdm(p.imap(random_crop_partial, corpus_dataset)):
                f.write(json.dumps(item) + "\n")