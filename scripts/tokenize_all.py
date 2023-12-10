import json
import os
import random
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from openmatch.arguments import DataArguments
from openmatch.dataset import InferenceDataset


def reformat_output(x, all_markers):
    result = {}
    for marker in all_markers:
        result[marker] = x[marker]["input_ids"] if x[marker] is not None else []
    return {"id": x["text_id"], **result}


if __name__ == "__main__":
    parser = HfArgumentParser(DataArguments)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--is_query", action="store_true")
    parser.add_argument("--save_to", type=str)
    parser.add_argument("--num_workers", type=int, default=4)

    data_args, other_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(other_args.tokenizer_name, use_fast=True)
    dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=other_args.is_query,
        mode="dict_processed",
        full_tokenization=False,
        stream=True,
    )

    save_dir = os.path.split(other_args.save_to)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    reformat_output_partial = partial(reformat_output, all_markers=data_args.all_markers.split(","))
    with open(other_args.save_to, "w") as f:
        with Pool(other_args.num_workers) as p:
            for x in tqdm(p.imap(reformat_output_partial, dataset)):
                f.write(json.dumps(x) + "\n")
