import argparse
import json

from transformers import AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_q_file", type=str)
    parser.add_argument("--output_d_file", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    idx = 0
    with open(args.input_file, "r") as fin, open(args.output_q_file, "w") as fout_q, open(args.output_d_file, "w") as fout_d:
        for line in fin:
            obj = json.loads(line)
            query = tokenizer.decode(obj["query"], skip_special_tokens=True)
            doc = tokenizer.decode(obj["positives"][0], skip_special_tokens=True)
            fout_q.write("original_q_{}\t{}\n".format(idx, query))
            fout_d.write("d_{}\t{}\n".format(idx, doc))
            idx += 1