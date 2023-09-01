import argparse
import json
import os

import sentence_transformers
from transformers import AutoTokenizer, T5EncoderModel

from openmatch.modeling import LinearHead

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--st_ckpt_path", type=str)
    parser.add_argument("--om_ckpt_path", type=str)
    args = parser.parse_args()

    linear_dir = os.path.join(args.st_ckpt_path, "2_Dense")
    dense_model = sentence_transformers.models.Dense.load(linear_dir)
    with open(os.path.join(linear_dir, "config.json"), "r") as f:
        dense_config = json.load(f)
    in_features, out_features = dense_config["in_features"], dense_config["out_features"]
    print("Sentence-transformers linear weight:")
    print(dense_model.state_dict())
    new_linear = LinearHead(in_features, out_features)
    new_linear.linear.weight.data = dense_model.linear.weight.data
    print("OpenMatch linear weight:")
    print(new_linear.state_dict())
    if not os.path.exists(args.om_ckpt_path):
        os.makedirs(args.om_ckpt_path)
    new_linear.save(args.om_ckpt_path)

    st_model = T5EncoderModel.from_pretrained(args.st_ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(args.st_ckpt_path)
    st_model.save_pretrained(args.om_ckpt_path)
    tokenizer.save_pretrained(args.om_ckpt_path)
    config = {
        "tied": True,
        "plm_backbone": {
            "type": type(st_model).__name__,
            "feature": "last_hidden_state",
        },
        "pooling": "mean",
        "linear_head": True,
        "normalize": True,
    }
    with open(os.path.join(args.om_ckpt_path, "openmatch_config.json"), "w") as f:
        json.dump(config, f, indent=4)
