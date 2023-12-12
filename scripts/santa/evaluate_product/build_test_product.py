import json
import os
from argparse import ArgumentParser

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

""" Init variables """
col_query = "query"
col_query_id = "query_id"
col_product_id = "product_id"
col_product_title = "product_title"
col_product_description = "product_description"
form = "title :" + " {} " + "text :" + " {}"
col_scores = "scores"
col_iteration = "iteration"
col_product_locale = "product_locale"
col_small_version = "small_version"
col_split = "split"
col_esci_label = "esci_label"


def load_product_data(args, version, split):
    # input directory path includes shopping_queries_dataset_examples.parquet and shopping_queries_dataset_products.parquet
    df_examples = pd.read_parquet(
        os.path.join(args.input, "shopping_queries_dataset_examples.parquet")
    )
    df_products = pd.read_parquet(
        os.path.join(args.input, "shopping_queries_dataset_products.parquet")
    )

    df_examples_products = pd.merge(
        df_examples,
        df_products,
        how="left",
        left_on=[col_product_locale, col_product_id],
        right_on=[col_product_locale, col_product_id],
    )
    df_examples_products = df_examples_products[df_examples_products[col_small_version] == version]
    df_examples_products = df_examples_products[df_examples_products[col_split] == split]
    df_examples_products = df_examples_products[df_examples_products[col_product_locale] == "us"]
    return df_examples_products


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="D:\\T5数据\\CSN\\test.jsonl")
    parser.add_argument("--output", type=str, default="D:\\T5数据\\CSN\\codequery.jsonl")
    parser.add_argument("--data_type", type=str, default="query", help="query or doc")

    args = parser.parse_args()

    """ Load data """
    # version small
    # use test data inference
    # language us
    df_examples_products = load_product_data(args, version=1, split="test")
    features_querys = df_examples_products[col_query].to_list()  # query
    features_products = df_examples_products[col_product_title].to_list()  # doc title
    features_product_des = df_examples_products[
        col_product_description
    ].to_list()  # doc description

    col_query_ids = df_examples_products[col_query_id].to_list()  # query id
    col_product_ids = df_examples_products[col_product_id].to_list()  # doc id

    with open(args.output, "w") as f:
        for i in tqdm(range(len(features_querys))):
            group = {}
            doc = str(features_product_des[i])
            if doc != "None":
                soup = BeautifulSoup(doc, "html.parser")  # delete html
                doc = soup.text
                doc = form.format(features_products[i], doc)
                query = features_querys[i]
                query_id = col_query_ids[i]
                product_id = col_product_ids[i]
            else:
                doc = ""
                doc = form.format(features_products[i], doc)
                query = features_querys[i]
                query_id = col_query_ids[i]
                product_id = col_product_ids[i]

            if args.data_type == "doc":
                group["product"] = doc
                group["id"] = product_id
            else:
                group["query"] = query
                group["id"] = query_id
            f.write(json.dumps(group) + "\n")


if __name__ == "__main__":
    main()
