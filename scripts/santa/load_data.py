import argparse
import logging
from tqdm import tqdm
import pickle
import torch
import pandas as pd
import numpy as np
import os
import json

""" Init variables """
col_scores = "scores"
col_iteration = "iteration"
col_query_id = "query_id"
col_product_id = "product_id"
col_product_locale = "product_locale"
col_small_version = "small_version"
col_split = "split"
col_esci_label = "esci_label"
col_relevance_pos = "relevance_pos"
col_ranking_postion = "ranking_postion"
col_score = "score"
col_conf = "conf"

def load_product_data(args,version,split):
    # input directory path includes shopping_queries_dataset_examples.parquet and shopping_queries_dataset_products.parquet
    df_examples = pd.read_parquet(os.path.join(args.input, 'shopping_queries_dataset_examples.parquet'))
    df_products = pd.read_parquet(os.path.join(args.input, 'shopping_queries_dataset_products.parquet'))

    df_examples_products = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=[col_product_locale, col_product_id],
        right_on=[col_product_locale, col_product_id]
    )
    df_examples_products = df_examples_products[df_examples_products[col_small_version] == version]
    df_examples_products = df_examples_products[df_examples_products[col_split] == split]
    df_examples_products = df_examples_products[df_examples_products[col_product_locale] == 'us']
    return df_examples_products

def load_code_data(file_path=None):
    data=[]
    with open(file_path) as f:
        for line in f:
            line= line.strip()
            js= json.loads(line)
            data.append(js)
    return data