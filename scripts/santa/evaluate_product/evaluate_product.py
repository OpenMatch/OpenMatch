import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
import pytrec_eval

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

esci_label2relevance_pos = {
    "E": 4,
    "S": 2,
    "C": 3,
    "I": 1,
}


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


def generate_product_result(args, df_hypothesis):
    df_results = pd.DataFrame()
    df_results = pd.concat([df_results, df_hypothesis])
    max_trec_eval_score = 128
    min_trec_eval_score = 0
    df_results_product_id = df_results.groupby(by=[col_query_id])
    l_query_id = []
    l_product_id = []
    l_ranking_postion = []
    l_score = []
    for (query_id, rows) in df_results_product_id:
        n = len(rows)
        l_query_id += [query_id for _ in range(n)]
        l_product_id += rows[col_product_id].to_list()
        l_ranking_postion += [i for i in range(n)]
        l_score += list(
            np.arange(min_trec_eval_score, max_trec_eval_score, max_trec_eval_score / n).round(3)[
                ::-1
            ][:n]
        )

    df_res = pd.DataFrame(
        {
            col_query_id: l_query_id,
            col_product_id: l_product_id,
            col_ranking_postion: l_ranking_postion,
            col_score: l_score,
        }
    )
    model_name_value = "baseline"
    iteration_value = "Q0"
    df_res[col_conf] = model_name_value
    df_res[col_iteration] = iteration_value

    logger.info("***** get results file *****")
    df_res[
        [
            col_query_id,
            col_iteration,
            col_product_id,
            col_ranking_postion,
            col_score,
            col_conf,
        ]
    ].to_csv(
        os.path.join(args.output_path, "hypothesis.results"),
        index=False,
        header=False,
        sep=" ",
    )


def generate_product_qrels(df_examples_products, args):
    df_examples_products[col_iteration] = 0
    df_examples_products[col_relevance_pos] = df_examples_products[col_esci_label].apply(
        lambda esci_label: esci_label2relevance_pos[esci_label]
    )
    df_examples_products = df_examples_products[
        [
            col_query_id,
            col_iteration,
            col_product_id,
            col_relevance_pos,
        ]
    ]
    logger.info("***** get qrels file *****")
    df_examples_products.to_csv(
        os.path.join(args.output_path, "test.qrels"),
        index=False,
        header=False,
        sep=" ",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_query_data", type=str, default="code", help="code or product")
    parser.add_argument("--pickle_doc_data", type=str, default="code", help="code or product")
    parser.add_argument("--input", type=str, default=None, help="the path of the product")
    parser.add_argument(
        "--hypothesis_path_file",
        type=str,
        default=None,
        help="the path of the hypothesis_path_file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="the path of the hypothesis.results and test.qrels",
    )

    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    query_base = open(args.pickle_query_data, "rb")
    doc_base = open(args.pickle_doc_data, "rb")

    info_query = pickle.load(query_base)
    query_vec = torch.tensor(info_query[0])
    info_doc = pickle.load(doc_base)
    doc_vec = torch.tensor(info_doc[0])
    query_id = info_query[1]
    doc_id = info_doc[1]

    # 之后要对query_id排序，而open-match中得到的id为str类型，必须要转换成int类型。
    query_id = list(map(int, query_id))
    # query dot product
    score = torch.diagonal(torch.mm(query_vec, doc_vec.transpose(0, 1)).to("cpu")).tolist()

    logger.info("***** prepare data *****")
    df_examples_products = load_product_data(args, version=1, split="test")

    """ Prepare hypothesis file """
    df_hypothesis = pd.DataFrame(
        {
            col_query_id: query_id,
            col_product_id: doc_id,
            col_scores: score,
        }
    )
    df_hypothesis = df_hypothesis.sort_values(by=[col_query_id, col_scores], ascending=False)

    logger.info("***** get hypothesis_path_file *****")
    df_hypothesis[[col_query_id, col_product_id]].to_csv(
        args.hypothesis_path_file,
        index=False,
        sep=",",
    )

    # prepare trec_eval files codes are copied and modified from https://github.com/amazon-science/esci-data/blob/main/ranking/prepare_trec_eval_files.py
    """ Generate RESULTS file """
    generate_product_result(args, df_hypothesis)

    """ Generate QRELS file """
    generate_product_qrels(df_examples_products, args)
    # use qrel and result file to calculate ndcg. The details can be found in https://github.com/amazon-science/esci-data
    # ../code/terrier-project-5.5/bin/terrier trec_eval ${QRELS_FILE} ${RES_FILE} -c -J -m 'ndcg.1=0,2=0.01,3=0.1,4=1'


if __name__ == "__main__":
    main()
