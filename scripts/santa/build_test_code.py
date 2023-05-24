import json
from argparse import ArgumentParser
import os
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from load_data import load_product_data, load_code_data

def main():

    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default="D:\\T5数据\\CSN\\test.jsonl")
    parser.add_argument('--output', type=str, default="D:\\T5数据\\CSN\\codequery.jsonl")
    parser.add_argument('--data_type', type=str, default="query", help="query or doc")
    parser.add_argument('--inference_data', type=str, default="code", help="code or product")

    args = parser.parse_args()
    if args.inference_data == 'code':
        data = load_code_data(args.input)
        with open(args.output, 'w') as f:
            for idx, item in enumerate(tqdm(data)):
                group = {}
                url = item['url']
                if args.data_type == 'doc':
                    if 'code_tokens' in item:
                        code = ' '.join(item['code_tokens'])
                    else:
                        code = ' '.join(item['function_tokens'])
                    group['code'] = code
                    group['id'] = url
                else:
                    nl = ' '.join(item['docstring_tokens'])
                    group['nl'] = nl
                    group['id'] = url
                f.write(json.dumps(group) + '\n')
    else:
        """ Init variables """
        col_query = "query"
        col_query_id = "query_id"
        col_product_id = "product_id"
        col_product_title = "product_title"
        col_product_description = "product_description"
        form = 'title :' + " {} " + 'text :' + " {}"

        """ Load data """
        # version small
        # use test data inference
        # language us
        df_examples_products = load_product_data(args, version=1, split='test')
        features_querys = df_examples_products[col_query].to_list()  # query
        features_products = df_examples_products[col_product_title].to_list()  # doc title
        features_product_des = df_examples_products[col_product_description].to_list()  # doc description

        col_query_ids = df_examples_products[col_query_id].to_list()  # query id
        col_product_ids = df_examples_products[col_product_id].to_list()  # doc id

        with open(args.output, 'w') as f:
            for i in tqdm(range(len(features_querys))):
                group = {}
                doc = str(features_product_des[i])
                if doc != "None":
                    soup = BeautifulSoup(doc, 'html.parser')  # delete html
                    doc = soup.text
                    doc = form.format(features_products[i], doc)
                    query = features_querys[i]
                    query_id = col_query_ids[i]
                    product_id = col_product_ids[i]
                else:
                    doc = ''
                    doc = form.format(features_products[i], doc)
                    query = features_querys[i]
                    query_id = col_query_ids[i]
                    product_id = col_product_ids[i]

                if args.data_type == 'doc':
                    group['product'] = doc
                    group['id'] = product_id
                else:
                    group['query'] = query
                    group['id'] = query_id
                f.write(json.dumps(group) + '\n')
if __name__ == '__main__':
    main()