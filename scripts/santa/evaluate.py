import pandas as pd
import pandas as pd
import argparse
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def calculate_mrr(data_samples):
    ranks = []
    for item in data_samples:
        rank = 0
        find = False
        url = item['query']
        for idx in item['code'][:100]:
            # MRR@100
            if find is False:
                rank += 1
            if idx == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)
    result = {
        "eval_mrr": float(np.mean(ranks))
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_save_path", type=str, default=None, help="The path to the inference.trec")
    parser.add_argument("--data_type", type=str, default='code',help='code or product')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    if args.data_type=='code':
        # read csv
        new_columns = ['query_id', 'Q0', 'passage_id', 'rank', 'score', 'tool']
        df = pd.read_csv(args.trec_save_path, delimiter=" ", header=None)
        df.columns = new_columns
        query2id = {}
        data_samples = []
        q_last = ''
        # get data samples for mrr
        for (index, row) in tqdm(df.iterrows()):
            if q_last != row['query_id']:
                query2id = {}
                query2id["query"] = row['query_id']
                query2id['code'] = []
                q_last = row['query_id']
                data_samples.append(query2id)
            query2id['code'].append(row['passage_id'])

        # The mrr code copy from Unxicode https://github.com/microsoft/CodeBERT/blob/master/UniXcoder/downstream-tasks/code-search/run.py
        result=calculate_mrr(data_samples)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 3)))

if __name__ == "__main__":
    main()
