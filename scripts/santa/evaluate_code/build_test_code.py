import json
from argparse import ArgumentParser
import os
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

def load_code_data(file_path=None):
    data=[]
    with open(file_path) as f:
        for line in f:
            line= line.strip()
            js= json.loads(line)
            data.append(js)
    return data

def main():

    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default="D:\\T5数据\\CSN\\test.jsonl")
    parser.add_argument('--output', type=str, default="D:\\T5数据\\CSN\\codequery.jsonl")
    parser.add_argument('--data_type', type=str, default="query", help="query or doc")

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
if __name__ == "__main__":
    main()