# for all negative samples of querys in input 1,
# append the positives and negatives of the corresponding query in input 2.
#  Input_folder 1 and 2 should contain train files
# (jsonl file with each line a dict of {"query":[<tokenized_query>],"positives":[[tokenized_positive_1],[tokenized_positive_2],...],"negatives":[[tokenized_negative_1],[tokenized_negative_2],...]})
# NOTE that the positive of input 1 will be discarded.
#
# The original version is too memory costly because it loads all negatives in input 1 into memory;
# This solves this issue by only saving the offset value of the corresponding data, and read the data only when accessed.
# However, the performance of this method is limited by speed of I/O of the corresponding hard drive. On most standard drives, this didn't slow the processing speed by much.
#
import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="")
    parser.add_argument('--input_folder_1', required=True)
    parser.add_argument('--input_folder_2', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    
    input_path_1 = os.path.join(args.data_dir, args.input_folder_1)
    input_path_2 = os.path.join(args.data_dir, args.input_folder_2)
    output_path = os.path.join(args.data_dir, args.output_folder)
    
    # create output data
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    # load input 1
    print("Indexing input 1...")
    file_list_1 = [listx for listx in os.listdir(input_path_1) if "json" in listx]
    # Load all negatives for each query in file_list 1
    query_data_dict = {}
    file_ops_list = []
    i=1
    for file in file_list_1:
        print("Current file: {} ({:02d}/{:02d})".format(file,i,len(file_list_1)))
        i+=1
        file_1 = os.path.join(input_path_1, file)
        fi = open(file_1, "r", encoding="utf-8")
        while True:
            offset = fi.tell()
            line = fi.readline()
            if not line:
                break
            data = json.loads(line)
            query = "_".join(str(ids) for ids in data["query"])
            query_data_dict[query] = (fi,offset)
        fi.seek(0,0)
        file_ops_list.append(fi)
    # load input 2 & write mix
    file_list_2 = [listx for listx in os.listdir(input_path_2) if "json" in listx]
    print("Appending to input 2...")
    # append all negative of input 2 to list
    neg_num_list = []
    diff_num = 0
    i=1
    for file in file_list_2:
        print("Current file: {} ({:02d}/{:02d})".format(file,i,len(file_list_2)))
        i+=1
        file_2 = os.path.join(input_path_2, file)
        output_file = os.path.join(output_path, file)
        with open(file_2, "r", encoding="utf-8") as fi, \
            open(output_file, "w", encoding="utf-8") as fw:
            for line in tqdm(fi):
                data = json.loads(line)
                query = data["query"]
                positives = data["positives"]
                qid = "_".join(str(ids) for ids in query)
                if qid in query_data_dict:
                    f1,offset = query_data_dict[qid]
                    f1.seek(offset,0)
                    data1 = json.loads(f1.readline())
                    negatives = data["negatives"] + data1["negatives"]
                    neg_num_list.append(len(negatives))
                else:
                    negatives = data["negatives"]
                    diff_num += 1

                mix_example = {
                    'query': query,
                    'positives': positives,
                    'negatives': negatives,
                }
                fw.write(json.dumps(mix_example) + '\n')
    for f in file_ops_list:
        f.close()           
    # number of querys that only appears in input 2
    print("Mixing done.")
    print(diff_num, " new queries appeared in input 2.")
    # average negatives of the joined file
    print(np.mean(neg_num_list)," negatives is appended to each query in average in combined data.")
