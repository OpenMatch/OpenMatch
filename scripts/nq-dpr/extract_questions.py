import argparse
import json
import csv
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train_file")
    parser.add_argument("--output_queries_file")
    parser.add_argument("--output_qrels_file")
    args = parser.parse_args()

    
    with open(args.input_train_file, "r") as fin:
        train_data = json.load(fin)

    qid = 0
    with open(args.output_queries_file, "w") as fqueries, open(args.output_qrels_file, "w") as fqrels:
        csv_writer = csv.writer(fqueries, delimiter="\t")
        for item in tqdm(train_data):
            qid_str = "train_" + str(qid)
            query = item["question"]
            answers = item["answers"]
            # print(item["positive_ctxs"])
            for pos_doc in item["positive_ctxs"]:
                if pos_doc["score"] == 1000:
                    fqrels.write("{}\t0\t{}\t1\n".format(qid_str, pos_doc["passage_id"]))
            csv_writer.writerow([qid_str, query, answers])
            # for _id in pos_doc_ids:
            #     fqrels.write("{}\t0\t{}\t1\n".format(qid_str, _id))
            qid += 1