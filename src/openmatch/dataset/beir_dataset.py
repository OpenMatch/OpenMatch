import csv
import logging
import os

from transformers import PreTrainedTokenizer

from ..arguments import DataArguments
from .inference_dataset import InferenceDataset

logger = logging.getLogger(__name__)


def load_beir_qrels(qrels_file):
    qrels = {}
    with open(qrels_file) as f:
        tsvreader = csv.DictReader(f, delimiter="\t")
        for row in tsvreader:
            qid = row["query-id"]
            pid = row["corpus-id"]
            rel = int(row["score"])
            if qid in qrels:
                qrels[qid][pid] = rel
            else:
                qrels[qid] = {pid: rel}
    return qrels


class BEIRDataset():

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        data_args: DataArguments, 
        full_tokenization: bool = True, 
        mode: str = "processed",
        stream: bool = True,
        batch_size: int = 1,
        num_processes: int = 1,
        process_index: int = 0,
        cache_dir: str = None
    ):
        logger.info("Loading corpus")
        self.corpus_dataset = InferenceDataset.load(
            tokenizer=tokenizer,
            data_args=data_args,
            data_files=os.path.join(data_args.data_dir, "corpus.jsonl"),
            is_query=False,
            full_tokenization=full_tokenization,
            mode=mode,
            stream=stream,
            batch_size=batch_size,
            num_processes=num_processes,
            process_index=process_index,
            cache_dir=cache_dir
        )

        split_names = ["train", "dev", "test"]
        self.query_datasets = {}
        self.qrels = {}
        for split_name in split_names:
            qrels_path = os.path.join(data_args.data_dir, "qrels", f"{split_name}.tsv")
            if os.path.exists(qrels_path):
                logger.info(f"Loading {split_name} queries and qrels")
                qrels = load_beir_qrels(qrels_path)
                self.qrels[split_name] = qrels
                qids = list(qrels.keys())
                self.query_datasets[split_name] = InferenceDataset.load(
                    tokenizer=tokenizer,
                    data_args=data_args,
                    data_files=os.path.join(data_args.data_dir, "queries.jsonl"),
                    is_query=True,
                    full_tokenization=full_tokenization,
                    mode=mode,
                    stream=stream,
                    batch_size=batch_size,
                    num_processes=num_processes,
                    process_index=process_index,
                    filter_fn=lambda x: x["_id"] in qids,
                    cache_dir=cache_dir
                )
            else:
                logger.info(f"{split_name} queries and qrels not found")
                self.query_datasets[split_name] = None
                self.qrels[split_name] = None