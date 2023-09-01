import logging
import os
from contextlib import nullcontext
from typing import Dict, List, Tuple
import random

import torch
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, T5ForConditionalGeneration
from transformers.trainer_pt_utils import IterableDatasetShard

from ..arguments import DataArguments, InferenceArguments as EncodingArguments
from ..dataset import InferenceDataset, CQGInferenceCollator
from ..modeling import RRModel
from ..utils import (load_from_trec, merge_retrieval_results_by_score,
                     save_as_trec)

logger = logging.getLogger(__name__)


def encode_pair(tokenizer, item1, item2, max_len_1=32, max_len_2=128):
    return tokenizer.encode_plus(
        item1 + item2,
        truncation='longest_first',
        padding='max_length',
        max_length=max_len_1 + max_len_2,
    )


class CQGPredictDataset(IterableDataset):

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        corpus_dataset_pos: InferenceDataset, 
        corpus_dataset_neg: InferenceDataset,
        run: Dict[str, List[Tuple[str, float]]],
        qrels: Dict[str, List[str]] = None
    ):
        super(CQGPredictDataset, self).__init__()
        self.tokenizer = tokenizer
        self.corpus_dataset_pos = corpus_dataset_pos
        self.corpus_dataset_neg = corpus_dataset_neg
        self.run = run
        self.qrels = qrels

    def __iter__(self):
        for qid, did_and_scores in self.run.items():
            doc_ids = [did for did, _ in did_and_scores]
            if self.qrels is not None and qid in self.qrels:
                pos_doc_id = random.choice(self.qrels[qid])
                neg_doc_id = random.choice(doc_ids)
            else:
                pos_doc_id = random.choice(doc_ids)
                neg_doc_id = random.choice(doc_ids.remove(pos_doc_id))
            yield {
                "query_id": qid,
                "pos_doc_id": pos_doc_id,
                "neg_doc_id": neg_doc_id, 
                **encode_pair(
                    self.tokenizer, 
                    self.corpus_dataset_pos[pos_doc_id]["input_ids"], 
                    self.corpus_dataset_neg[neg_doc_id]["input_ids"],
                    self.corpus_dataset_pos.max_len,
                    self.corpus_dataset_neg.max_len
                ),
            }


class ContrastiveQueryGenerator:

    def __init__(
        self, 
        model: T5ForConditionalGeneration, 
        tokenizer: PreTrainedTokenizer, 
        corpus_dataset_pos: Dataset, 
        corpus_dataset_neg: Dataset,
        args: EncodingArguments,
        data_args: DataArguments,
        qrels: Dict[str, List[str]] = None
    ):
        logger.info("Initializing reranker")
        self.model = model
        self.tokenizer = tokenizer
        self.corpus_dataset_pos = corpus_dataset_pos
        self.corpus_dataset_neg = corpus_dataset_neg
        self.args = args
        self.data_args = data_args
        self.qrels = qrels

        self.model = model.to(self.args.device)
        self.model.eval()

    def generate(self, run: Dict[str, List[Tuple[str, float]]]):
        dataset = CQGPredictDataset(self.tokenizer, self.corpus_dataset_pos, self.corpus_dataset_neg, run, self.qrels)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=CQGInferenceCollator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        qids = []
        pos_docids = []
        neg_docids = []
        generated_queries = []
        for (batch_ids_q, batch_ids_posd, batch_ids_negd, batch) in tqdm(dataloader, disable=self.args.process_index > 0):
            qids.extend(batch_ids_q)
            pos_docids.extend(batch_ids_posd)
            neg_docids.extend(batch_ids_negd)
            with amp.autocast() if self.args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.args.device)
                    outputs = self.model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        num_beams=self.args.generation_num_beams,
                        do_sample=self.args.do_sample,
                        top_k=self.args.top_k,
                        top_p=self.args.top_p,
                        max_new_tokens=self.data_args.q_max_len,
                        num_return_sequences=self.args.num_return_sequences,
                    )
                    # group outputs by doc
                    outputs = outputs.view(batch["input_ids"].shape[0], self.args.num_return_sequences, -1)
            for queries in outputs:
                generated_queries.append(self.tokenizer.batch_decode(queries, skip_special_tokens=True))
        
        return qids, pos_docids, neg_docids, generated_queries