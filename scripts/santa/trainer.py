import logging
import os
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union

import random
from transformers import BatchEncoding, PreTrainedTokenizer
from openmatch.trainer import DRTrainer
from openmatch.dataset.train_dataset import TrainDatasetBase, MappingTrainDatasetMixin,StreamTrainDatasetMixin


logger = logging.getLogger(__name__)

class STrainer(DRTrainer):
    def __init__(self, *args, **kwargs):
        super(STrainer, self).__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        query, passage, label = inputs
        outputs = model(query=query, passage=passage, label=label)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


class Sdataset(TrainDatasetBase):
    def __init__(self, santa_args,*args, **kwargs):
        super(Sdataset,self).__init__(*args, **kwargs)
        self.santa_args= santa_args

    def create_one_example(self, text_encoding: List[int], max_length):
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):

        def process_fn(example):
            qry = example['query']
            encoded_query = self.create_one_example(qry, self.data_args.q_max_len)
            encoded_passages = []
            group_positives = example['positives']
            group_negatives = example['negatives']

            if "labels" in example:
                label = example["labels"]
                encoder_label = self.create_one_example(label, self.santa_args.l_max_len)
            else:
                encoder_label = None

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
            encoded_passages.append(self.create_one_example(pos_psg,self.data_args.p_max_len))

            negative_size = self.data_args.train_n_passages - 1
            if len(group_negatives) < negative_size:
                if hashed_seed is not None:
                    negs = random.choices(group_negatives, k=negative_size)
                else:
                    negs = [x for x in group_negatives]
                    negs = negs * 2
                    negs = negs[:negative_size]
            elif self.data_args.train_n_passages == 1:
                negs = []
            elif self.data_args.negative_passage_no_shuffle:
                negs = group_negatives[:negative_size]
            else:
                _offset = epoch * negative_size % len(group_negatives)
                negs = [x for x in group_negatives]
                if hashed_seed is not None:
                    random.Random(hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset: _offset + negative_size]

            for neg_psg in negs:
                encoded_passages.append(self.create_one_example(neg_psg, self.data_args.p_max_len))

            assert len(encoded_passages) == self.data_args.train_n_passages

            return {"query": encoded_query, "passages": encoded_passages, "labels":encoder_label}

        return process_fn

class SStreamTrainDatasetMixin(StreamTrainDatasetMixin):
    def __iter__(self):
        if not self.is_eval:
            epoch = int(self.trainer.state.epoch)
            _hashed_seed = hash(self.trainer.args.seed)
            self.dataset.set_epoch(epoch)
            return iter(self.dataset.map(self.get_process_fn(epoch, _hashed_seed), remove_columns=["positives", "negatives"]))
        return iter(self.dataset.map(self.get_process_fn(0, None), remove_columns=self.all_columns))

class MappingDRTrainDataset(SStreamTrainDatasetMixin, Sdataset):
    pass


