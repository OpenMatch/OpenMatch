# Adapted from Tevatron (https://github.com/texttron/tevatron)

import glob
import os
import random
from typing import List

from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from ..arguments import DataArguments
from ..trainer import DRTrainer


class TrainDataset(IterableDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: DRTrainer = None, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(TrainDataset, self).__init__()
        self._prepare_data(data_args, shuffle_seed, cache_dir)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.trainer = trainer
        self.len = self.getLen()
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = None
        self.dataset = None
    # We only needs to prepare the len once, so we prepare the len here and store it
    def getLen(self):
        concat_filenames = " ".join(self.data_files)
        count = 0
        with os.popen("wc -l {}".format(concat_filenames)) as f:
            for line in f:
                lc, filename = line.strip().split()
                lc = int(lc)
                if filename != "total":
                    count += lc
        return count
    def __len__(self):
        return self.len
    def __iter__(self):
        raise NotImplementedError

def build_one_data(example,encode_fun,hashed_seed,epoch,data_args):
    qry = example['query']
    encoded_query = encode_fun(qry, is_query=True)
    encoded_passages = []
    group_positives = example['positives']
    group_negatives = example['negatives']
    if data_args.positive_passage_no_shuffle or hashed_seed is None:
        os_psg = group_positives[0]
    else:
        pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
    encoded_passages.append(encode_fun(pos_psg))
    negative_size = data_args.train_n_passages - 1
    if len(group_negatives) < negative_size:
        if hashed_seed is not None:
            negs = random.choices(group_negatives, k=negative_size)
        else:
            negs = [x for x in group_negatives]
            negs = negs * 2
            negs = negs[:negative_size]
    elif data_args.train_n_passages == 1:
        negs = []
    elif data_args.negative_passage_no_shuffle:
        negs = group_negatives[:negative_size]
    else:
        _offset = epoch * negative_size % len(group_negatives)
        negs = [x for x in group_negatives]
        if hashed_seed is not None:
            random.Random(hashed_seed).shuffle(negs)
        negs = negs * 2
        negs = negs[_offset: _offset + negative_size]

    for neg_psg in negs:
            encoded_passages.append(encode_fun(neg_psg))

    assert len(encoded_passages) == data_args.train_n_passages

    return {"query": encoded_query, "passages": encoded_passages}    

class DRTrainDataset(TrainDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: DRTrainer = None, shuffle_seed: int = None, cache_dir: str = None) -> None:
        
        super(DRTrainDataset, self).__init__(tokenizer, data_args, trainer, shuffle_seed, cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed, buffer_size=10_000) if shuffle_seed is not None else self.dataset

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        
        def process_fn(example):
            return build_one_data(example,self.create_one_example,hashed_seed,epoch,self.data_args)

        return process_fn

    def __iter__(self):
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(self.trainer.args.seed)
        self.dataset.set_epoch(epoch)
        return iter(self.dataset.map(self.get_process_fn(epoch, _hashed_seed), remove_columns=["positives", "negatives"]))

class DRMappingTrainDataset(TrainDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: DRTrainer = None,shuffle_seed = None, cache_dir: str = None) -> None:
        # No shuffle seed is needed for mapping datasets, but were keeped to maintain interface
        super(DRMappingTrainDataset, self).__init__(tokenizer, data_args, trainer, shuffle_seed, cache_dir)

        assert self.len == len(self.dataset)
    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]
    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tokenizer.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item
    def __getitem__(self, item) :
        group = self.dataset[item]
        epoch = int(self.trainer.state.epoch)
        
        _hashed_seed = hash(item + self.trainer.args.seed)
        return build_one_data(group,self.create_one_example,_hashed_seed,epoch,self.data_args)
    
    def __iter__(self):
        #__getitem__ defines the support of iteration by definition
        return self


class DREvalDataset(DRTrainDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str = None) -> None:
        super(DREvalDataset, self).__init__(tokenizer, data_args, None, cache_dir=cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]

    def __iter__(self):
        return iter(self.dataset.map(self.get_process_fn(0, None), remove_columns=["positives", "negatives"]))


class RRTrainDataset(TrainDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, trainer: DRTrainer = None, shuffle_seed: int = None, cache_dir: str = None) -> None:
        super(RRTrainDataset, self).__init__(tokenizer, data_args, trainer, shuffle_seed, cache_dir)
        self.neg_num = 1

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.train_path] if data_args.train_dir is None else glob.glob(os.path.join(data_args.train_dir, "*.jsonl"))
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.shuffle(seed=shuffle_seed, buffer_size=10_000) if shuffle_seed is not None else self.dataset

    def create_one_example(self, qry_encoding, psg_encoding):
        item = self.tokenizer.encode_plus(
            qry_encoding + psg_encoding,
            truncation='longest_first',
            max_length=self.data_args.q_max_len + self.data_args.p_max_len + 2,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def get_process_fn(self, epoch, hashed_seed):
        
        def process_fn(example):
            qry = example['query']
            group_positives = example['positives']
            group_negatives = example['negatives']

            if self.data_args.positive_passage_no_shuffle or hashed_seed is None:
                pos_psg = group_positives[0]
            else:
                pos_psg = group_positives[(hashed_seed + epoch) % len(group_positives)]
            encoded_pos_pair = self.create_one_example(qry, pos_psg)

            if hashed_seed is None:
                neg_psg = group_negatives[0]
            else:
                neg_psg = group_negatives[(hashed_seed + epoch) % len(group_negatives)]
            encoded_neg_pair = self.create_one_example(qry, neg_psg)
            return {"pos_pair": encoded_pos_pair, "neg_pair": encoded_neg_pair}

        return process_fn

    def __iter__(self):
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(self.trainer.args.seed)
        self.dataset.set_epoch(epoch)
        return iter(self.dataset.map(self.get_process_fn(epoch, _hashed_seed), remove_columns=["positives", "negatives"]))


class RREvalDataset(RRTrainDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str = None) -> None:
        super(RREvalDataset, self).__init__(tokenizer, data_args, None, cache_dir=cache_dir)

    def _prepare_data(self, data_args, shuffle_seed, cache_dir):
        self.data_files = [data_args.eval_path]
        self.dataset = load_dataset("json", data_files=self.data_files, streaming=True, cache_dir=cache_dir)["train"]

    def __iter__(self):
        return iter(self.dataset.map(self.get_process_fn(0, None), remove_columns=["positives", "negatives"]))