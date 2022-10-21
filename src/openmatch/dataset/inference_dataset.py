# Adapted from Tevatron (https://github.com/texttron/tevatron)

import os
from functools import lru_cache

from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer

from ..arguments import DataArguments
from ..utils import fill_template, find_all_markers


def get_idx(obj):
    example_id = obj.get("_id", None) or obj.get("id", None)
    example_id = str(example_id) if example_id is not None else None
    return example_id


class InferenceDataset():

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        data_args: DataArguments, 
        is_query: bool = False, 
        final: bool = True, 
        batch_size: int = 1,
        num_processes: int = 1,
        process_index: int = 0,
        cache_dir: str = None
    ):
        self.cache_dir = cache_dir
        self.is_query = is_query
        self.data_files = [data_args.query_path] if self.is_query else [data_args.corpus_path]
        self.tokenizer = tokenizer
        self.max_len = data_args.q_max_len if self.is_query else data_args.p_max_len
        self.template = data_args.query_template if self.is_query else data_args.doc_template
        self.all_markers = find_all_markers(self.template)
        self.final = final

        self.batch_size = batch_size
        self.num_processes = num_processes
        self.process_index = process_index

    @classmethod
    def load(
        cls, 
        tokenizer: PreTrainedTokenizer, 
        data_args: DataArguments, 
        is_query: bool = False, 
        final: bool = True, 
        stream: bool = True,
        batch_size: int = 1,
        num_processes: int = 1,
        process_index: int = 0,
        cache_dir: str = None
    ):
        data_files = [data_args.query_path] if is_query else [data_args.corpus_path]
        ext = os.path.splitext(data_files[0])[1]
        ext_to_cls = {
            ".json": StreamJsonlDataset if stream else MappingJsonlDataset,
            ".tsv": StreamTsvDataset if stream else MappingTsvDataset,
            ".txt": StreamTsvDataset if stream else MappingTsvDataset,
        }
        cls_ = ext_to_cls.get(ext, None)
        if cls_ is None:
            raise ValueError("Unsupported dataset file extension {}".format(ext))
        return cls_(
            tokenizer=tokenizer, 
            data_args=data_args, 
            is_query=is_query, 
            final=final, 
            batch_size=batch_size,
            num_processes=num_processes,
            process_index=process_index,
            cache_dir=cache_dir
        )

    def process_one(self, example):
        example_id = get_idx(example)
        full_text = fill_template(self.template, example, self.all_markers, allow_not_found=True)
        tokenized = self.tokenizer(
            full_text, 
            add_special_tokens=self.final, 
            padding='max_length' if self.final else False, 
            truncation=True, 
            max_length=self.max_len, 
            return_attention_mask=self.final, 
            return_token_type_ids=False
        )
        return {"text_id": example_id, **tokenized}


class StreamInferenceDataset(InferenceDataset, IterableDataset):

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        data_args: DataArguments, 
        **kwargs
    ):
        super(StreamInferenceDataset, self).__init__(tokenizer, data_args, **kwargs)

    def __iter__(self):
        real_batch_size = self.batch_size * self.num_processes
        process_slice = range(self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size)

        current_batch = []
        for element in self.dataset:
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield self.process_one(current_batch[i])
                current_batch = []

        if len(current_batch) > 0:
            for i in process_slice:
                if i < len(current_batch):
                    yield self.process_one(current_batch[i])


class StreamJsonlDataset(StreamInferenceDataset):

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        **kwargs
    ):
        super(StreamJsonlDataset, self).__init__(tokenizer, data_args, **kwargs)
        self.dataset = load_dataset(
            "json", 
            data_files=self.data_files, 
            streaming=True, 
            cache_dir=self.cache_dir
        )["train"]
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()


class StreamTsvDataset(StreamInferenceDataset):

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        **kwargs
    ):
        super(StreamTsvDataset, self).__init__(tokenizer, data_args, **kwargs)
        self.all_columns = data_args.query_column_names if self.is_query else data_args.doc_column_names
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(',')
        self.dataset = load_dataset(
            "csv", 
            data_files=self.data_files, 
            streaming=True, 
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=self.cache_dir
        )["train"]


class MappingInferenceDataset(InferenceDataset, Dataset):

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        data_args: DataArguments, 
        **kwargs
    ):
        super(MappingInferenceDataset, self).__init__(tokenizer, data_args, **kwargs)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        return self.process_one(self.dataset[index])

    def get_raw(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class MappingJsonlDataset(MappingInferenceDataset):

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        **kwargs
    ):
        super(MappingJsonlDataset, self).__init__(tokenizer, data_args, **kwargs)
        hf_dataset = load_dataset(
            "json", 
            data_files=self.data_files, 
            streaming=True, 
            cache_dir=self.cache_dir
        )["train"]
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()
        self.dataset = {}
        for item in hf_dataset:
            self.dataset[get_idx(item)] = item


class MappingTsvDataset(MappingInferenceDataset):

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        **kwargs
    ):
        super(MappingTsvDataset, self).__init__(tokenizer, data_args, **kwargs)
        self.all_columns = data_args.query_column_names if self.is_query else data_args.doc_column_names
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(',')
        hf_dataset = load_dataset(
            "csv",
            data_files=self.data_files,
            streaming=True,
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=self.cache_dir
        )["train"]
        self.dataset = {}
        for item in hf_dataset:
            self.dataset[get_idx(item)] = item
