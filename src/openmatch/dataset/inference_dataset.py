# Adapted from Tevatron (https://github.com/texttron/tevatron)

import os
from functools import lru_cache
from typing import Callable, List, Union, Dict

import datasets
from datasets import load_dataset, Dataset
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoProcessor, PreTrainedTokenizer, ProcessorMixin

from ..arguments import DataArguments
from ..utils import fill_template, find_all_markers


def get_idx(obj):
    example_id = obj.get("_id", None)
    if example_id is None: 
        example_id = obj.get("id", None)
    if example_id is None:
        example_id = obj.get("text_id", None)
    if example_id is None:
        example_id = obj.get("sample_id", None)
    if example_id is None:
        raise ValueError("No id field found in data, tried `_id`, `id`, `text_id`")
    example_id = str(example_id) if example_id is not None else None
    return example_id


class InferenceDataset():

    def __init__(
        self, 
        data_files: Union[str, List[str]],
        data: List[Dict] = None,
        max_len: int = 128,
        template: str = None,
        column_names: str = None,
        all_markers: str = None,
        tokenizer: PreTrainedTokenizer = None, 
        processor: ProcessorMixin = None,
        is_query: bool = False, 
        full_tokenization: bool = True, 
        mode: str = "processed",
        batch_size: int = 1,
        num_processes: int = 1,
        process_index: int = 0,
        filter_fn: Callable = lambda x: True,
        cache_dir: str = None
    ):
        self.cache_dir = cache_dir
        self.is_query = is_query
        self.data_files = data_files
        self.data = data
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = max_len
        self.is_image = False

        self.template = template
        self.column_names = column_names
        
        self.full_tokenization = full_tokenization
        modes = ["raw", "dict_processed", "processed"]
        if mode not in modes:
            raise ValueError(f"mode must be one of {modes}")
        self.mode = mode

        self.batch_size = batch_size
        self.num_processes = num_processes
        self.process_index = process_index

        self.filter_fn = filter_fn

        self._prepare_data()

        if not self.is_image:
            if self.template is None:
                self.all_markers = None
            else:
                self.all_markers = find_all_markers(self.template) if all_markers is None else all_markers.split(",")

    def _prepare_data(self):
        raise NotImplementedError

    @classmethod
    def load(
        cls, 
        data_args: DataArguments = None, 
        data: List[Dict] = None,
        data_files: Union[str, List[str]] = None,
        max_len: int = 128,
        template: str = None,
        column_names: str = None,
        all_markers: str = None,
        tokenizer: PreTrainedTokenizer = None, 
        processor: ProcessorMixin = None,
        is_query: bool = False, 
        full_tokenization: bool = True, 
        mode: str = "processed",
        stream: bool = True,
        batch_size: int = 1,
        num_processes: int = 1,
        process_index: int = 0,
        filter_fn: Callable = lambda x: True,
        cache_dir: str = None
    ):
        max_len = max_len if max_len is not None else data_args.q_max_len if is_query else data_args.p_max_len
        template = template if template is not None else data_args.query_template if is_query else data_args.doc_template
        column_names = column_names if column_names is not None else data_args.query_column_names if is_query else data_args.doc_column_names
        all_markers = all_markers if all_markers is not None else data_args.all_markers
        if data is not None:
            return StreamInMemoryDataset(
                tokenizer=tokenizer, 
                processor=processor,
                data_files=data_files,
                max_len=max_len,
                template=template,
                all_markers=all_markers,
                column_names=column_names,
                is_query=is_query, 
                full_tokenization=full_tokenization, 
                mode=mode,
                batch_size=batch_size,
                num_processes=num_processes,
                process_index=process_index,
                filter_fn=filter_fn,
                cache_dir=cache_dir
            )
        if data_files is not None:
            data_files = [data_files] if isinstance(data_files, str) else data_files
        else:
            data_files = [data_args.query_path] if is_query else [data_args.corpus_path]
        ext = os.path.splitext(data_files[0])[1]
        ext_to_cls = {
            ".jsonl": StreamJsonlDataset if stream else MappingJsonlDataset,
            ".tsv": StreamTsvDataset if stream else MappingTsvDataset,
            ".txt": StreamTsvDataset if stream else MappingTsvDataset,
        }
        cls_ = ext_to_cls.get(ext, None) if ext != "" else StreamImageDataset
        if cls_ is None:
            raise ValueError("Unsupported dataset file extension {}".format(ext))
        return cls_(
            tokenizer=tokenizer, 
            processor=processor,
            data_files=data_files,
            max_len=max_len,
            template=template,
            all_markers=all_markers,
            column_names=column_names,
            is_query=is_query, 
            full_tokenization=full_tokenization, 
            mode=mode,
            batch_size=batch_size,
            num_processes=num_processes,
            process_index=process_index,
            filter_fn=filter_fn,
            cache_dir=cache_dir
        )

    def _tokenize(self, example: str):
        return self.tokenizer(
            example, 
            add_special_tokens=self.full_tokenization, 
            padding='max_length' if self.full_tokenization else False, 
            truncation=True, 
            max_length=self.max_len, 
            return_attention_mask=self.full_tokenization, 
            return_token_type_ids=False
        )

    def process_one(self, example):
        if self.is_image:
            path = example["image"]["path"]
            img = Image.open(path)
            processed = self.processor(images=img)
            name = os.path.basename(path).split(".")[0]
            return {"text_id": name, "pixel_values": processed["pixel_values"][0]}
        elif self.mode == "raw":
            return example
        elif self.mode == "dict_processed":
            example_id = get_idx(example)
            tokenized = {}
            for marker in self.all_markers:
                tokenized[marker] = dict(self._tokenize(example[marker])) if (marker in example and example[marker] is not None) else None
            return {"text_id": example_id, **tokenized}
        else:
            example_id = get_idx(example)
            full_text = fill_template(self.template, example, self.all_markers, allow_not_found=True)
            tokenized = self._tokenize(full_text)
            return {"text_id": example_id, **tokenized}


class StreamInferenceDataset(IterableDataset):

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


class MappingInferenceDataset(Dataset):

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        return self.process_one(self.dataset[index])

    def get_raw(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class StreamJsonlDataset(StreamInferenceDataset, InferenceDataset):

    def _prepare_data(self):
        self.dataset = load_dataset(
            "json", 
            data_files=self.data_files, 
            streaming=True, 
            cache_dir=self.cache_dir
        )["train"].filter(self.filter_fn)
        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()


class MappingJsonlDataset(MappingInferenceDataset, InferenceDataset):
    
    def _prepare_data(self):
        hf_dataset = load_dataset(
            "json", 
            data_files=self.data_files, 
            streaming=True, 
            cache_dir=self.cache_dir
        )["train"].filter(self.filter_fn)
        sample = list(hf_dataset.take(1))[0]
        self.all_columns = sample.keys()
        self.dataset = {}
        for item in hf_dataset:
            self.dataset[get_idx(item)] = item


class StreamTsvDataset(StreamInferenceDataset, InferenceDataset):

    def _prepare_data(self):
        self.all_columns = self.column_names
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(',')
        self.dataset = load_dataset(
            "csv", 
            data_files=self.data_files, 
            streaming=True, 
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=self.cache_dir
        )["train"].filter(self.filter_fn)


class MappingTsvDataset(MappingInferenceDataset, InferenceDataset):
    
    def _prepare_data(self):
        self.all_columns = self.column_names
        if self.all_columns is not None:
            self.all_columns = self.all_columns.split(',')
        hf_dataset = load_dataset(
            "csv",
            data_files=self.data_files,
            streaming=True,
            column_names=self.all_columns,
            delimiter='\t',
            cache_dir=self.cache_dir
        )["train"].filter(self.filter_fn)
        self.dataset = {}
        for item in hf_dataset:
            self.dataset[get_idx(item)] = item


class StreamImageDataset(StreamInferenceDataset, InferenceDataset):

    def _prepare_data(self):
        self.is_image = True
        self.dataset = load_dataset(
            self.data_files[0],
            split="train",
            streaming=True,
        )
        self.dataset = self.dataset.cast_column("image", datasets.Image(decode=False))


class StreamInMemoryDataset(StreamInferenceDataset, InferenceDataset):

    def _prepare_data(self):
        self.dataset = Dataset.from_list(self.data).filter(self.filter_fn)
        sample = self.dataset[0]
        self.all_columns = sample.keys()