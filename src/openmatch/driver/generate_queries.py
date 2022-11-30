import logging
import os
import sys
from contextlib import nullcontext

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoConfig, AutoTokenizer, HfArgumentParser,
                          PreTrainedTokenizer, T5ForConditionalGeneration)

from openmatch.arguments import (DataArguments, ModelArguments,
                                 QGInferenceArguments)
from openmatch.dataset import DRInferenceCollator, InferenceDataset

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, QGInferenceArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        encoding_args: QGInferenceArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if encoding_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed inference: %s, 16-bits inference: %s",
        encoding_args.local_rank,
        encoding_args.device,
        encoding_args.n_gpu,
        bool(encoding_args.local_rank != -1),
        encoding_args.fp16,
    )
    logger.info("Encoding parameters %s", encoding_args)
    logger.info("MODEL parameters %s", model_args)

    num_labels = 1
    try:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
    except OSError:
        config = None
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    corpus_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        is_query=False,
        stream=True,
        batch_size=encoding_args.per_device_eval_batch_size,
        num_processes=encoding_args.world_size,
        process_index=encoding_args.process_index,
        cache_dir=model_args.cache_dir
    )

    model.to(encoding_args.device)
    model.eval()

    dataloader = DataLoader(
        corpus_dataset,
        batch_size=encoding_args.per_device_eval_batch_size,
        collate_fn=DRInferenceCollator(),
        num_workers=encoding_args.dataloader_num_workers,
        pin_memory=encoding_args.dataloader_pin_memory,
    )
    ids = []
    generated_queries = []
    for (batch_ids, batch) in tqdm(dataloader, disable=encoding_args.process_index > 0):
        ids.extend(batch_ids)
        with amp.autocast() if encoding_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(encoding_args.device)
                outputs = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_beams=encoding_args.generation_num_beams,
                    do_sample=encoding_args.do_sample,
                    top_k=encoding_args.top_k,
                    top_p=encoding_args.top_p,
                    max_new_tokens=data_args.q_max_len,
                    num_return_sequences=encoding_args.num_return_sequences,
                )
                # group outputs by doc
                outputs = outputs.view(batch["input_ids"].shape[0], encoding_args.num_return_sequences, -1)
        for queries in outputs:
            generated_queries.append(tokenizer.batch_decode(queries, skip_special_tokens=True))
    
    if encoding_args.world_size > 1:
        with open(encoding_args.queries_save_path + ".rank.{}".format(encoding_args.process_index), "w") as f_queries, \
             open(encoding_args.qrels_save_path + ".rank.{}".format(encoding_args.process_index), "w") as f_qrels:
            for docid, queries in zip(ids, generated_queries):
                for qid, q in enumerate(queries):
                    if len(q) < 10:
                        continue
                    f_queries.write(f"generated_q_{qid}_for_{docid}\t{q}\n")
                    f_qrels.write(f"generated_q_{qid}_for_{docid}\t0\t{docid}\t1\n")
        torch.distributed.barrier()
        if encoding_args.process_index == 0:
            ids, generated_queries = [], []
            for i in range(encoding_args.world_size):
                with open(encoding_args.queries_save_path + ".rank.{}".format(i), "r") as f:
                    for line in f:
                        id_, q = line.strip().split("\t")
                        ids.append(id_)
                        generated_queries.append(q)
    if encoding_args.process_index == 0:
        with open(encoding_args.queries_save_path, "w") as f_queries, \
             open(encoding_args.qrels_save_path, "w") as f_qrels:
            for docid, queries in zip(ids, generated_queries):
                for qid, q in enumerate(queries):
                    if len(q) < 10:
                        continue
                    f_queries.write(f"generated_q_{qid}_for_{docid}\t{q}\n")
                    f_qrels.write(f"generated_q_{qid}_for_{docid}\t0\t{docid}\t1\n")


if __name__ == '__main__':
    main()
