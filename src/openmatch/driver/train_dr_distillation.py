# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
import sys

from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import PairwiseDistillationCollator, StreamPairwiseDistillationTrainDataset, MappingPairwiseDistillationTrainDataset, ListwiseDistillationCollator, StreamListwiseDistillationTrainDataset, MappingListwiseDistillationTrainDataset
from openmatch.modeling import DRModel
from openmatch.trainer import DRTrainer as Trainer
from openmatch.trainer import GCDenseTrainer
from openmatch.utils import get_delta_model_class
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    training_args.distillation = True

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = DRModel.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if model_args.param_efficient_method:
        model_class = get_delta_model_class(model_args.param_efficient_method)
        delta_model = model_class(model)
        logger.info("Using param efficient method: %s",
                    model_args.param_efficient_method)

    if training_args.distil_mode == "pairwise":
        train_dataset_cls = MappingPairwiseDistillationTrainDataset if training_args.use_mapping_dataset else StreamPairwiseDistillationTrainDataset
    elif training_args.distil_mode == "listwise":
        train_dataset_cls = MappingListwiseDistillationTrainDataset if training_args.use_mapping_dataset else StreamListwiseDistillationTrainDataset
    else:
        raise ValueError(
            f"Invalid distillation mode: {training_args.distil_mode}")
    train_dataset = train_dataset_cls(
        tokenizer,
        data_args,
        shuffle_seed=training_args.seed,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir
    )
    eval_dataset = train_dataset_cls(
        tokenizer,
        data_args,
        is_eval=True,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir
    ) if data_args.eval_path is not None else None

    trainer_cls = GCDenseTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PairwiseDistillationCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ) if training_args.distil_mode == "pairwise" else ListwiseDistillationCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
        delta_model=delta_model if model_args.param_efficient_method else None
    )
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
