import copy
import importlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (AutoConfig, AutoModel, BatchEncoding,
                          PreTrainedModel, T5EncoderModel, T5ForConditionalGeneration)
from transformers.modeling_outputs import ModelOutput

from openmatch.arguments import DataArguments
from openmatch.arguments import DRTrainingArguments as TrainingArguments
from openmatch.arguments import ModelArguments
from santa_arguments import SantaArguments
from openmatch.utils import mean_pooling
from openmatch.modeling.linear import LinearHead

logger = logging.getLogger(__name__)


@dataclass
class DROutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class SModel(nn.Module):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            tied: bool = True,
            feature: str = "last_hidden_state",
            pooling: str = "first",
            head_q: nn.Module = None,
            head_p: nn.Module = None,
            normalize: bool = False,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            santa_args: SantaArguments = None,
    ):
        super().__init__()

        self.tied = tied
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.head_q = head_q
        self.head_p = head_p

        self.feature = feature
        self.pooling = pooling
        self.normalize = normalize

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args
        self.santa_args = santa_args

        if train_args is not None:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
            if train_args.negatives_x_device:
                if not dist.is_initialized():
                    raise ValueError('Distributed training has not been initialized for representation all gather.')
                self.process_rank = dist.get_rank()
                self.world_size = dist.get_world_size()

    def _get_config_dict(self):
        config = {
            "tied": self.tied,
            "plm_backbone": {
                "type": type(self.lm_q).__name__,
                "feature": self.feature,
            },
            "pooling": self.pooling,
            "linear_head": bool(self.head_q),
            "normalize": self.normalize,
        }
        return config

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
            label: Dict[str, Tensor] = None,

    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps, g_loss = self.encode_passage(passage, label)

        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))

        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )
        target = target * self.data_args.train_n_passages

        loss = self.loss_fn(scores, target)
        loss = loss + g_loss

        if self.training and self.train_args.negatives_x_device:
            loss = loss * self.world_size  # counter average weight reduction
        return DROutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps
        )

    def encode_q(self, items, model, head,):
        if items is None:
            return None, None
        items = BatchEncoding(items)
        decoder_input_ids = torch.zeros((items.input_ids.shape[0], 1), dtype=torch.long).to(items.input_ids.device)
        items_out = model(**items, decoder_input_ids=decoder_input_ids, output_hidden_states=True, return_dict=True)
        hidden = items_out.decoder_hidden_states[-1]
        reps = hidden[:, 0, :]
        if head is not None:
            reps = head(reps)  # D * d
        return hidden, reps

    def encode_p(self, items, model, head,labels):
        if items is None:
            return None, None
        items = BatchEncoding(items)
        g_loss=0
        decoder_input_ids = torch.zeros((items.input_ids.shape[0], 1), dtype=torch.long).to(items.input_ids.device)
        if self.santa_args.use_generate and not labels.equal(torch.tensor([0]).to(labels.device)):
            items_out = model(**items, output_hidden_states=True, return_dict=True, labels=labels)
            g_loss=items_out.loss
            hidden = items_out.decoder_hidden_states[-1]
            reps = hidden[:, 0, :]
        else:
            items_out = model(**items, decoder_input_ids=decoder_input_ids, output_hidden_states=True, return_dict=True)
            hidden = items_out.decoder_hidden_states[-1]
            reps = hidden[:, 0, :]
        if head is not None:
            reps = head(reps)  # D * d
        return hidden, reps, g_loss

    def encode_passage(self, psg, labels):
        return self.encode_p(psg, self.lm_p, self.head_p, labels)

    def encode_query(self, qry):
        return self.encode_q(qry, self.lm_q, self.head_q)

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            model_name_or_path: str = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            santa_args: SantaArguments = None,
            **hf_kwargs,
    ):
        model_name_or_path = model_name_or_path or model_args.model_name_or_path
        # load local
        config = None
        head_q = head_p = None
        if os.path.exists(os.path.join(model_name_or_path, "openmatch_config.json")):
            with open(os.path.join(model_name_or_path, "openmatch_config.json")) as f:
                config = json.load(f)

        if os.path.isdir(model_name_or_path) and config is not None:  # an OpenMatch model
            tied = config["tied"]
            if tied:
                logger.info(f'loading model weight from {model_name_or_path}')
                model_name = config["plm_backbone"]["type"]
                model_class = getattr(importlib.import_module("transformers"), model_name)
                lm_q = lm_p = model_class.from_pretrained(
                    model_name_or_path,
                    **hf_kwargs
                )
                if config["linear_head"]:
                    head_q = head_p = LinearHead.load(model_name_or_path)
            else:
                _qry_model_path = os.path.join(model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
                _qry_head_path = os.path.join(model_name_or_path, 'query_head')
                _psg_head_path = os.path.join(model_name_or_path, 'passage_head')

                logger.info(f'loading query model weight from {_qry_model_path}')
                model_name = config["plm_backbone"]["lm_q_type"]
                model_class = getattr(importlib.import_module("transformers"), model_name)
                if os.path.exists(os.path.join(_qry_model_path, "config.json")):
                    logger.info(f'loading query model config from {_qry_model_path}')
                    qry_model_config = AutoConfig.from_pretrained(_qry_model_path)
                    hf_kwargs["config"] = qry_model_config
                lm_q = model_class.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )

                logger.info(f'loading passage model weight from {_psg_model_path}')
                model_name = config["plm_backbone"]["lm_p_type"]
                model_class = getattr(importlib.import_module("transformers"), model_name)
                if os.path.exists(os.path.join(_psg_model_path, "config.json")):
                    logger.info(f'loading passage model config from {_psg_model_path}')
                    psg_model_config = AutoConfig.from_pretrained(_psg_model_path)
                    hf_kwargs["config"] = psg_model_config
                lm_p = model_class.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )

                if config["linear_head"]:
                    head_q = LinearHead.load(_qry_head_path)
                    head_p = LinearHead.load(_psg_head_path)
        else:  # a Huggingface model
            tied = not model_args.untie_encoder
            model_class = T5ForConditionalGeneration
            lm_q = model_class.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if not tied else lm_q
            if model_args.add_linear_head:
                head_q = LinearHead(model_args.projection_in_dim, model_args.projection_out_dim)
                head_p = copy.deepcopy(head_q) if not tied else head_q

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            tied=tied,
            feature=model_args.feature if config is None else config["plm_backbone"]["feature"],
            pooling=model_args.pooling if config is None else config["pooling"],
            head_q=head_q,
            head_p=head_p,
            normalize=model_args.normalize if config is None else config["normalize"],
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
            santa_args=santa_args,
        )
        return model

    def save(self, output_dir: str):
        if not self.tied:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
            if self.head_q is not None:
                self.head_q.save(os.path.join(output_dir, 'query_head'))
                self.head_p.save(os.path.join(output_dir, 'passage_head'))
        else:
            self.lm_q.save_pretrained(output_dir)
            if self.head_q is not None:
                self.head_q.save(output_dir)

        with open(os.path.join(output_dir, 'openmatch_config.json'), 'w') as f:
            json.dump(self._get_config_dict(), f, indent=4)


